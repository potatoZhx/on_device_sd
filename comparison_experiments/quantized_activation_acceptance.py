import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_prior_acceptance import (
    DEFAULT_DATA_FILE,
    DEFAULT_MODEL_PATH,
    LayerExpertCache,
    flatten_position_metrics,
    load_mtbench_samples,
    model_input_device,
    parse_float_list,
    parse_int_list,
    position_acceptance_metrics,
    prefix_match_len,
    resolve_dtype,
    update_position_acceptance_counters,
)


DEFAULT_QUANTIZED_MODEL_DIR = "/data2/group_谈海生/lagin/models/Qwen3-30B-A3B-Base-experts-int4-g128"


def unpack_int4(qweight: torch.Tensor, in_features: int) -> torch.Tensor:
    low = qweight & 0x0F
    high = (qweight >> 4) & 0x0F
    unpacked = torch.stack((low, high), dim=-1).reshape(qweight.shape[0], -1)
    return unpacked[:, :in_features].to(torch.int8) - 8


class QuantizedLinearWeight:
    def __init__(self, payload: dict, bias: torch.Tensor | None, storage_device: str):
        self.qweight = payload["qweight"].to(storage_device)
        self.scales = payload["scales"].to(storage_device)
        self.in_features = int(payload["in_features"])
        self.out_features = int(payload["out_features"])
        self.group_size = int(payload["group_size"])
        self.bias = bias

    def dequantize(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        qweight = self.qweight.to(device, non_blocking=True)
        scales = self.scales.to(device, non_blocking=True).float()
        q = unpack_int4(qweight, self.in_features).float()
        expanded_scales = scales.repeat_interleave(self.group_size, dim=1)[:, : self.in_features]
        return (q * expanded_scales).to(dtype=dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize(x.device, x.dtype)
        bias = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


class QuantizedExpert(nn.Module):
    def __init__(self, original_expert: nn.Module, payload: dict, expert_idx: int, storage_device: str):
        super().__init__()
        self.act_fn = original_expert.act_fn
        self.gate_proj = QuantizedLinearWeight(
            payload[f"experts.{expert_idx}.gate_proj"],
            getattr(original_expert.gate_proj, "bias", None),
            storage_device,
        )
        self.up_proj = QuantizedLinearWeight(
            payload[f"experts.{expert_idx}.up_proj"],
            getattr(original_expert.up_proj, "bias", None),
            storage_device,
        )
        self.down_proj = QuantizedLinearWeight(
            payload[f"experts.{expert_idx}.down_proj"],
            getattr(original_expert.down_proj, "bias", None),
            storage_device,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def load_quantization_metadata(quantized_model_dir: str) -> dict:
    metadata_path = os.path.join(quantized_model_dir, "quantization_config.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Quantized draft model metadata not found: {metadata_path}. "
            "Submit comparison_experiments/run_quantize_qwen3_experts.sh first."
        )
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


class QuantizedActivationQwenMoeWrapper(nn.Module):
    """Qwen MoE wrapper for MoE-SpeQ-style INT4 expert drafting.

    Draft mode uses pre-generated symmetric INT4 routed expert weights
    (group_size=128 by default) while keeping router/gate, shared experts and
    non-expert blocks in full precision. Target mode reproduces the original
    full-precision experts and measures whether target experts were already
    present in the predicted cache.
    """

    def __init__(
        self,
        original_block: nn.Module,
        layer_idx: int,
        cache_rate: float,
        layer_payload: dict,
        initial_cache_policy: str,
        quantized_weight_device: str,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = original_block.num_experts
        self.top_k = original_block.top_k
        self.norm_topk_prob = original_block.norm_topk_prob
        self.gate = original_block.gate
        self.experts = original_block.experts
        self.shared_expert = getattr(original_block, "shared_expert", None)
        self.shared_expert_gate = getattr(original_block, "shared_expert_gate", None)

        self.draft_enabled = False
        self.track_target_cache = False
        self.tracked_positions = None
        self.quantized_experts = nn.ModuleList(
            [
                QuantizedExpert(expert, layer_payload, expert_idx, quantized_weight_device)
                for expert_idx, expert in enumerate(self.experts)
            ]
        )
        self.cache = LayerExpertCache(self.num_experts, cache_rate, initial_cache_policy)

        self.reset_counters()
        self.clear_round_traces()

    def reset_counters(self) -> None:
        self.draft_cache_hits = 0
        self.draft_cache_queries = 0
        self.target_cache_hits = 0
        self.target_cache_queries = 0

    def clear_round_traces(self) -> None:
        self.predicted_trace = []
        self.target_trace = []

    def set_cache_rate(self, cache_rate: float, reset_counters: bool) -> None:
        self.cache.set_cache_rate(cache_rate)
        if reset_counters:
            self.reset_counters()
        self.clear_round_traces()

    def _active_positions(self, batch_size: int, sequence_length: int, device: torch.device) -> torch.Tensor:
        offsets = torch.arange(batch_size, device=device, dtype=torch.long) * sequence_length
        return offsets + (sequence_length - 1)

    def _target_positions(self, total_tokens: int, device: torch.device) -> torch.Tensor:
        if self.tracked_positions is None:
            return torch.empty(0, device=device, dtype=torch.long)
        valid_positions = [pos for pos in self.tracked_positions if 0 <= pos < total_tokens]
        return torch.tensor(valid_positions, device=device, dtype=torch.long)

    def _observe_cache(self, selected_experts: list[int], phase: str) -> None:
        hits = sum(1 for expert_idx in selected_experts if expert_idx in self.cache.cache)
        if phase == "draft":
            self.draft_cache_hits += hits
            self.draft_cache_queries += len(selected_experts)
        else:
            self.target_cache_hits += hits
            self.target_cache_queries += len(selected_experts)
        self.cache.update(selected_experts)

    def _record_draft_predictions(self, selected_experts: torch.Tensor, active_positions: torch.Tensor) -> None:
        for pos in active_positions.tolist():
            experts = selected_experts[pos].detach().cpu().tolist()
            self.predicted_trace.append(experts)
            self._observe_cache(experts, phase="draft")

    def _record_target_usage(self, selected_experts: torch.Tensor, positions: torch.Tensor) -> None:
        for pos in positions.tolist():
            experts = selected_experts[pos].detach().cpu().tolist()
            self.target_trace.append(experts)
            self._observe_cache(experts, phase="target")

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states_flat)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        if self.draft_enabled:
            active_positions = self._active_positions(batch_size, sequence_length, router_logits.device)
            self._record_draft_predictions(selected_experts, active_positions)
        elif self.track_target_cache:
            target_positions = self._target_positions(router_logits.size(0), router_logits.device)
            self._record_target_usage(selected_experts, target_positions)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0].item())
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states_flat[None, top_x].reshape(-1, hidden_dim)
            expert = self.quantized_experts[expert_idx] if self.draft_enabled else self.experts[expert_idx]
            current_hidden_states = expert(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states_flat)
            if self.shared_expert_gate is not None:
                shared_output = F.sigmoid(self.shared_expert_gate(hidden_states_flat)) * shared_output
            final_hidden_states += shared_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


def iter_quantized_wrappers(model: nn.Module):
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, QuantizedActivationQwenMoeWrapper):
            yield mlp


def apply_quantized_activation_wrappers(
    model: nn.Module,
    quantized_model_dir: str,
    metadata: dict,
    cache_rate: float,
    initial_cache_policy: str,
    quantized_weight_device: str,
) -> int:
    wrapped = 0
    layer_metadata = {int(layer["layer_idx"]): layer for layer in metadata["layers"]}
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, QuantizedActivationQwenMoeWrapper):
            mlp.set_cache_rate(cache_rate, reset_counters=True)
            wrapped += 1
            continue
        if mlp is not None and all(hasattr(mlp, attr) for attr in ("gate", "experts", "num_experts", "top_k")):
            if layer_idx not in layer_metadata:
                raise RuntimeError(f"Quantized expert shard metadata missing for MoE layer {layer_idx}.")
            shard_path = os.path.join(quantized_model_dir, layer_metadata[layer_idx]["shard"])
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Quantized expert shard not found: {shard_path}")
            layer_payload = torch.load(shard_path, map_location="cpu")
            layer.mlp = QuantizedActivationQwenMoeWrapper(
                mlp,
                layer_idx=layer_idx,
                cache_rate=cache_rate,
                layer_payload=layer_payload,
                initial_cache_policy=initial_cache_policy,
                quantized_weight_device=quantized_weight_device,
            )
            wrapped += 1
            del layer_payload
    return wrapped


def reset_quantized_state(model: nn.Module, cache_rate: float) -> None:
    for wrapper in iter_quantized_wrappers(model):
        wrapper.set_cache_rate(cache_rate, reset_counters=True)


def set_draft_mode(model: nn.Module, enabled: bool) -> None:
    for wrapper in iter_quantized_wrappers(model):
        wrapper.draft_enabled = enabled


def set_target_tracking(model: nn.Module, enabled: bool, tracked_positions: list[int] | None = None) -> None:
    for wrapper in iter_quantized_wrappers(model):
        wrapper.track_target_cache = enabled
        wrapper.tracked_positions = tracked_positions


def clear_round_traces(model: nn.Module) -> None:
    for wrapper in iter_quantized_wrappers(model):
        wrapper.clear_round_traces()


def sample_cache_metrics(model: nn.Module) -> dict[str, int]:
    metrics = defaultdict(int)
    for wrapper in iter_quantized_wrappers(model):
        metrics["draft_cache_hits"] += wrapper.draft_cache_hits
        metrics["draft_cache_queries"] += wrapper.draft_cache_queries
        metrics["target_cache_hits"] += wrapper.target_cache_hits
        metrics["target_cache_queries"] += wrapper.target_cache_queries
    return dict(metrics)


def round_expert_fidelity(model: nn.Module) -> dict[str, int]:
    hard = 0
    soft = 0
    total = 0
    for wrapper in iter_quantized_wrappers(model):
        for predicted, target in zip(wrapper.predicted_trace, wrapper.target_trace):
            total += 1
            hard += int(predicted == target)
            soft += int(set(predicted) == set(target))
    return {"expert_hard_matches": hard, "expert_soft_matches": soft, "expert_match_total": total}


@torch.inference_mode()
def draft_decode(model: nn.Module, context_ids: torch.Tensor, draft_len: int) -> list[int]:
    set_target_tracking(model, False)
    set_draft_mode(model, True)
    draft_tokens = []
    running_context = context_ids
    for _ in range(draft_len):
        outputs = model(input_ids=running_context, use_cache=False)
        next_token = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
        draft_tokens.append(next_token)
        next_token_tensor = torch.tensor([[next_token]], device=running_context.device, dtype=running_context.dtype)
        running_context = torch.cat([running_context, next_token_tensor], dim=1)
    return draft_tokens


@torch.inference_mode()
def verify_with_target(model: nn.Module, context_ids: torch.Tensor, draft_tokens: list[int]) -> list[int]:
    set_draft_mode(model, False)
    start = context_ids.size(1) - 1
    tracked_positions = list(range(start, start + len(draft_tokens)))
    set_target_tracking(model, True, tracked_positions)
    draft_tensor = torch.tensor([draft_tokens], device=context_ids.device, dtype=context_ids.dtype)
    combined = torch.cat([context_ids, draft_tensor], dim=1)
    outputs = model(input_ids=combined, use_cache=False)
    set_target_tracking(model, False)
    end = start + len(draft_tokens)
    return outputs.logits[:, start:end, :].argmax(dim=-1).squeeze(0).tolist()


@dataclass
class QuantizedConditionResult:
    cache_rate: float
    draft_len: int
    samples: int
    rounds: int
    drafted_tokens: int
    prefix_accepted_tokens: int
    position_matched_tokens: int
    fully_accepted_rounds: int
    generated_tokens: int
    draft_cache_hits: int
    draft_cache_queries: int
    target_cache_hits: int
    target_cache_queries: int
    expert_hard_matches: int
    expert_soft_matches: int
    expert_match_total: int
    elapsed_sec: float
    position_totals: list[int]
    position_matched_counts: list[int]
    position_prefix_accepted_counts: list[int]

    def as_dict(self) -> dict:
        row = {
            "cache_rate": self.cache_rate,
            "draft_len": self.draft_len,
            "samples": self.samples,
            "rounds": self.rounds,
            "drafted_tokens": self.drafted_tokens,
            "prefix_accepted_tokens": self.prefix_accepted_tokens,
            "position_matched_tokens": self.position_matched_tokens,
            "prefix_acceptance_rate": self.prefix_accepted_tokens / self.drafted_tokens
            if self.drafted_tokens
            else 0.0,
            "position_match_rate": self.position_matched_tokens / self.drafted_tokens if self.drafted_tokens else 0.0,
            "full_round_acceptance_rate": self.fully_accepted_rounds / self.rounds if self.rounds else 0.0,
            "avg_prefix_accepted_per_round": self.prefix_accepted_tokens / self.rounds if self.rounds else 0.0,
            "generated_tokens": self.generated_tokens,
            "draft_cache_hit_rate": self.draft_cache_hits / self.draft_cache_queries
            if self.draft_cache_queries
            else 0.0,
            "target_cache_hit_rate": self.target_cache_hits / self.target_cache_queries
            if self.target_cache_queries
            else 0.0,
            "expert_hard_match_rate": self.expert_hard_matches / self.expert_match_total
            if self.expert_match_total
            else 0.0,
            "expert_soft_match_rate": self.expert_soft_matches / self.expert_match_total
            if self.expert_match_total
            else 0.0,
            "elapsed_sec": self.elapsed_sec,
        }
        row.update(
            position_acceptance_metrics(
                self.position_totals,
                self.position_matched_counts,
                self.position_prefix_accepted_counts,
            )
        )
        return row


def run_condition(
    model: nn.Module,
    samples: list[torch.Tensor],
    cache_rate: float,
    draft_len: int,
    max_new_tokens: int,
    device: torch.device,
    detail_file,
) -> QuantizedConditionResult:
    start_time = time.time()
    totals = defaultdict(int)
    position_totals = [0 for _ in range(draft_len)]
    position_matched_counts = [0 for _ in range(draft_len)]
    position_prefix_accepted_counts = [0 for _ in range(draft_len)]

    for sample_idx, prompt_ids in enumerate(tqdm(samples, desc=f"speq cache={cache_rate:g}, draft={draft_len}", leave=False)):
        reset_quantized_state(model, cache_rate)
        context_ids = prompt_ids.to(device)
        generated = 0

        while generated < max_new_tokens:
            clear_round_traces(model)
            current_draft_len = min(draft_len, max_new_tokens - generated)
            draft_tokens = draft_decode(model, context_ids, current_draft_len)
            target_tokens = verify_with_target(model, context_ids, draft_tokens)

            prefix_accepted = prefix_match_len(draft_tokens, target_tokens)
            position_matches = sum(int(a == b) for a, b in zip(draft_tokens, target_tokens))
            position_match_flags = [int(a == b) for a, b in zip(draft_tokens, target_tokens)]
            position_prefix_accept_flags = [int(pos < prefix_accepted) for pos in range(current_draft_len)]
            update_position_acceptance_counters(
                position_totals,
                position_matched_counts,
                position_prefix_accepted_counts,
                draft_tokens,
                target_tokens,
                prefix_accepted,
            )
            expert_metrics = round_expert_fidelity(model)

            totals["rounds"] += 1
            totals["drafted_tokens"] += current_draft_len
            totals["prefix_accepted_tokens"] += prefix_accepted
            totals["position_matched_tokens"] += position_matches
            totals["fully_accepted_rounds"] += int(prefix_accepted == current_draft_len)
            totals["expert_hard_matches"] += expert_metrics["expert_hard_matches"]
            totals["expert_soft_matches"] += expert_metrics["expert_soft_matches"]
            totals["expert_match_total"] += expert_metrics["expert_match_total"]

            detail_file.write(
                json.dumps(
                    {
                        "sample_idx": sample_idx,
                        "round": totals["rounds"],
                        "cache_rate": cache_rate,
                        "draft_len": draft_len,
                        "context_len": int(context_ids.size(1)),
                        "draft_tokens": draft_tokens,
                        "target_tokens": target_tokens,
                        "prefix_accepted": prefix_accepted,
                        "position_matches": position_matches,
                        "position_match_flags": position_match_flags,
                        "position_prefix_accept_flags": position_prefix_accept_flags,
                        **expert_metrics,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            detail_file.flush()

            if prefix_accepted == current_draft_len:
                append_tokens = draft_tokens
            else:
                append_tokens = draft_tokens[:prefix_accepted] + [target_tokens[prefix_accepted]]

            append_tensor = torch.tensor([append_tokens], device=device, dtype=context_ids.dtype)
            context_ids = torch.cat([context_ids, append_tensor], dim=1)
            generated += len(append_tokens)

        totals["generated_tokens"] += generated
        cache_metrics = sample_cache_metrics(model)
        for key, value in cache_metrics.items():
            totals[key] += value

    return QuantizedConditionResult(
        cache_rate=cache_rate,
        draft_len=draft_len,
        samples=len(samples),
        rounds=totals["rounds"],
        drafted_tokens=totals["drafted_tokens"],
        prefix_accepted_tokens=totals["prefix_accepted_tokens"],
        position_matched_tokens=totals["position_matched_tokens"],
        fully_accepted_rounds=totals["fully_accepted_rounds"],
        generated_tokens=totals["generated_tokens"],
        draft_cache_hits=totals["draft_cache_hits"],
        draft_cache_queries=totals["draft_cache_queries"],
        target_cache_hits=totals["target_cache_hits"],
        target_cache_queries=totals["target_cache_queries"],
        expert_hard_matches=totals["expert_hard_matches"],
        expert_soft_matches=totals["expert_soft_matches"],
        expert_match_total=totals["expert_match_total"],
        elapsed_sec=time.time() - start_time,
        position_totals=position_totals,
        position_matched_counts=position_matched_counts,
        position_prefix_accepted_counts=position_prefix_accepted_counts,
    )


def write_summary_csv(path: str, rows: list[dict]) -> None:
    base_fieldnames = [
        "cache_rate",
        "draft_len",
        "samples",
        "rounds",
        "drafted_tokens",
        "prefix_accepted_tokens",
        "position_matched_tokens",
        "prefix_acceptance_rate",
        "position_match_rate",
        "full_round_acceptance_rate",
        "avg_prefix_accepted_per_round",
        "generated_tokens",
        "draft_cache_hit_rate",
        "target_cache_hit_rate",
        "expert_hard_match_rate",
        "expert_soft_match_rate",
        "elapsed_sec",
    ]
    max_positions = max((len(row.get("per_position_totals", [])) for row in rows), default=0)
    position_fieldnames = []
    for pos in range(max_positions):
        prefix = f"draft_pos_{pos + 1}"
        position_fieldnames.extend(
            [
                f"{prefix}_total",
                f"{prefix}_position_matches",
                f"{prefix}_position_match_rate",
                f"{prefix}_prefix_accepts",
                f"{prefix}_prefix_acceptance_rate",
            ]
        )
    fieldnames = base_fieldnames + position_fieldnames
    flat_rows = [flatten_position_metrics(row, max_positions) for row in rows]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare MoE-SpeQ-style INT4 expert draft acceptance on MTBench101."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--quantized-model-dir", default=DEFAULT_QUANTIZED_MODEL_DIR)
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--cache-rates", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--draft-lengths", default="1,2,4,8")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--initial-cache-policy", choices=["head", "tail", "even", "random"], default="tail")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--quantized-weight-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Store packed INT4 draft expert weights on CPU to save memory, or CUDA for faster small tests.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache_rates = parse_float_list(args.cache_rates)
    draft_lengths = parse_int_list(args.draft_lengths)
    if any(rate < 0 or rate > 1 for rate in cache_rates):
        raise ValueError("--cache-rates must be in [0, 1].")
    if any(length <= 0 for length in draft_lengths):
        raise ValueError("--draft-lengths must be positive integers.")
    if args.quantized_weight_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--quantized-weight-device=cuda requires CUDA.")

    metadata = load_quantization_metadata(args.quantized_model_dir)
    quant_config = metadata.get("quantization", {})
    if int(quant_config.get("bits", 0)) != 4 or int(quant_config.get("group_size", 0)) != 128:
        raise ValueError(
            "Experiment 2 expects MoE-SpeQ paper ratio: INT4 routed expert weights with group_size=128. "
            f"Got quantization={quant_config!r}."
        )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"./comparison_experiments/results/speq_int4_acceptance_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    detail_path = os.path.join(output_dir, "round_details.jsonl")
    summary_jsonl_path = os.path.join(output_dir, "summary.jsonl")
    summary_csv_path = os.path.join(output_dir, "summary.csv")
    config_path = os.path.join(output_dir, "config.json")

    print(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model once and switching FP target / INT4 expert draft modes: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=resolve_dtype(args.dtype),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    wrapped_layers = apply_quantized_activation_wrappers(
        model,
        quantized_model_dir=args.quantized_model_dir,
        metadata=metadata,
        cache_rate=cache_rates[0],
        initial_cache_policy=args.initial_cache_policy,
        quantized_weight_device=args.quantized_weight_device,
    )
    if wrapped_layers == 0:
        raise RuntimeError("No MoE layers were wrapped. Check the model architecture.")
    print(f"Wrapped {wrapped_layers} MoE layers.")

    samples = load_mtbench_samples(tokenizer, args.data_file, args.max_samples, args.max_prompt_tokens)
    if not samples:
        raise RuntimeError(f"No MTBench101 samples loaded from {args.data_file}")
    print(f"Loaded {len(samples)} MTBench101 samples.")

    config = vars(args).copy()
    config.update(
        {
            "cache_rates": cache_rates,
            "draft_lengths": draft_lengths,
            "wrapped_layers": wrapped_layers,
            "method": "moe_speq_int4_expert_draft",
            "quantized_model_dir": os.path.abspath(args.quantized_model_dir),
            "quantization": quant_config,
            "primary_metric": "prefix_acceptance_rate",
            "expert_metrics": ["expert_hard_match_rate", "expert_soft_match_rate", "target_cache_hit_rate"],
        }
    )
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    rows = []
    device = model_input_device(model)
    with open(detail_path, "w", encoding="utf-8") as detail_file, open(
        summary_jsonl_path, "w", encoding="utf-8"
    ) as summary_file:
        for cache_rate in cache_rates:
            for draft_len in draft_lengths:
                result = run_condition(
                    model=model,
                    samples=samples,
                    cache_rate=cache_rate,
                    draft_len=draft_len,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                    detail_file=detail_file,
                )
                row = result.as_dict()
                rows.append(row)
                summary_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                summary_file.flush()
                write_summary_csv(summary_csv_path, rows)
                print(
                    "cache_rate={cache_rate:g} draft_len={draft_len} "
                    "prefix_acceptance={prefix_acceptance_rate:.4f} "
                    "target_cache_hit={target_cache_hit_rate:.4f} "
                    "expert_soft_match={expert_soft_match_rate:.4f}".format(**row)
                )

    print("=" * 60)
    print("Experiment finished.")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"Summary CSV: {os.path.abspath(summary_csv_path)}")
    print(f"Summary JSONL: {os.path.abspath(summary_jsonl_path)}")
    print(f"Round details: {os.path.abspath(detail_path)}")


if __name__ == "__main__":
    main()
