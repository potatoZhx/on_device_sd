import argparse
import csv
import json
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = "/data2/group_谈海生/lagin/models/Qwen3-30B-A3B-Base"
DEFAULT_DATA_FILE = "/data2/group_谈海生/lagin/data/mtbench101/mtbench101.jsonl"


class LayerExpertCache:
    def __init__(self, num_experts: int, cache_rate: float, initial_policy: str = "tail"):
        self.num_experts = num_experts
        self.cache_rate = cache_rate
        self.initial_policy = initial_policy
        self.cache_size = self._resolve_cache_size(cache_rate)
        self.cache = OrderedDict()
        self.hits = 0
        self.total = 0
        self.reset()

    def _resolve_cache_size(self, cache_rate: float) -> int:
        if cache_rate <= 0:
            return 0
        return min(self.num_experts, max(1, int(round(self.num_experts * cache_rate))))

    def set_cache_rate(self, cache_rate: float) -> None:
        self.cache_rate = cache_rate
        self.cache_size = self._resolve_cache_size(cache_rate)
        self.reset()

    def reset(self) -> None:
        self.cache.clear()
        self.hits = 0
        self.total = 0
        for idx in self._initial_indices():
            self.cache[idx] = True

    def _initial_indices(self) -> list[int]:
        if self.cache_size <= 0:
            return []
        if self.initial_policy == "head":
            return list(range(self.cache_size))
        if self.initial_policy == "even":
            if self.cache_size == 1:
                return [0]
            step = (self.num_experts - 1) / (self.cache_size - 1)
            return sorted({int(round(i * step)) for i in range(self.cache_size)})[: self.cache_size]
        if self.initial_policy == "random":
            return sorted(random.sample(range(self.num_experts), self.cache_size))
        return list(range(self.num_experts - self.cache_size, self.num_experts))

    def mask(self, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(self.num_experts, device=device, dtype=torch.float32)
        if self.cache:
            indices = torch.tensor(list(self.cache.keys()), device=device, dtype=torch.long)
            mask[indices] = 1.0
        return mask

    def update(self, selected_experts: Iterable[int]) -> None:
        selected = [int(x) for x in selected_experts]
        if not selected:
            return

        for expert_idx in selected:
            if expert_idx in self.cache:
                self.hits += 1
                self.cache.move_to_end(expert_idx)
            elif self.cache_size > 0:
                while len(self.cache) >= self.cache_size:
                    self.cache.popitem(last=False)
                self.cache[expert_idx] = True
            self.total += 1

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


class CachePriorQwenMoeWrapper(nn.Module):
    """Cache-prior routing wrapper for Qwen MoE blocks.

    In standard mode it reproduces the original top-k routing. In cache-prior
    mode it boosts cached experts and current top-j experts only for the active
    decode token. This keeps long prompt tokens from polluting the simulated
    cache state while still making the next-token logits depend on method M.
    """

    def __init__(
        self,
        original_block: nn.Module,
        layer_idx: int,
        cache_rate: float,
        lambda_val: float,
        top_j: int,
        initial_cache_policy: str,
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

        self.enabled = False
        self.lambda_val = lambda_val
        self.top_j = top_j
        self.cache = LayerExpertCache(self.num_experts, cache_rate, initial_cache_policy)
        self.register_buffer("avg_range", torch.tensor(0.0), persistent=False)
        self.register_buffer("step_count", torch.tensor(0.0), persistent=False)

    def set_cache_rate(self, cache_rate: float) -> None:
        self.cache.set_cache_rate(cache_rate)
        self.reset_stats()

    def reset_stats(self) -> None:
        self.cache.reset()
        self.avg_range.zero_()
        self.step_count.zero_()

    def _active_positions(self, batch_size: int, sequence_length: int, device: torch.device) -> torch.Tensor:
        offsets = torch.arange(batch_size, device=device, dtype=torch.long) * sequence_length
        return offsets + (sequence_length - 1)

    def _apply_cache_prior(self, router_logits: torch.Tensor, active_positions: torch.Tensor) -> torch.Tensor:
        if not self.enabled or active_positions.numel() == 0:
            return router_logits

        router_logits_final = router_logits.clone()
        active_logits = router_logits[active_positions]

        with torch.no_grad():
            token_ranges = active_logits.max(dim=-1).values - active_logits.min(dim=-1).values
            current_avg = token_ranges.mean()
            self.step_count += 1.0
            if self.step_count.item() == 1.0:
                self.avg_range.copy_(current_avg)
            else:
                weight_new = 1.0 / self.step_count
                self.avg_range.copy_(self.avg_range * (1 - weight_new) + current_avg * weight_new)
            avg_range = float(self.avg_range.item())

        for pos in active_positions.tolist():
            current_logit = router_logits[pos]
            with torch.no_grad():
                priority_mask = self.cache.mask(current_logit.device)
                if self.top_j > 0:
                    _, top_j_indices = torch.topk(current_logit, min(self.top_j, self.num_experts))
                    priority_mask[top_j_indices] = 1.0
            boosted_logit = current_logit + self.lambda_val * avg_range * priority_mask
            router_logits_final[pos] = boosted_logit
            _, selected = torch.topk(boosted_logit, self.top_k)
            self.cache.update(selected.detach().cpu().tolist())

        return router_logits_final

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        router_logits_raw = self.gate(hidden_states_flat)

        active_positions = self._active_positions(batch_size, sequence_length, router_logits_raw.device)
        router_logits_final = self._apply_cache_prior(router_logits_raw, active_positions)

        routing_weights = F.softmax(router_logits_final, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0].item())
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states_flat[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states_flat)
            if self.shared_expert_gate is not None:
                shared_output = F.sigmoid(self.shared_expert_gate(hidden_states_flat)) * shared_output
            final_hidden_states += shared_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits_raw


def iter_cache_wrappers(model: nn.Module):
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, CachePriorQwenMoeWrapper):
            yield mlp


def apply_cache_prior_wrappers(
    model: nn.Module,
    cache_rate: float,
    lambda_val: float,
    top_j: int,
    initial_cache_policy: str,
) -> int:
    wrapped = 0
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, CachePriorQwenMoeWrapper):
            mlp.set_cache_rate(cache_rate)
            mlp.lambda_val = lambda_val
            mlp.top_j = top_j
            wrapped += 1
            continue
        if mlp is not None and all(hasattr(mlp, attr) for attr in ("gate", "experts", "num_experts", "top_k")):
            layer.mlp = CachePriorQwenMoeWrapper(
                mlp,
                layer_idx=layer_idx,
                cache_rate=cache_rate,
                lambda_val=lambda_val,
                top_j=top_j,
                initial_cache_policy=initial_cache_policy,
            )
            wrapped += 1
    return wrapped


def set_cache_prior_enabled(model: nn.Module, enabled: bool) -> None:
    for wrapper in iter_cache_wrappers(model):
        wrapper.enabled = enabled


def reset_cache_prior_state(model: nn.Module, cache_rate: float) -> None:
    for wrapper in iter_cache_wrappers(model):
        wrapper.set_cache_rate(cache_rate)


def cache_stats(model: nn.Module) -> tuple[int, int]:
    hits = 0
    total = 0
    for wrapper in iter_cache_wrappers(model):
        hits += wrapper.cache.hits
        total += wrapper.cache.total
    return hits, total


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def item_to_text(item: dict) -> str:
    history = item.get("history")
    if isinstance(history, list):
        parts = []
        for turn in history:
            if not isinstance(turn, dict):
                continue
            if "user" in turn:
                parts.append(f"User: {turn['user']}")
            if "bot" in turn:
                parts.append(f"Assistant: {turn['bot']}")
        if parts:
            return "\n".join(parts) + "\nAssistant:"

    conversations = item.get("conversations") or item.get("conversation")
    if isinstance(conversations, list):
        parts = []
        for turn in conversations:
            if isinstance(turn, dict):
                role = turn.get("role") or turn.get("from") or "user"
                value = turn.get("content") or turn.get("value") or ""
                parts.append(f"{role}: {value}")
        if parts:
            return "\n".join(parts)

    for key in ("prompt", "question", "instruction", "input"):
        if item.get(key):
            return str(item[key])
    return ""


def load_mtbench_samples(tokenizer, data_file: str, max_samples: int, max_prompt_tokens: int) -> list[torch.Tensor]:
    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            text = item_to_text(json.loads(line))
            if not text:
                continue
            input_ids = tokenizer(text, return_tensors="pt").input_ids
            if input_ids.size(1) > max_prompt_tokens:
                input_ids = input_ids[:, -max_prompt_tokens:]
            samples.append(input_ids)
            if max_samples > 0 and len(samples) >= max_samples:
                break
    return samples


def model_input_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.inference_mode()
def draft_decode(model: nn.Module, context_ids: torch.Tensor, draft_len: int) -> list[int]:
    set_cache_prior_enabled(model, True)
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
    set_cache_prior_enabled(model, False)
    draft_tensor = torch.tensor([draft_tokens], device=context_ids.device, dtype=context_ids.dtype)
    combined = torch.cat([context_ids, draft_tensor], dim=1)
    outputs = model(input_ids=combined, use_cache=False)
    start = context_ids.size(1) - 1
    end = start + len(draft_tokens)
    return outputs.logits[:, start:end, :].argmax(dim=-1).squeeze(0).tolist()


def prefix_match_len(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


POSITION_LIST_FIELDS = {
    "per_position_totals",
    "per_position_position_matches",
    "per_position_prefix_accepts",
    "per_position_position_match_rates",
    "per_position_prefix_acceptance_rates",
}


def update_position_acceptance_counters(
    position_totals: list[int],
    position_matches: list[int],
    position_prefix_accepts: list[int],
    draft_tokens: list[int],
    target_tokens: list[int],
    prefix_accepted: int,
) -> None:
    for pos, (draft_token, target_token) in enumerate(zip(draft_tokens, target_tokens)):
        position_totals[pos] += 1
        position_matches[pos] += int(draft_token == target_token)
        position_prefix_accepts[pos] += int(pos < prefix_accepted)


def position_acceptance_metrics(
    position_totals: list[int],
    position_matches: list[int],
    position_prefix_accepts: list[int],
) -> dict:
    position_match_rates = [
        position_matches[pos] / total if total else 0.0 for pos, total in enumerate(position_totals)
    ]
    prefix_acceptance_rates = [
        position_prefix_accepts[pos] / total if total else 0.0 for pos, total in enumerate(position_totals)
    ]
    return {
        "per_position_totals": position_totals,
        "per_position_position_matches": position_matches,
        "per_position_prefix_accepts": position_prefix_accepts,
        "per_position_position_match_rates": position_match_rates,
        "per_position_prefix_acceptance_rates": prefix_acceptance_rates,
    }


def flatten_position_metrics(row: dict, max_positions: int) -> dict:
    flat = {key: value for key, value in row.items() if key not in POSITION_LIST_FIELDS}
    totals = row.get("per_position_totals", [])
    position_matches = row.get("per_position_position_matches", [])
    prefix_accepts = row.get("per_position_prefix_accepts", [])
    position_match_rates = row.get("per_position_position_match_rates", [])
    prefix_acceptance_rates = row.get("per_position_prefix_acceptance_rates", [])

    for pos in range(max_positions):
        prefix = f"draft_pos_{pos + 1}"
        flat[f"{prefix}_total"] = totals[pos] if pos < len(totals) else 0
        flat[f"{prefix}_position_matches"] = position_matches[pos] if pos < len(position_matches) else 0
        flat[f"{prefix}_position_match_rate"] = (
            position_match_rates[pos] if pos < len(position_match_rates) else 0.0
        )
        flat[f"{prefix}_prefix_accepts"] = prefix_accepts[pos] if pos < len(prefix_accepts) else 0
        flat[f"{prefix}_prefix_acceptance_rate"] = (
            prefix_acceptance_rates[pos] if pos < len(prefix_acceptance_rates) else 0.0
        )
    return flat


@dataclass
class ConditionResult:
    cache_rate: float
    draft_len: int
    samples: int
    rounds: int
    drafted_tokens: int
    prefix_accepted_tokens: int
    position_matched_tokens: int
    fully_accepted_rounds: int
    generated_tokens: int
    cache_hit_rate: float
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
            "cache_hit_rate": self.cache_hit_rate,
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
) -> ConditionResult:
    start_time = time.time()
    rounds = 0
    drafted_total = 0
    prefix_accepted_total = 0
    position_matched_total = 0
    fully_accepted_rounds = 0
    generated_total = 0
    cache_hits_total = 0
    cache_queries_total = 0
    position_totals = [0 for _ in range(draft_len)]
    position_matched_counts = [0 for _ in range(draft_len)]
    position_prefix_accepted_counts = [0 for _ in range(draft_len)]

    for sample_idx, prompt_ids in enumerate(tqdm(samples, desc=f"cache={cache_rate:g}, draft={draft_len}", leave=False)):
        reset_cache_prior_state(model, cache_rate)
        context_ids = prompt_ids.to(device)
        generated = 0

        while generated < max_new_tokens:
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

            rounds += 1
            drafted_total += current_draft_len
            prefix_accepted_total += prefix_accepted
            position_matched_total += position_matches
            fully_accepted_rounds += int(prefix_accepted == current_draft_len)

            detail_file.write(
                json.dumps(
                    {
                        "sample_idx": sample_idx,
                        "round": rounds,
                        "cache_rate": cache_rate,
                        "draft_len": draft_len,
                        "context_len": int(context_ids.size(1)),
                        "draft_tokens": draft_tokens,
                        "target_tokens": target_tokens,
                        "prefix_accepted": prefix_accepted,
                        "position_matches": position_matches,
                        "position_match_flags": position_match_flags,
                        "position_prefix_accept_flags": position_prefix_accept_flags,
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

        generated_total += generated
        sample_cache_hits, sample_cache_queries = cache_stats(model)
        cache_hits_total += sample_cache_hits
        cache_queries_total += sample_cache_queries

    return ConditionResult(
        cache_rate=cache_rate,
        draft_len=draft_len,
        samples=len(samples),
        rounds=rounds,
        drafted_tokens=drafted_total,
        prefix_accepted_tokens=prefix_accepted_total,
        position_matched_tokens=position_matched_total,
        fully_accepted_rounds=fully_accepted_rounds,
        generated_tokens=generated_total,
        cache_hit_rate=cache_hits_total / cache_queries_total if cache_queries_total else 0.0,
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
        "cache_hit_rate",
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
        description="Compare cache-prior draft token acceptance rates on MTBench101."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--cache-rates", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--draft-lengths", default="1,2,4,8")
    parser.add_argument("--lambda-val", type=float, default=0.5)
    parser.add_argument("--top-j", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--initial-cache-policy", choices=["head", "tail", "even", "random"], default="tail")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return parser


def resolve_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


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

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"./comparison_experiments/results/cache_prior_acceptance_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    detail_path = os.path.join(output_dir, "round_details.jsonl")
    summary_jsonl_path = os.path.join(output_dir, "summary.jsonl")
    summary_csv_path = os.path.join(output_dir, "summary.csv")
    config_path = os.path.join(output_dir, "config.json")

    print(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model once and switching routing modes in-place: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=resolve_dtype(args.dtype),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    wrapped_layers = apply_cache_prior_wrappers(
        model,
        cache_rate=cache_rates[0],
        lambda_val=args.lambda_val,
        top_j=args.top_j,
        initial_cache_policy=args.initial_cache_policy,
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
            "primary_metric": "prefix_acceptance_rate",
            "secondary_metric": "position_match_rate",
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
                    "position_match={position_match_rate:.4f} "
                    "cache_hit={cache_hit_rate:.4f}".format(**row)
                )

    print("=" * 60)
    print("Experiment finished.")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"Summary CSV: {os.path.abspath(summary_csv_path)}")
    print(f"Summary JSONL: {os.path.abspath(summary_jsonl_path)}")
    print(f"Round details: {os.path.abspath(detail_path)}")


if __name__ == "__main__":
    main()
