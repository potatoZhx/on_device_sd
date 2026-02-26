import argparse
import time
import torch

from src.core.model import MoEConfig
from src.core.types import ExpertID, DeviceType, InferenceRequest, BatchedRequest, GenerationConfig
from src.execution.continuous_batch_engine import ContinuousBatchEngine, DecodeMode
from src.execution.standard_engine import StandardDecodeEngine
from src.memory.expert_cache import ExpertCache
from src.memory.paged_kv_cache import PagedKVCache
from src.model.qwen3_runner import Qwen3ModelRunner
from src.scheduling.cache_strategy import LRUCacheStrategy


class DummyParameterLoader:
    def __init__(self, config: MoEConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.static_params_gpu = {}
        self.shared_expert_ids = set()
        self.shared_experts_gpu = {}
        self.expert_params_gpu = {}
        self.expert_params_cpu = {}
        self._init_params()

    def _rand(self, shape, device=None):
        device = device or self.device
        return torch.randn(*shape, device=device, dtype=self.config.get_dtype()) * 0.02

    def _init_params(self):
        cfg = self.config
        hidden = cfg.hidden_size
        q_size = cfg.num_attention_heads * cfg.head_dim
        kv_size = cfg.num_key_value_heads * cfg.head_dim
        moe_hidden = cfg.moe_intermediate_size

        self.static_params_gpu["embed_tokens"] = self._rand((cfg.vocab_size, hidden))
        self.static_params_gpu["lm_head"] = self._rand((cfg.vocab_size, hidden))
        self.static_params_gpu["final_layernorm"] = self._rand((hidden,))

        for layer_idx in range(cfg.num_hidden_layers):
            prefix = f"layer_{layer_idx}"
            self.static_params_gpu[f"{prefix}.input_layernorm"] = self._rand((hidden,))
            self.static_params_gpu[f"{prefix}.post_attention_layernorm"] = self._rand((hidden,))
            self.static_params_gpu[f"{prefix}.self_attn.q_proj"] = self._rand((q_size, hidden))
            self.static_params_gpu[f"{prefix}.self_attn.k_proj"] = self._rand((kv_size, hidden))
            self.static_params_gpu[f"{prefix}.self_attn.v_proj"] = self._rand((kv_size, hidden))
            self.static_params_gpu[f"{prefix}.self_attn.o_proj"] = self._rand((hidden, q_size))
            self.static_params_gpu[f"{prefix}.self_attn.q_norm"] = self._rand((cfg.head_dim,))
            self.static_params_gpu[f"{prefix}.self_attn.k_norm"] = self._rand((cfg.head_dim,))
            self.static_params_gpu[f"{prefix}.router"] = self._rand((cfg.num_experts, hidden))

            for expert_idx in range(cfg.num_experts):
                expert_id = ExpertID(layer_idx, expert_idx)
                params_gpu = {
                    "gate_proj": self._rand((moe_hidden, hidden)),
                    "up_proj": self._rand((moe_hidden, hidden)),
                    "down_proj": self._rand((hidden, moe_hidden)),
                }
                self.expert_params_gpu[expert_id] = params_gpu
                self.expert_params_cpu[expert_id] = {k: v.cpu() for k, v in params_gpu.items()}

    def is_shared_expert(self, expert_id: ExpertID) -> bool:
        return expert_id in self.shared_expert_ids

    def get_shared_expert_params(self, expert_id: ExpertID):
        return self.shared_experts_gpu.get(expert_id)

    def get_expert_params(self, expert_id: ExpertID, device: DeviceType = DeviceType.GPU):
        if device == DeviceType.CPU:
            return self.expert_params_cpu.get(expert_id)
        return self.expert_params_gpu.get(expert_id)


def make_config():
    return MoEConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=256,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=0,
        moe_intermediate_size=16,
        max_position_embeddings=128,
        rope_theta=10000.0,
        torch_dtype="float16",
        model_type="qwen3_moe",
        draft_top_c=1,
        max_draft_tokens=4,
    )


def bench_continuous(engine: ContinuousBatchEngine, prompts, max_new_tokens: int, iters: int):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        engine.generate(prompts, max_new_tokens=max_new_tokens, top_k=1, top_p=1.0)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def bench_legacy(standard_engine: StandardDecodeEngine, prompts, max_new_tokens: int, iters: int):
    request_tensors = [torch.tensor(p, dtype=torch.long, device="cuda") for p in prompts]
    max_len = max(len(x) for x in request_tensors)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        for i, input_ids in enumerate(request_tensors):
            req = InferenceRequest(
                request_id=f"legacy_{i}",
                input_ids=input_ids,
                generation_config=GenerationConfig(max_new_tokens=max_new_tokens, top_k=1, top_p=1.0, temperature=1.0),
            )
            batch = BatchedRequest(
                batch_id=f"legacy_batch_{i}",
                requests=[req],
                input_ids=input_ids.unsqueeze(0),
                attention_mask=torch.ones(1, input_ids.shape[0], dtype=torch.long, device="cuda"),
                position_ids=torch.arange(input_ids.shape[0], dtype=torch.long, device="cuda").unsqueeze(0),
                current_lengths=[input_ids.shape[0]],
                finished=[False],
                max_batch_seq_len=max_len,
            )
            standard_engine.generate_batch(batch)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    cfg = make_config()
    loader = DummyParameterLoader(cfg)
    runner = Qwen3ModelRunner(config=cfg, parameter_loader=loader)
    expert_cache = ExpertCache(max_cache_size_gb=0.01, expert_size_mb=0.001, replacement_strategy=LRUCacheStrategy())

    prompts = [torch.randint(0, 100, (args.prompt_len,), device="cuda").tolist() for _ in range(args.batch_size)]

    kv_cache = PagedKVCache(config=cfg, block_size=256, dtype=cfg.get_dtype())
    continuous = ContinuousBatchEngine(
        model_runner=runner,
        kv_cache=kv_cache,
        expert_cache=expert_cache,
        parameter_loader=loader,
        decode_mode=DecodeMode.STANDARD,
    )
    standard_engine = StandardDecodeEngine(
        model_runner=runner,
        parameter_loader=loader,
        expert_cache=expert_cache,
    )

    t_cont = bench_continuous(continuous, prompts, args.max_new_tokens, args.iters)
    t_legacy = bench_legacy(standard_engine, prompts, args.max_new_tokens, args.iters)

    total_tokens = args.batch_size * args.max_new_tokens * args.iters
    print(f"legacy_time={t_legacy:.4f}s legacy_throughput={total_tokens / t_legacy:.2f} tok/s")
    print(f"continuous_time={t_cont:.4f}s continuous_throughput={total_tokens / t_cont:.2f} tok/s")


if __name__ == "__main__":
    main()
