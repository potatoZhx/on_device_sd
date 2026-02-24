import argparse
import time
import torch

from src.core.model import MoEConfig
from src.core.types import ExpertID, DeviceType
from src.execution.prefill_engine import PrefillEngine
from src.memory.expert_cache import ExpertCache
from src.memory.paged_kv_cache import PagedKVCache
from src.model.qwen3_runner import Qwen3ModelRunner
from src.scheduling.cache_strategy import LRUCacheStrategy
from src.scheduling.prefetcher import ExpertPrefetcher, SimplePrefetchStrategy
from src.utils.metrics import MetricsCollector


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
                params_cpu = {k: v.cpu() for k, v in params_gpu.items()}
                self.expert_params_gpu[expert_id] = params_gpu
                self.expert_params_cpu[expert_id] = params_cpu

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


def run_prefill(runner, loader, prefetcher, input_ids):
    cache = ExpertCache(
        max_cache_size_gb=0.01,
        expert_size_mb=0.001,
        replacement_strategy=LRUCacheStrategy(),
    )
    cache.transfer_stream = None
    kv_cache = PagedKVCache(
        config=loader.config,
        block_size=256,
        dtype=loader.config.get_dtype(),
    )
    engine = PrefillEngine(
        model_runner=runner,
        parameter_loader=loader,
        expert_cache=cache,
        prefetcher=prefetcher,
        metrics=MetricsCollector(),
    )
    engine.forward(input_ids, kv_cache, is_prefill=True)
    cache.complete_ready_transfers()
    return cache, engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--prefetch", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    config = make_config()
    loader = DummyParameterLoader(config)
    runner = Qwen3ModelRunner(config=config, parameter_loader=loader)
    input_ids = torch.randint(0, 100, (1, 6), device="cuda")
    prefetcher = ExpertPrefetcher(SimplePrefetchStrategy(num_experts_to_prefetch=2)) if args.prefetch else None

    torch.cuda.synchronize()
    start = time.perf_counter()
    cache = None
    for _ in range(args.iters):
        cache, _ = run_prefill(runner, loader, prefetcher, input_ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    cached = len(cache.cached_experts) if cache is not None else 0
    print(f"prefetch={args.prefetch} iters={args.iters} time={elapsed:.4f}s cached={cached}")


if __name__ == "__main__":
    main()
