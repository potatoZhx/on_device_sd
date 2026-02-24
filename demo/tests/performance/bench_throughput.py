import argparse
import time
import torch

from src.core.model import MoEConfig
from src.core.types import ExpertID, DeviceType
from src.execution.acceptance_strategy import StandardAcceptanceStrategy
from src.execution.continuous_batch_engine import ContinuousBatchEngine, DecodeMode
from src.memory.expert_cache import ExpertCache
from src.memory.paged_kv_cache import PagedKVCache
from src.model.qwen3_runner import Qwen3ModelRunner
from src.scheduling.cache_strategy import LRUCacheStrategy
from src.scheduling.draft_schduler import SimpleDraftScheduler


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "speculative"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    config = make_config()
    loader = DummyParameterLoader(config)
    runner = Qwen3ModelRunner(config=config, parameter_loader=loader)
    kv_cache = PagedKVCache(config=config, block_size=256, dtype=config.get_dtype())
    expert_cache = ExpertCache(
        max_cache_size_gb=0.01,
        expert_size_mb=0.001,
        replacement_strategy=LRUCacheStrategy(),
    )
    decode_mode = DecodeMode.SPECULATIVE if args.mode == "speculative" else DecodeMode.STANDARD
    engine = ContinuousBatchEngine(
        model_runner=runner,
        kv_cache=kv_cache,
        expert_cache=expert_cache,
        parameter_loader=loader,
        decode_mode=decode_mode,
        draft_scheduler=SimpleDraftScheduler(),
        acceptance_strategy=StandardAcceptanceStrategy(acceptance_threshold=0.0),
        max_draft_tokens=2,
    )

    prompts = [
        torch.randint(0, 100, (args.prompt_len,), device="cuda").tolist()
        for _ in range(args.batch_size)
    ]

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        engine.generate(
            prompts,
            max_new_tokens=args.max_new_tokens,
            top_k=1,
            top_p=1.0,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    total_tokens = args.iters * args.batch_size * args.max_new_tokens
    throughput = total_tokens / elapsed if elapsed > 0 else 0.0
    print(f"mode={args.mode} tokens={total_tokens} time={elapsed:.4f}s throughput={throughput:.2f} tok/s")


if __name__ == "__main__":
    main()
