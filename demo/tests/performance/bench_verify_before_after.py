import argparse
import time
import torch

from src.core.model import MoEConfig
from src.core.types import ExpertID, DeviceType
from src.execution.continuous_batch_engine import CBExecutor
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument("--draft-len", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    cfg = make_config()
    loader = DummyParameterLoader(cfg)
    runner = Qwen3ModelRunner(config=cfg, parameter_loader=loader)
    expert_cache = ExpertCache(max_cache_size_gb=0.01, expert_size_mb=0.001, replacement_strategy=LRUCacheStrategy())

    prompt = torch.randint(0, 100, (args.prompt_len,), device="cuda", dtype=torch.long)
    draft = torch.randint(0, 100, (args.draft_len,), device="cuda", dtype=torch.long)

    # before: full prefill verify (recompute prompt+draft)
    kv_before = PagedKVCache(config=cfg, block_size=256, dtype=cfg.get_dtype())
    ex_before = CBExecutor(runner, kv_before, expert_cache, loader)
    seq_before = 0
    full_ids = torch.cat([prompt, draft])
    full_pos = torch.arange(full_ids.shape[0], device="cuda", dtype=torch.long)
    kv_before.add_sequence(seq_before, prompt_len=full_ids.shape[0])

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        ex_before._forward(full_ids, full_pos, [seq_before], is_prefill=True)
    torch.cuda.synchronize()
    before_t = time.perf_counter() - t0

    # after: verify reuse KV (only last token + draft)
    kv_after = PagedKVCache(config=cfg, block_size=256, dtype=cfg.get_dtype())
    ex_after = CBExecutor(runner, kv_after, expert_cache, loader)
    seq_after = 1
    kv_after.add_sequence(seq_after, prompt_len=args.prompt_len)
    kv_after.start_draft(seq_after)
    for _ in range(args.draft_len):
        kv_after.append_token(seq_after)

    verify_in = torch.cat([prompt[-1:].clone(), draft])
    verify_pos = torch.arange(args.prompt_len - 1, args.prompt_len + args.draft_len, device="cuda", dtype=torch.long)

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(args.iters):
        ex_after._forward_verify(verify_in, verify_pos, [seq_after])
    torch.cuda.synchronize()
    after_t = time.perf_counter() - t1

    print(f"verify_before_full_prefill={before_t:.4f}s")
    print(f"verify_after_reuse_kv={after_t:.4f}s")
    if after_t > 0:
        print(f"speedup={before_t / after_t:.2f}x")


if __name__ == "__main__":
    main()
