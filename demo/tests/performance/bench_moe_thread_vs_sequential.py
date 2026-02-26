import argparse
import time
import torch

from src.core.model import MoEConfig
from src.core.model_runner import RoutingResult, ExpertPlacement
from src.core.types import ExpertID, DeviceType
from src.layers.mlp import expert_forward_with_weights
from src.model.qwen3_runner import Qwen3ModelRunner


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
        num_experts_per_token=1,
        num_shared_experts=0,
        moe_intermediate_size=16,
        max_position_embeddings=128,
        rope_theta=10000.0,
        torch_dtype="float16",
        model_type="qwen3_moe",
        draft_top_c=1,
        max_draft_tokens=4,
    )


def sequential_execute(hidden_states: torch.Tensor, placement: ExpertPlacement) -> torch.Tensor:
    routing = placement.routing_result
    flat = hidden_states
    h = flat.shape[-1]
    topk_indices = routing.topk_indices
    topk_scores = routing.topk_scores
    output = torch.zeros(flat.shape[0], h, device=flat.device, dtype=flat.dtype)

    for expert_id in routing.activated_expert_ids:
        idx = expert_id.expert_idx
        mask = topk_indices == idx
        pairs = torch.where(mask)
        tok_idx = pairs[0]
        if tok_idx.numel() == 0:
            continue
        k_idx = pairs[1]
        weights = topk_scores[tok_idx, k_idx]
        params = placement.cpu_expert_params[idx]
        inp = flat[tok_idx].to("cpu")
        out_cpu = expert_forward_with_weights(inp, params["gate_proj"], params["up_proj"], params["down_proj"])
        output[tok_idx] += out_cpu.to(flat.device, dtype=flat.dtype) * weights.unsqueeze(1)

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--tokens", type=int, default=16)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    cfg = make_config()
    loader = DummyParameterLoader(cfg)
    runner = Qwen3ModelRunner(config=cfg, parameter_loader=loader)

    hidden = torch.randn(args.tokens, cfg.hidden_size, device="cuda", dtype=cfg.get_dtype())
    topk_idx = torch.tensor([[0] for _ in range(args.tokens)], device="cuda")
    topk_scores = torch.ones_like(topk_idx, dtype=cfg.get_dtype())
    routing = RoutingResult(
        layer_idx=0,
        topk_indices=topk_idx,
        topk_scores=topk_scores,
        activated_expert_ids={ExpertID(0, 0)},
    )
    placement = ExpertPlacement(
        gpu_expert_params={},
        cpu_expert_params={0: loader.get_expert_params(ExpertID(0, 0), device=DeviceType.CPU)},
        routing_result=routing,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        runner._execute_moe_with_placement(hidden, placement)
    torch.cuda.synchronize()
    threaded_t = time.perf_counter() - t0

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(args.iters):
        sequential_execute(hidden, placement)
    torch.cuda.synchronize()
    seq_t = time.perf_counter() - t1

    print(f"threaded_impl={threaded_t:.4f}s")
    print(f"sequential_baseline={seq_t:.4f}s")
    print("note=stream_parallel_not_implemented_in_current_runner")


if __name__ == "__main__":
    main()
