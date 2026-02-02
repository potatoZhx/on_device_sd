#!/usr/bin/env python3
"""Compare Hugging Face MoE reference block with Tutel fused MoE implementation."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import torch
from torch import nn
import torch.nn.functional as F

from moe_expert_latency import _count_unique_experts, _load_first_moe_block
from tutel import moe as tutel_moe


@dataclass
class BackendResult:
    backend: str
    seq_len: int
    top_k: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    runs: int
    unique_experts: int


def _profile_callable(
    fn: Callable[[torch.Tensor], torch.Tensor],
    sample: torch.Tensor,
    warmup_runs: int,
    measure_runs: int,
) -> List[float]:
    device = sample.device
    timings: List[float] = []
    with torch.no_grad():
        for _ in range(warmup_runs):
            fn(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(measure_runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if start is not None and end is not None:
                start.record()
                fn(sample)
                end.record()
                torch.cuda.synchronize(device)
                timings.append(start.elapsed_time(end))
            else:
                import time
                t0 = time.perf_counter()
                fn(sample)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000.0)
    return timings


def _maybe_empty_cache(device: str | torch.device) -> None:
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

class QwenTutelExpert(nn.Module):
    """Custom Tutel expert with Qwen-style gated MLP weights."""

    def __init__(self, model_dim: int, num_experts_per_device: int, sharded_count: int, intermediate_size: int):
        super().__init__()
        if sharded_count != 1:
            raise NotImplementedError("This benchmark assumes single-device Tutel experts")
        self.num_experts = num_experts_per_device
        self.model_dim = model_dim
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Parameter(torch.empty(self.num_experts, model_dim, intermediate_size))
        self.up_proj = nn.Parameter(torch.empty(self.num_experts, model_dim, intermediate_size))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, intermediate_size, model_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.gate_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, ctx) -> torch.Tensor:
        gate = torch.matmul(x, self.gate_proj)
        up = torch.matmul(x, self.up_proj)
        activated = F.silu(gate)
        fused = activated * up
        out = torch.matmul(fused, self.down_proj)
        return out


def _stack_weights(modules: Iterable[nn.Module], attr: str, transpose: bool = False) -> torch.Tensor:
    tensors = [getattr(module, attr).weight.detach().to(torch.float32) for module in modules]
    stacked = torch.stack(tensors, dim=0)
    if transpose:
        stacked = stacked.transpose(1, 2)
    return stacked


def _load_qwen_weights_into_tutel(block: nn.Module, layer: nn.Module) -> None:
    with torch.no_grad():
        gate_dst = layer.gates[0].wg.weight
        gate_src = block.gate.weight.detach().to(device=gate_dst.device, dtype=gate_dst.dtype)
        gate_dst.copy_(gate_src)

        expert_module = layer.experts
        device = expert_module.gate_proj.device
        dtype = expert_module.gate_proj.dtype

        gate_proj = _stack_weights(block.experts, "gate_proj", transpose=True).to(device=device, dtype=dtype)
        up_proj = _stack_weights(block.experts, "up_proj", transpose=True).to(device=device, dtype=dtype)
        down_proj = _stack_weights(block.experts, "down_proj", transpose=True).to(device=device, dtype=dtype)
        expert_module.gate_proj.copy_(gate_proj)
        expert_module.up_proj.copy_(up_proj)
        expert_module.down_proj.copy_(down_proj)


def _build_tutel_layer(block: nn.Module, hidden_size: int, expert_hidden_size: int, num_experts: int,
                       top_k: int, dtype: torch.dtype, device: torch.device) -> nn.Module:
    layer = tutel_moe.moe_layer(
        gate_type={"type": "top", "k": top_k, "fp32_gate": False, "capacity_factor": 0.0, "gate_noise": 1e-3},
        model_dim=hidden_size,
        batch_prioritized_routing=False,
        normalize_gate=getattr(block, "norm_topk_prob", True),
        is_gshard_loss=False,
        experts={
            "type": "custom",
            "module": QwenTutelExpert,
            "num_experts_per_device": num_experts,
            "intermediate_size": expert_hidden_size,
        },
    )
    layer = layer.to(device=device, dtype=dtype)
    _load_qwen_weights_into_tutel(block, layer)
    layer.eval()
    return layer


def _count_unique_tutel(layer: nn.Module, sample: torch.Tensor) -> int:
    gate = layer.gates[0]
    hidden_dim = sample.shape[-1]
    flat = sample.view(-1, hidden_dim)
    logits = gate(flat)
    scores = torch.softmax(logits.float(), dim=-1)
    _, selected = torch.topk(scores, k=layer.gates[0].top_k, dim=-1)
    return torch.unique(selected).numel()


def benchmark_backends(model_path: str, device: str, dtype: torch.dtype, batch_size: int,
                       seq_lens: Iterable[int], topk_values: Iterable[int], warmup_runs: int,
                       measure_runs: int, seed: int) -> Dict[str, List[BackendResult]]:
    torch.manual_seed(seed)
    block = _load_first_moe_block(model_path, dtype=dtype).to(device)
    config_hidden = block.gate.in_features
    expert_hidden = block.experts[0].gate_proj.out_features
    num_experts = len(block.experts)

    results: List[BackendResult] = []

    class BlockWrapper(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.block = wrapped

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)[0]

    hf_module = BlockWrapper(block)

    for seq_len in seq_lens:
        sample = torch.randn(batch_size, seq_len, config_hidden, dtype=dtype, device=device)
        for topk in topk_values:
            block.top_k = topk
            _maybe_empty_cache(device)
            timings = _profile_callable(hf_module, sample, warmup_runs=warmup_runs, measure_runs=measure_runs)
            times_tensor = torch.tensor(timings, device="cpu")
            unique_experts = _count_unique_experts(block, sample)
            results.append(
                BackendResult(
                    backend="hf_reference",
                    seq_len=seq_len,
                    top_k=topk,
                    mean_ms=float(times_tensor.mean()),
                    std_ms=float(times_tensor.std(unbiased=False)),
                    min_ms=float(times_tensor.min()),
                    max_ms=float(times_tensor.max()),
                    median_ms=float(times_tensor.median()),
                    runs=measure_runs,
                    unique_experts=int(unique_experts),
                )
            )

        for topk in topk_values:
            _maybe_empty_cache(device)
            torch.manual_seed(seed)
            fused_layer = _build_tutel_layer(
                block=block,
                hidden_size=config_hidden,
                expert_hidden_size=expert_hidden,
                num_experts=num_experts,
                top_k=topk,
                dtype=dtype,
                device=torch.device(device),
            )
            timings = _profile_callable(fused_layer, sample, warmup_runs=warmup_runs, measure_runs=measure_runs)
            times_tensor = torch.tensor(timings, device="cpu")
            unique_experts = _count_unique_tutel(fused_layer, sample)
            results.append(
                BackendResult(
                    backend="tutel_fused",
                    seq_len=seq_len,
                    top_k=topk,
                    mean_ms=float(times_tensor.mean()),
                    std_ms=float(times_tensor.std(unbiased=False)),
                    min_ms=float(times_tensor.min()),
                    max_ms=float(times_tensor.max()),
                    median_ms=float(times_tensor.median()),
                    runs=measure_runs,
                    unique_experts=int(unique_experts),
                )
            )
    return {
        "hidden_size": config_hidden,
        "expert_hidden_size": expert_hidden,
        "num_experts": num_experts,
        "seq_lens": list(seq_lens),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare sequential and fused MoE latency")
    parser.add_argument("--model-path", default="/zx_data1/models/Qwen--Qwen3-30B-A3B-Base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=None,
                        help="Sequence lengths to benchmark. Defaults to [1, 64, 512] if omitted.")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="(Deprecated) single sequence length. Use --seq-lens instead.")
    parser.add_argument("--measure-runs", type=int, default=3,
                        help="Number of measured runs per test (default: 3).")
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="Number of warmup runs before measurement (default: 1).")
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="moe_fused_results.json")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise EnvironmentError("CUDA requested but not available")

    if args.seq_lens is not None:
        seq_lens = args.seq_lens
    elif args.seq_len is not None:
        seq_lens = [args.seq_len]
    else:
        seq_lens = [1, 64, 512]

    seq_lens_clean: List[int] = []
    for value in seq_lens:
        if value < 1:
            raise ValueError("Sequence lengths must be >= 1")
        if value not in seq_lens_clean:
            seq_lens_clean.append(value)
    if 1 not in seq_lens_clean:
        seq_lens_clean.insert(0, 1)

    warmup_runs = max(1, args.warmup_runs)
    measure_runs = max(1, args.measure_runs)

    payload = benchmark_backends(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
        batch_size=args.batch_size,
        seq_lens=seq_lens_clean,
        topk_values=args.topk,
        warmup_runs=warmup_runs,
        measure_runs=measure_runs,
        seed=args.seed,
    )

    output = {
        "config": {
            "model_path": args.model_path,
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "seq_lens": seq_lens_clean,
            "warmup_runs": warmup_runs,
            "measure_runs": measure_runs,
            "seed": args.seed,
        },
        "stats": [asdict(item) for item in payload["results"]],
        "meta": {
            "hidden_size": payload["hidden_size"],
            "expert_hidden_size": payload["expert_hidden_size"],
            "num_experts": payload["num_experts"],
            "seq_lens": payload["seq_lens"],
        },
    }

    Path(args.output).write_text(json.dumps(output, indent=2))

    print("backend | seq | topK | mean ms | median ms | unique experts")
    for item in payload["results"]:
        print(
            f"{item.backend:>12s} | {item.seq_len:>4d} | {item.top_k:>4d} | "
            f"{item.mean_ms:7.3f} | {item.median_ms:7.3f} | {item.unique_experts:>5d}"
        )
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
