#!/usr/bin/env python3
"""Benchmark Qwen3 MoE block latency under different expert counts."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock


def _load_first_moe_block(model_path: str, dtype: torch.dtype) -> Qwen3MoeSparseMoeBlock:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    block = Qwen3MoeSparseMoeBlock(config)
    state_dict = block.state_dict()
    prefix = "model.layers.0.mlp."
    required = set(state_dict.keys())
    tensors: Dict[str, torch.Tensor] = {}

    shard_paths = sorted(Path(model_path).glob("model-*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No *.safetensors shards under {model_path}")

    for shard in shard_paths:
        with safe_open(shard, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if not key.startswith(prefix):
                    continue
                sub_key = key[len(prefix) :]
                if sub_key in required and sub_key not in tensors:
                    tensors[sub_key] = reader.get_tensor(key)
        if len(tensors) == len(required):
            break

    missing = required.difference(tensors.keys())
    if missing:
        raise RuntimeError(f"Missing tensors: {sorted(missing)[:5]} (total {len(missing)})")

    block.load_state_dict({k: tensors[k] for k in state_dict.keys()})
    block = block.to(dtype=dtype)
    block.eval()
    return block


@dataclass
class TrialResult:
    top_k: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    iters: int
    unique_experts: int


def _count_unique_experts(block: Qwen3MoeSparseMoeBlock, hidden_states: torch.Tensor) -> int:
    flat_states = hidden_states.view(-1, hidden_states.shape[-1])
    router_logits = block.gate(flat_states)
    routing_weights = torch.softmax(router_logits.float(), dim=1)
    _, selected_experts = torch.topk(routing_weights, block.top_k, dim=-1)
    return torch.unique(selected_experts).numel()


def _profile(block: Qwen3MoeSparseMoeBlock, hidden_states: torch.Tensor, iters: int, warmup: int) -> List[float]:
    device = hidden_states.device
    timings: List[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            block(hidden_states)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            block(hidden_states)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end = time.perf_counter()
            timings.append((end - start) * 1000.0)
    return timings


def run_benchmark(model_path: str, device: str, dtype: torch.dtype, batch_size: int,
                  seq_len: int, topk_values: List[int], warmup: int, iters: int) -> Dict[str, List[TrialResult]]:
    block = _load_first_moe_block(model_path, dtype=dtype)
    block = block.to(device)
    sample = torch.randn(batch_size, seq_len, block.gate.in_features, dtype=dtype, device=device)

    results: List[TrialResult] = []
    for topk in topk_values:
        block.top_k = topk
        timings = _profile(block, sample, iters=iters, warmup=warmup)
        times_tensor = torch.tensor(timings)
        unique_experts = _count_unique_experts(block, sample)
        results.append(
            TrialResult(
                top_k=topk,
                mean_ms=float(times_tensor.mean()),
                std_ms=float(times_tensor.std(unbiased=False)),
                min_ms=float(times_tensor.min()),
                max_ms=float(times_tensor.max()),
                median_ms=float(times_tensor.median()),
                iters=iters,
                unique_experts=int(unique_experts),
            )
        )
    return {"results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="MoE expert-count latency study")
    parser.add_argument("--model-path", default="/zx_data1/models/Qwen--Qwen3-30B-A3B-Base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output", default="moe_latency_results.json")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise EnvironmentError("CUDA requested but not available")

    torch.manual_seed(0)
    payload = run_benchmark(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        topk_values=args.topk,
        warmup=args.warmup,
        iters=args.iters,
    )

    output_path = Path(args.output)
    output_path.write_text(
        json.dumps({"results": [asdict(item) for item in payload["results"]]}, indent=2)
    )

    print("TopK | Unique experts | mean ms | std ms | min ms | max ms | median ms")
    for item in payload["results"]:
        print(
            f"{item.top_k:>4d} | {item.unique_experts:>14d} | "
            f"{item.mean_ms:7.3f} | {item.std_ms:6.3f} | {item.min_ms:6.3f} | "
            f"{item.max_ms:6.3f} | {item.median_ms:7.3f}"
        )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
