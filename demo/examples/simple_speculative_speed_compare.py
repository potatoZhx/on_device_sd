"""
运行示例：
1) 复用 simple_inference 风格的单条/批量推理示例；
2) 同时演示 standard 与 speculative 两种模式；
3) 输出两种模式的推理速度对比（tokens/s 与 speedup）。

示例：
python examples/simple_speculative_speed_compare.py \
  --model-path /zx_data1/models/Qwen--Qwen3-30B-A3B-Base \
  --max-new-tokens 50 \
  --warmup 1
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

DEMO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DEMO_ROOT not in sys.path:
    sys.path.insert(0, DEMO_ROOT)

from src.core.model import MoEConfig
from src.core.types import GenerationConfig, InferenceMode, InferenceRequest
from src.execution.acceptance_strategy import StandardAcceptanceStrategy
from src.execution.orchestrator import EnhancedInferenceOrchestrator
from src.memory.expert_cache import ExpertCache
from src.memory.parameter_loader import ParameterLoader
from src.model import Qwen3ModelRunner
from src.scheduling.cache_strategy import LRUCacheStrategy
from src.scheduling.draft_schduler import SimpleDraftScheduler


PROMPTS = [
    "Once upon a time",
    "In a distant future, humans",
    "Write a short poem about the sea",
]


def _build_full_gpu_placement(config: MoEConfig) -> dict:
    gpu_experts = {
        layer_idx: list(range(config.num_experts))
        for layer_idx in range(config.num_hidden_layers)
    }
    return {"gpu_experts": gpu_experts}


def _build_orchestrator(
    config: MoEConfig,
    parameter_loader: ParameterLoader,
    model_runner: Qwen3ModelRunner,
) -> EnhancedInferenceOrchestrator:
    expert_size_mb = config.get_expert_weight_size_bytes() / (1024 * 1024)
    expert_cache = ExpertCache(
        max_cache_size_gb=1.0,
        expert_size_mb=expert_size_mb,
        replacement_strategy=LRUCacheStrategy(),
        pin_shared_experts=True,
    )

    return EnhancedInferenceOrchestrator(
        config=config,
        parameter_loader=parameter_loader,
        expert_cache=expert_cache,
        prefetcher=None,
        draft_scheduler=SimpleDraftScheduler(),
        acceptance_strategy=StandardAcceptanceStrategy(acceptance_threshold=0.0),
        model_runner=model_runner,
        default_mode=InferenceMode.SPECULATIVE,
    )


def _build_requests(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    generation_config: GenerationConfig,
) -> Tuple[List[InferenceRequest], InferenceRequest]:
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"].to(torch.long)
    attention_mask = tokenized["attention_mask"].to(torch.long)
    lengths = attention_mask.sum(dim=1).tolist()

    requests: List[InferenceRequest] = []
    for idx, prompt in enumerate(prompts):
        seq_len = int(lengths[idx])
        seq_ids = input_ids[idx, :seq_len].to(device="cuda")
        requests.append(
            InferenceRequest(
                request_id=f"batch_{idx}",
                input_ids=seq_ids,
                max_new_tokens=generation_config.max_new_tokens,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                generation_config=generation_config,
            )
        )

    single_request = InferenceRequest(
        request_id="single_0",
        input_ids=requests[0].input_ids.clone(),
        max_new_tokens=generation_config.max_new_tokens,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        top_k=generation_config.top_k,
        generation_config=generation_config,
    )
    return requests, single_request


def _run_mode(
    orchestrator: EnhancedInferenceOrchestrator,
    tokenizer: AutoTokenizer,
    mode: InferenceMode,
    single_request: InferenceRequest,
    batch_requests: List[InferenceRequest],
    warmup: int,
) -> Dict[str, float]:
    for _ in range(max(0, warmup)):
        _ = orchestrator.generate([single_request], mode=mode)
        torch.cuda.synchronize()

    # 单条请求（对齐 simple_inference 的 single 例子）
    start = time.perf_counter()
    single_out = orchestrator.generate([single_request], mode=mode)[0]
    torch.cuda.synchronize()
    single_time = time.perf_counter() - start

    single_ids = single_out.tolist()
    print(f"\n[{mode.value.upper()}][Single] Generated {len(single_ids)} tokens")
    print(f"[{mode.value.upper()}][Single] Output IDs: {single_ids}")
    print(
        f"[{mode.value.upper()}][Single] Decoded: "
        f"{tokenizer.decode(single_ids, skip_special_tokens=True)}"
    )

    # 批量请求（对齐 simple_inference 的 batch 例子）
    start = time.perf_counter()
    batch_out = orchestrator.generate(batch_requests, mode=mode)
    torch.cuda.synchronize()
    batch_time = time.perf_counter() - start

    batch_total_tokens = 0
    for idx, output_ids in enumerate(batch_out):
        ids_list = output_ids.tolist()
        batch_total_tokens += len(ids_list)
        print(f"[{mode.value.upper()}][Batch Prompt {idx}] Generated {len(ids_list)} tokens")
        print(f"[{mode.value.upper()}][Batch Prompt {idx}] Output IDs: {ids_list}")
        print(
            f"[{mode.value.upper()}][Batch Prompt {idx}] Decoded: "
            f"{tokenizer.decode(ids_list, skip_special_tokens=True)}"
        )

    single_tps = len(single_ids) / max(single_time, 1e-8)
    batch_tps = batch_total_tokens / max(batch_time, 1e-8)
    return {
        "single_tokens": float(len(single_ids)),
        "single_time": single_time,
        "single_tps": single_tps,
        "batch_tokens": float(batch_total_tokens),
        "batch_time": batch_time,
        "batch_tps": batch_tps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Standard vs Speculative speed comparison example")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/zx_data1/models/Qwen--Qwen3-30B-A3B-Base",
        help="HuggingFace model path",
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=1, help="warmup runs per mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exact-draft",
        action="store_true",
        help="Set draft_top_c=0 to disable CPU/substitution approximation in draft stage",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this example")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_path = args.model_path
    config = MoEConfig.from_pretrained(model_path)
    if args.exact_draft:
        config.draft_top_c = 0
    placement_config = _build_full_gpu_placement(config)

    parameter_loader = ParameterLoader(
        model_path=model_path,
        placement_config=placement_config,
    )
    parameter_loader.load_parameters()

    model_runner = Qwen3ModelRunner(config=config, parameter_loader=parameter_loader)
    orchestrator = _build_orchestrator(config, parameter_loader, model_runner)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    base_gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=False,
    )

    print("=" * 80)
    print("SIMPLE + SPECULATIVE EXAMPLE (CURRENT ARCHITECTURE)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    print(f"seed={args.seed}, exact_draft={args.exact_draft}, draft_top_c={config.draft_top_c}")

    standard_requests, standard_single = _build_requests(tokenizer, PROMPTS, base_gen_config)
    speculative_gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=False,
        use_speculative=True,
    )
    speculative_requests, speculative_single = _build_requests(tokenizer, PROMPTS, speculative_gen_config)

    print("\n" + "-" * 80)
    print("Running STANDARD mode")
    print("-" * 80)
    standard_stats = _run_mode(
        orchestrator=orchestrator,
        tokenizer=tokenizer,
        mode=InferenceMode.STANDARD,
        single_request=standard_single,
        batch_requests=standard_requests,
        warmup=args.warmup,
    )

    print("\n" + "-" * 80)
    print("Running SPECULATIVE mode")
    print("-" * 80)
    speculative_stats = _run_mode(
        orchestrator=orchestrator,
        tokenizer=tokenizer,
        mode=InferenceMode.SPECULATIVE,
        single_request=speculative_single,
        batch_requests=speculative_requests,
        warmup=args.warmup,
    )

    single_speedup = speculative_stats["single_tps"] / max(standard_stats["single_tps"], 1e-8)
    batch_speedup = speculative_stats["batch_tps"] / max(standard_stats["batch_tps"], 1e-8)

    print("\n" + "=" * 80)
    print("SPEED COMPARISON SUMMARY")
    print("=" * 80)
    print(
        "[Single] "
        f"standard={standard_stats['single_tps']:.2f} tok/s, "
        f"speculative={speculative_stats['single_tps']:.2f} tok/s, "
        f"speculative_speedup={single_speedup:.2f}x"
    )
    print(
        "[Batch]  "
        f"standard={standard_stats['batch_tps']:.2f} tok/s, "
        f"speculative={speculative_stats['batch_tps']:.2f} tok/s, "
        f"speculative_speedup={batch_speedup:.2f}x"
    )


if __name__ == "__main__":
    main()
