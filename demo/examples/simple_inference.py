"""
Simple inference example using StandardDecodeEngine (ModelRunner path).
Loads the full MoE model onto GPU.
"""

import os
import sys

from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoTokenizer

# Ensure demo root is on sys.path (so `src.*` imports work)
DEMO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DEMO_ROOT not in sys.path:
    sys.path.insert(0, DEMO_ROOT)

from src.core.model import MoEConfig
from src.core.types import BatchedRequest, GenerationConfig
from src.execution.standard_engine import StandardDecodeEngine
from src.memory.parameter_loader import ParameterLoader
from src.memory.expert_cache import ExpertCache
from src.scheduling.cache_strategy import LRUCacheStrategy
from src.model import Qwen3ModelRunner


@dataclass
class SimpleRequest:
    input_ids: torch.Tensor
    generation_config: GenerationConfig


def _build_full_gpu_placement(config: MoEConfig) -> dict:
    gpu_experts = {
        layer_idx: list(range(config.num_experts))
        for layer_idx in range(config.num_hidden_layers)
    }
    return {"gpu_experts": gpu_experts}


def main():
    model_path = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"

    # Load config
    config = MoEConfig.from_pretrained(model_path)

    # Placement: load all experts to GPU
    placement_config = _build_full_gpu_placement(config)

    # Load parameters
    parameter_loader = ParameterLoader(
        model_path=model_path,
        placement_config=placement_config,
    )
    parameter_loader.load_parameters()

    # Expert cache (not used when fully on GPU, but required by engine)
    expert_size_mb = config.get_expert_weight_size_bytes() / (1024 * 1024)
    expert_cache = ExpertCache(
        max_cache_size_gb=1.0,
        expert_size_mb=expert_size_mb,
        replacement_strategy=LRUCacheStrategy(),
        pin_shared_experts=True,
    )

    # Model runner
    model_runner = Qwen3ModelRunner(config=config, parameter_loader=parameter_loader)

    # Standard decode engine
    engine = StandardDecodeEngine(
        model_runner=model_runner,
        parameter_loader=parameter_loader,
        expert_cache=expert_cache,
        prefetcher=None,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    prompts = [
        "Once upon a time",
        "In a distant future, humans",
        "Write a short poem about the sea",
    ]
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids_batch = tokenized["input_ids"].to(torch.long)
    attention_mask = tokenized["attention_mask"].to(torch.long)
    lengths = attention_mask.sum(dim=1).tolist()

    # Build request
    gen_config = GenerationConfig(
        max_new_tokens=50,
        temperature=0.8,
        top_p=1.0,
        top_k=50,
        do_sample=True,
        use_speculative=False,
    )
    # Single request (first prompt) for comparison
    single_len = int(lengths[0])
    single_ids = input_ids_batch[0, :single_len]
    single_request = SimpleRequest(input_ids=single_ids, generation_config=gen_config)

    # Batch requests (trim to actual lengths to avoid padding affecting prefill)
    requests = []
    for i in range(input_ids_batch.shape[0]):
        seq_len_i = int(lengths[i])
        seq_ids = input_ids_batch[i, :seq_len_i]
        requests.append(SimpleRequest(input_ids=seq_ids, generation_config=gen_config))

    # Build batch
    seq_len = input_ids_batch.shape[1]
    batch = BatchedRequest(
        batch_id="single_request",
        requests=requests,
        input_ids=input_ids_batch,
        attention_mask=attention_mask,
        position_ids=torch.arange(seq_len).unsqueeze(0),
        current_lengths=[int(x) for x in attention_mask.sum(dim=1).tolist()],
        finished=[False] * input_ids_batch.shape[0],
        max_batch_seq_len=seq_len,
        padding_token_id=0,
    )

    # Generate
    # Single inference
    single_batch = BatchedRequest(
        batch_id="single_request",
        requests=[single_request],
        input_ids=single_ids.unsqueeze(0),
        attention_mask=torch.ones(1, single_len, dtype=torch.long),
        position_ids=torch.arange(single_len).unsqueeze(0),
        current_lengths=[single_len],
        finished=[False],
        max_batch_seq_len=single_len,
        padding_token_id=0,
    )

    single_output = engine.generate_batch(single_batch)
    single_ids_out = single_output["generated_sequences"][0].tolist()
    print("[Single] Generated", len(single_ids_out), "tokens")
    print("[Single] Output IDs:", single_ids_out)
    print("[Single] Decoded:", tokenizer.decode(single_ids_out, skip_special_tokens=True))

    # Batch inference
    output = engine.generate_batch(batch)
    generated_sequences = output["generated_sequences"]
    for idx, output_ids in enumerate(generated_sequences):
        output_ids_list: List[int] = output_ids.tolist()
        print(f"[Batch Prompt {idx}] Generated {len(output_ids_list)} tokens")
        print(f"[Batch Prompt {idx}] Output IDs: {output_ids_list}")
        print(f"[Batch Prompt {idx}] Decoded: {tokenizer.decode(output_ids_list, skip_special_tokens=True)}")


if __name__ == "__main__":
    main()