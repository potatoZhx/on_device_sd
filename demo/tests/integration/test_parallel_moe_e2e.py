import pytest
import torch

from src.execution.prefill_engine import PrefillEngine
from src.memory.expert_cache import ExpertCache
from src.memory.paged_kv_cache import PagedKVCache
from src.scheduling.cache_strategy import LRUCacheStrategy
from src.utils.metrics import MetricsCollector


def _make_expert_cache():
    return ExpertCache(
        max_cache_size_gb=0.01,
        expert_size_mb=0.001,
        replacement_strategy=LRUCacheStrategy(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_parallel_moe_mixed_cpu_gpu_close_to_all_gpu(qwen3_runner, dummy_parameter_loader, small_config):
    torch.manual_seed(0)
    input_ids = torch.randint(0, 100, (1, 6), device="cuda")

    kv_cache = PagedKVCache(
        config=small_config,
        block_size=256,
        dtype=small_config.get_dtype(),
    )
    engine = PrefillEngine(
        model_runner=qwen3_runner,
        parameter_loader=dummy_parameter_loader,
        expert_cache=_make_expert_cache(),
        prefetcher=None,
        metrics=MetricsCollector(),
    )
    baseline = engine.forward(input_ids, kv_cache, is_prefill=True)["logits"]

    to_restore = {}
    removed = []
    for expert_id in list(dummy_parameter_loader.expert_params_gpu.keys())[:2]:
        to_restore[expert_id] = dummy_parameter_loader.expert_params_gpu.pop(expert_id)
        removed.append(expert_id)

    kv_cache2 = PagedKVCache(
        config=small_config,
        block_size=256,
        dtype=small_config.get_dtype(),
    )
    engine2 = PrefillEngine(
        model_runner=qwen3_runner,
        parameter_loader=dummy_parameter_loader,
        expert_cache=_make_expert_cache(),
        prefetcher=None,
        metrics=MetricsCollector(),
    )
    mixed = engine2.forward(input_ids, kv_cache2, is_prefill=True)["logits"]

    for expert_id, params in to_restore.items():
        dummy_parameter_loader.expert_params_gpu[expert_id] = params

    torch.testing.assert_close(mixed, baseline, rtol=1e-2, atol=1e-2)
