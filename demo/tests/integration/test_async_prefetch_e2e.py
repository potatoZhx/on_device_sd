import pytest
import torch

from src.execution.prefill_engine import PrefillEngine
from src.memory.expert_cache import ExpertCache
from src.memory.paged_kv_cache import PagedKVCache
from src.scheduling.cache_strategy import LRUCacheStrategy
from src.scheduling.prefetcher import ExpertPrefetcher, SimplePrefetchStrategy
from src.utils.metrics import MetricsCollector


def _make_expert_cache():
    return ExpertCache(
        max_cache_size_gb=0.01,
        expert_size_mb=0.001,
        replacement_strategy=LRUCacheStrategy(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_async_prefetch_populates_cache(qwen3_runner, dummy_parameter_loader, small_config):
    expert_cache = _make_expert_cache()
    expert_cache.transfer_stream = None
    prefetcher = ExpertPrefetcher(SimplePrefetchStrategy(num_experts_to_prefetch=2))
    kv_cache = PagedKVCache(
        config=small_config,
        block_size=256,
        dtype=small_config.get_dtype(),
    )
    engine = PrefillEngine(
        model_runner=qwen3_runner,
        parameter_loader=dummy_parameter_loader,
        expert_cache=expert_cache,
        prefetcher=prefetcher,
        metrics=MetricsCollector(),
    )
    input_ids = torch.randint(0, 100, (1, 6), device="cuda")
    engine.forward(input_ids, kv_cache, is_prefill=True)
    expert_cache.complete_ready_transfers()

    layer_acts = engine.activation_history[0]
    predicted = prefetcher.strategy.predict_next_experts(0, layer_acts, engine.activation_history)
    assert predicted
    assert any(expert_cache.is_cached(exp) for exp in predicted)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_async_prefetch_improves_cache_presence(qwen3_runner, dummy_parameter_loader, small_config):
    input_ids = torch.randint(0, 100, (1, 6), device="cuda")

    no_prefetch_cache = _make_expert_cache()
    no_prefetch_cache.transfer_stream = None
    kv_cache = PagedKVCache(
        config=small_config,
        block_size=256,
        dtype=small_config.get_dtype(),
    )
    engine = PrefillEngine(
        model_runner=qwen3_runner,
        parameter_loader=dummy_parameter_loader,
        expert_cache=no_prefetch_cache,
        prefetcher=None,
        metrics=MetricsCollector(),
    )
    engine.forward(input_ids, kv_cache, is_prefill=True)
    no_prefetch_cache.complete_ready_transfers()

    prefetch_cache = _make_expert_cache()
    prefetch_cache.transfer_stream = None
    prefetcher = ExpertPrefetcher(SimplePrefetchStrategy(num_experts_to_prefetch=2))
    kv_cache2 = PagedKVCache(
        config=small_config,
        block_size=256,
        dtype=small_config.get_dtype(),
    )
    engine2 = PrefillEngine(
        model_runner=qwen3_runner,
        parameter_loader=dummy_parameter_loader,
        expert_cache=prefetch_cache,
        prefetcher=prefetcher,
        metrics=MetricsCollector(),
    )
    engine2.forward(input_ids, kv_cache2, is_prefill=True)
    prefetch_cache.complete_ready_transfers()

    assert len(prefetch_cache.cached_experts) > len(no_prefetch_cache.cached_experts)
