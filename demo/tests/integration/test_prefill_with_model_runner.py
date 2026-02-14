import torch

from src.execution.prefill_engine import PrefillEngine
from src.utils.metrics import MetricsCollector


def test_prefill_produces_logits(qwen3_runner, dummy_parameter_loader, dummy_expert_cache, kv_cache):
    engine = PrefillEngine(
        model_runner=qwen3_runner,
        parameter_loader=dummy_parameter_loader,
        expert_cache=dummy_expert_cache,
        prefetcher=None,
        metrics=MetricsCollector(),
    )
    input_ids = torch.randint(0, 100, (1, 6), device="cuda")

    output = engine.forward(input_ids, kv_cache, is_prefill=True)

    assert "logits" in output
    assert "next_token_id" in output
    assert output["logits"].dim() == 3


def test_prefill_kv_cache_updated(qwen3_runner, dummy_parameter_loader, dummy_expert_cache, kv_cache):
    engine = PrefillEngine(
        model_runner=qwen3_runner,
        parameter_loader=dummy_parameter_loader,
        expert_cache=dummy_expert_cache,
        prefetcher=None,
        metrics=MetricsCollector(),
    )
    input_ids = torch.randint(0, 100, (1, 4), device="cuda")

    engine.forward(input_ids, kv_cache, is_prefill=True)

    assert len(kv_cache.sequences) > 0
