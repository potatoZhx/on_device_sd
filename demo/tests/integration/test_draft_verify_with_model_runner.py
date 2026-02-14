import torch

from src.execution.draft_engine import DraftEngine
from src.execution.verify_engine import VerifyEngine
from src.scheduling.draft_schduler import SimpleDraftScheduler
from src.utils.metrics import MetricsCollector


def test_draft_produces_tokens(qwen3_runner, dummy_parameter_loader, dummy_expert_cache, kv_cache):
    engine = DraftEngine(
        model_runner=qwen3_runner,
        parameter_loader=dummy_parameter_loader,
        expert_cache=dummy_expert_cache,
        draft_scheduler=SimpleDraftScheduler(),
        metrics=MetricsCollector(),
    )

    input_ids = torch.tensor([1], device="cuda")
    result = engine.forward(input_ids, kv_cache, max_draft_tokens=2)

    assert "drafted_tokens" in result
    assert 0 < len(result["drafted_tokens"]) <= 2


def test_verify_produces_logits(qwen3_runner, dummy_parameter_loader, dummy_expert_cache, kv_cache):
    engine = VerifyEngine(
        model_runner=qwen3_runner,
        parameter_loader=dummy_parameter_loader,
        expert_cache=dummy_expert_cache,
        prefetcher=None,
        metrics=MetricsCollector(),
    )

    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    output = engine.forward(input_ids, kv_cache)

    assert "logits" in output
    assert output["logits"].dim() == 3
