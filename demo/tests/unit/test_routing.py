import torch

from src.core.types import ExpertID
from src.core.model_runner import RoutingResult


def test_route_experts_output_format(qwen3_runner):
    config = qwen3_runner.get_config()
    hidden = torch.randn(1, 4, config.hidden_size, device="cuda", dtype=config.get_dtype())
    result = qwen3_runner.route_experts(layer_idx=0, hidden_states=hidden)

    assert isinstance(result, RoutingResult)
    assert result.layer_idx == 0
    assert result.topk_indices.shape[-1] == config.num_experts_per_token
    assert result.topk_scores.shape == result.topk_indices.shape


def test_routing_scores_sum_to_one(qwen3_runner):
    config = qwen3_runner.get_config()
    hidden = torch.randn(1, 4, config.hidden_size, device="cuda", dtype=config.get_dtype())
    result = qwen3_runner.route_experts(layer_idx=0, hidden_states=hidden)

    score_sums = result.topk_scores.sum(dim=-1)
    assert torch.allclose(score_sums, torch.ones_like(score_sums), atol=1e-5)


def test_activated_expert_ids_consistency(qwen3_runner):
    config = qwen3_runner.get_config()
    hidden = torch.randn(1, 4, config.hidden_size, device="cuda", dtype=config.get_dtype())
    result = qwen3_runner.route_experts(layer_idx=0, hidden_states=hidden)

    expected_ids = {ExpertID(0, idx) for idx in result.topk_indices.flatten().unique().tolist()}
    assert result.activated_expert_ids == expected_ids
