import torch

from src.core.model_runner import RoutingResult
from src.core.types import ExpertID
from src.execution.model_runner_utils import build_prefill_placement, build_draft_placement
from src.scheduling.base_scheduler import DraftSchedulingStrategy
from src.core.types import LayerExpertActivations


class DummyDraftScheduler(DraftSchedulingStrategy):
    def select_cpu_experts(self, activations: LayerExpertActivations, top_c: int):
        return [act.expert_id for act in activations.activations[:top_c]]

    def select_gpu_substitutes(self, requested_experts, cached_experts, all_experts):
        substitutes = {}
        cached_list = list(cached_experts)
        for req in requested_experts:
            if cached_list:
                substitutes[req] = cached_list[0]
        return substitutes

    def select_experts_to_transfer(self, recent_activations, cached_experts, cache_capacity):
        return []

    def should_trigger_verify(self, num_drafted_tokens, perplexity, cache_hit_rate, max_draft_tokens):
        return False


def _make_routing(layer_idx: int, topk_indices: torch.Tensor, topk_scores: torch.Tensor):
    activated = {ExpertID(layer_idx, idx) for idx in topk_indices.flatten().unique().tolist()}
    return RoutingResult(
        layer_idx=layer_idx,
        topk_indices=topk_indices,
        topk_scores=topk_scores,
        activated_expert_ids=activated,
    )


def test_all_gpu_placement(dummy_expert_cache, dummy_parameter_loader):
    topk_indices = torch.tensor([[0, 1], [1, 0]], device="cuda")
    topk_scores = torch.full_like(topk_indices, 0.5, dtype=torch.float32)
    routing = _make_routing(0, topk_indices, topk_scores)

    # Cache both experts
    for expert_idx in [0, 1]:
        expert_id = ExpertID(0, expert_idx)
        dummy_expert_cache.put(expert_id, dummy_parameter_loader.get_expert_params(expert_id))

    placement = build_prefill_placement(routing, dummy_expert_cache, dummy_parameter_loader)
    assert len(placement.gpu_expert_params) == 2
    assert len(placement.cpu_expert_params) == 0


def test_mixed_cpu_gpu_placement(dummy_expert_cache, dummy_parameter_loader):
    topk_indices = torch.tensor([[0, 2]], device="cuda")
    topk_scores = torch.full_like(topk_indices, 0.5, dtype=torch.float32)
    routing = _make_routing(0, topk_indices, topk_scores)

    # Cache only expert 0
    expert0 = ExpertID(0, 0)
    dummy_expert_cache.put(expert0, dummy_parameter_loader.get_expert_params(expert0))

    # Remove GPU params for expert 2 to force CPU path
    expert2 = ExpertID(0, 2)
    if expert2 in dummy_parameter_loader.expert_params_gpu:
        dummy_parameter_loader.expert_params_gpu.pop(expert2)

    placement = build_prefill_placement(routing, dummy_expert_cache, dummy_parameter_loader)
    assert 0 in placement.gpu_expert_params
    assert 2 in placement.cpu_expert_params


def test_draft_substitution_placement(dummy_expert_cache, dummy_parameter_loader, small_config):
    topk_indices = torch.tensor([[0, 1], [2, 3]], device="cuda")
    topk_scores = torch.full_like(topk_indices, 0.5, dtype=torch.float32)
    routing = _make_routing(0, topk_indices, topk_scores)

    # Cache expert 0 for substitution
    expert0 = ExpertID(0, 0)
    dummy_expert_cache.put(expert0, dummy_parameter_loader.get_expert_params(expert0))

    placement = build_draft_placement(
        routing_result=routing,
        expert_cache=dummy_expert_cache,
        parameter_loader=dummy_parameter_loader,
        draft_scheduler=DummyDraftScheduler(),
        top_c=1,
        num_experts=small_config.num_experts,
    )

    for _, sub_idx in placement.substitution_map.items():
        assert sub_idx in placement.gpu_expert_params
