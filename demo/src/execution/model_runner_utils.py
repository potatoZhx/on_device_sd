from __future__ import annotations

from typing import Dict, List, Set
import torch

from ..core.model_runner import RoutingResult, ExpertPlacement
from ..core.types import ExpertID, ExpertActivation, LayerExpertActivations, DeviceType
from ..memory.expert_cache import ExpertCache
from ..memory.parameter_loader import ParameterLoader
from ..scheduling.draft_schduler import DraftSchedulingStrategy


def build_layer_activations(
    routing_result: RoutingResult,
    num_experts: int,
) -> LayerExpertActivations:
    """将 RoutingResult 转换为 LayerExpertActivations。"""
    topk_indices = routing_result.topk_indices
    topk_scores = routing_result.topk_scores

    activations = []
    for k in range(topk_indices.shape[1]):
        for expert_idx in range(num_experts):
            mask = topk_indices[:, k] == expert_idx
            token_indices = torch.where(mask)[0]
            if token_indices.numel() == 0:
                continue
            activations.append(
                ExpertActivation(
                    expert_id=ExpertID(routing_result.layer_idx, expert_idx),
                    token_indices=token_indices,
                    scores=topk_scores[token_indices, k],
                    top_k_rank=k,
                )
            )

    return LayerExpertActivations(
        layer_idx=routing_result.layer_idx,
        activations=activations,
        routing_scores=topk_scores,
    )


def _get_gpu_params(
    expert_id: ExpertID,
    expert_cache: ExpertCache,
    parameter_loader: ParameterLoader,
):
    if parameter_loader.is_shared_expert(expert_id):
        return parameter_loader.get_shared_expert_params(expert_id)
    if expert_cache.is_cached(expert_id):
        return expert_cache.get(expert_id)
    return parameter_loader.get_expert_params(expert_id, device=DeviceType.GPU)


def build_prefill_placement(
    routing_result: RoutingResult,
    expert_cache: ExpertCache,
    parameter_loader: ParameterLoader,
) -> ExpertPlacement:
    """构建 prefill/verify 阶段的 placement。"""
    gpu_expert_params: Dict[int, Dict[str, torch.Tensor]] = {}
    cpu_expert_params: Dict[int, Dict[str, torch.Tensor]] = {}

    for expert_id in routing_result.activated_expert_ids:
        gpu_params = _get_gpu_params(expert_id, expert_cache, parameter_loader)
        if gpu_params is not None:
            gpu_expert_params[expert_id.expert_idx] = gpu_params
            continue

        cpu_params = parameter_loader.get_expert_params(expert_id, device=DeviceType.CPU)
        if cpu_params is not None:
            cpu_expert_params[expert_id.expert_idx] = cpu_params

    return ExpertPlacement(
        gpu_expert_params=gpu_expert_params,
        cpu_expert_params=cpu_expert_params,
        routing_result=routing_result,
    )


def build_draft_placement(
    routing_result: RoutingResult,
    expert_cache: ExpertCache,
    parameter_loader: ParameterLoader,
    draft_scheduler: DraftSchedulingStrategy,
    *,
    top_c: int,
    num_experts: int,
) -> ExpertPlacement:
    """构建 draft 阶段的 placement（含替换）。"""
    gpu_available_experts: Set[ExpertID] = set()
    for expert_id in routing_result.activated_expert_ids:
        if _get_gpu_params(expert_id, expert_cache, parameter_loader) is not None:
            gpu_available_experts.add(expert_id)

    activated_expert_ids: Set[ExpertID] = set(routing_result.activated_expert_ids)
    missing_gpu_experts = activated_expert_ids - gpu_available_experts

    cpu_expert_ids = set()
    if missing_gpu_experts and top_c > 0:
        layer_acts = build_layer_activations(routing_result, num_experts)
        cpu_candidate_acts = [
            act for act in layer_acts.activations
            if act.expert_id in missing_gpu_experts
        ]
        if cpu_candidate_acts:
            cpu_layer_acts = LayerExpertActivations(
                layer_idx=layer_acts.layer_idx,
                activations=cpu_candidate_acts,
                routing_scores=layer_acts.routing_scores,
            )
            cpu_expert_ids = set(draft_scheduler.select_cpu_experts(cpu_layer_acts, top_c))

    cached_experts: Set[ExpertID] = set(gpu_available_experts)

    needs_substitution = activated_expert_ids - cached_experts - cpu_expert_ids

    all_layer_experts = [ExpertID(routing_result.layer_idx, i) for i in range(num_experts)]
    substitution_map = draft_scheduler.select_gpu_substitutes(
        requested_experts=list(needs_substitution),
        cached_experts=cached_experts,
        all_experts=all_layer_experts,
    )

    gpu_expert_params: Dict[int, Dict[str, torch.Tensor]] = {}
    cpu_expert_params: Dict[int, Dict[str, torch.Tensor]] = {}

    for expert_id in activated_expert_ids:
        if expert_id in cpu_expert_ids:
            cpu_params = parameter_loader.get_expert_params(expert_id, device=DeviceType.CPU)
            if cpu_params is not None:
                cpu_expert_params[expert_id.expert_idx] = cpu_params
            continue

        gpu_params = _get_gpu_params(expert_id, expert_cache, parameter_loader)
        if gpu_params is not None:
            gpu_expert_params[expert_id.expert_idx] = gpu_params

    # 确保替代 expert 的参数在 GPU params 中
    for _, sub_id in substitution_map.items():
        gpu_params = _get_gpu_params(sub_id, expert_cache, parameter_loader)
        if gpu_params is not None:
            gpu_expert_params[sub_id.expert_idx] = gpu_params

    return ExpertPlacement(
        gpu_expert_params=gpu_expert_params,
        cpu_expert_params=cpu_expert_params,
        substitution_map={orig.expert_idx: sub.expert_idx for orig, sub in substitution_map.items()},
        routing_result=routing_result,
    )
