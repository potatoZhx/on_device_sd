from __future__ import annotations

from typing import Dict, List, Tuple

from ..core.types import ExpertID, LayerExpertActivations


def select_experts_to_prefetch(
    current_step: int,
    max_draft_tokens: int,
    step_activations: List[LayerExpertActivations],
    cached_experts: set[ExpertID],
    pending_transfers: set[ExpertID],
    cache_capacity: int,
    num_experts_per_layer: int,
    *,
    max_prefetch_per_step: int = 4,
) -> List[ExpertID]:
    del current_step, max_draft_tokens, num_experts_per_layer

    if not step_activations:
        return []

    expert_stats: Dict[ExpertID, Tuple[int, float]] = {}
    for layer_acts in step_activations:
        for act in layer_acts.activations:
            eid = act.expert_id
            count, total_score = expert_stats.get(eid, (0, 0.0))
            expert_stats[eid] = (count + 1, total_score + act.scores.max().item())

    already_available = cached_experts | pending_transfers
    candidates = {
        eid: (count, total_score)
        for eid, (count, total_score) in expert_stats.items()
        if eid not in already_available
    }

    if not candidates:
        return []

    available_slots = cache_capacity - len(cached_experts) - len(pending_transfers)
    if available_slots <= 0:
        return []

    scored = [
        (eid, count * (total_score / count))
        for eid, (count, total_score) in candidates.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    num_to_prefetch = min(max_prefetch_per_step, available_slots, len(scored))
    return [eid for eid, _ in scored[:num_to_prefetch]]
