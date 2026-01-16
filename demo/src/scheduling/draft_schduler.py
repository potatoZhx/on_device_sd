from typing import List, Dict, Set, Optional
import torch
import numpy as np
from .base_scheduler import DraftSchedulingStrategy
from ..core.types import ExpertID, LayerExpertActivations
from ..utils.logger import get_logger

logger = get_logger(__name__)

# TODO： 仿照这两个重写一个
class SimpleDraftScheduler(DraftSchedulingStrategy):
    """
    Simple draft scheduling strategy:
    - Run top-c highest scored experts on CPU
    - Substitute remaining with random GPU-cached experts
    - Transfer frequently activated experts
    """
    
    def __init__(
        self,
        perplexity_threshold: float = 1.5,
        min_cache_hit_rate: float = 0.5
    ):
        # TODO: threshold；cache hit rate作用？；expert access不应该从expert cache获取吗
        self.perplexity_threshold = perplexity_threshold
        self.min_cache_hit_rate = min_cache_hit_rate
        
        # Track expert usage for transfer decisions
        self.expert_access_count: Dict[ExpertID, int] = {}
    
    def select_cpu_experts(
        self,
        activations: LayerExpertActivations,
        top_c: int
    ) -> List[ExpertID]:
        """
        Select top-c experts with highest activation scores for CPU execution.
        """
        # TODO: 需要根据GPU内情况
        # Sort activations by score
        sorted_activations = sorted(
            activations.activations,
            key=lambda x: x.scores.max().item(),
            reverse=True
        )
        
        # Take top-c
        cpu_experts = [act.expert_id for act in sorted_activations[:top_c]]
        
        logger.debug(f"Selected {len(cpu_experts)} experts for CPU execution: "
                    f"{[str(e) for e in cpu_experts]}")
        
        return cpu_experts
    
    def select_gpu_substitutes(
        self,
        requested_experts: List[ExpertID],
        cached_experts: Set[ExpertID],
        all_experts: List[ExpertID]
    ) -> Dict[ExpertID, ExpertID]:
        """
        For each requested expert not in cache, randomly select a cached substitute.
        """
        # TODO：替换策略
        substitutes = {}
        cached_list = list(cached_experts)
        
        if not cached_list:
            logger.warning("No cached experts available for substitution")
            return substitutes
        
        for req_expert in requested_experts:
            if req_expert not in cached_experts:
                # Random substitution (simple strategy)
                substitute = np.random.choice(cached_list)
                substitutes[req_expert] = substitute
                
                logger.debug(f"Substitute {req_expert} -> {substitute}")
        
        return substitutes
    
    def select_experts_to_transfer(
        self,
        recent_activations: List[LayerExpertActivations],
        cached_experts: Set[ExpertID],
        cache_capacity: int
    ) -> List[ExpertID]:
        """
        Select experts to transfer based on activation frequency.
        """
        # TODO：根据draft结果
        # Count activations across recent history
        activation_counts = {}
        
        for layer_acts in recent_activations:
            for act in layer_acts.activations:
                expert_id = act.expert_id
                activation_counts[expert_id] = activation_counts.get(expert_id, 0) + 1
        
        # Filter out already cached experts
        candidates = {
            exp: count for exp, count in activation_counts.items()
            if exp not in cached_experts
        }
        
        # Sort by frequency
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top experts up to available cache space
        available_slots = cache_capacity - len(cached_experts)
        to_transfer = [exp for exp, _ in sorted_candidates[:available_slots]]
        
        logger.debug(f"Selected {len(to_transfer)} experts for transfer: "
                    f"{[str(e) for e in to_transfer]}")
        
        return to_transfer
    
    def should_trigger_verify(
        self,
        num_drafted_tokens: int,
        perplexity: float,
        cache_hit_rate: float,
        max_draft_tokens: int
    ) -> bool:
        """
        Trigger verify if:
        - Max draft tokens reached
        - Perplexity too high (quality degradation)
        - Cache hit rate too low (too many substitutions)
        """
        # TODO：预测模型
        if num_drafted_tokens >= max_draft_tokens:
            logger.info(f"Triggering verify: max draft tokens reached ({num_drafted_tokens})")
            return True
        
        if perplexity > self.perplexity_threshold:
            logger.info(f"Triggering verify: high perplexity ({perplexity:.3f})")
            return True
        
        if cache_hit_rate < self.min_cache_hit_rate:
            logger.info(f"Triggering verify: low cache hit rate ({cache_hit_rate:.3f})")
            return True
        
        return False


class AdaptiveDraftScheduler(DraftSchedulingStrategy):
    """
    Advanced draft scheduler with adaptive thresholds and smart substitution.
    """
    
    def __init__(
        self,
        initial_perplexity_threshold: float = 1.5,
        initial_cache_hit_threshold: float = 0.5,
        adaptation_rate: float = 0.1
    ):
        self.perplexity_threshold = initial_perplexity_threshold
        self.cache_hit_threshold = initial_cache_hit_threshold
        self.adaptation_rate = adaptation_rate
        
        # Expert similarity matrix (for smart substitution)
        self.expert_similarity: Dict[tuple, float] = {}
        
        # Performance tracking
        self.acceptance_rates: List[float] = []
    
    def select_cpu_experts(
        self,
        activations: LayerExpertActivations,
        top_c: int
    ) -> List[ExpertID]:
        """Select top-c experts, same as simple strategy"""
        sorted_activations = sorted(
            activations.activations,
            key=lambda x: x.scores.max().item(),
            reverse=True
        )
        
        return [act.expert_id for act in sorted_activations[:top_c]]
    
    def select_gpu_substitutes(
        self,
        requested_experts: List[ExpertID],
        cached_experts: Set[ExpertID],
        all_experts: List[ExpertID]
    ) -> Dict[ExpertID, ExpertID]:
        """
        Smart substitution: select most similar cached expert.
        """
        substitutes = {}
        
        for req_expert in requested_experts:
            if req_expert not in cached_experts:
                # Find most similar cached expert
                best_substitute = self._find_most_similar(
                    req_expert,
                    cached_experts,
                    all_experts
                )
                
                if best_substitute:
                    substitutes[req_expert] = best_substitute
        
        return substitutes
    
    def _find_most_similar(
        self,
        target: ExpertID,
        candidates: Set[ExpertID],
        all_experts: List[ExpertID]
    ) -> Optional[ExpertID]:
        """
        Find most similar expert from candidates.
        Uses learned similarity matrix, falls back to same-index heuristic.
        """
        # Filter candidates to same layer
        same_layer_candidates = [
            c for c in candidates if c.layer_idx == target.layer_idx
        ]
        
        if not same_layer_candidates:
            # No same-layer candidates, pick any
            return list(candidates)[0] if candidates else None
        
        # Try similarity matrix
        best_sim = -1
        best_candidate = None
        
        for candidate in same_layer_candidates:
            key = (target, candidate)
            sim = self.expert_similarity.get(key, 0.0)
            
            if sim > best_sim:
                best_sim = sim
                best_candidate = candidate
        
        # If no learned similarity, use heuristic: closest expert index
        if best_candidate is None:
            best_candidate = min(
                same_layer_candidates,
                key=lambda c: abs(c.expert_idx - target.expert_idx)
            )
        
        return best_candidate
    
    def select_experts_to_transfer(
        self,
        recent_activations: List[LayerExpertActivations],
        cached_experts: Set[ExpertID],
        cache_capacity: int
    ) -> List[ExpertID]:
        """Transfer selection with recency weighting"""
        activation_scores = {}
        
        # Weight recent activations more heavily
        for i, layer_acts in enumerate(reversed(recent_activations)):
            weight = 1.0 / (i + 1)  # More recent = higher weight
            
            for act in layer_acts.activations:
                expert_id = act.expert_id
                score = act.scores.max().item() * weight
                activation_scores[expert_id] = activation_scores.get(expert_id, 0) + score
        
        # Filter and sort
        candidates = {
            exp: score for exp, score in activation_scores.items()
            if exp not in cached_experts
        }
        
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        available_slots = cache_capacity - len(cached_experts)
        return [exp for exp, _ in sorted_candidates[:available_slots]]
    
    def should_trigger_verify(
        self,
        num_drafted_tokens: int,
        perplexity: float,
        cache_hit_rate: float,
        max_draft_tokens: int
    ) -> bool:
        """Adaptive verification trigger with learning"""
        if num_drafted_tokens >= max_draft_tokens:
            return True
        
        # Adaptive thresholds based on recent performance
        if perplexity > self.perplexity_threshold:
            return True
        
        if cache_hit_rate < self.cache_hit_threshold:
            return True
        
        return False
    
    def update_from_verify_result(
        self,
        acceptance_rate: float,
        actual_perplexity: float
    ) -> None:
        """
        Update adaptive thresholds based on verify results.
        """
        self.acceptance_rates.append(acceptance_rate)
        
        # Adapt thresholds
        if acceptance_rate < 0.5:
            # Too many rejections, tighten thresholds
            self.perplexity_threshold *= (1 - self.adaptation_rate)
            self.cache_hit_threshold *= (1 + self.adaptation_rate)
        elif acceptance_rate > 0.9:
            # High acceptance, can relax th# High acceptance, can relax thresholds
            self.perplexity_threshold *= (1 + self.adaptation_rate)
            self.cache_hit_threshold *= (1 - self.adaptation_rate)
        
        logger.debug(f"Updated thresholds: perplexity={self.perplexity_threshold:.3f}, "
                    f"cache_hit={self.cache_hit_threshold:.3f}")