from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Set
from ..core.types import (
    ExpertID, ExpertActivation, LayerExpertActivations,
    TransferRequest, DeviceType
)

class PrefetchStrategy(ABC):
    """
    Abstract base class for expert prefetching strategies.
    Predicts which experts will be needed next and initiates transfers.
    """
    # TODO: prefetch应该是用draft预测verify，不是跨层
    @abstractmethod
    def predict_next_experts(
        self,
        current_layer_idx: int,
        current_activations: LayerExpertActivations,
        history: Optional[List[LayerExpertActivations]] = None
    ) -> List[ExpertID]:
        """
        Predict which experts will be activated in the next layer(s).
        
        Args:
            current_layer_idx: Current layer being executed
            current_activations: Expert activations for current layer
            history: Historical activation patterns (optional)
        
        Returns:
            List of expert IDs predicted to be needed next
        """
        pass
    
    @abstractmethod
    def get_prefetch_priority(
        self,
        expert_id: ExpertID,
        predicted_score: float
    ) -> float:
        """
        Calculate priority for prefetching an expert.
        Higher priority = more urgent to transfer.
        
        Args:
            expert_id: Expert to prioritize
            predicted_score: Prediction confidence score
        
        Returns:
            Priority value (higher = more important)
        """
        pass


class SchedulingStrategy(ABC):
    """
    Abstract base class for runtime scheduling decisions.
    Decides whether to execute on CPU or wait for GPU transfer.
    """
    # TODO: 批量调度？
    @abstractmethod
    def decide_execution_device(
        self,
        expert_id: ExpertID,
        is_in_gpu_cache: bool,
        is_transferring: bool,
        transfer_eta_ms: Optional[float],
        cpu_compute_time_ms: float,
        gpu_compute_time_ms: float
    ) -> DeviceType:
        """
        Decide where to execute an expert.
        
        Args:
            expert_id: Expert to schedule
            is_in_gpu_cache: Whether expert is already in GPU cache
            is_transferring: Whether transfer is in progress
            transfer_eta_ms: Estimated time until transfer completes
            cpu_compute_time_ms: Estimated CPU computation time
            gpu_compute_time_ms: Estimated GPU computation time
        
        Returns:
            Device to execute on
        """
        pass


class CacheReplacementStrategy(ABC):
    """
    Abstract base class for GPU expert cache replacement policies.
    Determines which experts to evict when cache is full.
    """
    # TODO：参考一下hybrimoe的cache实现
    
    def __init__(self):
        self.access_history: Dict[ExpertID, List[float]] = {}
        self.insertion_time: Dict[ExpertID, float] = {}
    
    @abstractmethod
    def select_victim(
        self,
        cached_experts: List[ExpertID],
        pinned_experts: Set[ExpertID]
    ) -> Optional[ExpertID]:
        """
        Select an expert to evict from cache.
        
        Args:
            cached_experts: List of currently cached experts
            pinned_experts: Set of experts that cannot be evicted
        
        Returns:
            Expert ID to evict, or None if no eviction possible
        """
        pass
    
    def on_access(self, expert_id: ExpertID) -> None:
        """Called when an expert is accessed"""
        import time
        if expert_id not in self.access_history:
            self.access_history[expert_id] = []
        self.access_history[expert_id].append(time.time())
    
    def on_insert(self, expert_id: ExpertID) -> None:
        """Called when an expert is inserted into cache"""
        import time
        self.insertion_time[expert_id] = time.time()
        if expert_id not in self.access_history:
            self.access_history[expert_id] = []
    
    def on_evict(self, expert_id: ExpertID) -> None:
        """Called when an expert is evicted"""
        if expert_id in self.access_history:
            del self.access_history[expert_id]
        if expert_id in self.insertion_time:
            del self.insertion_time[expert_id]


class DraftSchedulingStrategy(ABC):
    """
    Abstract base class for draft phase scheduling.
    Decides expert selection and replacement during speculative decoding.
    """
    # TODO
    @abstractmethod
    def select_cpu_experts(
        self,
        activations: LayerExpertActivations,
        top_c: int
    ) -> List[ExpertID]:
        """
        Select top-c experts to execute on CPU during draft.
        
        Args:
            activations: Expert activations for current layer
            top_c: Number of experts to run on CPU
        
        Returns:
            List of expert IDs to execute on CPU
        """
        pass
    
    @abstractmethod
    def select_gpu_substitutes(
        self,
        requested_experts: List[ExpertID],
        cached_experts: Set[ExpertID],
        all_experts: List[ExpertID]
    ) -> Dict[ExpertID, ExpertID]:
        """
        Select GPU-cached experts to substitute for CPU experts.
        
        Args:
            requested_experts: Experts that should ideally be used
            cached_experts: Experts currently in GPU cache
            all_experts: All available experts for this layer
        
        Returns:
            Mapping from requested expert -> substitute expert
        """
        pass
    
    @abstractmethod
    def select_experts_to_transfer(
        self,
        recent_activations: List[LayerExpertActivations],
        cached_experts: Set[ExpertID],
        cache_capacity: int
    ) -> List[ExpertID]:
        """
        Select experts to transfer to GPU cache based on draft activations.
        
        Args:
            recent_activations: Recent expert activation patterns
            cached_experts: Currently cached experts
            cache_capacity: Maximum number of experts that can be cached
        
        Returns:
            List of expert IDs to transfer
        """
        pass
    
    @abstractmethod
    def should_trigger_verify(
        self,
        num_drafted_tokens: int,
        perplexity: float,
        cache_hit_rate: float,
        max_draft_tokens: int
    ) -> bool:
        """
        Decide whether to switch from draft to verify phase.
        
        Args:
            num_drafted_tokens: Number of tokens drafted so far
            perplexity: Current perplexity metric
            cache_hit_rate: Expert cache hit rate
            max_draft_tokens: Maximum allowed draft tokens
        
        Returns:
            True if should verify, False to continue drafting
        """
        pass