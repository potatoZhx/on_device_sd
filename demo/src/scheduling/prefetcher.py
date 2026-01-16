from typing import List, Dict, Optional
import numpy as np
from .base_scheduler import PrefetchStrategy
from ..core.types import ExpertID, LayerExpertActivations
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SimplePrefetchStrategy(PrefetchStrategy):
    """
    Simple prefetching strategy: predict next layer will activate 
    the same experts as current layer (temporal locality).
    """
    # TODO prefetch实现的是verify期间的prefetch｜队列
    def __init__(self, num_experts_to_prefetch: int = 4):
        self.num_experts_to_prefetch = num_experts_to_prefetch
    
    # TODO 是否需要
    def predict_next_experts(
        self,
        current_layer_idx: int,
        current_activations: LayerExpertActivations,
        history: Optional[List[LayerExpertActivations]] = None
    ) -> List[ExpertID]:
        """
        Predict experts for next layer based on current activations.
        Simple strategy: same expert indices in next layer.
        """
        if current_activations is None:
            return []
        
        # Get top-k activated experts from current layer
        top_experts = sorted(
            current_activations.activations,
            key=lambda x: x.scores.max(),
            reverse=True
        )[:self.num_experts_to_prefetch]
        
        # Map to next layer
        next_layer_idx = current_layer_idx + 1
        predicted = [
            ExpertID(next_layer_idx, exp.expert_id.expert_idx)
            for exp in top_experts
        ]
        
        logger.debug(f"Prefetch prediction for layer {next_layer_idx}: "
                    f"{[str(e) for e in predicted]}")
        
        return predicted
    
    def get_prefetch_priority(
        self,
        expert_id: ExpertID,
        predicted_score: float
    ) -> float:
        """Priority based on prediction score"""
        return predicted_score


class HistoryBasedPrefetchStrategy(PrefetchStrategy):
    """
    Advanced prefetching using historical activation patterns.
    Learns which experts tend to be activated together.
    """
    
    def __init__(
        self,
        num_experts_to_prefetch: int = 4,
        history_window: int = 10
    ):
        self.num_experts_to_prefetch = num_experts_to_prefetch
        self.history_window = history_window
        
        # Co-occurrence matrix: expert_i, expert_j -> frequency
        self.co_occurrence: Dict[tuple, int] = {}
    
    def predict_next_experts(
        self,
        current_layer_idx: int,
        current_activations: LayerExpertActivations,
        history: Optional[List[LayerExpertActivations]] = None
    ) -> List[ExpertID]:
        """
        Predict using historical co-occurrence patterns.
        """
        if not history or len(history) < 2:
            # Fallback to simple strategy
            return self._simple_prediction(current_layer_idx, current_activations)
        
        # Update co-occurrence matrix
        self._update_co_occurrence(history[-self.history_window:])
        
        # Get currently activated experts
        current_experts = [act.expert_id for act in current_activations.activations]
        
        # Predict based on co-occurrence
        next_layer_idx = current_layer_idx + 1
        predictions = {}
        
        for curr_expert in current_experts:
            for expert_idx in range(32):  # Assume max 32 experts per layer
                next_expert = ExpertID(next_layer_idx, expert_idx)
                key = (curr_expert, next_expert)
                
                if key in self.co_occurrence:
                    score = self.co_occurrence[key]
                    predictions[next_expert] = predictions.get(next_expert, 0) + score
        
        # Sort by score and return top-k
        sorted_predictions = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        result = [exp for exp, _ in sorted_predictions[:self.num_experts_to_prefetch]]
        
        logger.debug(f"History-based prefetch for layer {next_layer_idx}: "
                    f"{[str(e) for e in result]}")
        
        return result
    
    def _simple_prediction(
        self,
        current_layer_idx: int,
        current_activations: LayerExpertActivations
    ) -> List[ExpertID]:
        """Fallback to simple prediction"""
        next_layer_idx = current_layer_idx + 1
        top_experts = sorted(
            current_activations.activations,
            key=lambda x: x.scores.max(),
            reverse=True
        )[:self.num_experts_to_prefetch]
        
        return [
            ExpertID(next_layer_idx, exp.expert_id.expert_idx)
            for exp in top_experts
        ]
    
    def _update_co_occurrence(
        self,
        history: List[LayerExpertActivations]
    ) -> None:
        """Update co-occurrence matrix from history"""
        for i in range(len(history) - 1):
            curr_experts = [act.expert_id for act in history[i].activations]
            next_experts = [act.expert_id for act in history[i + 1].activations]
            
            for curr in curr_experts:
                for nxt in next_experts:
                    key = (curr, nxt)
                    self.co_occurrence[key] = self.co_occurrence.get(key, 0) + 1
    
    def get_prefetch_priority(
        self,
        expert_id: ExpertID,
        predicted_score: float
    ) -> float:
        """Priority based on co-occurrence frequency"""
        return predicted_score


class ExpertPrefetcher:
    """
    Coordinates expert prefetching using a pluggable strategy.
    Manages async transfers in background.
    """
    # TODO：draft时的draft在draftSchduler中｜分别实现/结合/协调draft和verify期间的prefetch
    # TODO: 重新设计实现
    
    def __init__(
        self,
        strategy: PrefetchStrategy,
        max_concurrent_transfers: int = 4
    ):
        self.strategy = strategy
        self.max_concurrent_transfers = max_concurrent_transfers
        
        # Track ongoing transfers
        self.active_transfers: Dict[ExpertID, float] = {}  # expert_id -> start_time
        self.transfer_queue: List[tuple] = []  # (priority, expert_id)
    
    def prefetch_for_next_layer(
        self,
        current_layer_idx: int,
        current_activations: LayerExpertActivations,
        history: Optional[List[LayerExpertActivations]],
        expert_cache,
        parameter_loader
    ) -> List[ExpertID]:
        """
        Trigger prefetch for next layer's experts.
        
        Returns:
            List of expert IDs that were queued for transfer
        """
        # Get predictions
        predicted_experts = self.strategy.predict_next_experts(
            current_layer_idx,
            current_activations,
            history
        )
        
        # Filter out experts already in cache or being transferred
        to_prefetch = [
            exp for exp in predicted_experts
            if not expert_cache.is_cached(exp) and exp not in self.active_transfers
        ]
        
        if not to_prefetch:
            return []
        
        # Calculate priorities
        priorities = [
            self.strategy.get_prefetch_priority(exp, 1.0)
            for exp in to_prefetch
        ]
        
        # Queue transfers
        for exp, priority in zip(to_prefetch, priorities):
            self.transfer_queue.append((priority, exp))
        
        # Sort queue by priority
        self.transfer_queue.sort(key=lambda x: x[0], reverse=True)
        
        # Start transfers up to concurrent limit
        self._start_pending_transfers(expert_cache, parameter_loader)
        
        return to_prefetch
    
    def _start_pending_transfers(self, expert_cache, parameter_loader) -> None:
        """Start pending transfers up to concurrent limit"""
        import time
        
        while (len(self.active_transfers) < self.max_concurrent_transfers 
               and self.transfer_queue):
            
            priority, expert_id = self.transfer_queue.pop(0)
            
            # Get parameters from CPU
            cpu_params = parameter_loader.get_expert_params(
                expert_id, 
                device=DeviceType.CPU
            )
            
            if cpu_params:
                # Async transfer to GPU cache
                expert_cache.prefetch_async([expert_id], {expert_id: cpu_params})
                self.active_transfers[expert_id] = time.time()
                
                logger.debug(f"Started prefetch transfer for {expert_id}")
    
    def on_transfer_complete(self, expert_id: ExpertID) -> None:
        """Notify that a transfer has completed"""
        if expert_id in self.active_transfers:
            del self.active_transfers[expert_id]