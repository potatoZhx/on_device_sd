import time
from typing import List, Set, Optional, Dict
from collections import deque
from .base_scheduler import CacheReplacementStrategy
from ..core.types import ExpertID
from ..utils.logger import get_logger

logger = get_logger(__name__)

# TODO: 替换需要层感知
# TODO: 测试不同策略效果（命中率，推理速度）
class LRUCacheStrategy(CacheReplacementStrategy):
    """Least Recently Used cache replacement strategy"""
    
    def __init__(self):
        super().__init__()
        self.access_order: deque = deque()  # Most recent at the end
    
    def select_victim(
        self,
        cached_experts: List[ExpertID],
        pinned_experts: Set[ExpertID]
    ) -> Optional[ExpertID]:
        """Select least recently used expert"""
        # Find LRU expert that is not pinned
        for expert_id in self.access_order:
            if expert_id in cached_experts and expert_id not in pinned_experts:
                return expert_id
        
        # Fallback: select first non-pinned expert
        for expert_id in cached_experts:
            if expert_id not in pinned_experts:
                return expert_id
        
        return None
    
    def on_access(self, expert_id: ExpertID) -> None:
        """Update access order"""
        super().on_access(expert_id)
        
        # Remove if exists and re-add to end
        if expert_id in self.access_order:
            self.access_order.remove(expert_id)
        self.access_order.append(expert_id)
    
    def on_insert(self, expert_id: ExpertID) -> None:
        """Track insertion"""
        super().on_insert(expert_id)
        if expert_id not in self.access_order:
            self.access_order.append(expert_id)
    
    def on_evict(self, expert_id: ExpertID) -> None:
        """Remove from tracking"""
        super().on_evict(expert_id)
        if expert_id in self.access_order:
            self.access_order.remove(expert_id)


class LFUCacheStrategy(CacheReplacementStrategy):
    """Least Frequently Used cache replacement strategy"""
    
    def __init__(self):
        super().__init__()
        self.access_count: Dict[ExpertID, int] = {}
    
    def select_victim(
        self,
        cached_experts: List[ExpertID],
        pinned_experts: Set[ExpertID]
    ) -> Optional[ExpertID]:
        """Select least frequently used expert"""
        candidates = [
            exp for exp in cached_experts 
            if exp not in pinned_experts
        ]
        
        if not candidates:
            return None
        
        # Find expert with minimum access count
        return min(candidates, key=lambda e: self.access_count.get(e, 0))
    
    def on_access(self, expert_id: ExpertID) -> None:
        """Increment access count"""
        super().on_access(expert_id)
        self.access_count[expert_id] = self.access_count.get(expert_id, 0) + 1
    
    def on_insert(self, expert_id: ExpertID) -> None:
        """Initialize count"""
        super().on_insert(expert_id)
        if expert_id not in self.access_count:
            self.access_count[expert_id] = 0
    
    def on_evict(self, expert_id: ExpertID) -> None:
        """Remove from tracking"""
        super().on_evict(expert_id)
        if expert_id in self.access_count:
            del self.access_count[expert_id]


class AdaptiveCacheStrategy(CacheReplacementStrategy):
    """
    Adaptive cache replacement combining frequency and recency.
    Uses a weighted score: score = frequency * recency_weight
    """
    
    def __init__(self, recency_weight: float = 0.5):
        super().__init__()
        self.recency_weight = recency_weight
        self.access_count: Dict[ExpertID, int] = {}
        self.last_access_time: Dict[ExpertID, float] = {}
    
    def select_victim(
        self,
        cached_experts: List[ExpertID],
        pinned_experts: Set[ExpertID]
    ) -> Optional[ExpertID]:
        """Select expert with lowest combined score"""
        candidates = [
            exp for exp in cached_experts 
            if exp not in pinned_experts
        ]
        
        if not candidates:
            return None
        
        current_time = time.time()
        
        # Calculate scores
        scores = {}
        for expert_id in candidates:
            freq = self.access_count.get(expert_id, 0)
            last_access = self.last_access_time.get(expert_id, 0)
            recency = 1.0 / (current_time - last_access + 1)  # More recent = higher
            
            scores[expert_id] = freq * (1 - self.recency_weight) + recency * self.recency_weight
        
        # Return expert with lowest score
        victim = min(scores.items(), key=lambda x: x[1])[0]
        
        logger.debug(f"Selected victim {victim} with score {scores[victim]:.4f}")
        return victim
    
    def on_access(self, expert_id: ExpertID) -> None:
        """Update frequency and recency"""
        super().on_access(expert_id)
        self.access_count[expert_id] = self.access_count.get(expert_id, 0) + 1
        self.last_access_time[expert_id] = time.time()
    
    def on_insert(self, expert_id: ExpertID) -> None:
        """Initialize tracking"""
        super().on_insert(expert_id)
        self.access_count[expert_id] = 0
        self.last_access_time[expert_id] = time.time()
    
    def on_evict(self, expert_id: ExpertID) -> None:
        """Remove from tracking"""
        super().on_evict(expert_id)
        if expert_id in self.access_count:
            del self.access_count[expert_id]
        if expert_id in self.last_access_time:
            del self.last_access_time[expert_id]


class PredictiveCacheStrategy(CacheReplacementStrategy):
    """
    Predictive cache replacement using activation patterns.
    Predicts future usage and keeps likely-to-be-used experts.
    """
    
    def __init__(self, prediction_window: int = 5):
        super().__init__()
        self.prediction_window = prediction_window
        self.access_sequences: List[ExpertID] = []
        self.future_usage_prob: Dict[ExpertID, float] = {}
    
    def select_victim(
        self,
        cached_experts: List[ExpertID],
        pinned_experts: Set[ExpertID]
    ) -> Optional[ExpertID]:
        """Select expert with lowest predicted future usage"""
        candidates = [
            exp for exp in cached_experts 
            if exp not in pinned_experts
        ]
        
        if not candidates:
            return None
        
        # Update predictions based on recent access patterns
        self._update_predictions()
        
        # Select expert with lowest future usage probability
        victim = min(
            candidates,
            key=lambda e: self.future_usage_prob.get(e, 0.0)
        )
        
        logger.debug(f"Selected victim {victim} with future usage prob "
                    f"{self.future_usage_prob.get(victim, 0.0):.4f}")
        
        return victim
    
    def on_access(self, expert_id: ExpertID) -> None:
        """Track access sequence"""
        super().on_access(expert_id)
        self.access_sequences.append(expert_id)
        
        # Keep only recent accesses
        if len(self.access_sequences) > 1000:
            self.access_sequences = self.access_sequences[-1000:]
    
    # TODO: 学习，可参考
    def _update_predictions(self) -> None:
        """Update future usage predictions based on patterns"""
        if len(self.access_sequences) < self.prediction_window * 2:
            return
        
        # Simple Markov chain prediction
        transition_counts: Dict[tuple, int] = {}
        
        for i in range(len(self.access_sequences) - self.prediction_window):
            pattern = tuple(self.access_sequences[i:i + self.prediction_window])
            next_expert = self.access_sequences[i + self.prediction_window]
            
            key = (pattern, next_expert)
            transition_counts[key] = transition_counts.get(key, 0) + 1
        
        # Calculate probabilities
        recent_pattern = tuple(self.access_sequences[-self.prediction_window:])
        
        for expert_id in set(self.access_sequences):
            key = (recent_pattern, expert_id)
            count = transition_counts.get(key, 0)
            self.future_usage_prob[expert_id] = count / max(len(transition_counts), 1)