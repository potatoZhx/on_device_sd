"""
Expert scheduling optimization for heterogeneous inference
"""

import torch
import numpy as np
import time
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from ..utils.logger import get_logger
from ..utils.config import SchedulerConfig


@dataclass
class ExpertStats:
    """Statistics for an expert"""
    usage_history: List[float] = None
    selection_count: int = 0
    total_invocations: int = 0
    last_used: float = 0.0
    
    def __post_init__(self):
        if self.usage_history is None:
            self.usage_history = []
    
    def update_usage(self, usage: float, is_selected: bool):
        """Update usage statistics"""
        self.usage_history.append(usage)
        self.total_invocations += 1
        if is_selected:
            self.selection_count += 1
        self.last_used = time.time()
    
    def get_hit_rate(self) -> float:
        """Get hit rate"""
        if self.total_invocations == 0:
            return 0.0
        return self.selection_count / self.total_invocations
    
    def get_avg_usage(self) -> float:
        """Get average usage"""
        if not self.usage_history:
            return 0.0
        return sum(self.usage_history) / len(self.usage_history)


@dataclass
class ExpertSchedule:
    """Expert scheduling decision"""
    gpu_experts: Set[int] = None
    cpu_experts: Set[int] = None
    migration_decisions: List[Tuple[int, str]] = None  # (expert_idx, target_device)
    
    def __post_init__(self):
        if self.gpu_experts is None:
            self.gpu_experts = set()
        if self.cpu_experts is None:
            self.cpu_experts = set()
        if self.migration_decisions is None:
            self.migration_decisions = []
    
    def add_gpu_expert(self, expert_idx: int):
        """Add expert to GPU"""
        self.gpu_experts.add(expert_idx)
        if expert_idx in self.cpu_experts:
            self.cpu_experts.remove(expert_idx)
            self.migration_decisions.append((expert_idx, "cuda"))
    
    def add_cpu_expert(self, expert_idx: int):
        """Add expert to CPU"""
        self.cpu_experts.add(expert_idx)
        if expert_idx in self.gpu_experts:
            self.gpu_experts.remove(expert_idx)
            self.migration_decisions.append((expert_idx, "cpu"))


class ExpertPopularityTracker:
    """Track expert popularity for scheduling decisions"""
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.logger = get_logger()
        self.layer_expert_stats: Dict[str, ExpertStats] = {}
        self.popularity_window = config.popularity_window
    
    def update_popularity(self, layer_idx: int, routing_weights: torch.Tensor):
        """Update expert popularity based on routing weights"""
        batch_size, seq_len, num_experts = routing_weights.shape
        
        for expert_idx in range(num_experts):
            expert_key = f"{layer_idx}_{expert_idx}"
            
            # Calculate usage intensity
            expert_usage = routing_weights[:, :, expert_idx].sum().item()
            
            # Update statistics
            if expert_key not in self.layer_expert_stats:
                self.layer_expert_stats[expert_key] = ExpertStats()
            
            stats = self.layer_expert_stats[expert_key]
            stats.update_usage(expert_usage, expert_usage > 0)
            
            # Maintain sliding window
            if len(stats.usage_history) > self.popularity_window:
                stats.usage_history.pop(0)
    
    def get_popular_experts(self, layer_idx: int, top_k: int = None) -> List[int]:
        """Get most popular experts for a layer"""
        if top_k is None:
            top_k = self.config.top_k_experts
        
        layer_experts = {}
        for key, stats in self.layer_expert_stats.items():
            if key.startswith(f"{layer_idx}_"):
                expert_idx = int(key.split("_")[1])
                # Calculate weighted popularity score
                if stats.usage_history:
                    # Recent usage has higher weight
                    weights = torch.softmax(torch.arange(len(stats.usage_history), dtype=torch.float), dim=0)
                    weighted_score = sum(w * u for w, u in zip(weights, stats.usage_history))
                    layer_experts[expert_idx] = weighted_score.item()
                else:
                    layer_experts[expert_idx] = 0.0
        
        # Sort by popularity
        sorted_experts = sorted(layer_experts.items(), key=lambda x: x[1], reverse=True)
        return [expert_idx for expert_idx, _ in sorted_experts[:top_k]]
    
    def get_expert_hit_rate(self, layer_idx: int, expert_idx: int) -> float:
        """Get hit rate for a specific expert"""
        expert_key = f"{layer_idx}_{expert_idx}"
        if expert_key in self.layer_expert_stats:
            return self.layer_expert_stats[expert_key].get_hit_rate()
        return 0.0
    
    def get_expert_popularity_score(self, layer_idx: int, expert_idx: int) -> float:
        """Get popularity score for a specific expert"""
        expert_key = f"{layer_idx}_{expert_idx}"
        if expert_key in self.layer_expert_stats:
            stats = self.layer_expert_stats[expert_key]
            if stats.usage_history:
                return sum(stats.usage_history) / len(stats.usage_history)
        return 0.0


class HeterogeneousScheduler:
    """Main heterogeneous scheduler for expert placement"""
    
    def __init__(self, config: SchedulerConfig, device_manager, memory_manager):
        self.config = config
        self.logger = get_logger()
        self.device_manager = device_manager
        self.memory_manager = memory_manager
        self.expert_tracker = ExpertPopularityTracker(config)
        
        # Expert location tracking
        self.expert_locations: Dict[str, str] = {}  # {expert_key: device}
        self.layer_expert_counts = defaultdict(int)
        
        self.logger.info("Initialized HeterogeneousScheduler")
    
    def schedule_experts(self, layer_idx: int, routing_weights: torch.Tensor) -> ExpertSchedule:
        """Schedule experts based on routing weights and popularity"""
        # Update popularity statistics
        self.expert_tracker.update_popularity(layer_idx, routing_weights)
        
        # Get popular experts
        popular_experts = self.expert_tracker.get_popular_experts(layer_idx)
        
        # Create scheduling decision
        schedule = ExpertSchedule()
        
        # Allocate experts to devices
        self._allocate_experts_to_devices(layer_idx, popular_experts, schedule)
        
        return schedule
    
    def _allocate_experts_to_devices(self, layer_idx: int, popular_experts: List[int], 
                                   schedule: ExpertSchedule):
        """Allocate experts to CPU/GPU devices"""
        gpu_capacity = self.device_manager.get_gpu_memory_capacity()
        cpu_capacity = self.device_manager.get_cpu_memory_capacity()
        
        # First, ensure each layer has at least some experts on GPU
        min_gpu_experts = max(1, len(popular_experts) // 3)  # At least 1/3 on GPU
        
        for i, expert_idx in enumerate(popular_experts):
            expert_key = f"{layer_idx}_{expert_idx}"
            current_device = self.expert_locations.get(expert_key, "cpu")
            
            # Estimate memory usage
            memory_usage = self._estimate_expert_memory_usage(expert_idx)
            
            # Decision logic
            should_use_gpu = False
            
            if i < min_gpu_experts:
                # Force popular experts to GPU if possible
                should_use_gpu = gpu_capacity >= memory_usage
            else:
                # Use popularity and memory availability
                popularity_score = self.expert_tracker.get_expert_popularity_score(layer_idx, expert_idx)
                should_use_gpu = (popularity_score > 0.5 and gpu_capacity >= memory_usage)
            
            if should_use_gpu:
                schedule.add_gpu_expert(expert_idx)
                self.expert_locations[expert_key] = "cuda"
                gpu_capacity -= memory_usage
            else:
                schedule.add_cpu_expert(expert_idx)
                self.expert_locations[expert_key] = "cpu"
                cpu_capacity -= memory_usage
    
    def _estimate_expert_memory_usage(self, expert_idx: int) -> int:
        """Estimate memory usage for an expert (simplified)"""
        # This is a simplified estimation - in practice, you'd get the actual expert module
        # For now, assume a fixed size per expert
        return 100 * 1024 * 1024  # 100MB per expert
    
    def update_expert_locations(self, layer_idx: int):
        """Update expert locations based on current popularity"""
        popular_experts = self.expert_tracker.get_popular_experts(layer_idx)
        
        # Get current GPU experts for this layer
        current_gpu_experts = set()
        for expert_key, device in self.expert_locations.items():
            if expert_key.startswith(f"{layer_idx}_") and device == "cuda":
                expert_idx = int(expert_key.split("_")[1])
                current_gpu_experts.add(expert_idx)
        
        # Calculate which experts should be on GPU
        target_gpu_experts = set(popular_experts[:self.config.top_k_experts])
        
        # Determine migrations needed
        experts_to_gpu = target_gpu_experts - current_gpu_experts
        experts_to_cpu = current_gpu_experts - target_gpu_experts
        
        # Log migration decisions
        if experts_to_gpu:
            self.logger.info(f"Layer {layer_idx}: Moving experts {experts_to_gpu} to GPU")
        if experts_to_cpu:
            self.logger.info(f"Layer {layer_idx}: Moving experts {experts_to_cpu} to CPU")
        
        # Update locations
        for expert_idx in experts_to_gpu:
            expert_key = f"{layer_idx}_{expert_idx}"
            self.expert_locations[expert_key] = "cuda"
        
        for expert_idx in experts_to_cpu:
            expert_key = f"{layer_idx}_{expert_idx}"
            self.expert_locations[expert_key] = "cpu"
    
    def get_expert_device(self, layer_idx: int, expert_idx: int) -> str:
        """Get current device for an expert"""
        expert_key = f"{layer_idx}_{expert_idx}"
        return self.expert_locations.get(expert_key, "cpu")
    
    def get_layer_expert_distribution(self, layer_idx: int) -> Dict[str, int]:
        """Get expert distribution for a layer"""
        gpu_count = 0
        cpu_count = 0
        
        for expert_key, device in self.expert_locations.items():
            if expert_key.startswith(f"{layer_idx}_"):
                if device == "cuda":
                    gpu_count += 1
                else:
                    cpu_count += 1
        
        return {"gpu": gpu_count, "cpu": cpu_count}
    
    def log_scheduling_stats(self):
        """Log current scheduling statistics"""
        total_experts = len(self.expert_locations)
        gpu_experts = sum(1 for device in self.expert_locations.values() if device == "cuda")
        cpu_experts = total_experts - gpu_experts
        
        self.logger.info(f"Scheduling Stats - Total: {total_experts}, GPU: {gpu_experts}, CPU: {cpu_experts}")
        
        # Log per-layer distribution
        layer_stats = defaultdict(lambda: {"gpu": 0, "cpu": 0})
        for expert_key, device in self.expert_locations.items():
            layer_idx = int(expert_key.split("_")[0])
            layer_stats[layer_idx][device] += 1
        
        for layer_idx, stats in sorted(layer_stats.items()):
            self.logger.info(f"  Layer {layer_idx}: GPU={stats['gpu']}, CPU={stats['cpu']}") 