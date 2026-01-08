"""
Performance metrics for Heterogeneous Inference Engine
"""

import time
import psutil
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    prefill_time: float = 0.0
    decode_time: float = 0.0
    total_time: float = 0.0
    tokens_per_second: float = 0.0
    memory_usage: Dict[str, float] = field(default_factory=dict)
    migration_stats: Dict[str, int] = field(default_factory=dict)
    expert_hit_rate: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0


class MetricsCollector:
    """Collect and track performance metrics"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.start_time = None
        self.prefill_start = None
        self.decode_start = None
        self.migration_events = []
        self.expert_hits = 0
        self.expert_total = 0
    
    def start_timing(self):
        """Start overall timing"""
        self.start_time = time.time()
    
    def start_prefill(self):
        """Start prefill timing"""
        self.prefill_start = time.time()
    
    def end_prefill(self):
        """End prefill timing"""
        if self.prefill_start:
            self.metrics.prefill_time = time.time() - self.prefill_start
    
    def start_decode(self):
        """Start decode timing"""
        self.decode_start = time.time()
    
    def end_decode(self):
        """End decode timing"""
        if self.decode_start:
            self.metrics.decode_time = time.time() - self.decode_start
    
    def end_timing(self, num_tokens: int):
        """End overall timing and calculate metrics"""
        if self.start_time:
            self.metrics.total_time = time.time() - self.start_time
            if num_tokens > 0:
                self.metrics.tokens_per_second = num_tokens / self.metrics.total_time
    
    def record_memory_usage(self):
        """Record current memory usage"""
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
            self.metrics.memory_usage.update({
                'gpu_allocated_gb': gpu_memory_allocated,
                'gpu_reserved_gb': gpu_memory_reserved
            })
        
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        self.metrics.memory_usage.update({
            'cpu_used_gb': cpu_memory.used / (1024**3),
            'cpu_available_gb': cpu_memory.available / (1024**3)
        })
    
    def record_migration_event(self, event_type: str, layer_idx: int, expert_idx: int):
        """Record expert migration event"""
        self.migration_events.append({
            'type': event_type,
            'layer_idx': layer_idx,
            'expert_idx': expert_idx,
            'timestamp': time.time()
        })
        
        # Update migration stats
        if event_type not in self.metrics.migration_stats:
            self.metrics.migration_stats[event_type] = 0
        self.metrics.migration_stats[event_type] += 1
    
    def record_expert_usage(self, is_hit: bool):
        """Record expert usage (hit or miss)"""
        self.expert_total += 1
        if is_hit:
            self.expert_hits += 1
        
        if self.expert_total > 0:
            self.metrics.expert_hit_rate = self.expert_hits / self.expert_total
    
    def record_utilization(self):
        """Record GPU and CPU utilization"""
        # GPU utilization (simplified)
        if torch.cuda.is_available():
            # This is a simplified approach - in practice you might want to use nvidia-ml-py
            self.metrics.gpu_utilization = 0.8  # Placeholder
        
        # CPU utilization
        self.metrics.cpu_utilization = psutil.cpu_percent()
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current metrics"""
        return self.metrics
    
    def print_summary(self):
        """Print metrics summary"""
        print("\n" + "="*50)
        print("PERFORMANCE METRICS SUMMARY")
        print("="*50)
        print(f"Prefill Time: {self.metrics.prefill_time:.3f}s")
        print(f"Decode Time: {self.metrics.decode_time:.3f}s")
        print(f"Total Time: {self.metrics.total_time:.3f}s")
        print(f"Tokens per Second: {self.metrics.tokens_per_second:.2f}")
        print(f"Expert Hit Rate: {self.metrics.expert_hit_rate:.3f}")
        print(f"GPU Utilization: {self.metrics.gpu_utilization:.1f}%")
        print(f"CPU Utilization: {self.metrics.cpu_utilization:.1f}%")
        
        print("\nMemory Usage:")
        for key, value in self.metrics.memory_usage.items():
            print(f"  {key}: {value:.2f} GB")
        
        print("\nMigration Statistics:")
        for key, value in self.metrics.migration_stats.items():
            print(f"  {key}: {value}")
        print("="*50)


class Profiler:
    """Simple profiler for timing operations"""
    
    def __init__(self):
        self.timers = defaultdict(list)
        self.current_timers = {}
    
    def start(self, name: str):
        """Start timing an operation"""
        self.current_timers[name] = time.time()
    
    def end(self, name: str):
        """End timing an operation"""
        if name in self.current_timers:
            duration = time.time() - self.current_timers[name]
            self.timers[name].append(duration)
            del self.current_timers[name]
    
    def get_metrics(self) -> Dict[str, float]:
        """Get timing metrics"""
        metrics = {}
        for name, durations in self.timers.items():
            if durations:
                metrics[name] = {
                    'mean': sum(durations) / len(durations),
                    'total': sum(durations),
                    'count': len(durations),
                    'min': min(durations),
                    'max': max(durations)
                }
        return metrics
    
    def reset(self):
        """Reset all timers"""
        self.timers.clear()
        self.current_timers.clear() 