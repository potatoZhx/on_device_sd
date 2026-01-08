"""
Memory manager for heterogeneous inference
"""

import torch
import time
from typing import Dict, List, Optional, Any
from collections import OrderedDict
from ..utils.logger import get_logger
from ..utils.config import MemoryConfig


class KVCache:
    """Key-Value cache for attention mechanism"""
    
    def __init__(self, batch_size: int, seq_len: int, hidden_size: int, device: torch.device):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.device = device
        
        # Initialize cache tensors
        self.k_cache = torch.zeros(batch_size, seq_len, hidden_size, device=device)
        self.v_cache = torch.zeros(batch_size, seq_len, hidden_size, device=device)
        self.valid_length = 0
    
    def update(self, new_k: torch.Tensor, new_v: torch.Tensor, position: int):
        """Update cache with new key-value pairs"""
        if position < self.seq_len:
            self.k_cache[:, position:position+new_k.size(1), :] = new_k
            self.v_cache[:, position:position+new_v.size(1), :] = new_v
            self.valid_length = max(self.valid_length, position + new_k.size(1))
    
    def get_valid_cache(self):
        """Get valid portion of cache"""
        return (
            self.k_cache[:, :self.valid_length, :],
            self.v_cache[:, :self.valid_length, :]
        )
    
    def clear(self):
        """Clear cache"""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.valid_length = 0


class KVCacheManager:
    """Manage KV cache allocation and eviction"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = get_logger()
        self.cache_pool: Dict[str, KVCache] = {}
        self.lru_order: List[str] = []
        self.max_cache_size = config.max_kv_cache_size * (1024**3)  # Convert to bytes
        self.current_cache_size = 0
    
    def allocate(self, batch_size: int, seq_len: int, hidden_size: int, device: torch.device) -> KVCache:
        """Allocate KV cache"""
        cache_key = f"{batch_size}_{seq_len}_{hidden_size}_{device}"
        
        if cache_key in self.cache_pool:
            # Reuse existing cache
            cache = self.cache_pool[cache_key]
            self._update_lru(cache_key)
            return cache
        
        # Calculate cache size
        cache_size = batch_size * seq_len * hidden_size * 2 * 2  # 2 for k,v, 2 for float16
        
        # Check if we need to evict caches
        while self.current_cache_size + cache_size > self.max_cache_size and self.lru_order:
            self._evict_oldest_cache()
        
        # Create new cache
        cache = KVCache(batch_size, seq_len, hidden_size, device)
        self.cache_pool[cache_key] = cache
        self.lru_order.append(cache_key)
        self.current_cache_size += cache_size
        
        self.logger.debug(f"Allocated KV cache: {cache_key}, size: {cache_size / (1024**3):.2f} GB")
        
        return cache
    
    def _evict_oldest_cache(self):
        """Evict oldest cache"""
        if not self.lru_order:
            return
        
        oldest_key = self.lru_order.pop(0)
        if oldest_key in self.cache_pool:
            cache = self.cache_pool[oldest_key]
            cache_size = cache.batch_size * cache.seq_len * cache.hidden_size * 2 * 2
            del self.cache_pool[oldest_key]
            self.current_cache_size -= cache_size
            
            self.logger.debug(f"Evicted KV cache: {oldest_key}")
    
    def _update_lru(self, cache_key: str):
        """Update LRU order"""
        if cache_key in self.lru_order:
            self.lru_order.remove(cache_key)
        self.lru_order.append(cache_key)
    
    def get_memory_usage(self) -> float:
        """Get current cache memory usage in GB"""
        return self.current_cache_size / (1024**3)
    
    def clear_all(self):
        """Clear all caches"""
        self.cache_pool.clear()
        self.lru_order.clear()
        self.current_cache_size = 0
        self.logger.info("Cleared all KV caches")


class ExpertCacheManager:
    """Manage expert module caching"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = get_logger()
        self.expert_cache: Dict[str, torch.nn.Module] = {}
        self.expert_usage: Dict[str, int] = {}
        self.max_cache_size = config.max_expert_cache_size * (1024**3)  # Convert to bytes
        self.current_cache_size = 0
    
    def cache_expert(self, expert_id: str, expert_module: torch.nn.Module) -> bool:
        """Cache expert module"""
        if expert_id in self.expert_cache:
            return True  # Already cached
        
        # Estimate memory usage
        memory_usage = self._estimate_expert_memory(expert_module)
        
        # Check if we need to evict experts
        while self.current_cache_size + memory_usage > self.max_cache_size and self.expert_cache:
            self._evict_least_used_expert()
        
        # Cache the expert
        self.expert_cache[expert_id] = expert_module
        self.expert_usage[expert_id] = 0
        self.current_cache_size += memory_usage
        
        self.logger.debug(f"Cached expert: {expert_id}, memory: {memory_usage / (1024**3):.2f} GB")
        
        return True
    
    def get_cached_expert(self, expert_id: str) -> Optional[torch.nn.Module]:
        """Get cached expert module"""
        if expert_id in self.expert_cache:
            self.expert_usage[expert_id] += 1
            return self.expert_cache[expert_id]
        return None
    
    def _estimate_expert_memory(self, expert_module: torch.nn.Module) -> int:
        """Estimate memory usage of expert module"""
        total_params = sum(p.numel() for p in expert_module.parameters())
        # Assume float16 precision
        return total_params * 2
    
    def _evict_least_used_expert(self):
        """Evict least used expert"""
        if not self.expert_usage:
            return
        
        # Find least used expert
        least_used_id = min(self.expert_usage.keys(), key=lambda x: self.expert_usage[x])
        
        # Remove from cache
        expert_module = self.expert_cache[least_used_id]
        memory_usage = self._estimate_expert_memory(expert_module)
        
        del self.expert_cache[least_used_id]
        del self.expert_usage[least_used_id]
        self.current_cache_size -= memory_usage
        
        self.logger.debug(f"Evicted expert: {least_used_id}")
    
    def evict_lru(self):
        """Evict least recently used expert"""
        self._evict_least_used_expert()
    
    def get_memory_usage(self) -> float:
        """Get current cache memory usage in GB"""
        return self.current_cache_size / (1024**3)
    
    def clear_all(self):
        """Clear all cached experts"""
        self.expert_cache.clear()
        self.expert_usage.clear()
        self.current_cache_size = 0
        self.logger.info("Cleared all expert caches")


class MemoryManager:
    """Main memory manager for heterogeneous inference"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = get_logger()
        self.kv_cache_manager = KVCacheManager(config)
        self.expert_cache_manager = ExpertCacheManager(config)
        
        self.logger.info("Initialized MemoryManager")
    
    def allocate_kv_cache(self, batch_size: int, seq_len: int, hidden_size: int, device: torch.device) -> KVCache:
        """Allocate KV cache"""
        return self.kv_cache_manager.allocate(batch_size, seq_len, hidden_size, device)
    
    def cache_expert(self, expert_id: str, expert_module: torch.nn.Module) -> bool:
        """Cache expert module"""
        return self.expert_cache_manager.cache_expert(expert_id, expert_module)
    
    def get_cached_expert(self, expert_id: str) -> Optional[torch.nn.Module]:
        """Get cached expert module"""
        return self.expert_cache_manager.get_cached_expert(expert_id)
    
    def evict_least_used(self):
        """Evict least used caches"""
        self.expert_cache_manager.evict_lru()
        self.kv_cache_manager._evict_oldest_cache()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information"""
        return {
            "kv_cache_gb": self.kv_cache_manager.get_memory_usage(),
            "expert_cache_gb": self.expert_cache_manager.get_memory_usage(),
            "total_cache_gb": (self.kv_cache_manager.get_memory_usage() + 
                              self.expert_cache_manager.get_memory_usage())
        }
    
    def log_memory_status(self):
        """Log current memory status"""
        usage = self.get_memory_usage()
        self.logger.info("Memory Manager Status:")
        for key, value in usage.items():
            self.logger.info(f"  {key}: {value:.2f} GB")
    
    def clear_all(self):
        """Clear all caches"""
        self.kv_cache_manager.clear_all()
        self.expert_cache_manager.clear_all() 