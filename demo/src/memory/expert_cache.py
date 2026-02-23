from typing import Dict, List, Optional, Set
import torch
from collections import OrderedDict
from ..core.types import ExpertID, DeviceType
from ..scheduling.cache_strategy import CacheReplacementStrategy
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ExpertCache:
    """
    Manages the GPU cache for expert parameters.
    Handles loading, eviction, and tracking of experts in GPU memory.
    """
    
    def __init__(
        self,
        max_cache_size_gb: float,
        expert_size_mb: float,
        replacement_strategy: CacheReplacementStrategy,
        pin_shared_experts: bool = True
    ):
        self.max_cache_size = int(max_cache_size_gb * 1024)  # Convert to MB
        self.expert_size = expert_size_mb
        self.max_experts = int(self.max_cache_size / self.expert_size)
        self.replacement_strategy = replacement_strategy
        self.pin_shared_experts = pin_shared_experts
        
        # ? TODO Dict[str, torch.Tensor] str是什么；这里的实现中cache的gpu内存是预分配好的吗
        # Cache storage
        self.cached_experts: OrderedDict[ExpertID, Dict[str, torch.Tensor]] = OrderedDict()
        self.pinned_experts: Set[ExpertID] = set()  # Shared experts that cannot be evicted
        self.pending_transfers: Dict[ExpertID, tuple] = {}
        self.transfer_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"ExpertCache initialized: max {self.max_experts} experts "
                   f"({max_cache_size_gb:.2f} GB)")
    
    def is_cached(self, expert_id: ExpertID) -> bool:
        """Check if expert is in GPU cache"""
        return expert_id in self.cached_experts
    
    # TODO expert每次计算都要用get overhead是否会很大
    def get(self, expert_id: ExpertID) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieve expert from cache.
        Updates access patterns for replacement strategy.
        """
        if expert_id in self.cached_experts:
            self.cache_hits += 1
            self.replacement_strategy.on_access(expert_id)
            return self.cached_experts[expert_id]
        if expert_id in self.pending_transfers:
            event, gpu_params = self.pending_transfers[expert_id]
            if event is None or event.query():
                self._finalize_transfer(expert_id)
                if expert_id in self.cached_experts:
                    self.cache_hits += 1
                    self.replacement_strategy.on_access(expert_id)
                    return self.cached_experts[expert_id]
            self.cache_misses += 1
            return None
        else:
            self.cache_misses += 1
            return None
    
    def put(
        self, 
        expert_id: ExpertID, 
        expert_params: Dict[str, torch.Tensor],
        is_pinned: bool = False
    ) -> bool:
        """
        Add expert to cache. Returns True if successful.
        May trigger eviction if cache is full.
        """
        # Check if already cached
        if expert_id in self.cached_experts:
            logger.debug(f"Expert {expert_id} already in cache")
            return True
        
        # Make room if necessary
        while len(self.cached_experts) >= self.max_experts:
            if not self._evict_one():
                logger.warning("Cannot evict any expert, cache full")
                return False
        
        # TODO：现在的实现是用.cuda实现参数加载
        # Ensure parameters are on GPU
        gpu_params = {
            k: v.cuda() if not v.is_cuda else v 
            for k, v in expert_params.items()
        }
        
        # Add to cache
        self.cached_experts[expert_id] = gpu_params
        
        if is_pinned:
            self.pinned_experts.add(expert_id)
        
        # Notify replacement strategy
        self.replacement_strategy.on_insert(expert_id)
        
        logger.debug(f"Cached expert {expert_id} (pinned={is_pinned})")
        return True
    
    def put_batch(
        self, 
        expert_params: Dict[ExpertID, Dict[str, torch.Tensor]]
    ) -> Dict[ExpertID, bool]:
        """
        Batch insertion of multiple experts.
        Returns dict mapping expert_id -> success status.
        """
        # TODO： 并未batch
        results = {}
        for expert_id, params in expert_params.items():
            results[expert_id] = self.put(expert_id, params)
        return results
    
    def evict(self, expert_id: ExpertID) -> bool:
        """Manually evict a specific expert"""
        if expert_id in self.pinned_experts:
            logger.warning(f"Cannot evict pinned expert {expert_id}")
            return False
        
        if expert_id in self.cached_experts:
            # TODO：check put传入的对象是否为副本，直接删掉会不会导致专家丢失；
            # TODO：每一个expert的加载驱逐都需要创建/销毁对象，传输完全依赖torch，是否需要更底层地管理gpu mem
            del self.cached_experts[expert_id]
            self.replacement_strategy.on_evict(expert_id)
            logger.debug(f"Evicted expert {expert_id}")
            return True
        
        return False
    
    def _evict_one(self) -> bool:
        """Evict one expert according to replacement strategy"""
        # Get eviction candidate from strategy
        candidate = self.replacement_strategy.select_victim(
            cached_experts=list(self.cached_experts.keys()),
            pinned_experts=self.pinned_experts
        )
        
        if candidate is None:
            return False
        
        return self.evict(candidate)

    def _put_gpu_params(
        self,
        expert_id: ExpertID,
        expert_params: Dict[str, torch.Tensor],
        is_pinned: bool = False
    ) -> bool:
        if expert_id in self.cached_experts:
            return True
        while len(self.cached_experts) >= self.max_experts:
            if not self._evict_one():
                return False
        self.cached_experts[expert_id] = expert_params
        if is_pinned:
            self.pinned_experts.add(expert_id)
        self.replacement_strategy.on_insert(expert_id)
        return True

    def _finalize_transfer(self, expert_id: ExpertID) -> bool:
        if expert_id not in self.pending_transfers:
            return False
        event, gpu_params = self.pending_transfers[expert_id]
        if event is not None:
            event.synchronize()
        success = self._put_gpu_params(expert_id, gpu_params)
        del self.pending_transfers[expert_id]
        return success
    
    def prefetch_async(
        self, 
        expert_ids: List[ExpertID],
        source_params: Dict[ExpertID, Dict[str, torch.Tensor]]
    ) -> None:
        """
        Asynchronously prefetch experts to GPU cache.
        Non-blocking operation using CUDA streams.
        """
        for expert_id in expert_ids:
            if expert_id in self.cached_experts or expert_id in self.pending_transfers:
                continue
            if expert_id not in source_params:
                continue
            if self.transfer_stream is None:
                self.put(expert_id, source_params[expert_id])
                continue
            while len(self.cached_experts) + len(self.pending_transfers) >= self.max_experts:
                if not self._evict_one():
                    return
            with torch.cuda.stream(self.transfer_stream):
                gpu_params = {
                    k: v.to(device="cuda", non_blocking=True)
                    for k, v in source_params[expert_id].items()
                }
                event = torch.cuda.Event()
                event.record(self.transfer_stream)
            self.pending_transfers[expert_id] = (event, gpu_params)

    def complete_ready_transfers(self) -> List[ExpertID]:
        completed = []
        for expert_id, (event, _) in list(self.pending_transfers.items()):
            if event is None or event.query():
                if self._finalize_transfer(expert_id):
                    completed.append(expert_id)
        return completed


class AsyncExpertTransfer:
    def __init__(self, max_concurrent: int = 2):
        self.max_concurrent = max_concurrent
        self.transfer_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.pending_events: Dict[ExpertID, tuple] = {}

    def start_transfer(
        self,
        expert_id: ExpertID,
        cpu_params: Dict[str, torch.Tensor],
        expert_cache: ExpertCache,
    ) -> None:
        if expert_id in self.pending_events:
            return
        if len(self.pending_events) >= self.max_concurrent:
            self._wait_oldest(expert_cache)
        if self.transfer_stream is None:
            expert_cache.put(expert_id, cpu_params)
            return
        with torch.cuda.stream(self.transfer_stream):
            gpu_params = {
                k: v.to('cuda', non_blocking=True)
                for k, v in cpu_params.items()
            }
            event = torch.cuda.Event()
            event.record(self.transfer_stream)
        self.pending_events[expert_id] = (event, gpu_params)

    def poll_completed(self, expert_cache: ExpertCache) -> List[ExpertID]:
        completed = []
        for expert_id, (event, gpu_params) in list(self.pending_events.items()):
            if event.query():
                expert_cache.put(expert_id, gpu_params)
                completed.append(expert_id)
        for expert_id in completed:
            del self.pending_events[expert_id]
        return completed

    def wait_all(self, expert_cache: ExpertCache) -> None:
        for expert_id, (event, gpu_params) in list(self.pending_events.items()):
            event.synchronize()
            expert_cache.put(expert_id, gpu_params)
        self.pending_events.clear()

    def _wait_oldest(self, expert_cache: ExpertCache) -> None:
        if not self.pending_events:
            return
        oldest_id = next(iter(self.pending_events))
        event, gpu_params = self.pending_events.pop(oldest_id)
        event.synchronize()
        expert_cache.put(oldest_id, gpu_params)
    
    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache performance statistics"""
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0.0
        
        return {
            'hit_rate': hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cached_experts': len(self.cached_experts),
            'utilization': len(self.cached_experts) / self.max_experts
        }
    
    def reset_stats(self) -> None:
        """Reset cache statistics"""
        self.cache_hits = 0
        self.cache_misses = 0
    
    def clear(self, keep_pinned: bool = True) -> None:
        """Clear cache, optionally keeping pinned experts"""
        if keep_pinned:
            to_remove = [
                eid for eid in self.cached_experts.keys() 
                if eid not in self.pinned_experts
            ]
            for eid in to_remove:
                del self.cached_experts[eid]
        else:
            self.cached_experts.clear()
            self.pinned_experts.clear()
        
        logger.info("Expert cache cleared")
