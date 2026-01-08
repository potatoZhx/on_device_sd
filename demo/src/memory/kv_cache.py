from typing import Dict, List, Optional, Tuple
import torch
from ..core.types import ExecutionPhase
from ..core.model import MoEConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)

class KVCache:
    """
    Manages Key-Value cache for attention layers.
    Supports draft-verify with cache replacement.
    """
    
    def __init__(self, config: MoEConfig, max_batch_size: int = 1):
        self.config = config
        self.max_batch_size = max_batch_size
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_seq_len = config.max_seq_length
        
        # TODO：目前的实现预先分配了所有cache；？形状对应？ 参考nano-vllm考虑修改
        # KV cache storage: [num_layers][2][batch, num_heads, seq_len, head_dim]
        # [2] for key and value
        self.cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self.current_length = 0  # Current sequence length in cache
        
        # TODO：？verify的时候不需要draft cache
        # Draft cache backup (for verify phase)
        self.draft_cache_backup: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self.draft_start_position: int = 0
        
        self._initialize_cache()
    
    def _initialize_cache(self) -> None:
        """Initialize empty KV cache on GPU"""
        # TODO：固定形状可能导致bs小时有内存但无法推理更长
        self.cache = []
        for _ in range(self.num_layers):
            key_cache = torch.zeros(
                self.max_batch_size, 
                self.num_heads, 
                self.max_seq_len, 
                self.head_dim,
                dtype=torch.float16,
                device='cuda'
            )
            value_cache = torch.zeros(
                self.max_batch_size, 
                self.num_heads, 
                self.max_seq_len, 
                self.head_dim,
                dtype=torch.float16,
                device='cuda'
            )
            self.cache.append((key_cache, value_cache))
        
        self.current_length = 0
        logger.debug("KV cache initialized")
    
    def append(
        self        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        start_pos: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new key-value states to cache for a specific layer.
        
        Args:
            layer_idx: Layer index
            key_states: [batch, num_heads, seq_len, head_dim]
            value_states: [batch, num_heads, seq_len, head_dim]
            start_pos: Starting position in cache (if None, use current_length)
        
        Returns:
            Full key and value tensors including cached values
        """
        if start_pos is None:
            start_pos = self.current_length
        
        seq_len = key_states.shape[2]
        end_pos = start_pos + seq_len
        
        # Store in cache
        self.cache[layer_idx][0][:, :, start_pos:end_pos, :] = key_states
        self.cache[layer_idx][1][:, :, start_pos:end_pos, :] = value_states
        
        # Return full cache up to end_pos
        full_keys = self.cache[layer_idx][0][:, :, :end_pos, :]
        full_values = self.cache[layer_idx][1][:, :, :end_pos, :]
        
        return full_keys, full_values
    
    def get(self, layer_idx: int, end_pos: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve cached key-value states for a layer.
        
        Args:
            layer_idx: Layer index
            end_pos: End position (if None, use current_length)
        
        Returns:
            Cached key and value tensors
        """
        if end_pos is None:
            end_pos = self.current_length
        
        keys = self.cache[layer_idx][0][:, :, :end_pos, :]
        values = self.cache[layer_idx][1][:, :, :end_pos, :]
        
        return keys, values
    
    def update_length(self, new_length: int) -> None:
        """Update the current sequence length"""
        # TODO：没有使用
        self.current_length = new_length
    
    def backup_for_draft(self) -> None:
        """
        Backup current cache state before starting draft phase.
        This allows reverting if draft tokens are rejected.
        """
        self.draft_start_position = self.current_length
        # TODO：备份verify的kv cache不应该复制，记录位置即可，draft不会改变前面的cache，参考预实验代码
        # Deep copy the cache
        self.draft_cache_backup = [
            (k.clone(), v.clone()) for k, v in self.cache
        ]
        logger.debug(f"Backed up KV cache at position {self.draft_start_position}")
    
    def restore_from_backup(self) -> None:
        """Restore cache from backup (when draft tokens rejected)"""
        if self.draft_cache_backup is None:
            logger.warning("No backup to restore from")
            return
        
        # TODO：原cache是否会自动释放
        self.cache = self.draft_cache_backup
        self.current_length = self.draft_start_position
        self.draft_cache_backup = None
        logger.debug(f"Restored KV cache to position {self.current_length}")
    
    def replace_draft_with_verify(
        self,
        verify_cache: 'KVCache',
        num_accepted_tokens: int
    ) -> None:
        """
        Replace draft cache with verified cache.
        
        Args:
            verify_cache: KV cache from verify phase
            num_accepted_tokens: Number of tokens accepted from draft
        """
        # The verify cache contains the correct KV for all tokens
        # Replace our cache with the verify cache up to the accepted position
        # TODO：似乎不是增量更新
        for layer_idx in range(self.num_layers):
            end_pos = self.draft_start_position + num_accepted_tokens
            
            # Copy verified KV states
            self.cache[layer_idx][0][:, :, :end_pos, :] = \
                verify_cache.cache[layer_idx][0][:, :, :end_pos, :]
            self.cache[layer_idx][1][:, :, :end_pos, :] = \
                verify_cache.cache[layer_idx][1][:, :, :end_pos, :]
        
        self.current_length = self.draft_start_position + num_accepted_tokens
        self.draft_cache_backup = None
        
        logger.debug(f"Replaced draft cache with verified cache, "
                    f"new length: {self.current_length}")
    
    def clear(self) -> None:
        """Clear all cached states"""
        self._initialize_cache()
    
    def get_memory_usage_mb(self) -> float:
        """Calculate current memory usage in MB"""
        if self.cache is None:
            return 0.0
        
        total_elements = (
            self.num_layers * 2 *  # key and value
            self.max_batch_size * 
            self.num_heads * 
            self.current_length * 
            self.head_dim
        )
        
        # Assuming float16 (2 bytes per element)
        return (total_elements * 2) / (1024 * 1024)

