"""
PagedAttention KV Cache Manager
参考 nano-vllm 实现，支持 flash_attn 和 draft-verify 特殊逻辑
"""

from typing import Dict, List, Optional, Tuple
from collections import deque
import torch

from ..core.model import MoEConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


# ==================== KV Storage Helper ====================

def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor
):
    """
    Store KV states into cache using slot mapping
    使用简单的 PyTorch 实现，后续可以优化为 triton kernel
    
    Args:
        key, value: [N, num_heads, head_dim]
        k_cache, v_cache: [total_slots, num_heads*head_dim] (flattened cache)
        slot_mapping: [N]
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    
    # Reshape to [N, D]
    key_flat = key.reshape(N, D)
    value_flat = value.reshape(N, D)
    
    # Store using slot mapping
    for i in range(N):
        slot = slot_mapping[i].item()
        if slot != -1:
            k_cache[slot] = key_flat[i]
            v_cache[slot] = value_flat[i]


# ==================== Block Manager ====================

class Block:
    """Physical memory block for KV cache"""
    
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
    
    def reset(self):
        """Reset block for reuse"""
        self.ref_count = 1
    
    def inc_ref(self):
        """Increment reference count"""
        self.ref_count += 1
    
    def dec_ref(self):
        """Decrement reference count"""
        self.ref_count -= 1
        return self.ref_count == 0


class BlockManager:
    """
    Manages physical blocks of KV cache
    参考 nano-vllm 的实现，简化了 prefix caching 逻辑
    """
    
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque = deque(range(num_blocks))
        self.used_block_ids: set = set()
    
    def can_allocate(self, num_blocks: int) -> bool:
        """Check if we can allocate num_blocks"""
        return len(self.free_block_ids) >= num_blocks
    
    def allocate_block(self) -> int:
        """Allocate a single block"""
        if not self.free_block_ids:
            raise RuntimeError("No free blocks available")
        
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id
    
    def deallocate_block(self, block_id: int):
        """Deallocate a single block"""
        assert block_id in self.used_block_ids
        block = self.blocks[block_id]
        assert block.ref_count == 0
        
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
    
    def inc_ref(self, block_id: int):
        """Increment block reference count"""
        self.blocks[block_id].inc_ref()
    
    def dec_ref(self, block_id: int) -> bool:
        """Decrement block reference count, return True if should deallocate"""
        return self.blocks[block_id].dec_ref()
    
    def get_num_free_blocks(self) -> int:
        """Get number of free blocks"""
        return len(self.free_block_ids)


# ==================== Sequence State ====================

class SequenceState:
    """
    Tracks KV cache state for a single sequence
    包含 draft/verify 的特殊状态管理
    """
    
    def __init__(self, seq_id: int, block_size: int):
        self.seq_id = seq_id
        self.block_size = block_size
        
        # Block allocation
        self.block_table: List[int] = []  # Physical block IDs
        self.num_tokens = 0
        
        # Draft/verify state
        self.is_in_draft = False
        self.draft_start_num_tokens = 0
        self.draft_start_num_blocks = 0
        # 不需要备份 block_table，只需记录位置
    
    @property
    def num_blocks(self) -> int:
        """Number of blocks needed for current tokens"""
        return (self.num_tokens + self.block_size - 1) // self.block_size
    
    @property
    def last_block_num_tokens(self) -> int:
        """Number of tokens in the last block"""
        if self.num_tokens == 0:
            return 0
        return self.num_tokens - (self.num_blocks - 1) * self.block_size
    
    def get_slot_mapping(self, start_pos: int, num_new_tokens: int) -> List[int]:
        """
        Get slot indices for new tokens
        
        Args:
            start_pos: Starting position in sequence
            num_new_tokens: Number of new tokens to add
            
        Returns:
            List of slot indices (block_id * block_size + offset_in_block)
        """
        slot_mapping = []
        for i in range(num_new_tokens):
            pos = start_pos + i
            block_idx = pos // self.block_size
            offset_in_block = pos % self.block_size
            
            if block_idx >= len(self.block_table):
                # This shouldn't happen if blocks are allocated properly
                raise RuntimeError(f"Block index {block_idx} out of range")
            
            block_id = self.block_table[block_idx]
            slot = block_id * self.block_size + offset_in_block
            slot_mapping.append(slot)
        
        return slot_mapping
    
    def get_block_table_tensor(self, max_num_blocks: int) -> torch.Tensor:
        """
        Get block table as padded tensor for flash_attn
        
        Args:
            max_num_blocks: Maximum number of blocks to pad to
            
        Returns:
            Tensor of shape [max_num_blocks] with -1 padding
        """
        block_table = self.block_table + [-1] * (max_num_blocks - len(self.block_table))
        return torch.tensor(block_table[:max_num_blocks], dtype=torch.int32, device='cuda')
    
    def start_draft(self):
        """Mark the start of draft phase"""
        self.is_in_draft = True
        self.draft_start_num_tokens = self.num_tokens
        self.draft_start_num_blocks = len(self.block_table)
    
    def accept_draft_tokens(self, num_accepted: int):
        """
        Accept some draft tokens
        
        Args:
            num_accepted: Number of tokens accepted (0 means all rejected)
        """
        # 计算接受后的最终 token 数
        final_num_tokens = self.draft_start_num_tokens + num_accepted
        self.num_tokens = final_num_tokens
        
        # 更新 block_table（移除未使用的 blocks）
        final_num_blocks = (final_num_tokens + self.block_size - 1) // self.block_size
        # block_table 应该已经由 verify 阶段更新，这里只需截断
        # 注意：实际的 block 释放由 PagedKVCache 处理
        
        self.is_in_draft = False
        self.draft_start_num_tokens = 0
        self.draft_start_num_blocks = 0


# ==================== Paged KV Cache Manager ====================

class PagedKVCache:
    """
    Paged KV Cache Manager with support for flash_attn and draft-verify
    
    Key differences from nano-vllm:
    1. Support draft-verify workflow
    2. Simplified prefix caching (can be added later)
    3. Integrated with flash_attn APIs
    """
    
    def __init__(
        self,
        config: MoEConfig,
        block_size: int = 256,
        gpu_memory_utilization: float = 0.9,
        dtype: torch.dtype = torch.float16
    ):
        self.config = config
        self.block_size = block_size
        self.dtype = dtype
        
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        
        # Allocate KV cache blocks
        self.num_blocks = self._calculate_num_blocks(gpu_memory_utilization)
        self.block_manager = BlockManager(self.num_blocks, block_size)
        
        # Physical KV cache storage
        # Shape: [num_layers, 2, num_blocks, block_size, num_kv_heads * head_dim]
        self.kv_cache = self._allocate_kv_cache()
        
        # Sequence states
        self.sequences: Dict[int, SequenceState] = {}
        
        logger.info(f"Initialized PagedKVCache: {self.num_blocks} blocks, "
                   f"block_size={block_size}, memory={self.get_memory_usage_mb():.2f} MB")
    
    def _calculate_num_blocks(self, gpu_memory_utilization: float) -> int:
        """Calculate number of blocks based on available GPU memory"""
        free, total = torch.cuda.mem_get_info()
        available = free * 0.8  # Use 80% of free memory
        
        # Each block stores: num_layers * 2 (K,V) * block_size * num_kv_heads * head_dim
        block_bytes = (
            self.num_layers * 2 * self.block_size * 
            self.num_kv_heads * self.head_dim * 
            self.dtype.itemsize
        )
        
        # Reserve memory for model weights and activations
        kv_cache_memory = available * 0.3  # Use 30% of available for KV cache
        num_blocks = int(kv_cache_memory / block_bytes)
        num_blocks = max(16, num_blocks)  # At least 16 blocks
        
        logger.info(f"Calculated {num_blocks} KV cache blocks "
                   f"({num_blocks * block_bytes / (1024**2):.2f} MB)")
        
        return num_blocks
    
    def _allocate_kv_cache(self) -> torch.Tensor:
        """Allocate physical KV cache storage"""
        # Shape: [num_layers, 2, num_blocks, block_size, num_kv_heads * head_dim]
        kv_cache = torch.empty(
            2,  # K and V
            self.num_layers,
            self.num_blocks,
            self.block_size,
            self.num_kv_heads * self.head_dim,
            dtype=self.dtype,
            device='cuda'
        )
        return kv_cache
    
    # ==================== Sequence Management ====================
    
    def add_sequence(self, seq_id: int, prompt_len: int) -> bool:
        """
        Add a new sequence and allocate blocks
        
        Returns:
            True if successful, False if not enough memory
        """
        if seq_id in self.sequences:
            logger.warning(f"Sequence {seq_id} already exists")
            return False
        
        # Calculate required blocks
        num_blocks_needed = (prompt_len + self.block_size - 1) // self.block_size
        
        if not self.block_manager.can_allocate(num_blocks_needed):
            return False
        
        # Create sequence state
        seq_state = SequenceState(seq_id, self.block_size)
        
        # Allocate blocks
        for _ in range(num_blocks_needed):
            block_id = self.block_manager.allocate_block()
            seq_state.block_table.append(block_id)
        
        seq_state.num_tokens = prompt_len
        self.sequences[seq_id] = seq_state
        
        logger.debug(f"Added sequence {seq_id} with {num_blocks_needed} blocks")
        return True
    
    def remove_sequence(self, seq_id: int):
        """Remove a sequence and deallocate its blocks"""
        if seq_id not in self.sequences:
            return
        
        seq_state = self.sequences[seq_id]
        
        # Deallocate blocks
        for block_id in seq_state.block_table:
            self.block_manager.dec_ref(block_id)
            if self.block_manager.blocks[block_id].ref_count == 0:
                self.block_manager.deallocate_block(block_id)
        
        del self.sequences[seq_id]
        logger.debug(f"Removed sequence {seq_id}")
    
    def can_append_token(self, seq_id: int) -> bool:
        """Check if we can append one more token"""
        if seq_id not in self.sequences:
            return False
        
        seq_state = self.sequences[seq_id]
        
        # Check if we need a new block
        if (seq_state.num_tokens + 1) > len(seq_state.block_table) * self.block_size:
            return self.block_manager.can_allocate(1)
        
        return True
    
    def append_token(self, seq_id: int) -> bool:
        """
        Append one token slot (allocate new block if needed)
        
        Returns:
            True if successful
        """
        if seq_id not in self.sequences:
            return False
        
        seq_state = self.sequences[seq_id]
        seq_state.num_tokens += 1
        
        # Check if we need a new block
        if seq_state.num_tokens > len(seq_state.block_table) * self.block_size:
            if not self.block_manager.can_allocate(1):
                return False
            block_id = self.block_manager.allocate_block()
            seq_state.block_table.append(block_id)
        
        return True
    
    # ==================== KV Storage ====================
    
    def store_kv(
        self,
        layer_idx: int,
        seq_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: int
    ):
        """
        Store KV states for a layer
        
        Args:
            layer_idx: Layer index
            seq_id: Sequence ID
            key, value: [num_tokens, num_kv_heads, head_dim]
            start_pos: Starting position in sequence
        """
        if seq_id not in self.sequences:
            raise ValueError(f"Sequence {seq_id} not found")
        
        seq_state = self.sequences[seq_id]
        num_tokens = key.shape[0]
        
        # Get slot mapping
        slot_mapping = seq_state.get_slot_mapping(start_pos, num_tokens)
        slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int32, device='cuda')
        
        # Get cache for this layer
        k_cache = self.kv_cache[0, layer_idx]  # [num_blocks, block_size, D]
        v_cache = self.kv_cache[1, layer_idx]

        # Store using triton kernel
        store_kvcache(key, value, k_cache, v_cache, slot_mapping_tensor)
    
    def get_kv_cache_for_layer(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get K and V cache tensors for a layer
        
        Returns:
            k_cache, v_cache: [num_blocks, block_size, num_kv_heads * head_dim]
        """
        k_cache = self.kv_cache[0, layer_idx]
        v_cache = self.kv_cache[1, layer_idx]
        return k_cache, v_cache
    
    # ==================== Context for flash_attn ====================
    
    def get_attention_context(
        self,
        seq_ids: List[int],
        is_prefill: bool
    ) -> Dict:
        """
        Get context dict for attention computation
        
        Returns dict with:
            - slot_mapping: [total_num_tokens]
            - block_tables: [num_seqs, max_num_blocks] (for decode)
            - context_lens: [num_seqs] (for decode)
            - cu_seqlens_q/k: cumulative sequence lengths (for prefill)
            - max_seqlen_q/k: max sequence length (for prefill)
        """
        if is_prefill:
            return self._get_prefill_context(seq_ids)
        else:
            return self._get_decode_context(seq_ids)
    
    def _get_prefill_context(self, seq_ids: List[int]) -> Dict:
        """Get context for prefill (variable length)"""
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        slot_mapping = []
        max_seqlen_q = 0
        max_seqlen_k = 0
        
        for seq_id in seq_ids:
            seq_state = self.sequences[seq_id]
            seqlen = seq_state.num_tokens
            
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen)
            max_seqlen_q = max(max_seqlen_q, seqlen)
            max_seqlen_k = max(max_seqlen_k, seqlen)
            
            # Get slot mapping for all tokens
            slots = seq_state.get_slot_mapping(0, seqlen)
            slot_mapping.extend(slots)
        
        return {
            'slot_mapping': torch.tensor(slot_mapping, dtype=torch.int32, device='cuda'),
            'cu_seqlens_q': torch.tensor(cu_seqlens_q, dtype=torch.int32, device='cuda'),
            'cu_seqlens_k': torch.tensor(cu_seqlens_k, dtype=torch.int32, device='cuda'),
            'max_seqlen_q': max_seqlen_q,
            'max_seqlen_k': max_seqlen_k,
        }
    
    def _get_decode_context(self, seq_ids: List[int]) -> Dict:
        """Get context for decode (single token per sequence)"""
        slot_mapping = []
        context_lens = []
        max_num_blocks = 0
        
        for seq_id in seq_ids:
            seq_state = self.sequences[seq_id]
            
            # Last token position
            pos = seq_state.num_tokens - 1
            slots = seq_state.get_slot_mapping(pos, 1)
            slot_mapping.extend(slots)
            
            context_lens.append(seq_state.num_tokens)
            max_num_blocks = max(max_num_blocks, len(seq_state.block_table))
        
        # Build block tables
        block_tables = []
        for seq_id in seq_ids:
            seq_state = self.sequences[seq_id]
            block_table = seq_state.get_block_table_tensor(max_num_blocks)
            block_tables.append(block_table)
        
        return {
            'slot_mapping': torch.tensor(slot_mapping, dtype=torch.int32, device='cuda'),
            'context_lens': torch.tensor(context_lens, dtype=torch.int32, device='cuda'),
            'block_tables': torch.stack(block_tables),  # [num_seqs, max_num_blocks]
        }

    def get_verify_context(self, seq_ids: List[int]) -> Dict:
        slot_mapping = []
        context_lens = []
        max_num_blocks = 0

        for seq_id in seq_ids:
            seq_state = self.sequences[seq_id]
            verify_start = seq_state.draft_start_num_tokens
            num_new_tokens = seq_state.num_tokens - verify_start

            slots = seq_state.get_slot_mapping(verify_start, num_new_tokens)
            slot_mapping.extend(slots)

            context_lens.append(seq_state.num_tokens)
            max_num_blocks = max(max_num_blocks, len(seq_state.block_table))

        block_tables = []
        for seq_id in seq_ids:
            seq_state = self.sequences[seq_id]
            block_table = seq_state.get_block_table_tensor(max_num_blocks)
            block_tables.append(block_table)

        return {
            'slot_mapping': torch.tensor(slot_mapping, dtype=torch.int32, device='cuda'),
            'context_lens': torch.tensor(context_lens, dtype=torch.int32, device='cuda'),
            'block_tables': torch.stack(block_tables),
        }
    
    # ==================== Draft-Verify Support ====================
    
    def start_draft(self, seq_id: int):
        """
        Mark the start of draft phase for a sequence
        不需要物理复制，只记录当前状态
        """
        if seq_id not in self.sequences:
            raise ValueError(f"Sequence {seq_id} not found")
        
        seq_state = self.sequences[seq_id]
        seq_state.start_draft()
        
        logger.debug(f"Started draft for seq {seq_id} at position {seq_state.draft_start_num_tokens}")
    
    def replace_draft_with_verify(
        self,
        seq_id: int,
        verify_seq_id: int,
        num_accepted_tokens: int
    ):
        """
        Replace draft cache with verify cache
        
        Args:
            seq_id: Draft sequence ID
            verify_seq_id: Verify sequence ID (has the correct KV cache)
            num_accepted_tokens: Number of tokens accepted from draft
        """
        if seq_id not in self.sequences:
            raise ValueError(f"Draft sequence {seq_id} not found")
        if verify_seq_id not in self.sequences:
            raise ValueError(f"Verify sequence {verify_seq_id} not found")
        
        draft_seq = self.sequences[seq_id]
        verify_seq = self.sequences[verify_seq_id]
        
        if not draft_seq.is_in_draft:
            logger.warning(f"Sequence {seq_id} is not in draft mode")
            return
        
        # 计算最终的 token 数量
        final_num_tokens = draft_seq.draft_start_num_tokens + num_accepted_tokens
        
        # 释放 draft 阶段多余的 blocks
        final_num_blocks = (final_num_tokens + self.block_size - 1) // self.block_size
        for i in range(final_num_blocks, len(draft_seq.block_table)):
            block_id = draft_seq.block_table[i]
            if self.block_manager.dec_ref(block_id):
                self.block_manager.deallocate_block(block_id)
        
        # 更新 block_table（使用 verify 的 blocks）
        # 注意：verify_seq 应该已经有正确的 KV cache
        draft_seq.block_table = verify_seq.block_table[:final_num_blocks]
        
        # 增加引用计数
        for block_id in draft_seq.block_table:
            self.block_manager.inc_ref(block_id)
        
        # 更新状态
        draft_seq.accept_draft_tokens(num_accepted_tokens)
        
        # 清理 verify sequence
        self.remove_sequence(verify_seq_id)
        
        logger.debug(f"Replaced draft cache for seq {seq_id}, "
                    f"accepted {num_accepted_tokens} tokens, "
                    f"final_tokens={final_num_tokens}")

    def accept_draft(self, seq_id: int, num_accepted_tokens: int):
        if seq_id not in self.sequences:
            raise ValueError(f"Sequence {seq_id} not found")

        seq_state = self.sequences[seq_id]
        if not seq_state.is_in_draft:
            logger.warning(f"Sequence {seq_id} is not in draft mode")
            return

        final_num_tokens = seq_state.draft_start_num_tokens + num_accepted_tokens
        final_num_blocks = (final_num_tokens + self.block_size - 1) // self.block_size

        for i in range(final_num_blocks, len(seq_state.block_table)):
            block_id = seq_state.block_table[i]
            if self.block_manager.dec_ref(block_id):
                self.block_manager.deallocate_block(block_id)

        seq_state.block_table = seq_state.block_table[:final_num_blocks]
        seq_state.num_tokens = final_num_tokens
        seq_state.is_in_draft = False
        seq_state.draft_start_num_tokens = 0
        seq_state.draft_start_num_blocks = 0
    
    # ==================== Utilities ====================
    
    def get_memory_usage_mb(self) -> float:
        """Get total KV cache memory usage in MB"""
        total_bytes = self.kv_cache.numel() * self.kv_cache.element_size()
        return total_bytes / (1024 ** 2)
    
    def get_num_free_blocks(self) -> int:
        """Get number of free blocks"""
        return self.block_manager.get_num_free_blocks()
    
    def clear(self):
        """Clear all sequences"""
        seq_ids = list(self.sequences.keys())
        for seq_id in seq_ids:
            self.remove_sequence(seq_id)
