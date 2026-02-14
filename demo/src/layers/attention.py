"""
Attention Layer Implementation with flash_attn
参考 nano-vllm 和 transformers 实现
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple
import triton
import triton.language as tl

# 尝试导入 flash_attn，如果不可用则使用普通实现
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
except (ImportError, OSError) as e:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
    import warnings
    warnings.warn(f"flash_attn not available: {e}. Will use standard attention implementation.")

from .rotary_embedding import get_rope
from .layernorm import RMSNorm
from ..memory.paged_kv_cache import PagedKVCache
from ..utils.logger import get_logger

logger = get_logger(__name__)


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1
    assert k_cache.stride(-2) == D and v_cache.stride(-2) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D,
    )


class Qwen3Attention(nn.Module):
    """
    Qwen3 Attention Layer with flash_attn
    支持 GQA (Grouped Query Attention) 和 QK Norm
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 1000000.0,
        rope_scaling: dict | None = None,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.qkv_bias = qkv_bias
        
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        
        # QKV Projections (weights loaded separately)
        self.q_proj = nn.Linear(hidden_size, self.q_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, self.kv_size, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, self.kv_size, bias=qkv_bias)
        self.o_proj = nn.Linear(self.q_size, hidden_size, bias=False)
        
        # QK Norm (Qwen3 specific)
        if not qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        
        # RoPE
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
    
    def load_weights(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        o_weight: torch.Tensor,
        q_norm_weight: Optional[torch.Tensor] = None,
        k_norm_weight: Optional[torch.Tensor] = None,
    ):
        """
        Load weights from ParameterLoader
        
        Args:
            q_weight: [q_size, hidden_size]
            k_weight: [kv_size, hidden_size]
            v_weight: [kv_size, hidden_size]
            o_weight: [hidden_size, q_size]
            q_norm_weight, k_norm_weight: Optional norm weights
        """
        # Ensure weights match the module's dtype
        target_dtype = next(self.parameters()).dtype
        
        self.q_proj.weight.data = q_weight.to(target_dtype)
        self.k_proj.weight.data = k_weight.to(target_dtype)
        self.v_proj.weight.data = v_weight.to(target_dtype)
        self.o_proj.weight.data = o_weight.to(target_dtype)
        
        if q_norm_weight is not None and self.q_norm is not None:
            self.q_norm.weight.data = q_norm_weight.to(target_dtype)
        if k_norm_weight is not None and self.k_norm is not None:
            self.k_norm.weight.data = k_norm_weight.to(target_dtype)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: PagedKVCache,
        seq_ids: list[int],
        is_prefill: bool,
    ) -> torch.Tensor:
        """
        Forward pass with flash_attn
        
        Args:
            hidden_states: [num_tokens, hidden_size]
            positions: [num_tokens] position indices
            kv_cache: PagedKVCache instance
            seq_ids: List of sequence IDs
            is_prefill: Whether in prefill mode
        
        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        # QKV projection
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to multi-head format
        num_tokens = hidden_states.shape[0]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        
        # Apply QK norm if needed
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        
        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)
        
        # Store KV in cache
        context = kv_cache.get_attention_context(seq_ids, is_prefill)
        k_cache_layer, v_cache_layer = kv_cache.get_kv_cache_for_layer(self.layer_idx)
        
        # Store current KV
        if context['slot_mapping'].numel() > 0:
            store_kvcache(k, v, k_cache_layer, v_cache_layer, context['slot_mapping'])
        
        # Compute attention using flash_attn
        if is_prefill:
            # Prefill: use varlen API
            attn_output = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=context['cu_seqlens_q'],
                cu_seqlens_k=context['cu_seqlens_k'],
                max_seqlen_q=context['max_seqlen_q'],
                max_seqlen_k=context['max_seqlen_k'],
                softmax_scale=self.scaling,
                causal=True,
            )
        else:
            # Decode: use kvcache API
            # flash_attn_with_kvcache expects KV cache in [num_blocks, block_size, num_kv_heads, head_dim]
            k_cache_view = k_cache_layer.view(
                k_cache_layer.shape[0],
                k_cache_layer.shape[1],
                self.num_kv_heads,
                self.head_dim,
            )
            v_cache_view = v_cache_layer.view(
                v_cache_layer.shape[0],
                v_cache_layer.shape[1],
                self.num_kv_heads,
                self.head_dim,
            )
            attn_output = flash_attn_with_kvcache(
                q.unsqueeze(1),  # [num_tokens, 1, num_heads, head_dim]
                k_cache_view,
                v_cache_view,
                cache_seqlens=context['context_lens'],
                block_table=context['block_tables'],
                softmax_scale=self.scaling,
                causal=True,
            )
            attn_output = attn_output.squeeze(1)  # [num_tokens, num_heads, head_dim]
        
        # Reshape and apply output projection
        attn_output = attn_output.view(num_tokens, self.num_heads * self.head_dim)
        output = self.o_proj(attn_output)
        
        return output


class Qwen3AttentionWithWeights:
    """
    Qwen3 Attention wrapper that uses external weights
    不使用 nn.Module，直接使用预加载的权重
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 1000000.0,
        layer_idx: int = 0,
    ):
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.qkv_bias = qkv_bias
        self.rms_norm_eps = rms_norm_eps
        
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        
        # RoPE
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
        )
        
        # QK Norm weights (loaded externally)
        self.has_qk_norm = not qkv_bias
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        o_weight: torch.Tensor,
        q_norm_weight: Optional[torch.Tensor],
        k_norm_weight: Optional[torch.Tensor],
        kv_cache: PagedKVCache,
        seq_ids: list[int],
        is_prefill: bool,
    ) -> torch.Tensor:
        """
        Forward pass using external weights
        
        Args:
            hidden_states: [num_tokens, hidden_size]
            positions: [num_tokens]
            q_weight, k_weight, v_weight, o_weight: Projection weights
            q_norm_weight, k_norm_weight: Optional norm weights
            kv_cache: PagedKVCache instance
            seq_ids: List of sequence IDs
            is_prefill: Whether in prefill mode
        
        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        num_tokens = hidden_states.shape[0]
        
        # QKV projection
        q = F.linear(hidden_states, q_weight)
        k = F.linear(hidden_states, k_weight)
        v = F.linear(hidden_states, v_weight)
        
        # Reshape to multi-head
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        
        # Apply QK norm if needed
        if self.has_qk_norm:
            q = self._apply_rms_norm(q, q_norm_weight)
            k = self._apply_rms_norm(k, k_norm_weight)
        
        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)
        
        # Get attention context
        context = kv_cache.get_attention_context(seq_ids, is_prefill)
        k_cache_layer, v_cache_layer = kv_cache.get_kv_cache_for_layer(self.layer_idx)
        
        # Store KV in cache
        if context['slot_mapping'].numel() > 0:
            store_kvcache(k, v, k_cache_layer, v_cache_layer, context['slot_mapping'])
        
        # Compute attention
        if is_prefill:
            attn_output = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=context['cu_seqlens_q'],
                cu_seqlens_k=context['cu_seqlens_k'],
                max_seqlen_q=context['max_seqlen_q'],
                max_seqlen_k=context['max_seqlen_k'],
                softmax_scale=self.scaling,
                causal=True,
            )
        else:
            k_cache_view = k_cache_layer.view(
                k_cache_layer.shape[0],
                k_cache_layer.shape[1],
                self.num_kv_heads,
                self.head_dim,
            )
            v_cache_view = v_cache_layer.view(
                v_cache_layer.shape[0],
                v_cache_layer.shape[1],
                self.num_kv_heads,
                self.head_dim,
            )
            attn_output = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache_view,
                v_cache_view,
                cache_seqlens=context['context_lens'],
                block_table=context['block_tables'],
                softmax_scale=self.scaling,
                causal=True,
            )
            attn_output = attn_output.squeeze(1)
        
        # Output projection
        attn_output = attn_output.view(num_tokens, self.q_size)
        output = F.linear(attn_output, o_weight)
        
        return output
    
    def _apply_rms_norm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply RMS normalization
        
        Args:
            x: [num_tokens, num_heads, head_dim]
            weight: [head_dim]
        """
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.rms_norm_eps)
        x = x.to(orig_dtype) * weight
        return x
