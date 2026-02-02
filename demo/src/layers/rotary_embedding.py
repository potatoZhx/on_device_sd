"""
Rotary Position Embedding (RoPE)
参考 nano-vllm 实现
"""

from functools import lru_cache
import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dims of the input.
    This matches transformers implementation.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary positional embedding
    Using transformers-compatible implementation: (x * cos) + (rotate_half(x) * sin)
    
    Args:
        x: [num_tokens, num_heads, head_dim]
        cos, sin: [num_tokens, 1, head_dim] (with head_dim repeated from head_dim//2)
    
    Returns:
        Rotated tensor
    """
    # Match transformers implementation exactly
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding
    """
    
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size, "Currently only support full rotary_dim"
        
        # Compute inverse frequencies (only for half of head_dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        
        # Precompute cos and sin for all positions
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # [max_pos, head_dim//2]
        
        # Repeat freqs to match transformers format
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_pos, head_dim]
        cos = emb.cos()
        sin = emb.sin()
        
        # Cache: [max_position, 1, head_dim]
        self.register_buffer("cos_cache", cos.unsqueeze(1), persistent=False)
        self.register_buffer("sin_cache", sin.unsqueeze(1), persistent=False)
    
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key
        
        Args:
            positions: [num_tokens] position indices
            query: [num_tokens, num_q_heads, head_dim]
            key: [num_tokens, num_kv_heads, head_dim]
        
        Returns:
            Rotated query and key
        """
        # Move cache to same device as query if needed
        if self.cos_cache.device != query.device:
            self.cos_cache = self.cos_cache.to(query.device)
            self.sin_cache = self.sin_cache.to(query.device)
        
        cos = self.cos_cache[positions]  # [num_tokens, 1, head_dim]
        sin = self.sin_cache[positions]  # [num_tokens, 1, head_dim]
        
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        
        return query, key


@lru_cache(maxsize=8)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
) -> RotaryEmbedding:
    """
    Get or create a cached RoPE module
    
    Args:
        head_size: Dimension of each head
        rotary_dim: Rotary dimension (usually == head_size)
        max_position: Maximum position
        base: RoPE base (theta)
        rope_scaling: Scaling configuration (not yet supported)
    
    Returns:
        RotaryEmbedding module
    """
    if rope_scaling is not None:
        raise NotImplementedError("rope_scaling not yet supported")
    
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
