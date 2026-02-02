import torch
import torch.nn.functional as F
from typing import Dict, Optional
from .base_operator import ExpertOperator, AttentionOperator
from ..core.model import MoEConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GPUOperators:
    """
    GPU operator implementations.
    Uses PyTorch operations on CUDA tensors.
    """
    
    def __init__(self, config: MoEConfig):
        self.config = config
    
    def embedding(
        self,
        input_ids: torch.Tensor,
        embed_weight: torch.Tensor
    ) -> torch.Tensor:
        """Embedding lookup"""
        return F.embedding(input_ids, embed_weight)
    
    def linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Linear transformation"""
        return F.linear(input, weight, bias)
    
    def layernorm(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        rms = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        inv = torch.rsqrt(rms + eps)
        hidden_normed = (hidden_states.float() * inv).to(hidden_states.dtype)
        return hidden_normed * weight
    
    def router(
        self,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Router computation for MoE.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            router_weight: [num_experts, hidden_size]
        
        Returns:
            Routing scores [batch*seq_len, num_experts]
        """
        batch, seq_len, hidden = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden)
        
        # Linear projection
        scores = F.linear(flat_hidden, router_weight)
        
        return scores
    
    def expert_forward(
        self,
        hidden_states: torch.Tensor,
        expert_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Expert FFN forward pass (SwiGLU activation).
        
        Args:
            hidden_states: [num_tokens, hidden_size]
            expert_params: {'gate_proj', 'up_proj', 'down_proj'}
        
        Returns:
            Output [num_tokens, hidden_size]
        """
        # Gate and up projections
        gate_output = F.linear(hidden_states, expert_params['gate_proj'])
        up_output = F.linear(hidden_states, expert_params['up_proj'])
        
        # SwiGLU activation
        intermediate = F.silu(gate_output) * up_output
        
        # Down projection
        output = F.linear(intermediate, expert_params['down_proj'])
        
        return output
    
    def self_attention(
        self,
        hidden_states: torch.Tensor,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor,
        o_proj: torch.Tensor,
        kv_cache: Optional[any],
        layer_idx: int,
        q_norm: Optional[torch.Tensor] = None,
        k_norm: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Self-attention with KV caching.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            q_proj, k_proj, v_proj, o_proj: Projection weights
            kv_cache: KV cache object
            layer_idx: Current layer index
        
        Returns:
            Attention output [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_heads = self.config.num_attention_heads
        kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        q = F.linear(hidden_states, q_proj)
        k = F.linear(hidden_states, k_proj)
        v = F.linear(hidden_states, v_proj)
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, kv_heads, head_dim).transpose(1, 2)
        if q_norm is not None:
            q = self._rms_norm_head(q, q_norm)
        if k_norm is not None:
            k = self._rms_norm_head(k, k_norm)
        start_pos = kv_cache.current_length if kv_cache is not None else 0
        q, k = self._apply_rope(q, k, start_pos)
        if kv_cache is not None:
            k_cached, v_cached = kv_cache.append(layer_idx, k, v, start_pos=start_pos)
        else:
            k_cached, v_cached = k, v
        if num_heads != kv_heads:
            groups = num_heads // kv_heads
            k_cached = k_cached.repeat_interleave(groups, dim=1)
            v_cached = v_cached.repeat_interleave(groups, dim=1)
        attn_weights = torch.matmul(q, k_cached.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_cached)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, hidden_size)
        output = F.linear(attn_output, o_proj)
        return output

    def _rms_norm_head(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True)
        inv = torch.rsqrt(rms + self.config.rms_norm_eps)
        x_normed = (x.float() * inv).to(x.dtype)
        return x_normed * weight
    
    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: int
    ) -> tuple:
        b, h, t, d = q.shape
        half = d // 2
        theta = self.config.rope_theta
        pos = torch.arange(start_pos, start_pos + t, device=q.device).float()
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=q.device).float() / d))
        freqs = torch.einsum("t,f->tf", pos, inv_freq)
        cos = torch.cos(freqs)[None, None, :, :]
        sin = torch.sin(freqs)[None, None, :, :]
        def rotate(x):
            x1 = x[..., :half]
            x2 = x[..., half:]
            x_rot_1 = x1 * cos - x2 * sin
            x_rot_2 = x1 * sin + x2 * cos
            return torch.cat([x_rot_1, x_rot_2], dim=-1)
        q = rotate(q)
        k = rotate(k)
        return q, k
