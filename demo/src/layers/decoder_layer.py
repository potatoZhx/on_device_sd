"""
Qwen3 Decoder Layer Implementation
包含 Attention + MoE Layer + LayerNorm
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .attention import Qwen3Attention
from .moe_layer import Qwen3MoELayer
from .layernorm import RMSNorm


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3 单个 Decoder 层
    包含：
    1. Self-Attention (with QK Norm)
    2. MoE Layer
    3. 两个 RMSNorm
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_intermediate_size: int,
        max_position_embeddings: int,
        rms_norm_eps: float,
        rope_theta: float,
        layer_idx: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        
        # 1. Input LayerNorm
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # 2. Self-Attention
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            qkv_bias=qkv_bias,
            rope_theta=rope_theta,
            layer_idx=layer_idx,
        )
        
        # 3. Post-Attention LayerNorm
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # 4. MoE Layer
        self.mlp = Qwen3MoELayer(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            moe_intermediate_size=moe_intermediate_size,
            layer_idx=layer_idx,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[any] = None,
        positions: Optional[torch.Tensor] = None,
        expert_weights_dict: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        seq_ids: Optional[list[int]] = None,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        """
        Decoder Layer 前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] or [num_tokens, hidden_size]
            kv_cache: KV cache 对象
            positions: [batch_size, seq_len] or [num_tokens] 位置索引
            expert_weights_dict: 当前层的 expert 权重字典
            seq_ids: sequence IDs for KV cache
            is_prefill: whether this is prefill phase
            
        Returns:
            output: [batch_size, seq_len, hidden_size] or [num_tokens, hidden_size]
        """
        # 1. Self-Attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Reshape for attention if needed
        if hidden_states.dim() == 3:
            batch_size, seq_len, hidden_size = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, hidden_size)  # [batch*seq, hidden]
            if positions is not None:
                positions = positions.reshape(-1)  # [batch*seq]
            reshape_output = True
        else:
            batch_size, seq_len = 1, hidden_states.shape[0]
            reshape_output = False
        
        # Generate default seq_ids if not provided
        if seq_ids is None:
            seq_ids = list(range(batch_size))
        
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            kv_cache=kv_cache,
            seq_ids=seq_ids,
            is_prefill=is_prefill,
        )
        
        # Reshape back if needed
        if reshape_output:
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1)
        
        hidden_states = residual + hidden_states
        
        # 2. MoE Layer with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if expert_weights_dict is not None:
            hidden_states = self.mlp(
                hidden_states=hidden_states,
                expert_weights_dict=expert_weights_dict,
            )
        else:
            # 如果没有提供 expert 权重，跳过 MoE 计算
            # （用于测试或特殊情况）
            pass
        
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def load_weights(
        self,
        input_layernorm_weight: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        o_weight: torch.Tensor,
        q_norm_weight: Optional[torch.Tensor],
        k_norm_weight: Optional[torch.Tensor],
        post_attention_layernorm_weight: torch.Tensor,
        gate_weight: torch.Tensor,
    ):
        """加载预训练权重"""
        target_dtype = next(self.parameters()).dtype
        
        # Load LayerNorm weights
        self.input_layernorm.weight.data = input_layernorm_weight.to(target_dtype)
        self.post_attention_layernorm.weight.data = post_attention_layernorm_weight.to(target_dtype)
        
        # Load Attention weights
        self.self_attn.load_weights(
            q_weight=q_weight,
            k_weight=k_weight,
            v_weight=v_weight,
            o_weight=o_weight,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
        )
        
        # Load MoE gate weights
        self.mlp.gate.load_weights(gate_weight)
