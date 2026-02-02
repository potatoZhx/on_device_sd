"""
Qwen3 MoE Layer Implementation
支持 CPU/GPU 混合执行和路由
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .mlp import expert_forward_with_weights


class Qwen3MoEGate(nn.Module):
    """
    MoE 路由门控网络
    计算每个 token 应该路由到哪些 experts
    """
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, hidden_states: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算路由权重
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            top_k: 选择前 k 个 experts
            
        Returns:
            expert_indices: [batch_size, seq_len, top_k] - 选中的 expert 索引
            expert_weights: [batch_size, seq_len, top_k] - 对应的权重（已归一化）
        """
        # 计算路由分数
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # [batch_size * seq_len, hidden_size]
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        # [batch_size * seq_len, num_experts]
        router_logits = self.gate(hidden_states_flat)
        
        # Top-k 选择
        # routing_weights: [batch_size * seq_len, top_k]
        # selected_experts: [batch_size * seq_len, top_k]
        routing_weights, selected_experts = torch.topk(
            router_logits, top_k, dim=-1
        )
        
        # Softmax 归一化
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Reshape 回 [batch_size, seq_len, top_k]
        expert_indices = selected_experts.view(batch_size, seq_len, top_k)
        expert_weights = routing_weights.view(batch_size, seq_len, top_k)
        
        return expert_indices, expert_weights
    
    def load_weights(self, gate_weight: torch.Tensor):
        """从预训练权重加载门控网络参数"""
        target_dtype = next(self.parameters()).dtype
        self.gate.weight.data = gate_weight.to(target_dtype)


class Qwen3MoELayer(nn.Module):
    """
    Qwen3 MoE 层
    暂时实现简单的全 GPU 版本，后续扩展 CPU/GPU 混合执行
    """
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_intermediate_size: int,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_intermediate_size = moe_intermediate_size
        self.layer_idx = layer_idx
        
        # 路由门控
        self.gate = Qwen3MoEGate(hidden_size, num_experts)
        
        # 暂时不在这里创建 experts，使用外部权重
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_weights_dict: Dict[int, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        MoE 前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            expert_weights_dict: {expert_idx: {'gate_proj': ..., 'up_proj': ..., 'down_proj': ...}}
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 1. 计算路由
        expert_indices, expert_weights = self.gate(
            hidden_states, self.num_experts_per_token
        )
        # expert_indices: [batch_size, seq_len, top_k]
        # expert_weights: [batch_size, seq_len, top_k]
        
        # 2. 准备输出
        final_output = torch.zeros_like(hidden_states)
        
        # 3. 对每个 token 执行其选中的 experts
        # 将数据展平便于处理
        hidden_states_flat = hidden_states.view(-1, hidden_size)  # [B*S, H]
        expert_indices_flat = expert_indices.view(-1, self.num_experts_per_token)  # [B*S, K]
        expert_weights_flat = expert_weights.view(-1, self.num_experts_per_token)  # [B*S, K]
        
        # 遍历每个 expert，批量处理所有使用该 expert 的 tokens
        for expert_idx in range(self.num_experts):
            # 找到所有选择了这个 expert 的 token 位置
            # expert_mask: [B*S, K] -> [B*S]
            expert_mask = (expert_indices_flat == expert_idx).any(dim=-1)
            
            if not expert_mask.any():
                continue
            
            # 获取该 expert 的权重
            if expert_idx not in expert_weights_dict:
                continue  # Skip if expert weights not loaded
            
            expert_params = expert_weights_dict[expert_idx]
            
            # 选择使用该 expert 的 tokens
            token_indices = expert_mask.nonzero(as_tuple=False).squeeze(1)
            if token_indices.dim() == 0:
                token_indices = token_indices.unsqueeze(0)
            
            selected_hidden_states = hidden_states_flat[token_indices]  # [N, H]
            
            # 执行 expert 计算
            expert_output = expert_forward_with_weights(
                selected_hidden_states,
                gate_weight=expert_params['gate_proj'],
                up_weight=expert_params['up_proj'],
                down_weight=expert_params['down_proj'],
            )  # [N, H]
            
            # 获取对应的权重并加权累加
            # 对于每个选中该 expert 的 token，找到其对应的权重
            for i, token_idx in enumerate(token_indices):
                # 在 expert_indices_flat[token_idx] 中找到 expert_idx 的位置
                expert_positions = (expert_indices_flat[token_idx] == expert_idx).nonzero(as_tuple=False).squeeze(1)
                if expert_positions.dim() == 0:
                    expert_positions = expert_positions.unsqueeze(0)
                
                # 对应的权重
                weight = expert_weights_flat[token_idx][expert_positions].sum()
                
                # 加权累加到最终输出
                final_output.view(-1, hidden_size)[token_idx] += weight * expert_output[i]
        
        return final_output


class Qwen3MoELayerWithWeights(nn.Module):
    """
    无状态的 MoE Layer wrapper
    使用外部提供的权重进行前向传播
    """
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_intermediate_size: int,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_intermediate_size = moe_intermediate_size
        self.layer_idx = layer_idx
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        gate_weight: torch.Tensor,
        expert_weights_dict: Dict[int, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        完全无状态的前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            gate_weight: [num_experts, hidden_size] - 路由门控权重
            expert_weights_dict: {expert_idx: {'gate_proj': ..., 'up_proj': ..., 'down_proj': ...}}
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 1. 计算路由（无状态）
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        router_logits = F.linear(hidden_states_flat, gate_weight)  # [B*S, num_experts]
        
        routing_weights, selected_experts = torch.topk(
            router_logits, self.num_experts_per_token, dim=-1
        )
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        expert_indices = selected_experts.view(batch_size, seq_len, self.num_experts_per_token)
        expert_weights = routing_weights.view(batch_size, seq_len, self.num_experts_per_token)
        
        # 2. Expert 执行（同上）
        final_output = torch.zeros_like(hidden_states)
        expert_indices_flat = expert_indices.view(-1, self.num_experts_per_token)
        expert_weights_flat = expert_weights.view(-1, self.num_experts_per_token)
        
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices_flat == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
            
            if expert_idx not in expert_weights_dict:
                continue
            
            expert_params = expert_weights_dict[expert_idx]
            token_indices = expert_mask.nonzero(as_tuple=False).squeeze(1)
            if token_indices.dim() == 0:
                token_indices = token_indices.unsqueeze(0)
            
            selected_hidden_states = hidden_states_flat[token_indices]
            expert_output = expert_forward_with_weights(
                selected_hidden_states,
                gate_weight=expert_params['gate_proj'],
                up_weight=expert_params['up_proj'],
                down_weight=expert_params['down_proj'],
            )
            
            for i, token_idx in enumerate(token_indices):
                expert_positions = (expert_indices_flat[token_idx] == expert_idx).nonzero(as_tuple=False).squeeze(1)
                if expert_positions.dim() == 0:
                    expert_positions = expert_positions.unsqueeze(0)
                weight = expert_weights_flat[token_idx][expert_positions].sum()
                final_output.view(-1, hidden_size)[token_idx] += weight * expert_output[i]
        
        return final_output
