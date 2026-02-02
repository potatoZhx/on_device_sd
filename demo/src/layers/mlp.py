"""
MLP (Multi-Layer Perceptron) implementation for Qwen3MoE
参考 nano-vllm 和 transformers 实现
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


class SiluAndMul(nn.Module):
    """
    SiLU activation followed by element-wise multiplication
    Fused operation: silu(gate) * up
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, intermediate_size * 2] concatenated gate and up projections
        
        Returns:
            [*, intermediate_size] after silu(gate) * up
        """
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up


class Qwen3MLP(nn.Module):
    """
    Standard MLP with gated activation (SiLU)
    Used in non-MoE layers
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Projections
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        # Activation
        if hidden_act != "silu":
            raise ValueError(f"Only silu activation is supported, got {hidden_act}")
        self.act_fn = F.silu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [*, hidden_size]
        
        Returns:
            [*, hidden_size]
        """
        # Gate and up projections
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # Activation + element-wise multiply
        intermediate = self.act_fn(gate) * up
        
        # Down projection
        output = self.down_proj(intermediate)
        
        return output
    
    def load_weights(
        self,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ):
        """
        Load weights from ParameterLoader
        
        Args:
            gate_weight: [intermediate_size, hidden_size]
            up_weight: [intermediate_size, hidden_size]
            down_weight: [hidden_size, intermediate_size]
        """
        # Ensure weights match the module's dtype
        target_dtype = next(self.parameters()).dtype
        
        self.gate_proj.weight.data = gate_weight.to(target_dtype)
        self.up_proj.weight.data = up_weight.to(target_dtype)
        self.down_proj.weight.data = down_weight.to(target_dtype)


class Qwen3MLPWithWeights:
    """
    MLP wrapper that uses external weights (no nn.Module)
    Useful for dynamic weight management
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        if hidden_act != "silu":
            raise ValueError(f"Only silu activation is supported, got {hidden_act}")
        self.act_fn = F.silu
    
    def forward(
        self,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with external weights
        
        Args:
            x: [*, hidden_size]
            gate_weight: [intermediate_size, hidden_size]
            up_weight: [intermediate_size, hidden_size]
            down_weight: [hidden_size, intermediate_size]
        
        Returns:
            [*, hidden_size]
        """
        # Gate and up projections
        gate = F.linear(x, gate_weight)
        up = F.linear(x, up_weight)
        
        # Activation + multiply
        intermediate = self.act_fn(gate) * up
        
        # Down projection
        output = F.linear(intermediate, down_weight)
        
        return output


class Qwen3Expert(nn.Module):
    """
    Single Expert (FFN) for MoE layer
    Identical to Qwen3MLP but used in MoE context
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        expert_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        self.expert_id = expert_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Projections
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        self.act_fn = F.silu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [*, hidden_size]
        
        Returns:
            [*, hidden_size]
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        intermediate = self.act_fn(gate) * up
        output = self.down_proj(intermediate)
        return output
    
    def load_weights(
        self,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ):
        """
        Load weights from ParameterLoader
        """
        target_dtype = next(self.parameters()).dtype
        
        self.gate_proj.weight.data = gate_weight.to(target_dtype)
        self.up_proj.weight.data = up_weight.to(target_dtype)
        self.down_proj.weight.data = down_weight.to(target_dtype)


def expert_forward_with_weights(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Expert forward pass using external weights (functional API)
    
    Args:
        x: [*, hidden_size]
        gate_weight: [intermediate_size, hidden_size]
        up_weight: [intermediate_size, hidden_size]
        down_weight: [hidden_size, intermediate_size]
    
    Returns:
        [*, hidden_size]
    """
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    intermediate = F.silu(gate) * up
    output = F.linear(intermediate, down_weight)
    return output
