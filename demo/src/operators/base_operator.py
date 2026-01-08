from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch


class BaseOperator(ABC):
    """Abstract base class for all operators"""
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward computation"""
        pass


class ExpertOperator(BaseOperator):
    """Abstract base for expert FFN operations"""
    
    @abstractmethod
    def expert_forward(
        self,
        hidden_states: torch.Tensor,
        expert_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Expert FFN forward pass.
        
        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]
            expert_params: Dict with 'gate_proj', 'up_proj', 'down_proj'
        
        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        pass


class AttentionOperator(BaseOperator):
    """Abstract base for attention operations"""
    
    @abstractmethod
    def self_attention(
        self,
        hidden_states: torch.Tensor,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor,
        o_proj: torch.Tensor,
        kv_cache: Optional[any],
        layer_idx: int
    ) -> torch.Tensor:
        """Self-attention with KV caching"""
        pass