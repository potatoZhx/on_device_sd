import torch
import torch.nn.functional as F
from typing import Dict, Optional
from .base_operator import ExpertOperator
from ..core.model import MoEConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CPUOperators:
    """
    CPU operator implementations.
    Optimized for CPU execution.
    """
    
    def __init__(self, config: MoEConfig):
        self.config = config
    
    def expert_forward(
        self,
        hidden_states: torch.Tensor,
        expert_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Expert FFN forward pass on CPU.
        
        Args:
            hidden_states: [num_tokens, hidden_size] on CPU
            expert_params: {'gate_proj', 'up_proj', 'down_proj'} on CPU
        
        Returns:
            Output [num_tokens, hidden_size] on CPU
        """
        # Ensure everything is on CPU
        assert not hidden_states.is_cuda, "Hidden states should be on CPU"
        assert all(not p.is_cuda for p in expert_params.values()), "Params should be on CPU"
        
        # Gate and up projections
        gate_output = F.linear(hidden_states, expert_params['gate_proj'])
        up_output = F.linear(hidden_states, expert_params['up_proj'])
        
        # SwiGLU activation
        intermediate = F.silu(gate_output) * up_output
        
        # Down projection
        output = F.linear(intermediate, expert_params['down_proj'])
        
        return output
    
    def batched_expert_forward(
        self,
        hidden_states_list: list,
        expert_params: Dict[str, torch.Tensor]
    ) -> list:
        """
        Batch multiple expert forward passes for efficiency.
        
        Args:
            hidden_states_list: List of hidden state tensors
            expert_params: Expert parameters
        
        Returns:
            List of output tensors
        """
        # Stack inputs
        if not hidden_states_list:
            return []
        
        batched_input = torch.cat(hidden_states_list, dim=0)
        
        # Single forward pass
        batched_output = self.expert_forward(batched_input, expert_params)
        
        # Split outputs
        sizes = [h.shape[0] for h in hidden_states_list]
        outputs = torch.split(batched_output, sizes, dim=0)
        
        return list(outputs)