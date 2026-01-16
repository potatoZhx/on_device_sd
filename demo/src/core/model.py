"""
尝试transformer套壳， kt
"""


from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn

@dataclass
class MoEConfig:
    """MoE model configuration"""
    # Model dimensions
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    
    # MoE specific
    num_experts: int
    num_experts_per_token: int  # top-k
    num_shared_experts: int = 0
    
    # Inference specific
    max_seq_length: int = 2048
    rope_theta: float = 10000.0
    
    # TODO
    # Draft-verify
    draft_top_c: int = 2  # 需删除，每一步由sched决定，不属于config # Top-c experts for CPU during draft
    max_draft_tokens: int = 8 # 删除？
    verify_threshold_perplexity: float = 1.5 # threshold重新设计

class MoEModelStructure:
    """
    Represents the structure of an MoE model without actual parameters.
    Used for planning and coordination.
    """
    def __init__(self, config: MoEConfig):
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.num_experts_per_layer = config.num_experts
        
    def get_expert_ids(self, layer_idx: Optional[int] = None) -> List[ExpertID]:
        """Get all expert IDs, optionally filtered by layer"""
        if layer_idx is not None:
            return [
                ExpertID(layer_idx, exp_idx) 
                for exp_idx in range(self.num_experts_per_layer)
            ]
        else:
            return [
                ExpertID(layer_idx, exp_idx)
                for layer_idx in range(self.num_layers)
                for exp_idx in range(self.num_experts_per_layer)
            ]
    
    def get_num_experts(self) -> int:
        """Total number of experts in the model"""
        return self.num_layers * self.num_experts_per_layer
    
    def is_shared_expert(self, expert_id: ExpertID) -> bool:
        """Check if expert is a shared expert"""
        return expert_id.expert_idx < self.config.num_shared_experts

# TODO
class MoELayer(nn.Module):
    """
    Single MoE layer (placeholder for actual implementation).
    Actual operators will be in operators/ module.
    """
    def __init__(self, config: MoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
    def forward(self, hidden_states, expert_cache, device_assignments):
        """Forward pass coordinating CPU/GPU execution"""
        raise NotImplementedError("Use execution engines instead")