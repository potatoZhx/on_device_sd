from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch

class DeviceType(Enum):
    """Device type enumeration"""
    CPU = "cpu"
    GPU = "gpu"

class ExecutionPhase(Enum):
    """Inference execution phase"""
    PREFILL = "prefill"
    DRAFT = "draft"
    VERIFY = "verify"

@dataclass
class ExpertID:
    """Unique identifier for an expert"""
    layer_idx: int
    expert_idx: int
    
    def __hash__(self):
        return hash((self.layer_idx, self.expert_idx))
    
    def __str__(self):
        return f"L{self.layer_idx}_E{self.expert_idx}"

@dataclass
class ExpertLocation:
    """Location information for an expert"""
    expert_id: ExpertID
    device: DeviceType
    is_cached: bool  # Whether in GPU cache
    memory_address: Optional[int] = None

@dataclass
class ExpertActivation:
    """Expert activation information"""
    expert_id: ExpertID
    token_indices: torch.Tensor  # Which tokens activate this expert
    scores: torch.Tensor  # Activation scores
    top_k_rank: int  # Rank in top-k selection

@dataclass
class LayerExpertActivations:
    """All expert activations for a layer"""
    layer_idx: int
    activations: List[ExpertActivation]
    routing_scores: torch.Tensor  # Full routing scores [batch, num_experts]

@dataclass
class TransferRequest:
    """Request to transfer expert parameters"""
    expert_id: ExpertID
    source_device: DeviceType
    target_device: DeviceType
    priority: float
    async_transfer: bool = True

@dataclass
class InferenceRequest:
    """User inference request"""
    request_id: str
    input_ids: torch.Tensor
    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50

@dataclass
class DraftMetrics:
    """Metrics collected during draft phase"""
    num_drafted_tokens: int
    perplexity: float
    expert_hit_rate: float  # GPU cache hit rate
    cpu_compute_ratio: float  # Ratio of CPU computation
    
@dataclass
class VerifyResult:
    """Result from verify phase"""
    num_accepted_tokens: int
    accepted_token_ids: torch.Tensor
    new_kv_cache: Any  # KV cache after verification
    should_continue: bool