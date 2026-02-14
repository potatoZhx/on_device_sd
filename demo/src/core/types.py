from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import torch
import threading
from queue import Queue

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
    generation_config: Optional["GenerationConfig"] = None

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




# batch related types

class InferenceMode(Enum):
    """Inference mode selection"""
    STANDARD = "standard"  # Standard autoregressive decoding
    SPECULATIVE = "speculative"  # Draft-verify speculative decoding


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    
    # Stop conditions
    eos_token_id: Optional[int] = None
    stop_strings: List[str] = field(default_factory=list)
    
    # Speculative decoding specific
    use_speculative: bool = True
    max_draft_tokens: int = 8


@dataclass
class BatchedRequest:
    """A batch of requests processed together"""
    batch_id: str
    requests: List[Any]  # List of InferenceRequest
    
    # Batched tensors
    input_ids: torch.Tensor  # [batch_size, max_seq_len] padded
    attention_mask: torch.Tensor  # [batch_size, max_seq_len]
    position_ids: torch.Tensor  # [batch_size, max_seq_len]
    
    # Per-request state
    current_lengths: List[int]  # Current generation length for each request
    finished: List[bool]  # Whether each request is finished
    
    # Batch metadata
    max_batch_seq_len: int
    padding_token_id: int = 0
    
    def is_complete(self) -> bool:
        """Check if all requests in batch are finished"""
        return all(self.finished)
    
    def active_request_count(self) -> int:
        """Count how many requests are still active"""
        return sum(1 for f in self.finished if not f)


@dataclass
class InferenceResponse:
    """Response for a completed request"""
    request_id: str
    generated_ids: List[int]
    generated_text: Optional[str] = None
    
    # Generation statistics
    num_tokens_generated: int = 0
    generation_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    
    # Detailed metrics
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    
    # For speculative decoding
    num_draft_rounds: Optional[int] = None
    avg_acceptance_rate: Optional[float] = None
    
    # Error handling
    error: Optional[str] = None
    success: bool = True


class RequestStatus(Enum):
    """Status of an inference request"""
    QUEUED = "queued"
    BATCHED = "batched"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchStatistics:
    """Statistics for batch processing"""
    batch_id: str
    batch_size: int
    total_input_tokens: int
    total_output_tokens: int
    
    # Timing
    batch_formation_time_ms: float
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    
    # Efficiency metrics
    average_tokens_per_second: float
    gpu_utilization: float
    
    # Expert statistics (for MoE)
    expert_cache_hit_rate: float
    cpu_compute_ratio: float
