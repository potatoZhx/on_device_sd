import time
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PhaseMetrics:
    """Metrics for a single execution phase"""
    phase_name: str
    duration_ms: float
    start_time: float
    end_time: float


@dataclass
class RequestMetrics:
    """Metrics for a complete inference request"""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    total_tokens_generated: int = 0
    prefill_time_ms: float = 0.0
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    total_draft_rounds: int = 0
    total_verify_rounds: int = 0
    avg_acceptance_rate: float = 0.0
    phases: List[PhaseMetrics] = field(default_factory=list)


class MetricsCollector:
    """
    Collects and aggregates performance metrics across inference.
    """
    
    def __init__(self):
        # Request-level metrics
        self.requests: Dict[str, RequestMetrics] = {}
        self.current_request_id: Optional[str] = None
        
        # Phase timing
        self.phase_starts: Dict[str, float] = {}
        
        # Aggregate statistics
        self.total_tokens_generated = 0
        self.total_requests = 0
        
        # Expert cache statistics
        self.expert_cache_hits = 0
        self.expert_cache_misses = 0
        
        # Transfer statistics
        self.total_transfers = 0
        self.total_transfer_time_ms = 0.0
    
    def start_request(self, request_id: str) -> None:
        """Start tracking a new request"""
        self.current_request_id = request_id
        self.requests[request_id] = RequestMetrics(
            request_id=request_id,
            start_time=time.time()
        )
        logger.debug(f"Started tracking request {request_id}")
    
    def end_request(self, request_id: str) -> None:
        """End tracking a request"""
        if request_id in self.requests:
            self.requests[request_id].end_time = time.time()
            self.total_requests += 1
            logger.debug(f"Ended tracking request {request_id}")
    
    def start_phase(self, phase_name: str) -> None:
        """Start timing a phase"""
        self.phase_starts[phase_name] = time.time()
    
    def end_phase(self, phase_name: str) -> None:
        """End timing a phase and record metrics"""
        if phase_name not in self.phase_starts:
            logger.warning(f"Phase {phase_name} was not started")
            return
        
        start_time = self.phase_starts[phase_name]
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        phase_metrics = PhaseMetrics(
            phase_name=phase_name,
            duration_ms=duration_ms,
            start_time=start_time,
            end_time=end_time
        )
        
        # Add to current request
        if self.current_request_id and self.current_request_id in self.requests:
            self.requests[self.current_request_id].phases.append(phase_metrics)
            
            # Update phase-specific timings
            if phase_name == 'prefill':
                self.requests[self.current_request_id].prefill_time_ms += duration_ms
            elif phase_name == 'draft':
                self.requests[self.current_request_id].draft_time_ms += duration_ms
                self.requests[self.current_request_id].total_draft_rounds += 1
            elif phase_name == 'verify':
                self.requests[self.current_request_id].verify_time_ms += duration_ms
                self.requests[self.current_request_id].total_verify_rounds += 1
        
        del self.phase_starts[phase_name]
        logger.debug(f"Phase {phase_name} took {duration_ms:.2f}ms")
    
    def record_tokens_generated(self, num_tokens: int) -> None:
        """Record number of tokens generated"""
        self.total_tokens_generated += num_tokens
        
        if self.current_request_id and self.current_request_id in self.requests:
            self.requests[self.current_request_id].total_tokens_generated += num_tokens
    
    def record_cache_hit(self) -> None:
        """Record an expert cache hit"""
        self.expert_cache_hits += 1
    
    def record_cache_miss(self) -> None:
        """Record an expert cache miss"""
        self.expert_cache_misses += 1
    
    def record_transfer(self, duration_ms: float) -> None:
        """Record a data transfer"""
        self.total_transfers += 1
        self.total_transfer_time_ms += duration_ms
    
    def get_request_metrics(self, request_id: str) -> Optional[RequestMetrics]:
        """Get metrics for a specific request"""
        return self.requests.get(request_id)
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        total_cache_accesses = self.expert_cache_hits + self.expert_cache_misses
        cache_hit_rate = (
            self.expert_cache_hits / total_cache_accesses 
            if total_cache_accesses > 0 else 0.0
        )
        
        # Calculate average times
        if self.total_requests > 0:
            avg_prefill_time = sum(
                r.prefill_time_ms for r in self.requests.values()
            ) / self.total_requests
            avg_draft_time = sum(
                r.draft_time_ms for r in self.requests.values()
            ) / self.total_requests
            avg_verify_time = sum(
                r.verify_time_ms for r in self.requests.values()
            ) / self.total_requests
        else:
            avg_prefill_time = avg_draft_time = avg_verify_time = 0.0
        
        return {
            'total_requests': self.total_requests,
            'total_tokens_generated': self.total_tokens_generated,
            'cache_hit_rate': cache_hit_rate,
            'expert_cache_hits': self.expert_cache_hits,
            'expert_cache_misses': self.expert_cache_misses,
            'total_transfers': self.total_transfers,
            'avg_transfer_time_ms': (
                self.total_transfer_time_ms / self.total_transfers 
                if self.total_transfers > 0 else 0.0
            ),
            'avg_prefill_time_ms': avg_prefill_time,
            'avg_draft_time_ms': avg_draft_time,
            'avg_verify_time_ms': avg_verify_time
        }
    
    def print_summary(self) -> None:
        """Print formatted summary"""
        summary = self.get_summary()
        
        print("\\n" + "="*60)
        print("INFERENCE METRICS SUMMARY")
        print("="*60)
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Total Tokens Generated: {summary['total_tokens_generated']}")
        print(f"\\nCache Statistics:")
        print(f"  Hit Rate: {summary['cache_hit_rate']:.2%}")
        print(f"  Hits: {summary['expert_cache_hits']}")
        print(f"  Misses: {summary['expert_cache_misses']}")
        print(f"\\nTransfer Statistics:")
        print(f"  Total Transfers: {summary['total_transfers']}")
        print(f"  Avg Transfer Time: {summary['avg_transfer_time_ms']:.2f}ms")
        print(f"\\nPhase Timings:")
        print(f"  Avg Prefill Time: {summary['avg_prefill_time_ms']:.2f}ms")
        print(f"  Avg Draft Time: {summary['avg_draft_time_ms']:.2f}ms")
        print(f"  Avg Verify Time: {summary['avg_verify_time_ms']:.2f}ms")
        print("="*60 + "\\n")