import time
import threading
from queue import PriorityQueue, Empty
from typing import List, Dict, Optional, Tuple
import torch
from collections import defaultdict

from ..core.types import (
    InferenceRequest, BatchedRequest, InferenceResponse,
    RequestStatus, GenerationConfig
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BatchManager:
    """
    Manages request batching and scheduling.
    Implements continuous batching for efficient throughput.
    """
    
    def __init__(
        self,
        max_batch_size: int = 32,
        max_waiting_time_ms: float = 100.0,
        enable_dynamic_batching: bool = True,
        padding_token_id: int = 0
    ):
        self.max_batch_size = max_batch_size
        self.max_waiting_time_ms = max_waiting_time_ms
        self.enable_dynamic_batching = enable_dynamic_batching
        self.padding_token_id = padding_token_id
        
        # Request queue (priority queue: higher priority first)
        self.request_queue: PriorityQueue = PriorityQueue()
        
        # Active batches
        self.active_batches: Dict[str, BatchedRequest] = {}
        self.active_requests: Dict[str, InferenceRequest] = {}
        
        # Request tracking
        self.request_status: Dict[str, RequestStatus] = {}
        self.request_responses: Dict[str, InferenceResponse] = {}
        
        # Threading
        self.lock = threading.Lock()
        
        logger.info(f"BatchManager initialized: max_batch_size={max_batch_size}")
    
    def add_request(self, request: InferenceRequest) -> None:
        """
        Add a new inference request to the queue.
        
        Args:
            request: Inference request to add
        """
        with self.lock:
            # Priority queue: lower number = higher priority
            # Negate priority so higher priority values come first
            self.request_queue.put((-request.priority, time.time(), request))
            self.request_status[request.request_id] = RequestStatus.QUEUED
        
        logger.debug(f"Added request {request.request_id} to queue "
                    f"(priority={request.priority})")
    
    def get_next_batch(
        self,
        timeout_ms: Optional[float] = None
    ) -> Optional[BatchedRequest]:
        """
        Get the next batch of requests to process.
        
        Args:
            timeout_ms: Maximum time to wait for batch formation
        
        Returns:
            BatchedRequest if available, None otherwise
        """
        if timeout_ms is None:
            timeout_ms = self.max_waiting_time_ms
        
        start_time = time.time()
        requests = []
        
        # Collect requests up to max_batch_size or timeout
        while len(requests) < self.max_batch_size:
            remaining_time = timeout_ms - (time.time() - start_time) * 1000
            
            if remaining_time <= 0 and len(requests) > 0:
                break
            
            try:
                # Wait for request with timeout
                timeout_sec = max(0.001, remaining_time / 1000.0)
                _, _, request = self.request_queue.get(timeout=timeout_sec)
                requests.append(request)
                
                # Mark as batched
                with self.lock:
                    self.request_status[request.request_id] = RequestStatus.BATCHED
                
            except Empty:
                # No more requests available
                if len(requests) > 0:
                    break
                else:
                    return None
        
        if not requests:
            return None
        
        # Form batch
        batch = self._form_batch(requests)
        
        # Track active batch
        with self.lock:
            self.active_batches[batch.batch_id] = batch
            for req in requests:
                self.request_status[req.request_id] = RequestStatus.PROCESSING
                req.started_at = time.time()
        
        logger.info(f"Formed batch {batch.batch_id} with {len(requests)} requests")
        
        return batch

    def pop_requests(
        self,
        timeout_ms: Optional[float] = None,
        max_requests: Optional[int] = None
    ) -> List[InferenceRequest]:
        if timeout_ms is None:
            timeout_ms = self.max_waiting_time_ms
        if max_requests is None:
            max_requests = self.max_batch_size

        start_time = time.time()
        requests: List[InferenceRequest] = []

        while len(requests) < max_requests:
            remaining_time = timeout_ms - (time.time() - start_time) * 1000

            if remaining_time <= 0 and len(requests) > 0:
                break

            try:
                timeout_sec = max(0.001, remaining_time / 1000.0)
                _, _, request = self.request_queue.get(timeout=timeout_sec)
                requests.append(request)

                with self.lock:
                    self.request_status[request.request_id] = RequestStatus.PROCESSING
                    self.active_requests[request.request_id] = request
                    request.started_at = time.time()
            except Empty:
                if len(requests) > 0:
                    break
                return []

        return requests
    
    def _form_batch(self, requests: List[InferenceRequest]) -> BatchedRequest:
        """
        Form a batched request from individual requests.
        
        Args:
            requests: List of inference requests
        
        Returns:
            BatchedRequest with padded tensors
        """
        batch_size = len(requests)
        
        # Find max sequence length in this batch
        max_seq_len = max(len(req.input_ids) for req in requests)
        
        # Initialize batched tensors
        input_ids = torch.full(
            (batch_size, max_seq_len),
            self.padding_token_id,
            dtype=torch.long
        )
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        position_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        
        # Fill in actual data
        for i, request in enumerate(requests):
            seq_len = len(request.input_ids)
            input_ids[i, :seq_len] = request.input_ids
            attention_mask[i, :seq_len] = 1
            position_ids[i, :seq_len] = torch.arange(seq_len)
        
        # Create batched request
        batch_id = f"batch_{int(time.time() * 1000)}_{id(requests[0])}"
        
        return BatchedRequest(
            batch_id=batch_id,
            requests=requests,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            current_lengths=[len(req.input_ids) for req in requests],
            finished=[False] * batch_size,
            max_batch_seq_len=max_seq_len,
            padding_token_id=self.padding_token_id
        )
    
    def update_batch(
        self,
        batch: BatchedRequest,
        new_token_ids: torch.Tensor,
        finished_mask: torch.Tensor
    ) -> None:
        """
        Update batch state after generating new tokens.
        
        Args:
            batch: Batch to update
            new_token_ids: New tokens for each request [batch_size]
            finished_mask: Boolean mask indicating finished requests [batch_size]
        """
        batch_size = len(batch.requests)
        
        # Update finished status
        for i in range(batch_size):
            if finished_mask[i].item():
                batch.finished[i] = True
        
        # Extend input_ids and attention_mask for continuing requests
        if not batch.is_complete():
            # Append new tokens
            new_input_ids = torch.cat([
                batch.input_ids,
                new_token_ids.unsqueeze(1)
            ], dim=1)
            
            # Update attention mask
            new_attention_mask = torch.cat([
                batch.attention_mask,
                (~finished_mask).long().unsqueeze(1)
            ], dim=1)
            
            batch.input_ids = new_input_ids
            batch.attention_mask = new_attention_mask
            batch.max_batch_seq_len += 1
            
            # Update lengths
            for i in range(batch_size):
                if not batch.finished[i]:
                    batch.current_lengths[i] += 1
    
    def complete_batch(
        self,
        batch: BatchedRequest,
        generated_sequences: List[torch.Tensor],
        statistics: Optional[Dict] = None
    ) -> List[InferenceResponse]:
        """
        Mark batch as complete and generate responses.
        
        Args:
            batch: Completed batch
            generated_sequences: Generated token sequences for each request
            statistics: Optional generation statistics
        
        Returns:
            List of InferenceResponse objects
        """
        responses = []
        
        for i, request in enumerate(batch.requests):
            # Calculate timing
            generation_time = (time.time() - request.started_at) * 1000
            num_tokens = len(generated_sequences[i])
            tokens_per_sec = num_tokens / (generation_time / 1000) if generation_time > 0 else 0
            
            # Create response
            response = InferenceResponse(
                request_id=request.request_id,
                generated_ids=generated_sequences[i].tolist(),
                num_tokens_generated=num_tokens,
                generation_time_ms=generation_time,
                tokens_per_second=tokens_per_sec,
                success=True
            )
            
            # Add batch statistics if available
            if statistics:
                response.prefill_time_ms = statistics.get('prefill_time_ms', 0.0)
                response.decode_time_ms = statistics.get('decode_time_ms', 0.0)
                if 'num_draft_rounds' in statistics:
                    response.num_draft_rounds = statistics['num_draft_rounds']
                    response.avg_acceptance_rate = statistics.get('avg_acceptance_rate', 0.0)
            
            responses.append(response)
            
            # Update tracking
            with self.lock:
                self.request_status[request.request_id] = RequestStatus.COMPLETED
                self.request_responses[request.request_id] = response
                request.completed_at = time.time()
        
        # Remove from active batches
        with self.lock:
            if batch.batch_id in self.active_batches:
                del self.active_batches[batch.batch_id]
        
        logger.info(f"Completed batch {batch.batch_id}")
        
        return responses

    def complete_requests(self, responses: List[InferenceResponse]) -> None:
        for response in responses:
            with self.lock:
                status = RequestStatus.COMPLETED if response.success else RequestStatus.FAILED
                self.request_status[response.request_id] = status
                self.request_responses[response.request_id] = response
                request = self.active_requests.pop(response.request_id, None)
                if request is not None:
                    request.completed_at = time.time()
    
    def get_request_status(self, request_id: str) -> Optional[RequestStatus]:
        """Get status of a specific request"""
        with self.lock:
            return self.request_status.get(request_id)
    
    def get_response(self, request_id: str) -> Optional[InferenceResponse]:
        """Get response for a completed request"""
        with self.lock:
            return self.request_responses.get(request_id)
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending request"""
        with self.lock:
            if request_id in self.request_status:
                status = self.request_status[request_id]
                if status in [RequestStatus.QUEUED, RequestStatus.BATCHED]:
                    self.request_status[request_id] = RequestStatus.CANCELLED
                    return True
        return False
    
    def get_queue_length(self) -> int:
        """Get current queue length"""
        return self.request_queue.qsize()
    
    def get_active_batch_count(self) -> int:
        """Get number of active batches"""
        with self.lock:
            return len(self.active_batches) + len(self.active_requests)
