from typing import Optional, List, Dict
import torch
from enum import Enum

from ..core.types import (
    InferenceRequest, InferenceResponse, BatchedRequest,
    InferenceMode, GenerationConfig,
    ExecutionPhase, ExpertID,
    LayerExpertActivations, DraftMetrics, VerifyResult
)
from ..core.model import MoEConfig
from ..core.model_runner import ModelRunner
from ..memory.parameter_loader import ParameterLoader
from ..memory.expert_cache import ExpertCache
from ..memory.paged_kv_cache import PagedKVCache
from ..scheduling.prefetcher import ExpertPrefetcher
from ..scheduling.draft_schduler import DraftSchedulingStrategy
from .prefill_engine import PrefillEngine
from .draft_engine import DraftEngine
from .verify_engine import VerifyEngine
from .standard_engine import StandardDecodeEngine
from .acceptance_strategy import AcceptanceStrategy
from .batch_manager import BatchManager
from ..model import Qwen3ModelRunner
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCollector

logger = get_logger(__name__)


class EnhancedInferenceOrchestrator:
    """
    Enhanced orchestrator supporting both standard and speculative decoding,
    with batch processing capabilities.
    """
    
    def __init__(
        self,
        config: MoEConfig,
        parameter_loader: ParameterLoader,
        expert_cache: ExpertCache,
        prefetcher: ExpertPrefetcher,
        draft_scheduler: DraftSchedulingStrategy,
        acceptance_strategy: AcceptanceStrategy,
        metrics_collector: Optional[MetricsCollector] = None,
        max_batch_size: int = 32,
        default_mode: InferenceMode = InferenceMode.SPECULATIVE,
        model_runner: Optional[ModelRunner] = None,
        kv_cache_block_size: int = 256,
    ):
        self.config = config
        self.parameter_loader = parameter_loader
        self.expert_cache = expert_cache
        self.prefetcher = prefetcher
        self.draft_scheduler = draft_scheduler
        self.acceptance_strategy = acceptance_strategy
        self.metrics = metrics_collector or MetricsCollector()
        self.default_mode = default_mode
        self.kv_cache_block_size = kv_cache_block_size

        self.model_runner = model_runner or Qwen3ModelRunner(
            config=config,
            parameter_loader=parameter_loader,
        )
        
        # Initialize batch manager
        self.batch_manager = BatchManager(
            max_batch_size=max_batch_size,
            max_waiting_time_ms=100.0,
            enable_dynamic_batching=True
        )
        
        self.standard_engine = StandardDecodeEngine(
            model_runner=self.model_runner,
            parameter_loader=parameter_loader,
            expert_cache=expert_cache,
            prefetcher=prefetcher,
            metrics=self.metrics,
        )

        self.prefill_engine = PrefillEngine(
            model_runner=self.model_runner,
            parameter_loader=parameter_loader,
            expert_cache=expert_cache,
            prefetcher=prefetcher,
            metrics=self.metrics,
        )

        self.draft_engine = DraftEngine(
            model_runner=self.model_runner,
            parameter_loader=parameter_loader,
            expert_cache=expert_cache,
            draft_scheduler=draft_scheduler,
            metrics=self.metrics,
        )

        self.verify_engine = VerifyEngine(
            model_runner=self.model_runner,
            parameter_loader=parameter_loader,
            expert_cache=expert_cache,
            prefetcher=prefetcher,
            metrics=self.metrics,
        )
        
        logger.info(f"EnhancedInferenceOrchestrator initialized "
                   f"(default_mode={default_mode.value}, max_batch_size={max_batch_size})")
    
    def submit_request(self, request: InferenceRequest) -> str:
        """
        Submit an inference request for batch processing.
        
        Args:
            request: Inference request
        
        Returns:
            Request ID for tracking
        """
        self.batch_manager.add_request(request)
        return request.request_id
    
    def get_request_status(self, request_id: str):
        """Get status of a submitted request"""
        return self.batch_manager.get_request_status(request_id)
    
    def get_response(self, request_id: str) -> Optional[InferenceResponse]:
        """Get response for a completed request"""
        return self.batch_manager.get_response(request_id)
    
    def process_batch(
        self,
        mode: Optional[InferenceMode] = None,
        timeout_ms: Optional[float] = None
    ) -> List[InferenceResponse]:
        """
        Process next available batch.
        
        Args:
            mode: Inference mode (standard or speculative)
            timeout_ms: Timeout for batch formation
        
        Returns:
            List of inference responses
        """
        # Get next batch
        batch = self.batch_manager.get_next_batch(timeout_ms=timeout_ms)
        
        if batch is None:
            return []
        
        # Determine inference mode
        inference_mode = mode or self.default_mode
        
        # Normalize generation configs for requests
        for req in batch.requests:
            if getattr(req, "generation_config", None) is None:
                req.generation_config = self._get_generation_config(req)

        # Check if all requests want the same mode
        if all(req.generation_config.use_speculative for req in batch.requests):
            inference_mode = InferenceMode.SPECULATIVE
        elif not any(req.generation_config.use_speculative for req in batch.requests):
            inference_mode = InferenceMode.STANDARD
        
        logger.info(f"Processing batch {batch.batch_id} with mode={inference_mode.value}")
        
        # Route to appropriate engine
        if inference_mode == InferenceMode.STANDARD:
            result = self.standard_engine.generate_batch(batch)
        else:
            result = self._generate_batch_speculative(batch)
        
        # Complete batch and generate responses
        responses = self.batch_manager.complete_batch(
            batch=batch,
            generated_sequences=result['generated_sequences'],
            statistics=result.get('statistics')
        )
        
        return responses

    def _get_generation_config(self, request: InferenceRequest) -> GenerationConfig:
        if getattr(request, "generation_config", None) is not None:
            return request.generation_config
        return GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=True,
            use_speculative=self.default_mode == InferenceMode.SPECULATIVE,
        )
    
    def generate(
        self,
        request: InferenceRequest,
        mode: Optional[InferenceMode] = None
    ) -> torch.Tensor:
        """
        Synchronous generation for single request (for compatibility).
        
        Args:
            request: Inference request
            mode: Inference mode
        
        Returns:
            Generated token IDs
        """
        inference_mode = mode or self.default_mode
        
        if inference_mode == InferenceMode.STANDARD:
            return self._generate_standard(request)
        else:
            return self._generate_speculative(request)
    
    def _generate_standard(self, request: InferenceRequest) -> torch.Tensor:
        """
        Standard autoregressive generation for single request.
        
        Args:
            request: Inference request
        
        Returns:
            Generated token IDs
        """
        logger.info(f"Starting standard generation for request {request.request_id}")
        self.metrics.start_request(request.request_id)
        
        request.generation_config = self._get_generation_config(request)

        # Create single-request batch
        batch = self._create_single_request_batch(request)
        
        # Generate
        result = self.standard_engine.generate_batch(batch)
        
        self.metrics.end_request(request.request_id)
        
        generated_ids = result['generated_sequences'][0]
        
        logger.info(f"Standard generation complete: {len(generated_ids)} tokens")
        
        return generated_ids
    
    def _generate_speculative(self, request: InferenceRequest) -> torch.Tensor:
        """
        Speculative generation (draft-verify) for single request.
        Uses the existing draft-verify implementation.
        
        Args:
            request: Inference request
        
        Returns:
            Generated token IDs
        """
        logger.info(f"Starting speculative generation for request {request.request_id}")
        self.metrics.start_request(request.request_id)
        
        # Initialize KV cache
        kv_cache = PagedKVCache(
            config=self.config,
            block_size=self.kv_cache_block_size,
            dtype=self.config.get_dtype(),
        )
        
        # Phase 1: Prefill
        logger.info("=== PREFILL PHASE ===")
        prefill_output = self.prefill_engine.forward(
            input_ids=request.input_ids,
            kv_cache=kv_cache
        )
        
        generated_ids = [prefill_output['next_token_id'].item()]
        
        # Phase 2: Decode (Draft-Verify loop)
        logger.info("=== DECODE PHASE ===")
        gen_config = self._get_generation_config(request)
        request.generation_config = gen_config
        max_new_tokens = gen_config.max_new_tokens
        
        while len(generated_ids) < max_new_tokens:
            # Draft phase
            draft_result = self._run_draft_phase(
                current_ids=torch.tensor(generated_ids, dtype=torch.long),
                kv_cache=kv_cache,
                config=gen_config
            )
            
            # Check if should verify
            if self.draft_scheduler.should_trigger_verify(
                num_drafted_tokens=len(draft_result['drafted_tokens']),
                perplexity=draft_result['metrics'].perplexity,
                cache_hit_rate=draft_result['metrics'].expert_hit_rate,
                max_draft_tokens=gen_config.max_draft_tokens
            ):
                # Verify phase
                verify_result = self._run_verify_phase(
                    prefill_ids=request.input_ids,
                    draft_tokens=draft_result['drafted_tokens'],
                    kv_cache=kv_cache,
                    config=gen_config
                )
                
                # Accept tokens
                accepted_tokens = verify_result.accepted_token_ids.tolist()
                generated_ids.extend(accepted_tokens)
                
                # Update KV cache
                kv_cache.replace_draft_with_verify(
                    seq_id=0,
                    verify_seq_id=1,
                    num_accepted_tokens=len(accepted_tokens),
                )
                
                logger.info(f"Accepted {len(accepted_tokens)}/{len(draft_result['drafted_tokens'])} "
                           f"draft tokens")
                
                if not verify_result.should_continue:
                    break
            else:
                # Continue drafting
                generated_ids.extend(draft_result['drafted_tokens'])

        self.metrics.end_request(request.request_id)
        
        logger.info(f"Speculative generation complete: {len(generated_ids)} tokens")
        
        return torch.tensor(generated_ids, dtype=torch.long)

    def _generate_batch_speculative(self, batch: BatchedRequest) -> Dict:
        """
        Speculative generation for batch.
        Note: This is a simplified version. Full batch speculative decoding
        is complex due to varying draft lengths per request.
        
        For now, process each request individually within the batch.
        """
        logger.info(f"Speculative generation for batch {batch.batch_id}")
        
        generated_sequences = []
        
        for request in batch.requests:
            # Generate for each request
            gen_ids = self._generate_speculative(request)
            generated_sequences.append(gen_ids)
        
        return {
            'generated_sequences': generated_sequences,
            'statistics': {}
        }
    
    def _run_draft_phase(
        self,
        current_ids: torch.Tensor,
        kv_cache,
        config: GenerationConfig
    ) -> Dict:
        """Execute draft phase (existing logic)"""
        logger.info("--- Draft Phase ---")
        kv_cache.start_draft(seq_id=0)
        
        draft_output = self.draft_engine.forward(
            input_ids=current_ids[-1:],
            kv_cache=kv_cache,
            max_draft_tokens=config.max_draft_tokens,
            temperature=config.temperature,
            seq_ids=[0],
        )
        
        return draft_output
    
    def _run_verify_phase(
        self,
        prefill_ids: torch.Tensor,
        draft_tokens: List[int],
        kv_cache,
        config: GenerationConfig
    ):
        """Execute verify phase (existing logic)"""
        from ..core.types import VerifyResult
        
        logger.info("--- Verify Phase ---")
        
        all_ids = torch.cat([
            prefill_ids,
            torch.tensor(draft_tokens, dtype=torch.long)
        ])
        
        verify_kv_cache = kv_cache
        
        verify_output = self.verify_engine.forward(
            input_ids=all_ids,
            kv_cache=verify_kv_cache,
            seq_ids=[1],
        )
        
        verify_logits = verify_output['logits']
        if verify_logits.dim() == 3:
            verify_logits = verify_logits[:, -len(draft_tokens):, :][0]
        draft_token_ids = torch.tensor(draft_tokens, dtype=torch.long)

        acceptance_result = self.acceptance_strategy.accept(
            draft_token_ids=draft_token_ids,
            verify_logits=verify_logits[-len(draft_tokens):],
            temperature=config.temperature
        )
        
        return VerifyResult(
            num_accepted_tokens=acceptance_result['num_accepted'],
            accepted_token_ids=acceptance_result['accepted_tokens'],
            new_kv_cache=verify_kv_cache,
            should_continue=True
        )
    
    def _create_single_request_batch(self, request: InferenceRequest) -> BatchedRequest:
        """Create a batch containing a single request"""
        input_ids = request.input_ids.unsqueeze(0)  # [1, seq_len]
        seq_len = len(request.input_ids)
        
        return BatchedRequest(
            batch_id=f"single_{request.request_id}",
            requests=[request],
            input_ids=input_ids,
            attention_mask=torch.ones(1, seq_len, dtype=torch.long),
            position_ids=torch.arange(seq_len).unsqueeze(0),
            current_lengths=[seq_len],
            finished=[False],
            max_batch_seq_len=seq_len,
            padding_token_id=0
        )
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            'cache_stats': self.expert_cache.get_cache_stats(),
            'batch_queue_length': self.batch_manager.get_queue_length(),
            'active_batches': self.batch_manager.get_active_batch_count(),
            'metrics': self.metrics.get_summary()
        }

class InferenceOrchestrator:
    """
    High-level orchestrator coordinating all inference phases.
    Manages phase transitions and resource allocation.
    """
    
    def __init__(
        self,
        config: MoEConfig,
        parameter_loader: ParameterLoader,
        expert_cache: ExpertCache,
        prefetcher: ExpertPrefetcher,
        draft_scheduler: DraftSchedulingStrategy,
        acceptance_strategy: AcceptanceStrategy,
        metrics_collector: Optional[MetricsCollector] = None,
        model_runner: Optional[ModelRunner] = None,
        kv_cache_block_size: int = 256,
    ):
        self.config = config
        self.parameter_loader = parameter_loader
        self.expert_cache = expert_cache
        self.prefetcher = prefetcher
        self.draft_scheduler = draft_scheduler
        self.acceptance_strategy = acceptance_strategy
        self.metrics = metrics_collector or MetricsCollector()
        self.kv_cache_block_size = kv_cache_block_size
        self.model_runner = model_runner or Qwen3ModelRunner(
            config=config,
            parameter_loader=parameter_loader,
        )
        
        # Initialize engines (new implementation)
        self.prefill_engine = PrefillEngine(
            model_runner=self.model_runner,
            parameter_loader=parameter_loader,
            expert_cache=expert_cache,
            prefetcher=prefetcher,
            metrics=self.metrics,
        )
        
        self.draft_engine = DraftEngine(
            model_runner=self.model_runner,
            parameter_loader=parameter_loader,
            expert_cache=expert_cache,
            draft_scheduler=draft_scheduler,
            metrics=self.metrics,
        )

        self.verify_engine = VerifyEngine(
            model_runner=self.model_runner,
            parameter_loader=parameter_loader,
            expert_cache=expert_cache,
            prefetcher=prefetcher,
            metrics=self.metrics,
        )
        
        # KV cache
        self.kv_cache: Optional[PagedKVCache] = None
        
        # Current phase
        self.current_phase = ExecutionPhase.PREFILL
        
        logger.info("InferenceOrchestrator initialized")
    
    def generate(
        self,
        request: InferenceRequest
    ) -> torch.Tensor:
        """
        Main entry point for text generation.
        
        Args:
            request: Inference request with input and parameters
        
        Returns:
            Generated token IDs
        """
        logger.info(f"Starting generation for request {request.request_id}")
        self.metrics.start_request(request.request_id)
        
        # Initialize KV cache
        self.kv_cache = PagedKVCache(
            config=self.config,
            block_size=self.kv_cache_block_size,
            dtype=self.config.get_dtype(),
        )
        
        # Phase 1: Prefill
        logger.info("=== PREFILL PHASE ===")
        self.current_phase = ExecutionPhase.PREFILL
        
        prefill_output = self.prefill_engine.forward(
            input_ids=request.input_ids,
            kv_cache=self.kv_cache
        )
        
        generated_ids = [prefill_output['next_token_id'].item()]
        
        # Phase 2: Decode (Draft-Verify loop)
        logger.info("=== DECODE PHASE ===")
        self.current_phase = ExecutionPhase.DRAFT
        
        while len(generated_ids) < request.max_new_tokens:
            # Draft phase
            draft_result = self._run_draft_phase(
                current_ids=torch.tensor(generated_ids, dtype=torch.long),
                request=request
            )
            
            # Check if should verify
            if self.draft_scheduler.should_trigger_verify(
                num_drafted_tokens=len(draft_result['drafted_tokens']),
                perplexity=draft_result['metrics'].perplexity,
                cache_hit_rate=draft_result['metrics'].expert_hit_rate,
                max_draft_tokens=self.config.max_draft_tokens
            ):
                # Verify phase
                verify_result = self._run_verify_phase(
                    prefill_ids=request.input_ids,
                    draft_tokens=draft_result['drafted_tokens'],
                    request=request
                )
                
                # Accept tokens
                accepted_tokens = verify_result.accepted_token_ids.tolist()
                generated_ids.extend(accepted_tokens)
                
                # Update KV cache
                self.kv_cache.replace_draft_with_verify(
                    seq_id=0,
                    verify_seq_id=1,
                    num_accepted_tokens=len(accepted_tokens),
                )
                
                logger.info(f"Accepted {len(accepted_tokens)}/{len(draft_result['drafted_tokens'])} "
                           f"draft tokens")
                
                if not verify_result.should_continue:
                    break
            else:
                # Continue drafting
                generated_ids.extend(draft_result['drafted_tokens'])
        
        self.metrics.end_request(request.request_id)
        
        logger.info(f"Generation complete: {len(generated_ids)} tokens generated")
        
        return torch.tensor(generated_ids, dtype=torch.long)
    
    def _run_draft_phase(
        self,
        current_ids: torch.Tensor,
        request: InferenceRequest
    ) -> Dict:
        """
        Execute draft phase.
        
        Returns:
            Dict with drafted tokens and metrics
        """
        logger.info("--- Draft Phase ---")
        self.current_phase = ExecutionPhase.DRAFT
        
        # Mark draft phase
        self.kv_cache.start_draft(seq_id=0)
        
        # Run draft engine
        draft_output = self.draft_engine.forward(
            input_ids=current_ids[-1:],  # Only last token
            kv_cache=self.kv_cache,
            max_draft_tokens=self.config.max_draft_tokens,
            temperature=request.temperature,
            seq_ids=[0],
        )
        
        return draft_output
    
    def _run_verify_phase(
        self,
        prefill_ids: torch.Tensor,
        draft_tokens: List[int],
        request: InferenceRequest
    ) -> VerifyResult:
        """
        Execute verify phase.
        
        Returns:
            VerifyResult with accepted tokens and new KV cache
        """
        logger.info("--- Verify Phase ---")
        self.current_phase = ExecutionPhase.VERIFY
        
        # Concatenate prefill + draft tokens
        all_ids = torch.cat([
            prefill_ids,
            torch.tensor(draft_tokens, dtype=torch.long)
        ])
        
        # Run verify engine (full model inference) on same cache with new seq_id
        verify_kv_cache = self.kv_cache
        verify_output = self.verify_engine.forward(
            input_ids=all_ids,
            kv_cache=verify_kv_cache,
            seq_ids=[1],
        )
        
        # Run acceptance strategy
        verify_logits = verify_output['logits']
        if verify_logits.dim() == 3:
            verify_logits = verify_logits[:, -len(draft_tokens):, :][0]
        draft_token_ids = torch.tensor(draft_tokens, dtype=torch.long)
        
        acceptance_result = self.acceptance_strategy.accept(
            draft_token_ids=draft_token_ids,
            verify_logits=verify_logits[-len(draft_tokens):],  # Only draft positions
            temperature=request.temperature
        )
        
        return VerifyResult(
            num_accepted_tokens=acceptance_result['num_accepted'],
            accepted_token_ids=acceptance_result['accepted_tokens'],
            new_kv_cache=verify_kv_cache,
            should_continue=True  # TODO: Add EOS detection
        )
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics from all components"""
        return {
            'cache_stats': self.expert_cache.get_cache_stats(),
            'kv_cache_memory_mb': self.kv_cache.get_memory_usage_mb() if self.kv_cache else 0,
            'metrics': self.metrics.get_summary()
        }
