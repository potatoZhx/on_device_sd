from typing import Optional, List, Dict, Union
import time
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
from .continuous_batch_engine import ContinuousBatchEngine, DecodeMode, Sequence
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

        max_batched_tokens = max_batch_size * self.config.max_position_embeddings
        self.cb_engines = {
            InferenceMode.STANDARD: self._create_continuous_engine(
                decode_mode=DecodeMode.STANDARD,
                max_num_seqs=max_batch_size,
                max_num_batched_tokens=max_batched_tokens,
                max_draft_tokens=GenerationConfig().max_draft_tokens,
            ),
            InferenceMode.SPECULATIVE: self._create_continuous_engine(
                decode_mode=DecodeMode.SPECULATIVE,
                max_num_seqs=max_batch_size,
                max_num_batched_tokens=max_batched_tokens,
                max_draft_tokens=GenerationConfig().max_draft_tokens,
            ),
        }
        self.cb_seq_request_map: Dict[InferenceMode, Dict[int, InferenceRequest]] = {
            InferenceMode.STANDARD: {},
            InferenceMode.SPECULATIVE: {},
        }
        
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
        requests = self.batch_manager.pop_requests(
            timeout_ms=timeout_ms,
            max_requests=self.batch_manager.max_batch_size,
        )

        if not requests and all(engine.scheduler.is_finished() for engine in self.cb_engines.values()):
            return []

        for request in requests:
            gen_config = self._get_generation_config(request)
            request.generation_config = gen_config
            request.max_new_tokens = gen_config.max_new_tokens
            engine_mode = self._select_mode_for_request(request, mode)
            if engine_mode == InferenceMode.SPECULATIVE:
                self.cb_engines[engine_mode].executor.max_draft_tokens = max(
                    self.cb_engines[engine_mode].executor.max_draft_tokens,
                    gen_config.max_draft_tokens,
                )
            seq = self._build_sequence(request)
            self.cb_seq_request_map[engine_mode][seq.seq_id] = request
            self.cb_engines[engine_mode].add_sequences([seq])

        finished_sequences: List[Sequence] = []
        for engine in self.cb_engines.values():
            if not engine.scheduler.is_finished():
                _, finished = engine.step()
                finished_sequences.extend(finished)

        responses = self._build_responses_from_sequences(finished_sequences)
        if responses:
            self.batch_manager.complete_requests(responses)

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
        requests: Union[InferenceRequest, List[InferenceRequest]],
        mode: Optional[InferenceMode] = None
    ) -> List[torch.Tensor]:
        if not isinstance(requests, list):
            requests = [requests]
        
        if not requests:
            return []
        
        for req in requests:
            self.metrics.start_request(req.request_id)
        
        result = self._execute_requests_continuous(requests, mode=mode)
        
        for req in requests:
            self.metrics.end_request(req.request_id)
        
        return result['generated_sequences']
    
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
    
    def _execute_batch(
        self,
        batch: BatchedRequest,
        mode: Optional[InferenceMode] = None
    ) -> Dict:
        return self._execute_requests_continuous(batch.requests, mode=mode)
    
    def _create_batch_from_requests(self, requests: List[InferenceRequest]) -> BatchedRequest:
        batch_size = len(requests)
        max_seq_len = max(len(req.input_ids) for req in requests)
        padding_token_id = self.batch_manager.padding_token_id
        
        input_ids = torch.full(
            (batch_size, max_seq_len),
            padding_token_id,
            dtype=torch.long
        )
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        position_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        
        for i, request in enumerate(requests):
            seq_len = len(request.input_ids)
            input_ids[i, :seq_len] = request.input_ids
            attention_mask[i, :seq_len] = 1
            position_ids[i, :seq_len] = torch.arange(seq_len)
        
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
            padding_token_id=padding_token_id
        )

    def _create_continuous_engine(
        self,
        decode_mode: DecodeMode,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        max_draft_tokens: int,
    ) -> ContinuousBatchEngine:
        kv_cache = PagedKVCache(
            config=self.config,
            block_size=self.kv_cache_block_size,
            dtype=self.config.get_dtype(),
        )
        return ContinuousBatchEngine(
            model_runner=self.model_runner,
            kv_cache=kv_cache,
            expert_cache=self.expert_cache,
            parameter_loader=self.parameter_loader,
            prefetcher=self.prefetcher,
            draft_scheduler=self.draft_scheduler,
            acceptance_strategy=self.acceptance_strategy,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            decode_mode=decode_mode,
            max_draft_tokens=max_draft_tokens,
        )

    def _select_mode_for_request(
        self,
        request: InferenceRequest,
        mode: Optional[InferenceMode],
    ) -> InferenceMode:
        if mode is not None:
            return mode
        gen_config = self._get_generation_config(request)
        return InferenceMode.SPECULATIVE if gen_config.use_speculative else InferenceMode.STANDARD

    def _build_sequence(self, request: InferenceRequest) -> Sequence:
        gen_config = self._get_generation_config(request)
        return Sequence(
            token_ids=request.input_ids.tolist(),
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            top_k=gen_config.top_k,
            eos_token_id=gen_config.eos_token_id,
        )

    def _execute_requests_continuous(
        self,
        requests: List[InferenceRequest],
        mode: Optional[InferenceMode] = None,
    ) -> Dict:
        if not requests:
            return {"generated_sequences": [], "statistics": {}}

        requests_by_mode: Dict[InferenceMode, List[tuple[int, InferenceRequest]]] = {
            InferenceMode.STANDARD: [],
            InferenceMode.SPECULATIVE: [],
        }
        for idx, request in enumerate(requests):
            gen_config = self._get_generation_config(request)
            request.generation_config = gen_config
            request.max_new_tokens = gen_config.max_new_tokens
            selected_mode = self._select_mode_for_request(request, mode)
            requests_by_mode[selected_mode].append((idx, request))

        generated_sequences: List[Optional[torch.Tensor]] = [None] * len(requests)
        for selected_mode, mode_requests in requests_by_mode.items():
            if not mode_requests:
                continue
            outputs = self._run_continuous_generation(mode_requests, selected_mode)
            for idx, output in outputs:
                generated_sequences[idx] = output

        return {
            "generated_sequences": generated_sequences,
            "statistics": {},
        }

    def _run_continuous_generation(
        self,
        requests: List[tuple[int, InferenceRequest]],
        mode: InferenceMode,
    ) -> List[tuple[int, torch.Tensor]]:
        max_draft_tokens = max(
            (self._get_generation_config(req).max_draft_tokens for _, req in requests),
            default=GenerationConfig().max_draft_tokens,
        )
        max_num_batched_tokens = len(requests) * self.config.max_position_embeddings
        engine = self._create_continuous_engine(
            decode_mode=DecodeMode.SPECULATIVE if mode == InferenceMode.SPECULATIVE else DecodeMode.STANDARD,
            max_num_seqs=len(requests),
            max_num_batched_tokens=max_num_batched_tokens,
            max_draft_tokens=max_draft_tokens,
        )

        seqs: List[Sequence] = []
        seq_id_to_index: Dict[int, int] = {}
        for idx, request in requests:
            seq = self._build_sequence(request)
            seqs.append(seq)
            seq_id_to_index[seq.seq_id] = idx

        engine.add_sequences(seqs)

        outputs: Dict[int, torch.Tensor] = {}
        while not engine.scheduler.is_finished():
            _, finished = engine.step()
            for seq in finished:
                outputs[seq_id_to_index[seq.seq_id]] = torch.tensor(
                    seq.output_token_ids,
                    dtype=torch.long,
                )

        return [(idx, outputs[idx]) for idx, _ in requests]

    def _build_responses_from_sequences(
        self,
        sequences: List[Sequence],
    ) -> List[InferenceResponse]:
        responses: List[InferenceResponse] = []
        for seq in sequences:
            request = None
            for mode_map in self.cb_seq_request_map.values():
                if seq.seq_id in mode_map:
                    request = mode_map.pop(seq.seq_id)
                    break
            if request is None:
                continue
            generation_time = 0.0
            if hasattr(request, "started_at"):
                generation_time = (time.time() - request.started_at) * 1000
            num_tokens = len(seq.output_token_ids)
            tokens_per_sec = num_tokens / (generation_time / 1000) if generation_time > 0 else 0.0
            success = seq.error_msg is None
            response = InferenceResponse(
                request_id=request.request_id,
                generated_ids=seq.output_token_ids,
                num_tokens_generated=num_tokens,
                generation_time_ms=generation_time,
                tokens_per_second=tokens_per_sec,
                success=success,
                error=seq.error_msg,
            )
            responses.append(response)
        return responses
    
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
