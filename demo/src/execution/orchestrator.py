from typing import Optional, List, Dict, Union
import time
import torch

from ..core.types import (
    InferenceRequest, InferenceResponse, BatchedRequest,
    InferenceMode, GenerationConfig,
    ExecutionPhase,
)
from ..core.model import MoEConfig
from ..core.model_runner import ModelRunner
from ..memory.parameter_loader import ParameterLoader
from ..memory.expert_cache import ExpertCache
from ..memory.paged_kv_cache import PagedKVCache
from ..scheduling.prefetcher import ExpertPrefetcher
from ..scheduling.draft_schduler import DraftSchedulingStrategy
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
        """Deprecated compatibility path: delegate to continuous engine."""
        logger.warning(
            "_generate_speculative is deprecated; delegating to continuous batch engine"
        )
        result = self._execute_requests_continuous(
            [request],
            mode=InferenceMode.SPECULATIVE,
        )
        return result["generated_sequences"][0]

    def _generate_batch_speculative(self, batch: BatchedRequest) -> Dict:
        """Deprecated compatibility path: delegate to continuous engine."""
        logger.warning(
            "_generate_batch_speculative is deprecated; delegating to continuous batch engine"
        )
        return self._execute_requests_continuous(
            batch.requests,
            mode=InferenceMode.SPECULATIVE,
        )
    
    def _execute_batch(
        self,
        batch: BatchedRequest,
        mode: Optional[InferenceMode] = None
    ) -> Dict:
        return self._execute_requests_continuous(batch.requests, mode=mode)
    
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
            do_sample=gen_config.do_sample,
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
