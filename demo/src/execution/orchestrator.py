from typing import Optional, List, Dict
import torch
from ..core.types import (
    InferenceRequest, ExecutionPhase, ExpertID,
    LayerExpertActivations, DraftMetrics, VerifyResult
)
from ..core.model import MoEConfig
from ..memory.parameter_loader import ParameterLoader
from ..memory.expert_cache import ExpertCache
from ..memory.kv_cache import KVCache
from ..scheduling.prefetcher import ExpertPrefetcher
from ..scheduling.draft_scheduler import DraftSchedulingStrategy
from .prefill_engine import PrefillEngine
from .draft_engine import DraftEngine
from .verify_engine import VerifyEngine
from .acceptance import AcceptanceStrategy
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCollector

logger = get_logger(__name__)


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
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.config = config
        self.parameter_loader = parameter_loader
        self.expert_cache = expert_cache
        self.prefetcher = prefetcher
        self.draft_scheduler = draft_scheduler
        self.acceptance_strategy = acceptance_strategy
        self.metrics = metrics_collector or MetricsCollector()
        
        # Initialize engines
        self.prefill_engine = PrefillEngine(
            config=config,
            parameter_loader=parameter_loader,
            expert_cache=expert_cache,
            prefetcher=prefetcher,
            metrics=self.metrics
        )
        
        self.draft_engine = DraftEngine(
            config=config,
            parameter_loader=parameter_loader,
            expert_cache=expert_cache,
            draft_scheduler=draft_scheduler,
            metrics=self.metrics
        )
        # TODO: 为什么不用acceptance_strategy
        self.verify_engine = VerifyEngine(
            config=config,
            parameter_loader=parameter_loader,
            expert_cache=expert_cache,
            prefetcher=prefetcher,
            metrics=self.metrics
        )
        
        # KV cache
        self.kv_cache: Optional[KVCache] = None
        
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
        self.kv_cache = KVCache(self.config, max_batch_size=1)
        
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
                    verify_cache=verify_result.new_kv_cache,
                    num_accepted_tokens=len(accepted_tokens)
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
        
        # Backup KV cache before drafting
        self.kv_cache.backup_for_draft()
        
        # Run draft engine
        draft_output = self.draft_engine.forward(
            input_ids=current_ids[-1:],  # Only last token
            kv_cache=self.kv_cache,
            max_draft_tokens=self.config.max_draft_tokens,
            temperature=request.temperature
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
        
        # Create new KV cache for verification
        verify_kv_cache = KVCache(self.config, max_batch_size=1)
        
        # Run verify engine (full model inference)
        verify_output = self.verify_engine.forward(
            input_ids=all_ids,
            kv_cache=verify_kv_cache
        )
        
        # Run acceptance strategy
        verify_logits = verify_output['logits']  # [seq_len, vocab_size]
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