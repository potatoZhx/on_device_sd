from typing import Optional, List, Union, Callable
import torch
import time
import threading
from queue import Queue

from ..core.model import MoEConfig
from ..core.types import (
    InferenceRequest, InferenceResponse, InferenceMode, 
    GenerationConfig, RequestStatus
)
from ..memory.parameter_loader import ParameterLoader
from ..memory.expert_cache import ExpertCache
from ..scheduling.prefetcher import ExpertPrefetcher, SimplePrefetchStrategy, HistoryBasedPrefetchStrategy
from ..scheduling.draft_schduler import SimpleDraftScheduler, AdaptiveDraftScheduler
from ..scheduling.cache_strategy import LRUCacheStrategy, LFUCacheStrategy, AdaptiveCacheStrategy
from ..execution.orchestrator import EnhancedInferenceOrchestrator
from ..execution.acceptance_strategy import StandardAcceptanceStrategy, AdaptiveAcceptanceStrategy
from ..utils.config import ConfigManager, InferenceConfig
from ..utils.logger import get_logger, configure_logging
from ..utils.metrics import MetricsCollector

logger = get_logger(__name__)


class MoEInferenceEngine:
    """
    Enhanced high-level API for MoE model inference.
    Supports both standard and speculative decoding with batch processing.
    """
    
    def __init__(
        self,
        model_path: str,
        config_dir: str = "configs",
        inference_config: Optional[InferenceConfig] = None,
        max_batch_size: int = 32,
        default_mode: InferenceMode = InferenceMode.SPECULATIVE,
        enable_batch_processing: bool = True
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to model parameters
            config_dir: Directory containing configuration files
            inference_config: Runtime inference configuration
            max_batch_size: Maximum batch size for batch processing
            default_mode: Default inference mode (standard or speculative)
            enable_batch_processing: Enable automatic batch processing
        """
        logger.info("Initializing Enhanced MoE Inference Engine...")
        
        # Load configurations
        config_manager = ConfigManager(config_dir)
        self.model_config = config_manager.load_model_config()
        self.inference_config = inference_config or config_manager.load_inference_config()
        self.expert_placement = config_manager.load_expert_placement()
        
        self.default_mode = default_mode
        self.enable_batch_processing = enable_batch_processing
        
        # Configure logging
        configure_logging(self.inference_config.log_level)
        
        # Initialize components
        self._init_components(model_path, max_batch_size)
        
        # Background batch processor
        self.batch_processor_thread = None
        self.processing_active = False
        
        if enable_batch_processing:
            self._start_batch_processor()
        
        logger.info("Enhanced MoE Inference Engine initialized successfully")
    
    def _init_components(self, model_path: str, max_batch_size: int) -> None:
        """Initialize all inference components"""
        
        # Parameter Loader
        logger.info("Loading model parameters...")
        self.parameter_loader = ParameterLoader(
            model_path=model_path,
            config=self.model_config,
            placement_config=self.expert_placement
        )
        self.parameter_loader.load_parameters()
        
        # Expert Cache
        cache_strategy = self._create_cache_strategy()
        self.expert_cache = ExpertCache(
            max_cache_size_gb=self.inference_config.expert_cache_size_gb,
            expert_size_mb=self.inference_config.expert_size_mb,
            replacement_strategy=cache_strategy,
            pin_shared_experts=self.inference_config.pin_shared_experts
        )
        
        # Prefetcher
        prefetch_strategy = self._create_prefetch_strategy()
        self.prefetcher = ExpertPrefetcher(
            strategy=prefetch_strategy,
            max_concurrent_transfers=self.inference_config.max_concurrent_transfers
        )
        
        # Draft Scheduler
        draft_scheduler = self._create_draft_scheduler()
        
        # Acceptance Strategy
        acceptance_strategy = self._create_acceptance_strategy()
        
        # Metrics Collector
        self.metrics = MetricsCollector()
        
        # Enhanced Orchestrator
        self.orchestrator = EnhancedInferenceOrchestrator(
            config=self.model_config,
            parameter_loader=self.parameter_loader,
            expert_cache=self.expert_cache,
            prefetcher=self.prefetcher,
            draft_scheduler=draft_scheduler,
            acceptance_strategy=acceptance_strategy,
            metrics_collector=self.metrics,
            max_batch_size=max_batch_size,
            default_mode=self.default_mode
        )
    
    def _create_cache_strategy(self):
        """Create cache replacement strategy"""
        strategy_name = self.inference_config.cache_strategy
        
        if strategy_name == "lru":
            return LRUCacheStrategy()
        elif strategy_name == "lfu":
            return LFUCacheStrategy()
        elif strategy_name == "adaptive":
            return AdaptiveCacheStrategy()
        else:
            logger.warning(f"Unknown cache strategy '{strategy_name}', using LRU")
            return LRUCacheStrategy()
    
    def _create_prefetch_strategy(self):
        """Create prefetch strategy"""
        strategy_name = self.inference_config.prefetch_strategy
        
        if strategy_name == "simple":
            return SimplePrefetchStrategy(
                num_experts_to_prefetch=self.inference_config.num_experts_to_prefetch
            )
        elif strategy_name == "history_based":
            return HistoryBasedPrefetchStrategy(
                num_experts_to_prefetch=self.inference_config.num_experts_to_prefetch
            )
        else:
            return SimplePrefetchStrategy()
    
    def _create_draft_scheduler(self):
        """Create draft scheduler"""
        scheduler_name = self.inference_config.draft_scheduler
        
        if scheduler_name == "simple":
            return SimpleDraftScheduler()
        elif scheduler_name == "adaptive":
            return AdaptiveDraftScheduler()
        else:
            return SimpleDraftScheduler()
    
    def _create_acceptance_strategy(self):
        """Create acceptance strategy"""
        strategy_name = self.inference_config.acceptance_strategy
        
        if strategy_name == "standard":
            return StandardAcceptanceStrategy(
                acceptance_threshold=self.inference_config.acceptance_threshold
            )
        elif strategy_name == "adaptive":
            return AdaptiveAcceptanceStrategy()
        else:
            return StandardAcceptanceStrategy()
    
    def generate(
        self,
        prompt: Union[str, List[int], List[str], List[List[int]]],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        mode: Optional[InferenceMode] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[List[int], List[List[int]], InferenceResponse]:
        """
        Generate text from prompts (synchronous).
        
        Args:
            prompt: Input prompt or list of prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            mode: Inference mode (standard or speculative)
            stream: Enable streaming (not yet implemented)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated token IDs or InferenceResponse
        """
        if isinstance(prompt, list):
            if len(prompt) == 0:
                return []
            if all(isinstance(item, int) for item in prompt):
                prompts = [prompt]
                return_single = True
            else:
                prompts = prompt
                return_single = False
        else:
            prompts = [prompt]
            return_single = True
        
        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_speculative=(mode != InferenceMode.STANDARD) if mode else True,
            **kwargs
        )
        
        requests = []
        timestamp = int(time.time() * 1000)
        for idx, item in enumerate(prompts):
            if isinstance(item, str):
                input_ids = self._tokenize(item)
            else:
                input_ids = torch.tensor(item, dtype=torch.long)
            requests.append(InferenceRequest(
                request_id=f"req_{timestamp}_{id(item)}_{idx}",
                input_ids=input_ids,
                generation_config=gen_config
            ))
        
        inference_mode = mode or self.default_mode
        output_ids = self.orchestrator.generate(requests, mode=inference_mode)
        output_lists = [ids.tolist() for ids in output_ids]
        
        if return_single:
            return output_lists[0]
        return output_lists
    
    def submit(
        self,
        prompt: Union[str, List[int]],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        priority: int = 0,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> str:
        """
        Submit an inference request for batch processing (asynchronous).
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            priority: Request priority (higher = more important)
            callback: Optional callback when complete
            **kwargs: Additional generation parameters
        
        Returns:
            Request ID for tracking
        """
        # Tokenize
        if isinstance(prompt, str):
            input_ids = self._tokenize(prompt)
        else:
            input_ids = torch.tensor(prompt, dtype=torch.long)
        
        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        # Create request
        request = InferenceRequest(
            request_id=f"req_{int(time.time() * 1000)}_{id(prompt)}",
            input_ids=input_ids,
            generation_config=gen_config,
            priority=priority,
            stream_callback=callback
        )
        
        # Submit to batch manager
        request_id = self.orchestrator.submit_request(request)
        
        logger.info(f"Submitted request {request_id} for batch processing")
        
        return request_id
    
    def get_result(
        self,
        request_id: str,
        timeout: Optional[float] = None
    ) -> Optional[InferenceResponse]:
        """
        Get result for a submitted request (blocking).
        
        Args:
            request_id: Request ID
            timeout: Timeout in seconds
        
        Returns:
            InferenceResponse when complete, or None if timeout
        """
        start_time = time.time()
        
        while True:
            # Check status
            status = self.orchestrator.get_request_status(request_id)
            
            if status == RequestStatus.COMPLETED:
                return self.orchestrator.get_response(request_id)
            elif status == RequestStatus.FAILED:
                response = self.orchestrator.get_response(request_id)
                logger.error(f"Request {request_id} failed: {response.error if response else 'Unknown'}")
                return response
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout waiting for request {request_id}")
                return None
            
            # Sleep briefly
            time.sleep(0.01)
    
    def _start_batch_processor(self) -> None:
        """Start background batch processor thread"""
        self.processing_active = True
        self.batch_processor_thread = threading.Thread(
            target=self._batch_processor_loop,
            daemon=True
        )
        self.batch_processor_thread.start()
        logger.info("Background batch processor started")
    
    def _batch_processor_loop(self) -> None:
        """Background loop for processing batches"""
        while self.processing_active:
            try:
                # Process next batch
                responses = self.orchestrator.process_batch(timeout_ms=50.0)
                
                # Trigger callbacks if any
                for response in responses:
                    request = next(
                        (req for req in self.orchestrator.batch_manager.request_status.keys()
                         if req == response.request_id),
                        None
                    )
                    if request and hasattr(request, 'stream_callback') and request.stream_callback:
                        request.stream_callback(response.generated_text)
                
            except Exception as e:
                logger.error(f"Error in batch processor: {e}", exc_info=True)
                time.sleep(0.1)
    
    def stop_batch_processor(self) -> None:
        """Stop background batch processor"""
        self.processing_active = False
        if self.batch_processor_thread:
            self.batch_processor_thread.join(timeout=5.0)
            logger.info("Background batch processor stopped")
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize input text (placeholder)"""
        # TODO: Implement actual tokenization
        logger.warning("Using placeholder tokenization")
        return torch.randint(0, self.model_config.vocab_size, (len(text.split()),))
    
    def get_statistics(self) -> dict:
        """Get inference statistics"""
        return self.orchestrator.get_statistics()
    
    def print_statistics(self) -> None:
        """Print formatted statistics"""
        self.metrics.print_summary()
        
        stats = self.get_statistics()
        print("\\n" + "="*60)
        print("BATCH PROCESSING STATISTICS")
        print("="*60)
        print(f"Queue Length: {stats['batch_queue_length']}")
        print(f"Active Batches: {stats['active_batches']}")
        print("="*60 + "\\n")
    
    def __del__(self):
        """Cleanup when engine is destroyed"""
        if self.enable_batch_processing:
            self.stop_batch_processor()
