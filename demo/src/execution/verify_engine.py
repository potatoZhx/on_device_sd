from typing import Dict, Optional
import torch
from ..core.model import MoEConfig
from ..memory.parameter_loader import ParameterLoader
from ..memory.expert_cache import ExpertCache
from ..memory.kv_cache import KVCache
from ..scheduling.prefetcher import ExpertPrefetcher
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCollector

logger = get_logger(__name__)


class VerifyEngine:
    """
    Verify phase execution engine.
    Performs full model inference similar to prefill, but on draft tokens.
    """
    
    def __init__(
        self,
        config: MoEConfig,
        parameter_loader: ParameterLoader,
        expert_cache: ExpertCache,
        prefetcher: ExpertPrefetcher,
        metrics: MetricsCollector
    ):
        self.config = config
        self.parameter_loader = parameter_loader
        self.expert_cache = expert_cache
        self.prefetcher = prefetcher
        self.metrics = metrics
        
        # Reuse prefill engine logic
        from .prefill_engine import PrefillEngine
        self.prefill_engine = PrefillEngine(
            config=config,
            parameter_loader=parameter_loader,
            expert_cache=expert_cache,
            prefetcher=prefetcher,
            metrics=metrics
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache
    ) -> Dict:
        """
        Verify draft tokens with full model inference.
        
        Args:
            input_ids: All tokens (prefill + draft) [batch, seq_len]
            kv_cache: Fresh KV cache for verification
        
        Returns:
            Dict with 'logits' for all positions
        """
        self.metrics.start_phase('verify')
        
        logger.info(f"Verifying {input_ids.shape[1]} tokens")
        
        # Use prefill engine for full inference
        output = self.prefill_engine.forward(
            input_ids=input_ids,
            kv_cache=kv_cache
        )
        
        self.metrics.end_phase('verify')
        
        return output