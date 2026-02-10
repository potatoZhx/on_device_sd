from typing import Dict, Optional
import torch

from ..core.model_runner import ModelRunner
from ..memory.expert_cache import ExpertCache
from ..memory.parameter_loader import ParameterLoader
from ..scheduling.prefetcher import ExpertPrefetcher
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCollector
from .prefill_engine import PrefillEngine

logger = get_logger(__name__)


class VerifyEngine:
    """
    Verify phase execution engine using ModelRunner.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        parameter_loader: ParameterLoader,
        expert_cache: ExpertCache,
        prefetcher: Optional[ExpertPrefetcher],
        metrics: MetricsCollector,
    ):
        self.model_runner = model_runner
        self.prefill_engine = PrefillEngine(
            model_runner=model_runner,
            parameter_loader=parameter_loader,
            expert_cache=expert_cache,
            prefetcher=prefetcher,
            metrics=metrics,
        )
        self.metrics = metrics

    def forward(self, input_ids: torch.Tensor, kv_cache, seq_ids: Optional[list[int]] = None) -> Dict:
        self.metrics.start_phase('verify')
        logger.info(f"Verifying {input_ids.shape[1]} tokens")
        output = self.prefill_engine.forward(
            input_ids=input_ids,
            kv_cache=kv_cache,
            seq_ids=seq_ids,
            is_prefill=True,
        )
        self.metrics.end_phase('verify')
        return output