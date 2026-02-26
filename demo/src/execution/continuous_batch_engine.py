"""
Backward-compatible facade for continuous batching components.

Implementations are split into:
- src/core/sequence.py
- src/execution/prefetch_selector.py
- src/execution/cb_scheduler.py
- src/execution/cb_executor.py
- src/execution/cb_engine.py
"""

from ..core.sequence import DecodeMode, Sequence, SequenceStatus
from .cb_engine import ContinuousBatchEngine
from .cb_executor import CBExecutor
from .cb_scheduler import CBScheduler, ScheduleResult
from .prefetch_selector import select_experts_to_prefetch

__all__ = [
    "SequenceStatus",
    "Sequence",
    "DecodeMode",
    "ScheduleResult",
    "CBScheduler",
    "CBExecutor",
    "ContinuousBatchEngine",
    "select_experts_to_prefetch",
]
