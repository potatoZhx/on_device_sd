"""
Heterogeneous Inference Engine for DeepSeek-V2-Lite
A flexible framework for CPU+GPU mixed inference with dynamic expert offloading
"""

from .core.engine import HeterogeneousInferenceEngine
from .core.scheduler import HeterogeneousScheduler
from .core.device_manager import DeviceManager
from .core.memory_manager import MemoryManager

__version__ = "0.1.0"
__author__ = "HeterSD Team"

__all__ = [
    "HeterogeneousInferenceEngine",
    "HeterogeneousScheduler", 
    "DeviceManager",
    "MemoryManager"
] 