"""
Configuration management for Heterogeneous Inference Engine
"""

import yaml
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch


@dataclass
class DeviceConfig:
    """Device configuration"""
    gpu_memory_limit: int = 24  # GB
    cpu_memory_limit: int = 64  # GB
    enable_mixed_precision: bool = True
    gpu_threshold: float = 0.5
    gpu_devices: list = None
    
    def __post_init__(self):
        if self.gpu_devices is None:
            self.gpu_devices = list(range(torch.cuda.device_count()))


@dataclass
class SchedulerConfig:
    """Scheduler configuration"""
    type: str = "popularity_based"
    expert_cache_size: int = 100
    popularity_window: int = 1000
    gpu_expert_ratio: float = 0.3
    top_k_experts: int = 2


@dataclass
class MemoryConfig:
    """Memory configuration"""
    kv_cache_strategy: str = "lru"
    expert_cache_strategy: str = "lru"
    max_kv_cache_size: int = 8  # GB
    max_expert_cache_size: int = 4  # GB
    prefer_gpu_for_kv_cache: bool = True


@dataclass
class RuntimeOffloadConfig:
    """Runtime offload configuration"""
    enable_dynamic_offloading: bool = True
    offload_threshold: float = 0.1
    migration_batch_size: int = 5
    async_migration: bool = True
    migration_cooldown: float = 0.1
    runtime_window: int = 100
    hotness_threshold: float = 0.7
    coldness_threshold: float = 0.3


@dataclass
class ProfilingConfig:
    """Profiling configuration"""
    enable_profiling: bool = True
    profile_memory: bool = True
    profile_computation: bool = True


@dataclass
class EngineConfig:
    """Main engine configuration"""
    model_path: str
    device_config: DeviceConfig
    scheduler_config: SchedulerConfig
    memory_config: MemoryConfig
    runtime_offload_config: RuntimeOffloadConfig
    profiling_config: ProfilingConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'EngineConfig':
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls._from_dict(config_dict)
    
    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> 'EngineConfig':
        """Create config from dictionary"""
        engine_config = config_dict.get('engine', {})
        
        return cls(
            model_path=engine_config.get('model_path', ''),
            device_config=DeviceConfig(**engine_config.get('device_config', {})),
            scheduler_config=SchedulerConfig(**engine_config.get('scheduler_config', {})),
            memory_config=MemoryConfig(**engine_config.get('memory_config', {})),
            runtime_offload_config=RuntimeOffloadConfig(**engine_config.get('runtime_offload_config', {})),
            profiling_config=ProfilingConfig(**engine_config.get('profiling_config', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'engine': {
                'model_path': self.model_path,
                'device_config': {
                    'gpu_memory_limit': self.device_config.gpu_memory_limit,
                    'cpu_memory_limit': self.device_config.cpu_memory_limit,
                    'enable_mixed_precision': self.device_config.enable_mixed_precision,
                    'gpu_threshold': self.device_config.gpu_threshold,
                    'gpu_devices': self.device_config.gpu_devices
                },
                'scheduler_config': {
                    'type': self.scheduler_config.type,
                    'expert_cache_size': self.scheduler_config.expert_cache_size,
                    'popularity_window': self.scheduler_config.popularity_window,
                    'gpu_expert_ratio': self.scheduler_config.gpu_expert_ratio,
                    'top_k_experts': self.scheduler_config.top_k_experts
                },
                'memory_config': {
                    'kv_cache_strategy': self.memory_config.kv_cache_strategy,
                    'expert_cache_strategy': self.memory_config.expert_cache_strategy,
                    'max_kv_cache_size': self.memory_config.max_kv_cache_size,
                    'max_expert_cache_size': self.memory_config.max_expert_cache_size,
                    'prefer_gpu_for_kv_cache': self.memory_config.prefer_gpu_for_kv_cache
                },
                'runtime_offload_config': {
                    'enable_dynamic_offloading': self.runtime_offload_config.enable_dynamic_offloading,
                    'offload_threshold': self.runtime_offload_config.offload_threshold,
                    'migration_batch_size': self.runtime_offload_config.migration_batch_size,
                    'async_migration': self.runtime_offload_config.async_migration,
                    'migration_cooldown': self.runtime_offload_config.migration_cooldown,
                    'runtime_window': self.runtime_offload_config.runtime_window,
                    'hotness_threshold': self.runtime_offload_config.hotness_threshold,
                    'coldness_threshold': self.runtime_offload_config.coldness_threshold
                },
                'profiling_config': {
                    'enable_profiling': self.profiling_config.enable_profiling,
                    'profile_memory': self.profiling_config.profile_memory,
                    'profile_computation': self.profiling_config.profile_computation
                }
            }
        }
    
    def save_to_yaml(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = self.to_dict()
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def create_default_config(model_path: str, output_path: str = "config.yaml"):
    """Create default configuration file"""
    config = EngineConfig(
        model_path=model_path,
        device_config=DeviceConfig(),
        scheduler_config=SchedulerConfig(),
        memory_config=MemoryConfig(),
        runtime_offload_config=RuntimeOffloadConfig(),
        profiling_config=ProfilingConfig()
    )
    config.save_to_yaml(output_path)
    return config 