from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any
import yaml
from pathlib import Path
from ..core.model import MoEConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InferenceConfig:
    """Runtime inference configuration"""
    # Cache settings
    expert_cache_size_gb: float = 8.0
    expert_size_mb: float = 100.0
    pin_shared_experts: bool = True
    
    # Prefetching
    prefetch_strategy: str = "simple"  # "simple" or "history_based"
    num_experts_to_prefetch: int = 4
    max_concurrent_transfers: int = 4
    
    # Draft-verify
    draft_scheduler: str = "simple"  # "simple" or "adaptive"
    acceptance_strategy: str = "standard"  # "standard" or "adaptive"
    acceptance_threshold: float = 0.7
    
    # Cache replacement
    cache_strategy: str = "lru"  # "lru", "lfu", "adaptive", "predictive"
    
    # Performance
    enable_profiling: bool = False
    log_level: str = "INFO"


class ConfigManager:
    """Manages loading and validation of all configurations"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
    
    def load_model_config(self, config_path: Optional[str] = None) -> MoEConfig:
        """Load model architecture configuration"""
        if config_path is None:
            config_path = self.config_dir / "model_config.yaml"
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return MoEConfig(**config_dict)
    
    def load_inference_config(self, config_path: Optional[str] = None) -> InferenceConfig:
        """Load inference runtime configuration"""
        if config_path is None:
            config_path = self.config_dir / "inference_config.yaml"
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return InferenceConfig(**config_dict)
    
    def load_expert_placement(self, config_path: Optional[str] = None) -> Dict:
        """Load expert placement configuration"""
        if config_path is None:
            config_path = self.config_dir / "expert_placement.yaml"
        
        if not Path(config_path).exists():
            logger.warning(f"Expert placement config not found at {config_path}, using defaults")
            return {}
        
        with open(config_path, 'r') as f:
            placement_dict = yaml.safe_load(f)
        
        return placement_dict
    
    def save_config(self, config: Any, output_path: str) -> None:
        """Save configuration to YAML file"""
        if hasattr(config, '__dataclass_fields__'):
            config_dict = asdict(config)
        else:
            config_dict = config
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"Saved configuration to {output_path}")