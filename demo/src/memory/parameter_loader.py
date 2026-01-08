"""
学习nano-vllm
"""

from typing import Dict, List, Optional
import torch
from ..core.types import ExpertID, DeviceType, ExpertLocation
from ..core.model import MoEConfig, MoEModelStructure
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ParameterLoader:
    """
    Responsible for loading model parameters from disk and 
    placing them in CPU/GPU memory according to configuration.
    """
    
    def __init__(
        self, 
        model_path: str,
        config: MoEConfig,
        placement_config: Optional[Dict] = None
    ):
        self.model_path = model_path
        self.config = config
        self.model_structure = MoEModelStructure(config)
        self.placement_config = placement_config or {}
        
        # Storage for loaded parameters
        self.static_params_gpu: Dict[str, torch.Tensor] = {}
        self.shared_experts_gpu: Dict[ExpertID, Dict[str, torch.Tensor]] = {}
        self.expert_params_cpu: Dict[ExpertID, Dict[str, torch.Tensor]] = {}
        self.expert_params_gpu: Dict[ExpertID, Dict[str, torch.Tensor]] = {}
        
        # Location tracking
        self.expert_locations: Dict[ExpertID, ExpertLocation] = {}
        
    def load_parameters(self) -> None:
        """
        Main entry point: load all parameters according to configuration.
        """
        logger.info("Starting parameter loading...")
        
        # Step 1: Load static parameters to GPU
        self._load_static_parameters()
        
        # Step 2: Load shared experts to GPU
        self._load_shared_experts()
        
        # Step 3: Load expert parameters according to placement strategy
        self._load_expert_parameters()
        
        logger.info(f"Parameter loading complete. "
                   f"GPU experts: {len(self.expert_params_gpu)}, "
                   f"CPU experts: {len(self.expert_params_cpu)}")
    
    # TODO
    # load 逻辑不对，文件是不对的，参考kt/nano-vllm修改
    def _load_static_parameters(self) -> None:
        """Load non-expert parameters (embeddings, attention, layernorm, etc.)"""
        logger.info("Loading static parameters to GPU...")
        
        # Load embedding layers
        self.static_params_gpu['embed_tokens'] = self._load_tensor(
            f"{self.model_path}/embed_tokens.pt", device='cuda'
        )
        
        # Load per-layer non-expert parameters
        for layer_idx in range(self.config.num_hidden_layers):
            layer_prefix = f"layer_{layer_idx}"
            
            # Attention weights
            self.static_params_gpu[f"{layer_prefix}.self_attn.q_proj"] = \
                self._load_tensor(f"{self.model_path}/{layer_prefix}/q_proj.pt", device='cuda')
            self.static_params_gpu[f"{layer_prefix}.self_attn.k_proj"] = \
                self._load_tensor(f"{self.model_path}/{layer_prefix}/k_proj.pt", device='cuda')
            self.static_params_gpu[f"{layer_prefix}.self_attn.v_proj"] = \
                self._load_tensor(f"{self.model_path}/{layer_prefix}/v_proj.pt", device='cuda')
            self.static_params_gpu[f"{layer_prefix}.self_attn.o_proj"] = \
                self._load_tensor(f"{self.model_path}/{layer_prefix}/o_proj.pt", device='cuda')
            
            # Layer norms
            self.static_params_gpu[f"{layer_prefix}.input_layernorm"] = \
                self._load_tensor(f"{self.model_path}/{layer_prefix}/input_layernorm.pt", device='cuda')
            self.static_params_gpu[f"{layer_prefix}.post_attention_layernorm"] = \
                self._load_tensor(f"{self.model_path}/{layer_prefix}/post_attention_layernorm.pt", device='cuda')
            
            # Router
            self.static_params_gpu[f"{layer_prefix}.router"] = \
                self._load_tensor(f"{self.model_path}/{layer_prefix}/router.pt", device='cuda')
        
        # Final layer norm and LM head
        self.static_params_gpu['final_layernorm'] = \
            self._load_tensor(f"{self.model_path}/final_layernorm.pt", device='cuda')
        self.static_params_gpu['lm_head'] = \
            self._load_tensor(f"{self.model_path}/lm_head.pt", device='cuda')
    
    # TODO
    # shared experts的load可以和static合并
    def _load_shared_experts(self) -> None:
        """Load shared experts to GPU"""
        if self.config.num_shared_experts == 0:
            return
            
        logger.info(f"Loading {self.config.num_shared_experts} shared experts to GPU...")
        
        for layer_idx in range(self.config.num_hidden_layers):
            for expert_idx in range(self.config.num_shared_experts):
                expert_id = ExpertID(layer_idx, expert_idx)
                self.shared_experts_gpu[expert_id] = self._load_expert_weights(
                    layer_idx, expert_idx, device='cuda'
                )
                self.expert_locations[expert_id] = ExpertLocation(
                    expert_id=expert_id,
                    device=DeviceType.GPU,
                    is_cached=True
                )
    
    # TODO
    # expert的load逻辑应该要全部加载到CPU，再根据放置配置复制到GPU
    # CPU-GPU间不是在“交换”expert，而是CPU中始终保存全部的expert副本, 除了shared_experts
    # GPU按需从CPU加载到GPU, GPU evict时是直接逐出（覆盖内存），而不是浪费pcie传输回CPU
    def _load_expert_parameters(self) -> None:
        """Load expert parameters according to placement configuration"""
        logger.info("Loading expert parameters...")
        
        for layer_idx in range(self.config.num_hidden_layers):
            layer_placement = self.placement_config.get(
                f"layer_{layer_idx}", 
                self._get_default_placement(layer_idx)
            )
            
            for expert_idx in range(self.config.num_shared_experts, self.config.num_experts):
                expert_id = ExpertID(layer_idx, expert_idx)
                
                # Skip if already loaded as shared expert
                if expert_id in self.shared_experts_gpu:
                    continue
                
                # Determine placement
                target_device = layer_placement.get(
                    f"expert_{expert_idx}", 
                    'cpu'  # Default to CPU
                )
                
                if target_device == 'gpu':
                    self.expert_params_gpu[expert_id] = self._load_expert_weights(
                        layer_idx, expert_idx, device='cuda'
                    )
                    self.expert_locations[expert_id] = ExpertLocation(
                        expert_id=expert_id,
                        device=DeviceType.GPU,
                        is_cached=True
                    )
                else:
                    self.expert_params_cpu[expert_id] = self._load_expert_weights(
                        layer_idx, expert_idx, device='cpu'
                    )
                    self.expert_locations[expert_id] = ExpertLocation(
                        expert_id=expert_id,
                        device=DeviceType.CPU,
                        is_cached=False
                    )
    
    # TODO
    # 同_load_static
    def _load_expert_weights(
        self, 
        layer_idx: int, 
        expert_idx: int, 
        device: str
    ) -> Dict[str, torch.Tensor]:
        """Load weights for a single expert"""
        expert_path = f"{self.model_path}/layer_{layer_idx}/expert_{expert_idx}"
        
        return {
            'gate_proj': self._load_tensor(f"{expert_path}/gate_proj.pt", device=device),
            'up_proj': self._load_tensor(f"{expert_path}/up_proj.pt", device=device),
            'down_proj': self._load_tensor(f"{expert_path}/down_proj.pt", device=device),
        }
    
    # TODO
    # 同_load_static
    def _load_tensor(self, path: str, device: str) -> torch.Tensor:
        """Load a single tensor from disk"""
        tensor = torch.load(path, map_location=device)
        return tensor
    
    def _get_default_placement(self, layer_idx: int) -> Dict[str, str]:
        """
        Default placement strategy: place all experts on CPU.
        Can be overridden by configuration.
        """
        return {f"expert_{i}": 'cpu' for i in range(self.config.num_experts)}
    
    def get_expert_location(self, expert_id: ExpertID) -> ExpertLocation:
        """Get current location of an expert"""
        return self.expert_locations.get(expert_id)
    
    def get_expert_params(
        self, 
        expert_id: ExpertID, 
        device: Optional[DeviceType] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieve expert parameters.
        If device specified, only return if expert is on that device.
        """
        if device == DeviceType.GPU or device is None:
            if expert_id in self.shared_experts_gpu:
                return self.shared_experts_gpu[expert_id]
            if expert_id in self.expert_params_gpu:
                return self.expert_params_gpu[expert_id]
        
        if device == DeviceType.CPU or device is None:
            if expert_id in self.expert_params_cpu:
                return self.expert_params_cpu[expert_id]
        
        return None