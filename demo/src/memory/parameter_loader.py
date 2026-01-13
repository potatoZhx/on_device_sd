"""
参数加载器 - 参考 nano-vllm 的实现
支持从 safetensors 格式加载 MoE 模型参数
"""

import os
import json
import re
from glob import glob
from typing import Dict, List, Optional, Set, Tuple, Any
import torch
from safetensors import safe_open

from ..core.types import ExpertID, DeviceType, ExpertLocation
from ..utils.logger import get_logger

logger = get_logger(__name__)


def load_config_json(model_path: str) -> Dict[str, Any]:
    """直接从 config.json 加载配置，避免依赖 transformers 版本"""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


class MoEModelConfig:
    """
    从 HuggingFace 配置加载的 MoE 模型配置
    直接从 config.json 加载以避免 transformers 版本依赖
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.config_dict = load_config_json(model_path)
        
        # 基本配置
        self.hidden_size = self.config_dict['hidden_size']
        self.num_hidden_layers = self.config_dict['num_hidden_layers']
        self.num_attention_heads = self.config_dict['num_attention_heads']
        self.num_key_value_heads = self.config_dict.get('num_key_value_heads', self.num_attention_heads)
        self.head_dim = self.config_dict.get('head_dim', self.hidden_size // self.num_attention_heads)
        self.vocab_size = self.config_dict['vocab_size']
        self.rms_norm_eps = self.config_dict.get('rms_norm_eps', 1e-6)
        
        # MoE 特定配置
        self.num_experts = self.config_dict.get('num_experts', 1)
        self.num_experts_per_token = self.config_dict.get('num_experts_per_tok', 1)
        self.num_shared_experts = self.config_dict.get('num_shared_experts', 0)
        self.moe_intermediate_size = self.config_dict.get(
            'moe_intermediate_size', 
            self.config_dict.get('intermediate_size', self.hidden_size * 4)
        )
        self.intermediate_size = self.config_dict.get('intermediate_size', self.hidden_size * 4)
        
        # 推理配置
        self.max_position_embeddings = self.config_dict.get('max_position_embeddings', 32768)
        self.rope_theta = self.config_dict.get('rope_theta', 10000.0)
        self.torch_dtype = self.config_dict.get('torch_dtype', 'float16')
        
        # 模型类型
        self.model_type = self.config_dict.get('model_type', 'unknown')
        
        # 是否为 MoE 模型
        self.is_moe = self.num_experts > 1
        
        logger.info(f"Loaded config for {self.model_type}: "
                   f"{self.num_hidden_layers} layers, {self.num_experts} experts, "
                   f"top-{self.num_experts_per_token}, shared_experts={self.num_shared_experts}")
    
    def get_expert_weight_size_bytes(self) -> int:
        """计算单个 expert 的参数大小（字节）"""
        # gate_proj: [moe_intermediate_size, hidden_size]
        # up_proj: [moe_intermediate_size, hidden_size]
        # down_proj: [hidden_size, moe_intermediate_size]
        dtype_size = 2 if 'float16' in str(self.torch_dtype) or 'bfloat16' in str(self.torch_dtype) else 4
        total_params = 3 * self.moe_intermediate_size * self.hidden_size
        return total_params * dtype_size


class SafetensorsWeightLoader:
    """
    从 safetensors 文件加载权重的工具类
    """
    def __init__(self, model_path: str, device: str = "cpu", use_pin_memory: bool = True):
        self.model_path = model_path
        self.device = device
        self.use_pin_memory = use_pin_memory
        self.weight_map: Dict[str, str] = {}  # weight_name -> safetensors_file
        self.file_handles: Dict[str, safe_open] = {}
        
        self._load_weight_map()
    
    def _load_weight_map(self):
        """加载 weight_map 索引"""
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index = json.load(f)
                self.weight_map = index.get("weight_map", {})
            logger.info(f"Loaded weight map with {len(self.weight_map)} entries")
        else:
            # 单个 safetensors 文件
            files = glob(os.path.join(self.model_path, "*.safetensors"))
            if files:
                single_file = os.path.basename(files[0])
                with safe_open(files[0], "pt", "cpu") as f:
                    for key in f.keys():
                        self.weight_map[key] = single_file
                logger.info(f"Single safetensors file with {len(self.weight_map)} weights")
    
    def _get_file_handle(self, filename: str) -> safe_open:
        """获取或创建 safetensors 文件句柄"""
        if filename not in self.file_handles:
            filepath = os.path.join(self.model_path, filename)
            self.file_handles[filename] = safe_open(filepath, "pt", self.device)
        return self.file_handles[filename]
    
    def get_tensor(self, weight_name: str, device: Optional[str] = None, 
                   pin_memory: Optional[bool] = None) -> torch.Tensor:
        """
        加载单个张量
        
        Args:
            weight_name: 权重名称
            device: 目标设备 ('cpu', 'cuda')
            pin_memory: 是否使用 pin_memory（仅对 CPU 张量有效，加速 CPU->GPU 传输）
        """
        if weight_name not in self.weight_map:
            raise KeyError(f"Weight '{weight_name}' not found in model")
        
        filename = self.weight_map[weight_name]
        f = self._get_file_handle(filename)
        tensor = f.get_tensor(weight_name)
        
        # 决定是否使用 pin_memory
        use_pin = pin_memory if pin_memory is not None else self.use_pin_memory
        
        if device is not None and device != self.device:
            if device == 'cuda':
                tensor = tensor.to(device)
            elif device == 'cpu' and use_pin and tensor.device.type == 'cpu':
                tensor = tensor.pin_memory()
        elif device == 'cpu' and use_pin and tensor.device.type == 'cpu':
            tensor = tensor.pin_memory()
        
        return tensor
    
    def has_weight(self, weight_name: str) -> bool:
        """检查权重是否存在"""
        return weight_name in self.weight_map
    
    def get_weight_names(self, pattern: Optional[str] = None) -> List[str]:
        """获取所有权重名称，可选按模式过滤"""
        if pattern is None:
            return list(self.weight_map.keys())
        regex = re.compile(pattern)
        return [name for name in self.weight_map.keys() if regex.search(name)]
    
    def close(self):
        """关闭所有文件句柄"""
        self.file_handles.clear()


class ParameterLoader:
    """
    负责从磁盘加载模型参数并根据配置放置在 CPU/GPU 内存中
    
    参考 nano-vllm 的实现，使用 safetensors 格式加载
    
    权重组织：
    - static_params_gpu: 静态参数（embedding, attention, layernorm, router, lm_head）
    - shared_experts_gpu: Shared experts 参数（始终在 GPU）
    - expert_params_cpu: Routed experts 参数（CPU 副本）
    - expert_params_gpu: Routed experts 参数（GPU 缓存）
    """
    
    def __init__(
        self, 
        model_path: str,
        config: Optional[MoEModelConfig] = None,
        placement_config: Optional[Dict] = None,
        device: str = "cpu",
        use_pin_memory: bool = True
    ):
        self.model_path = model_path
        self.config = config or MoEModelConfig(model_path)
        self.placement_config = placement_config or {}
        self.default_device = device
        self.use_pin_memory = use_pin_memory
        
        # 权重加载器
        self.weight_loader = SafetensorsWeightLoader(
            model_path, device="cpu", use_pin_memory=use_pin_memory
        )
        
        # 存储加载的参数
        self.static_params_gpu: Dict[str, torch.Tensor] = {}
        self.shared_experts_gpu: Dict[ExpertID, Dict[str, torch.Tensor]] = {}  # Shared experts 始终在 GPU
        self.expert_params_cpu: Dict[ExpertID, Dict[str, torch.Tensor]] = {}   # Routed experts CPU 副本
        self.expert_params_gpu: Dict[ExpertID, Dict[str, torch.Tensor]] = {}   # Routed experts GPU 缓存
        
        # 位置跟踪
        self.expert_locations: Dict[ExpertID, ExpertLocation] = {}
        
        # Shared expert IDs（用于快速查询）
        self.shared_expert_ids: Set[ExpertID] = set()
        
        # 加载状态
        self._loaded = False
        
    def load_parameters(self) -> None:
        """
        主入口：根据配置加载所有参数
        """
        if self._loaded:
            logger.warning("Parameters already loaded, skipping...")
            return
            
        logger.info(f"Starting parameter loading from {self.model_path}...")
        
        # Step 1: 加载静态参数（非 expert 参数）和 shared experts 到 GPU
        self._load_static_parameters()
        
        # Step 2: 加载所有 routed expert 参数到 CPU（跳过 shared experts）
        self._load_all_experts_to_cpu()
        
        # Step 3: 根据放置配置复制部分 routed expert 到 GPU
        self._copy_experts_to_gpu()
        
        self._loaded = True
        logger.info(f"Parameter loading complete. "
                   f"Static params: {len(self.static_params_gpu)}, "
                   f"Shared experts: {len(self.shared_experts_gpu)}, "
                   f"GPU routed experts: {len(self.expert_params_gpu)}, "
                   f"CPU routed experts: {len(self.expert_params_cpu)}")
    
    def _load_static_parameters(self) -> None:
        """
        加载非 expert 参数（embeddings, attention, layernorm 等）和 shared experts 到 GPU
        """
        logger.info("Loading static parameters and shared experts to GPU...")
        
        # Embedding 层
        embed_name = "model.embed_tokens.weight"
        if self.weight_loader.has_weight(embed_name):
            self.static_params_gpu['embed_tokens'] = self.weight_loader.get_tensor(
                embed_name, device='cuda'
            )
            logger.debug(f"Loaded {embed_name}")
        
        # 每层的非 expert 参数
        for layer_idx in range(self.config.num_hidden_layers):
            layer_prefix = f"model.layers.{layer_idx}"
            
            # Attention 权重
            attn_weights = [
                ('q_proj', 'self_attn.q_proj'),
                ('k_proj', 'self_attn.k_proj'),
                ('v_proj', 'self_attn.v_proj'),
                ('o_proj', 'self_attn.o_proj'),
            ]
            
            for short_name, full_name in attn_weights:
                weight_name = f"{layer_prefix}.{full_name}.weight"
                if self.weight_loader.has_weight(weight_name):
                    self.static_params_gpu[f"layer_{layer_idx}.self_attn.{short_name}"] = \
                        self.weight_loader.get_tensor(weight_name, device='cuda')
            
            # QK norm（如果存在）
            for norm_name in ['q_norm', 'k_norm']:
                weight_name = f"{layer_prefix}.self_attn.{norm_name}.weight"
                if self.weight_loader.has_weight(weight_name):
                    self.static_params_gpu[f"layer_{layer_idx}.self_attn.{norm_name}"] = \
                        self.weight_loader.get_tensor(weight_name, device='cuda')
            
            # Layer norms
            for norm_type in ['input_layernorm', 'post_attention_layernorm']:
                weight_name = f"{layer_prefix}.{norm_type}.weight"
                if self.weight_loader.has_weight(weight_name):
                    self.static_params_gpu[f"layer_{layer_idx}.{norm_type}"] = \
                        self.weight_loader.get_tensor(weight_name, device='cuda')
            
            # MoE Router (gate)
            router_name = f"{layer_prefix}.mlp.gate.weight"
            if self.weight_loader.has_weight(router_name):
                self.static_params_gpu[f"layer_{layer_idx}.router"] = \
                    self.weight_loader.get_tensor(router_name, device='cuda')
            
            # ==================== Shared Experts 加载 ====================
            # 方式 1: 独立命名空间的 shared experts (如 Qwen2-MoE, DeepSeek-V2)
            # 格式: model.layers.{layer}.mlp.shared_expert.{proj}.weight (单个)
            #       model.layers.{layer}.mlp.shared_experts.{idx}.{proj}.weight (多个)
            self._load_shared_experts_separate_namespace(layer_idx, layer_prefix)
            
            # 方式 2: 从 experts 列表的前几个作为 shared experts (某些模型变体)
            # 在 config 中通过 shared_expert_indices 或 num_shared_experts 指定
            self._load_shared_experts_from_experts_list(layer_idx, layer_prefix)
            
            # Shared expert gate（如果存在）
            shared_gate_name = f"{layer_prefix}.mlp.shared_expert_gate.weight"
            if self.weight_loader.has_weight(shared_gate_name):
                self.static_params_gpu[f"layer_{layer_idx}.shared_expert_gate"] = \
                    self.weight_loader.get_tensor(shared_gate_name, device='cuda')
        
        # 最终 layer norm
        final_norm_name = "model.norm.weight"
        if self.weight_loader.has_weight(final_norm_name):
            self.static_params_gpu['final_layernorm'] = \
                self.weight_loader.get_tensor(final_norm_name, device='cuda')
        
        # LM head
        lm_head_name = "lm_head.weight"
        if self.weight_loader.has_weight(lm_head_name):
            self.static_params_gpu['lm_head'] = \
                self.weight_loader.get_tensor(lm_head_name, device='cuda')
        
        logger.info(f"Loaded {len(self.static_params_gpu)} static parameters to GPU")
        logger.info(f"Loaded {len(self.shared_experts_gpu)} shared experts to GPU")
    
    def _load_shared_experts_separate_namespace(self, layer_idx: int, layer_prefix: str) -> None:
        """
        加载独立命名空间的 shared experts
        格式: mlp.shared_expert.{proj} 或 mlp.shared_experts.{idx}.{proj}
        """
        proj_names = ['gate_proj', 'up_proj', 'down_proj']
        
        # 单个 shared expert: mlp.shared_expert.{proj}.weight
        single_shared_prefix = f"{layer_prefix}.mlp.shared_expert"
        has_single_shared = self.weight_loader.has_weight(f"{single_shared_prefix}.gate_proj.weight")
        
        if has_single_shared:
            expert_weights = {}
            for proj in proj_names:
                weight_name = f"{single_shared_prefix}.{proj}.weight"
                if self.weight_loader.has_weight(weight_name):
                    expert_weights[proj] = self.weight_loader.get_tensor(weight_name, device='cuda')
            
            if len(expert_weights) == 3:
                # 使用特殊的 expert_idx = -1 表示单个 shared expert
                expert_id = ExpertID(layer_idx, -1)
                self.shared_experts_gpu[expert_id] = expert_weights
                self.shared_expert_ids.add(expert_id)
                self.expert_locations[expert_id] = ExpertLocation(
                    expert_id=expert_id,
                    device=DeviceType.GPU,
                    is_cached=True
                )
                logger.debug(f"Loaded shared expert for layer {layer_idx}")
        
        # 多个 shared experts: mlp.shared_experts.{idx}.{proj}.weight
        multi_shared_prefix = f"{layer_prefix}.mlp.shared_experts"
        for shared_idx in range(self.config.num_shared_experts):
            expert_prefix = f"{multi_shared_prefix}.{shared_idx}"
            has_expert = self.weight_loader.has_weight(f"{expert_prefix}.gate_proj.weight")
            
            if has_expert:
                expert_weights = {}
                for proj in proj_names:
                    weight_name = f"{expert_prefix}.{proj}.weight"
                    if self.weight_loader.has_weight(weight_name):
                        expert_weights[proj] = self.weight_loader.get_tensor(weight_name, device='cuda')
                
                if len(expert_weights) == 3:
                    # 使用负数 expert_idx 表示 shared experts (-1, -2, ...)
                    expert_id = ExpertID(layer_idx, -(shared_idx + 1))
                    self.shared_experts_gpu[expert_id] = expert_weights
                    self.shared_expert_ids.add(expert_id)
                    self.expert_locations[expert_id] = ExpertLocation(
                        expert_id=expert_id,
                        device=DeviceType.GPU,
                        is_cached=True
                    )
                    logger.debug(f"Loaded shared expert {shared_idx} for layer {layer_idx}")
    
    def _load_shared_experts_from_experts_list(self, layer_idx: int, layer_prefix: str) -> None:
        """
        从 experts 列表的前 N 个加载 shared experts（某些模型变体使用此方式）
        这些 experts 使用正常的索引 0, 1, ..., num_shared_experts-1
        """
        # 检查配置中是否指定了 shared_expert_indices
        shared_indices = self.config.config_dict.get('shared_expert_indices', [])
        
        # 如果没有指定且有 num_shared_experts，检查是否存在对应的 expert 权重
        # 注意：这种情况下 shared experts 和 routed experts 共享同一个 experts 列表
        # 只有当独立命名空间的 shared experts 不存在时才使用此方式
        if not shared_indices and self.config.num_shared_experts > 0:
            # 检查是否已经通过独立命名空间加载了 shared experts
            layer_shared_count = sum(
                1 for eid in self.shared_expert_ids if eid.layer_idx == layer_idx
            )
            if layer_shared_count > 0:
                return  # 已通过独立命名空间加载
            
            # 尝试从 experts 列表加载（假设前 num_shared_experts 个是 shared）
            proj_names = ['gate_proj', 'up_proj', 'down_proj']
            
            for shared_idx in range(self.config.num_shared_experts):
                expert_prefix = f"{layer_prefix}.mlp.experts.{shared_idx}"
                has_expert = self.weight_loader.has_weight(f"{expert_prefix}.gate_proj.weight")
                
                if has_expert:
                    expert_weights = {}
                    for proj in proj_names:
                        weight_name = f"{expert_prefix}.{proj}.weight"
                        if self.weight_loader.has_weight(weight_name):
                            expert_weights[proj] = self.weight_loader.get_tensor(weight_name, device='cuda')
                    
                    if len(expert_weights) == 3:
                        expert_id = ExpertID(layer_idx, shared_idx)
                        self.shared_experts_gpu[expert_id] = expert_weights
                        self.shared_expert_ids.add(expert_id)
                        self.expert_locations[expert_id] = ExpertLocation(
                            expert_id=expert_id,
                            device=DeviceType.GPU,
                            is_cached=True
                        )
                        logger.debug(f"Loaded shared expert {shared_idx} (from experts list) for layer {layer_idx}")
    
    def _load_all_experts_to_cpu(self) -> None:
        """
        加载所有 routed expert 参数到 CPU
        CPU 中始终保存全部的 routed expert 副本
        跳过已加载到 GPU 的 shared experts
        """
        if not self.config.is_moe:
            logger.info("Not a MoE model, skipping expert loading")
            return
        
        # 计算需要加载的 expert 数量
        total_routed = self.config.num_experts * self.config.num_hidden_layers - len(self.shared_expert_ids)
        logger.info(f"Loading routed experts to CPU ({total_routed} experts, "
                   f"skipping {len(self.shared_expert_ids)} shared experts)...")
        
        total_loaded = 0
        for layer_idx in range(self.config.num_hidden_layers):
            for expert_idx in range(self.config.num_experts):
                expert_id = ExpertID(layer_idx, expert_idx)
                
                # 跳过 shared experts（它们已经在 GPU 上）
                if expert_id in self.shared_expert_ids:
                    logger.debug(f"Skipping shared expert {expert_id}")
                    continue
                
                expert_weights = self._load_single_expert(layer_idx, expert_idx, device='cpu')
                if expert_weights:
                    self.expert_params_cpu[expert_id] = expert_weights
                    self.expert_locations[expert_id] = ExpertLocation(
                        expert_id=expert_id,
                        device=DeviceType.CPU,
                        is_cached=False
                    )
                    total_loaded += 1
        
        logger.info(f"Loaded {total_loaded} routed experts to CPU")
    
    def _load_single_expert(
        self, 
        layer_idx: int, 
        expert_idx: int, 
        device: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """加载单个 expert 的权重"""
        expert_prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
        
        weights = {}
        proj_names = ['gate_proj', 'up_proj', 'down_proj']
        
        # 使用 pin_memory 加速 CPU 到 GPU 的传输（仅对 CPU 设备）
        use_pin = (device == 'cpu' and self.use_pin_memory)
        
        for proj in proj_names:
            weight_name = f"{expert_prefix}.{proj}.weight"
            if self.weight_loader.has_weight(weight_name):
                weights[proj] = self.weight_loader.get_tensor(
                    weight_name, device=device, pin_memory=use_pin
                )
            else:
                # 如果任何一个权重不存在，返回 None
                return None
        
        return weights
    
    def _copy_experts_to_gpu(self) -> None:
        """
        根据放置配置将部分 routed expert 复制到 GPU
        注意：CPU 中仍保留完整副本
        """
        gpu_placement = self.placement_config.get('gpu_experts', {})
        
        if not gpu_placement:
            logger.info("No GPU placement config, all routed experts remain on CPU only")
            return
        
        copied_count = 0
        for layer_idx, expert_indices in gpu_placement.items():
            layer_idx = int(layer_idx) if isinstance(layer_idx, str) else layer_idx
            for expert_idx in expert_indices:
                expert_id = ExpertID(layer_idx, expert_idx)
                
                # 跳过 shared experts（它们已经在 shared_experts_gpu 中）
                if expert_id in self.shared_expert_ids:
                    continue
                
                if expert_id in self.expert_params_cpu:
                    # 复制到 GPU（使用 non_blocking 如果是 pinned memory）
                    cpu_weights = self.expert_params_cpu[expert_id]
                    is_pinned = cpu_weights['gate_proj'].is_pinned() if hasattr(cpu_weights['gate_proj'], 'is_pinned') else False
                    self.expert_params_gpu[expert_id] = {
                        k: v.to('cuda', non_blocking=is_pinned) for k, v in cpu_weights.items()
                    }
                    self.expert_locations[expert_id] = ExpertLocation(
                        expert_id=expert_id,
                        device=DeviceType.GPU,
                        is_cached=True
                    )
                    copied_count += 1
        
        if copied_count > 0:
            # 同步确保传输完成
            torch.cuda.synchronize()
        
        logger.info(f"Copied {copied_count} routed experts to GPU")
    
    def is_shared_expert(self, expert_id: ExpertID) -> bool:
        """检查是否为 shared expert"""
        return expert_id in self.shared_expert_ids
    
    def get_shared_expert_params(self, expert_id: ExpertID) -> Optional[Dict[str, torch.Tensor]]:
        """获取 shared expert 参数"""
        return self.shared_experts_gpu.get(expert_id)
    
    def get_all_shared_experts_for_layer(self, layer_idx: int) -> Dict[ExpertID, Dict[str, torch.Tensor]]:
        """获取指定层的所有 shared experts"""
        return {
            eid: params for eid, params in self.shared_experts_gpu.items()
            if eid.layer_idx == layer_idx
        }
    
    def get_expert_location(self, expert_id: ExpertID) -> Optional[ExpertLocation]:
        """获取 expert 的当前位置"""
        return self.expert_locations.get(expert_id)
    
    def get_expert_params(
        self, 
        expert_id: ExpertID, 
        device: Optional[DeviceType] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        获取 expert 参数
        如果指定 device，只返回该设备上的参数
        优先级：shared_experts_gpu > expert_params_gpu > expert_params_cpu
        """
        # 优先检查 shared experts（它们始终在 GPU）
        if expert_id in self.shared_experts_gpu:
            if device is None or device == DeviceType.GPU:
                return self.shared_experts_gpu[expert_id]
            return None  # shared experts 只在 GPU 上
        
        if device == DeviceType.GPU:
            return self.expert_params_gpu.get(expert_id)
        elif device == DeviceType.CPU:
            return self.expert_params_cpu.get(expert_id)
        else:
            # 优先返回 GPU 上的
            if expert_id in self.expert_params_gpu:
                return self.expert_params_gpu[expert_id]
            return self.expert_params_cpu.get(expert_id)
    
    def get_static_param(self, name: str) -> Optional[torch.Tensor]:
        """获取静态参数"""
        return self.static_params_gpu.get(name)
    
    def load_expert_to_gpu(self, expert_id: ExpertID, non_blocking: bool = True) -> bool:
        """
        将 expert 从 CPU 加载到 GPU
        返回是否成功加载
        """
        # Shared experts 已经在 GPU 上
        if expert_id in self.shared_experts_gpu:
            return True
        
        if expert_id in self.expert_params_gpu:
            return True  # 已在 GPU 上
        
        if expert_id not in self.expert_params_cpu:
            logger.warning(f"Expert {expert_id} not found in CPU cache")
            return False
        
        # 从 CPU 复制到 GPU
        cpu_weights = self.expert_params_cpu[expert_id]
        is_pinned = cpu_weights['gate_proj'].is_pinned() if hasattr(cpu_weights['gate_proj'], 'is_pinned') else False
        self.expert_params_gpu[expert_id] = {
            k: v.to('cuda', non_blocking=(non_blocking and is_pinned)) 
            for k, v in cpu_weights.items()
        }
        self.expert_locations[expert_id] = ExpertLocation(
            expert_id=expert_id,
            device=DeviceType.GPU,
            is_cached=True
        )
        return True
    
    def evict_expert_from_gpu(self, expert_id: ExpertID) -> bool:
        """
        从 GPU 驱逐 expert（直接释放 GPU 内存，CPU 副本保留）
        注意：不能驱逐 shared experts
        """
        # Shared experts 不能被驱逐
        if expert_id in self.shared_expert_ids:
            logger.warning(f"Cannot evict shared expert {expert_id}")
            return False
        
        if expert_id not in self.expert_params_gpu:
            return False
        
        # 释放 GPU 内存
        del self.expert_params_gpu[expert_id]
        
        # 更新位置
        self.expert_locations[expert_id] = ExpertLocation(
            expert_id=expert_id,
            device=DeviceType.CPU,
            is_cached=False
        )
        return True
    
    def get_gpu_expert_count(self) -> int:
        """获取当前 GPU 上的 routed expert 数量（不含 shared）"""
        return len(self.expert_params_gpu)
    
    def get_shared_expert_count(self) -> int:
        """获取 shared expert 数量"""
        return len(self.shared_experts_gpu)
    
    def get_cpu_expert_count(self) -> int:
        """获取 CPU 上的 routed expert 数量"""
        return len(self.expert_params_cpu)
    
    def get_expert_memory_usage(self) -> Dict[str, float]:
        """获取 expert 内存使用情况（字节和 MB）"""
        def calc_size(params: Dict[str, torch.Tensor]) -> int:
            return sum(p.numel() * p.element_size() for p in params.values())
        
        shared_gpu_usage = sum(calc_size(p) for p in self.shared_experts_gpu.values())
        routed_gpu_usage = sum(calc_size(p) for p in self.expert_params_gpu.values())
        cpu_usage = sum(calc_size(p) for p in self.expert_params_cpu.values())
        
        return {
            'shared_gpu_bytes': shared_gpu_usage,
            'shared_gpu_mb': shared_gpu_usage / (1024 * 1024),
            'routed_gpu_bytes': routed_gpu_usage,
            'routed_gpu_mb': routed_gpu_usage / (1024 * 1024),
            'total_gpu_bytes': shared_gpu_usage + routed_gpu_usage,
            'total_gpu_mb': (shared_gpu_usage + routed_gpu_usage) / (1024 * 1024),
            'cpu_bytes': cpu_usage,
            'cpu_mb': cpu_usage / (1024 * 1024),
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取完整的内存使用情况（包括静态参数和 experts）"""
        def calc_tensor_size(tensor: torch.Tensor) -> int:
            return tensor.numel() * tensor.element_size()
        
        def calc_dict_size(params: Dict[str, torch.Tensor]) -> int:
            return sum(p.numel() * p.element_size() for p in params.values())
        
        # 静态参数（GPU）
        static_gpu_usage = sum(calc_tensor_size(p) for p in self.static_params_gpu.values())
        
        # Shared experts（GPU）
        shared_gpu_usage = sum(calc_dict_size(p) for p in self.shared_experts_gpu.values())
        
        # Routed experts
        routed_gpu_usage = sum(calc_dict_size(p) for p in self.expert_params_gpu.values())
        cpu_usage = sum(calc_dict_size(p) for p in self.expert_params_cpu.values())
        
        total_gpu = static_gpu_usage + shared_gpu_usage + routed_gpu_usage
        
        return {
            'static_gpu_bytes': static_gpu_usage,
            'static_gpu_mb': static_gpu_usage / (1024 * 1024),
            'shared_expert_gpu_bytes': shared_gpu_usage,
            'shared_expert_gpu_mb': shared_gpu_usage / (1024 * 1024),
            'routed_expert_gpu_bytes': routed_gpu_usage,
            'routed_expert_gpu_mb': routed_gpu_usage / (1024 * 1024),
            'total_gpu_bytes': total_gpu,
            'total_gpu_mb': total_gpu / (1024 * 1024),
            'cpu_bytes': cpu_usage,
            'cpu_mb': cpu_usage / (1024 * 1024),
        }
    
    def close(self):
        """清理资源"""
        self.weight_loader.close()
