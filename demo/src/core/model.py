"""
MoE 模型配置和结构定义
支持从 HuggingFace 配置文件加载
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
import os
import json
import torch
import torch.nn as nn

from .types import ExpertID


def load_config_json(model_path: str) -> Dict[str, Any]:
    """直接从 config.json 加载配置"""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


@dataclass
class MoEConfig:
    """
    MoE 模型配置
    支持手动指定或从 HuggingFace 配置加载
    """
    # 模型维度
    hidden_size: int = 2048
    num_hidden_layers: int = 48
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    intermediate_size: int = 6144
    vocab_size: int = 151936
    
    # MoE 特定配置
    num_experts: int = 128
    num_experts_per_token: int = 8  # top-k
    num_shared_experts: int = 0
    moe_intermediate_size: int = 768  # 每个 expert 的中间维度
    
    # 归一化配置
    rms_norm_eps: float = 1e-6
    
    # 位置编码配置
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    
    # 数据类型
    torch_dtype: str = "bfloat16"
    
    # 模型类型
    model_type: str = "qwen3_moe"
    
    # TODO
    # Draft-verify
    draft_top_c: int = 2  # 需删除，每一步由sched决定，不属于config # Top-c experts for CPU during draft
    max_draft_tokens: int = 8 # 删除？
    verify_threshold_perplexity: float = 1.5 # threshold重新设计
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> "MoEConfig":
        """
        从 HuggingFace 模型路径加载配置
        直接从 config.json 加载以避免 transformers 版本依赖
        """
        config_json = load_config_json(model_path)
        
        # 提取配置
        hidden_size = config_json['hidden_size']
        num_attention_heads = config_json['num_attention_heads']
        
        config_dict = {
            'hidden_size': hidden_size,
            'num_hidden_layers': config_json['num_hidden_layers'],
            'num_attention_heads': num_attention_heads,
            'num_key_value_heads': config_json.get('num_key_value_heads', num_attention_heads),
            'head_dim': config_json.get('head_dim', hidden_size // num_attention_heads),
            'intermediate_size': config_json.get('intermediate_size', hidden_size * 4),
            'vocab_size': config_json['vocab_size'],
            'rms_norm_eps': config_json.get('rms_norm_eps', 1e-6),
            'max_position_embeddings': config_json.get('max_position_embeddings', 32768),
            'rope_theta': config_json.get('rope_theta', 10000.0),
            'model_type': config_json.get('model_type', 'unknown'),
        }
        
        # MoE 特定配置
        config_dict['num_experts'] = config_json.get('num_experts', 1)
        config_dict['num_experts_per_token'] = config_json.get('num_experts_per_tok', 1)
        config_dict['num_shared_experts'] = config_json.get('num_shared_experts', 0)
        config_dict['moe_intermediate_size'] = config_json.get(
            'moe_intermediate_size', config_dict['intermediate_size']
        )
        
        # 数据类型
        torch_dtype = config_json.get('torch_dtype', 'float16')
        config_dict['torch_dtype'] = str(torch_dtype)
        
        return cls(**config_dict)
    
    @property
    def is_moe(self) -> bool:
        """是否为 MoE 模型"""
        return self.num_experts > 1
    
    def get_expert_weight_size_bytes(self) -> int:
        """计算单个 expert 的参数大小（字节）"""
        dtype_size = 2 if 'float16' in self.torch_dtype or 'bfloat16' in self.torch_dtype else 4
        total_params = 3 * self.moe_intermediate_size * self.hidden_size
        return total_params * dtype_size
    
    def get_total_expert_params(self) -> int:
        """计算所有 expert 的总参数量"""
        params_per_expert = 3 * self.moe_intermediate_size * self.hidden_size
        return self.num_hidden_layers * self.num_experts * params_per_expert
    
    def get_dtype(self) -> torch.dtype:
        """获取 torch 数据类型"""
        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.float16)


class MoEModelStructure:
    """
    表示 MoE 模型的结构（不包含实际参数）
    用于规划和协调
    """
    def __init__(self, config: MoEConfig):
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.num_experts_per_layer = config.num_experts
        
    def get_expert_ids(self, layer_idx: Optional[int] = None) -> List[ExpertID]:
        """获取所有 expert ID，可选按层过滤"""
        if layer_idx is not None:
            return [
                ExpertID(layer_idx, exp_idx) 
                for exp_idx in range(self.num_experts_per_layer)
            ]
        else:
            return [
                ExpertID(l_idx, exp_idx)
                for l_idx in range(self.num_layers)
                for exp_idx in range(self.num_experts_per_layer)
            ]
    
    def get_num_experts(self) -> int:
        """模型中的 expert 总数"""
        return self.num_layers * self.num_experts_per_layer
    
    def is_shared_expert(self, expert_id: ExpertID) -> bool:
        """检查是否为 shared expert"""
        return expert_id.expert_idx < self.config.num_shared_experts


class MoELayer(nn.Module):
    """
    单个 MoE 层（实际实现的占位符）
    实际的算子在 operators/ 模块中
    """
    def __init__(self, config: MoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
    def forward(self, hidden_states, expert_cache, device_assignments):
        """前向传播，协调 CPU/GPU 执行"""
        raise NotImplementedError("Use execution engines instead")
