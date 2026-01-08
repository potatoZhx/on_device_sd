import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from typing import Dict, Tuple, Optional, List
import copy

class MOEModelWrapper:
    """
    原始MOE模型包装器
    仅封装原始模型，不进行任何路由修改
    """
    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "float16"):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        
        # 加载原始模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=getattr(torch, dtype),
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
    def prefill(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        执行prefill阶段，生成初始KV缓存
        注意：prefill始终使用原始路由，不使用修改后的路由
        """
        with torch.no_grad():
            # 直接使用模型的原生缓存机制
            outputs = self.model(input_ids, use_cache=True)
            
            # 对于Qwen3模型，确保KV cache是DynamicCache类型
            if 'qwen' in self.model_path.lower() and not isinstance(outputs.past_key_values, DynamicCache):
                # 如果是元组，转换为DynamicCache
                if isinstance(outputs.past_key_values, tuple):
                    kv_cache = DynamicCache.from_legacy_cache(outputs.past_key_values)
                else:
                    kv_cache = outputs.past_key_values
            else:
                kv_cache = outputs.past_key_values
                
            return outputs.logits, kv_cache
    
    def decode(self, input_ids: torch.Tensor, kv_cache: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """
        执行单步decode
        注意：这是原始模型的decode，不使用修改后的路由
        """
        with torch.no_grad():
            # 对于所有模型，我们需要确保只使用最后一个token
            if input_ids.shape[1] > 1 and kv_cache is not None:
                # 只使用最后一个token进行decode
                input_ids = input_ids[:, -1:]
            
            # 处理Qwen3模型的KV cache
            if 'qwen' in self.model_path.lower() and kv_cache is not None:
                # 确保kv_cache是DynamicCache类型
                if not isinstance(kv_cache, DynamicCache) and isinstance(kv_cache, tuple):
                    kv_cache = DynamicCache.from_legacy_cache(kv_cache)
            
            # 执行模型前向计算
            outputs = self.model(
                input_ids, 
                past_key_values=kv_cache,
                use_cache=True
            )
            
            # 确保返回的KV cache是正确类型
            if 'qwen' in self.model_path.lower() and not isinstance(outputs.past_key_values, DynamicCache):
                if isinstance(outputs.past_key_values, tuple):
                    new_kv_cache = DynamicCache.from_legacy_cache(outputs.past_key_values)
                else:
                    new_kv_cache = outputs.past_key_values
            else:
                new_kv_cache = outputs.past_key_values
                
            return outputs.logits, new_kv_cache
    
    def get_tokenizer(self):
        """获取tokenizer"""
        return self.tokenizer

class ModifiedMOEModel:
    """
    修改的MOE模型（draft模型）
    
    关键设计：
    - 与original_model共享同一个模型实例
    - 通过路由修改器的enable/disable方法控制路由行为
    - decode时临时启用路由修改，完成后立即禁用
    """
    def __init__(
        self,
        original_model: MOEModelWrapper,
        *,
        num_to_modify: int,
        src_positions: List[int],
        dst_positions: List[int],
    ):
        self.original_model = original_model
        self.num_to_modify = num_to_modify
        self.src_positions = src_positions
        self.dst_positions = dst_positions
        self.model = original_model.model  # 共享实例
        self.tokenizer = original_model.tokenizer
        self.device = original_model.device
        self.dtype = original_model.dtype
        
        # 初始化路由修改器
        from .moe_routing_modifier import create_moe_routing_modifier
        self.routing_modifier = create_moe_routing_modifier(
            model_name="qwen",  
            num_to_modify=num_to_modify,
            src_positions=src_positions,
            dst_positions=dst_positions,
        )
        
        # 应用路由修改（替换forward方法，但初始状态是禁用的）
        self.routing_modifier.modify_model(self.model)
        print(
            f"ModifiedMOEModel初始化完成，按位置替换 {num_to_modify} 个experts: "
            f"src_positions={src_positions}, dst_positions={dst_positions}"
        )
    
    def decode(self, input_ids: torch.Tensor, kv_cache: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """
        执行修改后的decode（draft模型）
        在decode前启用路由修改，完成后立即禁用
        """
        try:
            # 启用路由修改
            self.routing_modifier.enable_routing_modification(self.model)
            
            # 执行decode（使用原始模型的decode逻辑，但路由已被修改）
            with torch.no_grad():
                # 处理输入token
                if input_ids.shape[1] > 1 and kv_cache is not None:
                    input_ids = input_ids[:, -1:]
                
                # 处理KV cache
                if 'qwen' in self.original_model.model_path.lower() and kv_cache is not None:
                    if not isinstance(kv_cache, DynamicCache) and isinstance(kv_cache, tuple):
                        kv_cache = DynamicCache.from_legacy_cache(kv_cache)
                
                # 执行模型前向计算
                outputs = self.model(
                    input_ids,
                    past_key_values=kv_cache,
                    use_cache=True
                )
                
                # 确保返回的KV cache是正确类型
                if 'qwen' in self.original_model.model_path.lower() and not isinstance(outputs.past_key_values, DynamicCache):
                    if isinstance(outputs.past_key_values, tuple):
                        new_kv_cache = DynamicCache.from_legacy_cache(outputs.past_key_values)
                    else:
                        new_kv_cache = outputs.past_key_values
                else:
                    new_kv_cache = outputs.past_key_values
                
                return outputs.logits, new_kv_cache
        finally:
            # 无论成功还是失败，都要禁用路由修改
            self.routing_modifier.disable_routing_modification(self.model)
    
    def __del__(self):
        """清理路由修改器，恢复模型原始状态"""
        if hasattr(self, 'routing_modifier'):
            self.routing_modifier.restore_model(self.model)
            print("ModifiedMOEModel已清理，模型已恢复原始状态")
