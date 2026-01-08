"""
MOE路由修改器 - 实际的MOE路由修改实现

这个文件包含了真实的MOE路由修改逻辑，需要根据具体的MOE架构进行调整。
当前实现针对Qwen3模型，支持“基于排名位置的专家替换”路由修改：
- 接收三个参数：num_to_modify、src_positions、dst_positions
- 将top-k内指定排名的专家替换为top-k之外指定排名的专家
- 不修改路由分数，仅修改选择逻辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class MOERoutingModifier:
    """
    MOE路由修改器基类
    通过为MoE层添加属性和替换forward方法来实现路由修改
    """
    
    def __init__(
        self,
        *,
        num_to_modify: int,
        src_positions: List[int],
        dst_positions: List[int],
    ):
        self.num_to_modify = num_to_modify
        self.src_positions = src_positions
        self.dst_positions = dst_positions
        self.original_forwards = {}
        # 在modify_model中完成参数检查，并预计算列索引
        self._s_cols = None
        self._d_cols = None
    
    def enable_routing_modification(self, model):
        """启用路由修改"""
        raise NotImplementedError("子类必须实现enable_routing_modification方法")
    
    def disable_routing_modification(self, model):
        """禁用路由修改"""
        raise NotImplementedError("子类必须实现disable_routing_modification方法")
    
    def restore_model(self, model):
        """恢复模型的原始路由逻辑（完全移除修改）"""
        raise NotImplementedError("子类必须实现restore_model方法")

class QwenMOERoutingModifier(MOERoutingModifier):
    """
    Qwen3模型的MOE路由修改器
    
    针对Qwen3模型的MoE结构，修改expert选择逻辑：
    - 排除路由分数最高的top-2 experts
    - 从剩余experts中选择top-k experts
    - 不修改路由分数本身，只修改选择逻辑
    """
    
    def __init__(
        self,
        *,
        num_to_modify: int,
        src_positions: List[int],
        dst_positions: List[int],
    ):
        super().__init__(
            num_to_modify=num_to_modify,
            src_positions=src_positions,
            dst_positions=dst_positions,
        )
        self.moe_layers = []
    
    def modify_model(self, model):
        """
        修改Qwen3模型的MOE路由逻辑
        替换所有MoE层的forward方法，添加use_modified_routing属性控制
        """
        self.original_forwards = {}
        self.moe_layers = []
        
        # 遍历模型中的MOE层
        for name, module in model.named_modules():
            if self._is_moe_layer(name, module):
                # 保存模块引用
                self.moe_layers.append(module)
                
                # 添加控制属性
                module.use_modified_routing = False
                
                # 保存原始forward方法
                self.original_forwards[id(module)] = module.forward
                
        # 参数校验：基于模型层的top_k与num_experts（要求各层一致）
        if len(self.moe_layers) == 0:
            print("警告：未检测到MoE层，路由修改不会生效")
            return
        else:
            first = self.moe_layers[0]
            top_k = getattr(first, 'top_k')
            num_experts = getattr(first, 'num_experts')
            for layer in self.moe_layers:
                if getattr(layer, 'top_k') != top_k or getattr(layer, 'num_experts') != num_experts:
                    raise ValueError("所有MoE层的top_k与num_experts必须一致")
            
            # 校验新接口参数（一次性校验）
            if not (isinstance(self.num_to_modify, int) and self.num_to_modify >= 0):
                raise ValueError("num_to_modify 必须是非负整数")
            if not (isinstance(self.src_positions, list) and isinstance(self.dst_positions, list)):
                raise ValueError("src_positions 与 dst_positions 必须是列表")
            if not (len(self.src_positions) == len(self.dst_positions) == self.num_to_modify):
                raise ValueError(
                    f"路由修改参数长度不一致: num_to_modify={self.num_to_modify}, "
                    f"len(src_positions)={len(self.src_positions)}, len(dst_positions)={len(self.dst_positions)}"
                )
            if any(self.src_positions[i] >= self.src_positions[i+1] for i in range(len(self.src_positions)-1)):
                raise ValueError("src_positions 必须严格升序")
            if any(self.dst_positions[i] >= self.dst_positions[i+1] for i in range(len(self.dst_positions)-1)):
                raise ValueError("dst_positions 必须严格升序")
            if self.num_to_modify > top_k:
                raise ValueError(f"num_to_modify(={self.num_to_modify}) 不能大于 top_k(={top_k})")
            if not all(1 <= s <= top_k for s in self.src_positions):
                raise ValueError(f"src_positions 元素需在[1, top_k={top_k}]范围内")
            if not all((d > top_k) and (d <= num_experts) for d in self.dst_positions):
                raise ValueError(
                    f"dst_positions 元素需满足 top_k(={top_k}) < d ≤ num_experts(={num_experts})"
                )
            
            # 预计算0-based列索引
            self._s_cols = torch.tensor([s-1 for s in self.src_positions], dtype=torch.long)
            self._d_cols = torch.tensor([d-1 for d in self.dst_positions], dtype=torch.long)
        

        for module in self.moe_layers:
                # 创建新的forward方法
                module.forward = self._create_modified_forward(module)
        
        print(f"已为 {len(self.moe_layers)} 个MoE层添加路由修改支持")
    
    def _create_modified_forward(self, moe_module):
        """
        创建修改后的forward方法
        通过use_modified_routing属性控制是否应用修改
        """
        original_forward = self.original_forwards[id(moe_module)]
        
        def modified_forward(hidden_states):
            """修改后的forward方法"""
            # 如果不使用修改路由，直接调用原始forward
            if not getattr(moe_module, 'use_modified_routing', False):
                return original_forward(hidden_states)
            
            # 使用修改后的路由逻辑
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
            
            # 计算路由logits
            router_logits = moe_module.gate(hidden_states)
            
            # 计算路由权重（不修改）
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            
            # 全体专家的排名（降序）
            _, sorted_indices = torch.topk(
                routing_weights, moe_module.num_experts, dim=-1
            )
            
            # 基础选择：top_k
            top_k = moe_module.top_k
            selected_experts = sorted_indices[:, :top_k]
            
            # 按位置替换（使用在modify_model中预计算的列索引）
            if self._s_cols is not None and self._d_cols is not None and self._s_cols.numel() > 0:
                dest_idx = sorted_indices[:, self._d_cols]  # [N, num_to_modify]
                selected_experts[:, self._s_cols] = dest_idx
            
            # 禁止重复：逐行检查（确保运行时不出现意外重复）
            if selected_experts.shape[1] > 1:
                sorted_sel = torch.sort(selected_experts, dim=1).values
                has_dup = (sorted_sel[:, 1:] == sorted_sel[:, :-1]).any()
                if bool(has_dup.item() if has_dup.numel() == 1 else has_dup.any().item()):
                    raise ValueError("路由修改后selected_experts出现重复，违反禁止重复约束")
            
            # 收集对应的原始权重
            batch_indices = torch.arange(
                routing_weights.shape[0], device=routing_weights.device
            ).unsqueeze(1).expand(-1, top_k)
            routing_weights = routing_weights[batch_indices, selected_experts]
            
            # 归一化（如果需要）
            if moe_module.norm_topk_prob:
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            
            # 转换回原始数据类型
            routing_weights = routing_weights.to(hidden_states.dtype)
            
            # 创建最终的隐藏状态
            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), 
                dtype=hidden_states.dtype, 
                device=hidden_states.device
            )
            
            # 创建专家掩码
            expert_mask = F.one_hot(selected_experts, num_classes=moe_module.num_experts).permute(2, 1, 0)
            
            # 对每个专家执行计算
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit:
                expert_layer = moe_module.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                
                # 计算当前专家的隐藏状态
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                
                # 将结果添加到最终隐藏状态
                final_hidden_states.index_add_(
                    0, top_x, current_hidden_states.to(hidden_states.dtype)
                )
            
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states, router_logits
        
        return modified_forward
    
    def enable_routing_modification(self, model):
        """启用路由修改"""
        for module in self.moe_layers:
            module.use_modified_routing = True
    
    def disable_routing_modification(self, model):
        """禁用路由修改"""
        for module in self.moe_layers:
            module.use_modified_routing = False
    
    def restore_model(self, model):
        """恢复模型的原始路由逻辑"""
        # 恢复原始forward方法
        for module in self.moe_layers:
            if id(module) in self.original_forwards:
                module.forward = self.original_forwards[id(module)]
                # 删除添加的属性
                if hasattr(module, 'use_modified_routing'):
                    delattr(module, 'use_modified_routing')
        
        print(f"已恢复 {len(self.moe_layers)} 个MoE层的原始forward方法")
        
        # 清除保存的引用
        self.moe_layers.clear()
        self.original_forwards.clear()
    
    def _is_moe_layer(self, name: str, module: nn.Module) -> bool:
        """判断是否为Qwen3的MoE层"""
        # Qwen3的MoE层通常是model.layers.*.mlp
        if '.mlp' in name and hasattr(module, 'gate') and hasattr(module, 'experts'):
            return True
        return False

def create_moe_routing_modifier(model_name: str = "qwen", **kwargs) -> MOERoutingModifier:
    """
    创建MOE路由修改器工厂函数
    
    Args:
        model_name: 模型名称 ("qwen", "qwen3")
        **kwargs: 其他参数，如top_k_experts_to_remove
    
    Returns:
        MOERoutingModifier实例
    """
    if model_name.lower() in ["qwen", "qwen3"]:
        return QwenMOERoutingModifier(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_name}，当前仅支持Qwen3模型")
