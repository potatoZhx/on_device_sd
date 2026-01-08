"""
MOE路由修改器的单元测试
"""
import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from model.moe_spec.moe_routing_modifier import (
    MOERoutingModifier, 
    GenericMOERoutingModifier, 
    create_moe_routing_modifier
)

class TestGenericMOERoutingModifier:
    """通用MOE路由修改器测试类"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.modifier = GenericMOERoutingModifier(
            top_k_experts_to_remove=2, 
            noise_scale=0.01
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.modifier.top_k_experts_to_remove == 2
        assert self.modifier.noise_scale == 0.01
        assert self.modifier.hooks == []
    
    def test_modify_routing_scores_logic(self):
        """测试路由分数修改逻辑"""
        # 创建测试数据
        batch_size = 2
        num_experts = 6
        original_scores = torch.tensor([
            [0.5, 0.3, 0.1, 0.05, 0.03, 0.02],  # 第一个样本
            [0.4, 0.25, 0.15, 0.1, 0.06, 0.04]   # 第二个样本
        ])
        
        modified_scores = self.modifier.modify_routing_scores(original_scores)
        
        # 验证形状不变
        assert modified_scores.shape == original_scores.shape
        
        # 验证每行和为1（归一化）
        row_sums = modified_scores.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones(batch_size), atol=1e-6, rtol=1e-6)
        
        # 验证top-2专家的分数为0
        for i in range(batch_size):
            _, top_indices = torch.topk(original_scores[i], self.modifier.top_k_experts_to_remove)
            for idx in top_indices:
                assert modified_scores[i, idx] == 0.0
    
    def test_create_generic_modifier(self):
        """测试创建通用修改器"""
        modifier = create_moe_routing_modifier(
            model_name="generic",
            top_k_experts_to_remove=3,
            noise_scale=0.05
        )
        
        assert isinstance(modifier, GenericMOERoutingModifier)
        assert modifier.top_k_experts_to_remove == 3
        assert modifier.noise_scale == 0.05
    
    def test_create_unknown_modifier(self):
        """测试创建未知修改器（应该返回GenericMOERoutingModifier）"""
        # 根据实际实现，未知模型名会返回GenericMOERoutingModifier
        modifier = create_moe_routing_modifier(model_name="unknown_model")
        assert isinstance(modifier, GenericMOERoutingModifier)
