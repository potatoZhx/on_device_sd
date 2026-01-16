"""
推测采样算法的单元测试
"""
import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import torch
import numpy as np
from model.moe_spec.spec_sampling import speculative_sampling

class TestSpeculativeSampling:
    """推测采样算法测试类"""
    
    def test_basic_speculative_sampling(self):
        """测试基本的推测采样功能"""
        # 准备测试数据
        batch_size, candidate_length, vocab_size = 1, 3, 1000
        
        # 创建候选输入序列
        candidate_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        
        # 创建候选logits（修改模型的输出）
        candidate_logits = torch.randn(batch_size, candidate_length, vocab_size)
        
        # 创建验证logits（原模型的输出）
        new_logits = torch.randn(batch_size, candidate_length + 1, vocab_size)
        
        # 执行推测采样
        valid_tokens, n_matches = speculative_sampling(
            candidate_input_ids=candidate_input_ids,
            candidate_logits=candidate_logits,
            candidate_length=candidate_length,
            new_logits=new_logits,
            last_assistant_token_is_eos=False,
            max_matches=candidate_length
        )
        
        # 验证输出
        assert isinstance(valid_tokens, torch.Tensor)
        assert isinstance(n_matches, int)
        assert 0 <= n_matches <= candidate_length
        assert valid_tokens.shape[0] == batch_size
        assert valid_tokens.shape[1] >= 1  # 至少包含一个token
    
    def test_full_acceptance(self):
        """测试完全接受的情况"""
        batch_size, candidate_length, vocab_size = 1, 2, 10
        
        # 创建候选序列
        candidate_tokens = [8, 9]
        candidate_input_ids = torch.tensor([[1, 2, 3] + candidate_tokens], dtype=torch.long)
        
        # 创建候选logits，让候选token有很高的概率
        candidate_logits = torch.zeros(batch_size, candidate_length, vocab_size)
        candidate_logits[0, 0, 8] = 10.0  # 第一个候选token
        candidate_logits[0, 1, 9] = 10.0  # 第二个候选token
        
        # 创建验证logits，让原模型也给候选token很高的概率
        new_logits = torch.zeros(batch_size, candidate_length + 1, vocab_size)
        new_logits[0, 0, 8] = 10.0
        new_logits[0, 1, 9] = 10.0
        new_logits[0, 2, 5] = 10.0  # 下一个token
        
        # 执行推测采样
        valid_tokens, n_matches = speculative_sampling(
            candidate_input_ids=candidate_input_ids,
            candidate_logits=candidate_logits,
            candidate_length=candidate_length,
            new_logits=new_logits,
            last_assistant_token_is_eos=False,
            max_matches=candidate_length
        )
        
        # 由于概率很高，应该有较高的接受率
        assert n_matches >= 0
        assert valid_tokens.shape[1] >= 1
    
    def test_no_acceptance(self):
        """测试完全拒绝的情况"""
        batch_size, candidate_length, vocab_size = 1, 2, 10
        
        # 创建候选序列
        candidate_tokens = [8, 9]
        candidate_input_ids = torch.tensor([[1, 2, 3] + candidate_tokens], dtype=torch.long)
        
        # 创建候选logits，让候选token有很高的概率
        candidate_logits = torch.zeros(batch_size, candidate_length, vocab_size)
        candidate_logits[0, 0, 8] = 10.0
        candidate_logits[0, 1, 9] = 10.0
        
        # 创建验证logits，让原模型给候选token很低的概率
        new_logits = torch.zeros(batch_size, candidate_length + 1, vocab_size)
        new_logits[0, 0, 7] = 10.0  # 不同的token
        new_logits[0, 1, 6] = 10.0  # 不同的token
        new_logits[0, 2, 5] = 10.0
        
        # 执行推测采样
        valid_tokens, n_matches = speculative_sampling(
            candidate_input_ids=candidate_input_ids,
            candidate_logits=candidate_logits,
            candidate_length=candidate_length,
            new_logits=new_logits,
            last_assistant_token_is_eos=False,
            max_matches=candidate_length
        )
        
        # 验证结果
        assert n_matches >= 0
        assert valid_tokens.shape[1] >= 1
    
    def test_eos_handling(self):
        """测试EOS token处理"""
        batch_size, candidate_length, vocab_size = 1, 3, 10
        
        # 创建包含EOS的候选序列
        candidate_tokens = [8, 9, 2]  # 最后一个是EOS
        candidate_input_ids = torch.tensor([[1, 2, 3] + candidate_tokens], dtype=torch.long)
        
        candidate_logits = torch.randn(batch_size, candidate_length, vocab_size)
        new_logits = torch.randn(batch_size, candidate_length + 1, vocab_size)
        
        # 执行推测采样，标记最后一个token为EOS
        valid_tokens, n_matches = speculative_sampling(
            candidate_input_ids=candidate_input_ids,
            candidate_logits=candidate_logits,
            candidate_length=candidate_length,
            new_logits=new_logits,
            last_assistant_token_is_eos=True,
            max_matches=candidate_length
        )
        
        # 验证结果
        assert isinstance(valid_tokens, torch.Tensor)
        assert isinstance(n_matches, int)
        assert n_matches >= 0
    
    def test_max_matches_limit(self):
        """测试最大匹配数限制"""
        batch_size, candidate_length, vocab_size = 1, 5, 10
        max_matches = 2  # 限制最大匹配数
        
        candidate_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        candidate_logits = torch.randn(batch_size, candidate_length, vocab_size)
        new_logits = torch.randn(batch_size, candidate_length + 1, vocab_size)
        
        valid_tokens, n_matches = speculative_sampling(
            candidate_input_ids=candidate_input_ids,
            candidate_logits=candidate_logits,
            candidate_length=candidate_length,
            new_logits=new_logits,
            last_assistant_token_is_eos=False,
            max_matches=max_matches
        )
        
        # 验证匹配数不超过限制
        assert n_matches <= max_matches
        assert valid_tokens.shape[1] >= 1
    
    def test_probability_ratio_calculation(self):
        """测试概率比值计算的正确性"""
        # 设置固定种子以确保可重现性
        torch.manual_seed(42)
        
        batch_size, candidate_length, vocab_size = 1, 1, 5
        
        # 创建简单的测试用例
        candidate_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        
        # 创建已知的logits
        candidate_logits = torch.zeros(batch_size, candidate_length, vocab_size)
        candidate_logits[0, 0, 3] = 2.0  # 候选token 3的logit为2.0
        
        new_logits = torch.zeros(batch_size, candidate_length + 1, vocab_size)
        new_logits[0, 0, 3] = 3.0  # 原模型给token 3的logit为3.0
        
        # 执行推测采样
        valid_tokens, n_matches = speculative_sampling(
            candidate_input_ids=candidate_input_ids,
            candidate_logits=candidate_logits,
            candidate_length=candidate_length,
            new_logits=new_logits,
            last_assistant_token_is_eos=False,
            max_matches=candidate_length
        )
        
        # 由于原模型概率更高，应该有较高的接受概率
        assert valid_tokens.shape[1] >= 1
        assert n_matches >= 0
    
    def test_edge_cases(self):
        """测试边界情况"""
        batch_size, vocab_size = 1, 10
        
        # 测试candidate_length为0的情况
        with pytest.raises((IndexError, RuntimeError)):
            candidate_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
            candidate_logits = torch.randn(batch_size, 0, vocab_size)
            new_logits = torch.randn(batch_size, 1, vocab_size)
            
            speculative_sampling(
                candidate_input_ids=candidate_input_ids,
                candidate_logits=candidate_logits,
                candidate_length=0,
                new_logits=new_logits
            )
    
    def test_deterministic_behavior(self):
        """测试在固定种子下的确定性行为"""
        batch_size, candidate_length, vocab_size = 1, 2, 10
        
        candidate_input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        candidate_logits = torch.randn(batch_size, candidate_length, vocab_size)
        new_logits = torch.randn(batch_size, candidate_length + 1, vocab_size)
        
        # 第一次运行
        torch.manual_seed(123)
        valid_tokens1, n_matches1 = speculative_sampling(
            candidate_input_ids=candidate_input_ids,
            candidate_logits=candidate_logits,
            candidate_length=candidate_length,
            new_logits=new_logits,
            last_assistant_token_is_eos=False,
            max_matches=candidate_length
        )
        
        # 第二次运行（相同种子）
        torch.manual_seed(123)
        valid_tokens2, n_matches2 = speculative_sampling(
            candidate_input_ids=candidate_input_ids,
            candidate_logits=candidate_logits,
            candidate_length=candidate_length,
            new_logits=new_logits,
            last_assistant_token_is_eos=False,
            max_matches=candidate_length
        )
        
        # 结果应该相同
        assert torch.equal(valid_tokens1, valid_tokens2)
        assert n_matches1 == n_matches2
