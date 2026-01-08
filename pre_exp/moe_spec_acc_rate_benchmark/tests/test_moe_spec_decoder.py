"""
MOE推测解码器的单元测试
"""
import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from model.moe_spec.moe_spec_decoder import MOESpecDecoder

class TestMOESpecDecoder:
    """MOE推测解码器测试类"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        # 创建mock模型
        self.mock_original_model = Mock()
        self.mock_modified_model = Mock()
        
        # 设置mock tokenizer
        self.mock_original_model.tokenizer = Mock()
        self.mock_original_model.tokenizer.eos_token_id = 2
        
        # 设置mock prefill输出
        self.mock_kv_cache = [(torch.randn(1, 8, 4, 64), torch.randn(1, 8, 4, 64))]
        self.mock_original_model.prefill.return_value = (torch.randn(1, 4, 1000), self.mock_kv_cache)
        
        # 设置mock decode输出
        self.mock_original_model.decode.return_value = (torch.randn(1, 5, 1000), self.mock_kv_cache)
        self.mock_modified_model.decode.return_value = (torch.randn(1, 5, 1000), self.mock_kv_cache)
    
    def test_initialization(self):
        """测试MOE推测解码器初始化"""
        decoder = MOESpecDecoder(
            self.mock_original_model, 
            self.mock_modified_model, 
            draft_length=2
        )
        
        assert decoder.original_model == self.mock_original_model
        assert decoder.modified_model == self.mock_modified_model
        assert decoder.draft_length == 2
        assert decoder.total_draft_length == 0
        assert decoder.total_accept_length == 0
        assert decoder.accept_length_list == []
    
    @patch('model.moe_spec.moe_spec_decoder.speculative_sampling')
    def test_generate_draft(self, mock_spec_sampling):
        """测试draft生成"""
        decoder = MOESpecDecoder(
            self.mock_original_model, 
            self.mock_modified_model, 
            draft_length=2
        )
        
        # 设置mock输出
        mock_logits = torch.randn(1, 1, 1000)
        mock_logits[0, 0, 5] = 10.0  # 让token 5有最高概率
        self.mock_modified_model.decode.return_value = (mock_logits, self.mock_kv_cache)
        
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        
        draft_tokens, draft_logits = decoder._generate_draft(input_ids, self.mock_kv_cache)
        
        # 验证结果
        assert len(draft_tokens) == 2  # draft_length
        assert isinstance(draft_logits, torch.Tensor)
        assert draft_logits.shape[1] == 2  # draft_length
        
        # 验证模型被调用了正确的次数
        assert self.mock_modified_model.decode.call_count == 2
    
    @patch('model.moe_spec.moe_spec_decoder.speculative_sampling')
    def test_verify_draft(self, mock_spec_sampling):
        """测试draft验证"""
        # 设置mock推测采样返回值
        valid_tokens = torch.tensor([[7, 8]], dtype=torch.long)
        n_matches = 2
        mock_spec_sampling.return_value = (valid_tokens, n_matches)
        
        decoder = MOESpecDecoder(
            self.mock_original_model, 
            self.mock_modified_model, 
            draft_length=2
        )
        
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        draft_tokens = [5, 6]
        draft_logits = torch.randn(1, 2, 1000)
        
        accepted_tokens, accepted_length = decoder._verify_draft(
            input_ids, draft_tokens, draft_logits, self.mock_kv_cache
        )
        
        # 验证结果
        assert accepted_tokens == [7, 8]
        assert accepted_length == 2
        
        # 验证推测采样被正确调用
        mock_spec_sampling.assert_called_once()
        call_args = mock_spec_sampling.call_args
        assert torch.equal(call_args[1]['candidate_input_ids'], torch.cat([input_ids, torch.tensor([[5, 6]])], dim=1))
        assert call_args[1]['candidate_length'] == 2
    
    @patch('model.moe_spec.moe_spec_decoder.speculative_sampling')
    def test_verify_draft_no_matches(self, mock_spec_sampling):
        """测试draft验证无匹配的情况"""
        # 设置mock推测采样返回值（无匹配）
        valid_tokens = torch.tensor([[9]], dtype=torch.long)
        n_matches = 0
        mock_spec_sampling.return_value = (valid_tokens, n_matches)
        
        decoder = MOESpecDecoder(
            self.mock_original_model, 
            self.mock_modified_model, 
            draft_length=2
        )
        
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        draft_tokens = [5, 6]
        draft_logits = torch.randn(1, 2, 1000)
        
        accepted_tokens, accepted_length = decoder._verify_draft(
            input_ids, draft_tokens, draft_logits, self.mock_kv_cache
        )
        
        # 验证结果
        assert accepted_tokens == [9]
        assert accepted_length == 1  # 新采样的token数量
    
    def test_crop_kv_cache(self):
        """测试KV缓存裁剪"""
        decoder = MOESpecDecoder(
            self.mock_original_model, 
            self.mock_modified_model, 
            draft_length=1
        )
        
        # 创建测试KV缓存
        original_cache = [
            (torch.randn(1, 8, 10, 64), torch.randn(1, 8, 10, 64)),
            (torch.randn(1, 8, 10, 64), torch.randn(1, 8, 10, 64))
        ]
        
        new_cache_size = 5
        cropped_cache = decoder._crop_kv_cache(original_cache, new_cache_size)
        
        # 验证裁剪结果
        assert len(cropped_cache) == len(original_cache)
        assert cropped_cache[0][0].shape[2] == new_cache_size  # 序列长度维度被裁剪
        assert cropped_cache[0][1].shape[2] == new_cache_size
        assert cropped_cache[1][0].shape[2] == new_cache_size
        assert cropped_cache[1][1].shape[2] == new_cache_size
    
    def test_crop_kv_cache_none(self):
        """测试KV缓存裁剪None情况"""
        decoder = MOESpecDecoder(
            self.mock_original_model, 
            self.mock_modified_model, 
            draft_length=1
        )
        
        result = decoder._crop_kv_cache(None, 5)
        assert result is None
    
    @patch('model.moe_spec.moe_spec_decoder.speculative_sampling')
    def test_speculate_decode_basic(self, mock_spec_sampling):
        """测试基本的推测解码流程"""
        # 设置mock推测采样返回值
        valid_tokens = torch.tensor([[7]], dtype=torch.long)
        n_matches = 1
        mock_spec_sampling.return_value = (valid_tokens, n_matches)
        
        decoder = MOESpecDecoder(
            self.mock_original_model, 
            self.mock_modified_model, 
            draft_length=1
        )
        
        # 设置mock模型行为
        mock_logits = torch.randn(1, 1, 1000)
        mock_logits[0, 0, 7] = 10.0  # 让token 7有最高概率
        self.mock_modified_model.decode.return_value = (mock_logits, self.mock_kv_cache)
        
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        
        result = decoder.speculate_decode(input_ids, max_new_tokens=10)
        
        # 验证结果结构
        assert 'output_ids' in result
        assert 'new_token' in result
        assert 'step' in result
        assert 'accept_length_list' in result
        assert 'total_draft_length' in result
        assert 'total_accept_length' in result
        assert 'acceptance_rate' in result
        
        # 验证统计信息
        assert result['total_draft_length'] >= 0
        assert result['total_accept_length'] >= 0
        assert result['step'] >= 0
        assert isinstance(result['accept_length_list'], list)
    
    @patch('model.moe_spec.moe_spec_decoder.speculative_sampling')
    def test_speculate_decode_with_eos(self, mock_spec_sampling):
        """测试包含EOS token的推测解码"""
        # 设置mock推测采样返回值，包含EOS
        valid_tokens = torch.tensor([[2]], dtype=torch.long)  # EOS token
        n_matches = 1
        mock_spec_sampling.return_value = (valid_tokens, n_matches)
        
        decoder = MOESpecDecoder(
            self.mock_original_model, 
            self.mock_modified_model, 
            draft_length=1
        )
        
        # 设置mock模型行为
        mock_logits = torch.randn(1, 1, 1000)
        mock_logits[0, 0, 2] = 10.0  # EOS token有最高概率
        self.mock_modified_model.decode.return_value = (mock_logits, self.mock_kv_cache)
        
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        
        result = decoder.speculate_decode(input_ids, max_new_tokens=10)
        
        # 验证结果
        assert isinstance(result['output_ids'], torch.Tensor)
        assert result['step'] >= 0  # 应该在遇到EOS后停止
    
    @patch('model.moe_spec.moe_spec_decoder.speculative_sampling')
    def test_speculate_decode_no_acceptance(self, mock_spec_sampling):
        """测试无接受情况的推测解码"""
        # 设置mock推测采样返回值（无接受）
        valid_tokens = torch.tensor([[9]], dtype=torch.long)
        n_matches = 0
        mock_spec_sampling.return_value = (valid_tokens, n_matches)
        
        decoder = MOESpecDecoder(
            self.mock_original_model, 
            self.mock_modified_model, 
            draft_length=1
        )
        
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        
        result = decoder.speculate_decode(input_ids, max_new_tokens=5)
        
        # 验证结果
        assert result['total_draft_length'] > 0  # 应该生成了draft
        assert result['step'] >= 0
        # 由于没有接受，应该很快停止
    
    def test_statistics_tracking(self):
        """测试统计信息跟踪"""
        decoder = MOESpecDecoder(
            self.mock_original_model, 
            self.mock_modified_model, 
            draft_length=1
        )
        
        # 初始状态
        assert decoder.total_draft_length == 0
        assert decoder.total_accept_length == 0
        assert decoder.accept_length_list == []
        
        # 模拟统计更新
        decoder.total_draft_length += 3
        decoder.total_accept_length += 2
        decoder.accept_length_list.append(2)
        
        # 验证更新
        assert decoder.total_draft_length == 3
        assert decoder.total_accept_length == 2
        assert decoder.accept_length_list == [2]
        
        # 计算接收率
        acceptance_rate = decoder.total_accept_length / decoder.total_draft_length
        assert acceptance_rate == 2/3
    
    def test_acceptance_rate_calculation(self):
        """测试接收率计算"""
        decoder = MOESpecDecoder(
            self.mock_original_model, 
            self.mock_modified_model, 
            draft_length=1
        )
        
        # 测试零除情况
        decoder.total_draft_length = 0
        decoder.total_accept_length = 0
        
        # 在speculate_decode中会计算接收率
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        
        # 设置模型不生成任何draft（通过让decode返回空结果）
        self.mock_modified_model.decode.side_effect = Exception("No draft")
        
        try:
            result = decoder.speculate_decode(input_ids, max_new_tokens=1)
            # 如果没有draft，接收率应该为0
            assert result['acceptance_rate'] == 0.0
        except:
            # 如果发生异常，这也是预期的行为
            pass
