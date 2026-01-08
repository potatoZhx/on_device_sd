"""
pytest配置文件，定义共享的fixtures和测试配置
"""
import pytest
import torch
import tempfile
import os
from unittest.mock import Mock, MagicMock
from transformers import AutoTokenizer

@pytest.fixture
def device():
    """测试设备fixture"""
    return "cpu"  # 使用CPU进行测试以避免GPU依赖

@pytest.fixture
def dtype():
    """数据类型fixture"""
    return "float32"

@pytest.fixture
def mock_tokenizer():
    """模拟tokenizer fixture"""
    tokenizer = Mock()
    tokenizer.encode.return_value = torch.tensor([1, 2, 3, 4, 5])
    tokenizer.decode.return_value = "test output"
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    tokenizer.pad_token = "<pad>"
    return tokenizer

@pytest.fixture
def mock_model():
    """模拟模型fixture"""
    model = Mock()
    
    # 模拟模型输出
    mock_output = Mock()
    mock_output.logits = torch.randn(1, 5, 1000)  # batch_size=1, seq_len=5, vocab_size=1000
    mock_output.past_key_values = [
        (torch.randn(1, 8, 4, 64), torch.randn(1, 8, 4, 64))  # 模拟KV cache
        for _ in range(12)  # 12层
    ]
    
    model.return_value = mock_output
    model.eval.return_value = None
    model.device = "cpu"
    
    return model

@pytest.fixture
def sample_input_ids():
    """样本输入IDs fixture"""
    return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

@pytest.fixture
def sample_kv_cache():
    """样本KV缓存fixture"""
    return [
        (torch.randn(1, 8, 4, 64), torch.randn(1, 8, 4, 64))
        for _ in range(12)
    ]

@pytest.fixture
def sample_logits():
    """样本logits fixture"""
    return torch.randn(1, 5, 1000)

@pytest.fixture
def temp_dir():
    """临时目录fixture"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture(autouse=True)
def set_random_seed():
    """设置随机种子以确保测试可重现"""
    torch.manual_seed(42)
    
@pytest.fixture
def mock_model_path():
    """模拟模型路径"""
    return "/fake/model/path"
