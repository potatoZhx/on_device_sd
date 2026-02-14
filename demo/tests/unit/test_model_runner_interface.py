import torch

from src.core.model_runner import ModelRunner
from src.core.model import MoEConfig


def test_is_subclass(qwen3_runner):
    assert isinstance(qwen3_runner, ModelRunner)


def test_get_config(qwen3_runner):
    config = qwen3_runner.get_config()
    assert isinstance(config, MoEConfig)
    assert config.num_experts > 0
    assert config.num_hidden_layers > 0


def test_get_num_layers(qwen3_runner):
    assert qwen3_runner.get_num_layers() == qwen3_runner.get_config().num_hidden_layers


def test_embed_output_shape(qwen3_runner):
    config = qwen3_runner.get_config()
    batch, seq_len = 2, 8
    input_ids = torch.randint(0, 100, (batch, seq_len), device="cuda")
    output = qwen3_runner.embed(input_ids)
    assert output.shape == (batch, seq_len, config.hidden_size)


def test_compute_logits_output_shape(qwen3_runner):
    config = qwen3_runner.get_config()
    batch, seq_len = 2, 8
    hidden = torch.randn(batch, seq_len, config.hidden_size, device="cuda", dtype=config.get_dtype())
    logits = qwen3_runner.compute_logits(hidden)
    assert logits.shape == (batch, seq_len, config.vocab_size)
