import torch

from src.core.model_runner import AttentionOutput


def test_forward_attention_output_shape(qwen3_runner, kv_cache):
    config = qwen3_runner.get_config()
    batch, seq_len = 1, 6
    hidden = torch.randn(batch, seq_len, config.hidden_size, device="cuda", dtype=config.get_dtype())
    positions = torch.arange(seq_len, device="cuda").unsqueeze(0)

    attn_out = qwen3_runner.forward_attention(
        layer_idx=0,
        hidden_states=hidden,
        kv_cache=kv_cache,
        positions=positions,
        is_prefill=True,
    )

    assert isinstance(attn_out, AttentionOutput)
    assert attn_out.hidden_states.shape == hidden.shape
    assert attn_out.post_attn_normed.shape == hidden.shape
    assert attn_out.residual.shape == hidden.shape


def test_forward_attention_residual(qwen3_runner, kv_cache):
    config = qwen3_runner.get_config()
    hidden = torch.randn(1, 4, config.hidden_size, device="cuda", dtype=config.get_dtype())
    positions = torch.arange(4, device="cuda").unsqueeze(0)

    attn_out = qwen3_runner.forward_attention(
        layer_idx=0,
        hidden_states=hidden,
        kv_cache=kv_cache,
        positions=positions,
        is_prefill=True,
    )

    assert torch.allclose(attn_out.residual, attn_out.hidden_states)
