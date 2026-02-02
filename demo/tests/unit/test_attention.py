"""
测试 Attention 层实现
与 transformers 对比验证正确性
"""

import os
import sys
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig

# 添加项目路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.core.model import MoEConfig
from src.layers.rotary_embedding import RotaryEmbedding, get_rope, apply_rotary_emb
from src.layers.layernorm import RMSNorm
from src.layers.attention import Qwen3Attention, Qwen3AttentionWithWeights
from src.memory.paged_kv_cache import PagedKVCache
from src.memory.parameter_loader import ParameterLoader, MoEModelConfig

# 测试模型路径
QWEN3_MODEL_PATH = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"


class TestRoPE:
    """测试 RoPE 实现"""
    
    def test_rope_creation(self):
        """测试 RoPE 创建"""
        rope = get_rope(
            head_size=128,
            rotary_dim=128,
            max_position=32768,
            base=1000000.0
        )
        
        assert rope.head_size == 128
        assert rope.cos_cache.shape[0] == 32768
        assert rope.sin_cache.shape[0] == 32768
        print("✓ RoPE creation works")
    
    def test_apply_rotary_emb(self):
        """测试 RoPE 应用"""
        rope = get_rope(128, 128, 1024, 10000.0)
        
        # Create dummy Q and K
        num_tokens = 10
        num_heads = 4
        head_dim = 128
        
        positions = torch.arange(num_tokens, device='cuda')
        q = torch.randn(num_tokens, num_heads, head_dim, device='cuda')
        k = torch.randn(num_tokens, num_heads, head_dim, device='cuda')
        
        q_rot, k_rot = rope(positions, q, k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert not torch.allclose(q, q_rot)  # Should be different
        
        print("✓ apply_rotary_emb works")


class TestRMSNorm:
    """测试 RMSNorm 实现"""
    
    def test_rmsnorm_forward(self):
        """测试 RMSNorm 前向传播"""
        hidden_size = 128
        norm = RMSNorm(hidden_size, eps=1e-6).cuda()
        
        x = torch.randn(2, 10, hidden_size, device='cuda')
        output = norm(x)
        
        assert output.shape == x.shape
        print("✓ RMSNorm forward works")
    
    def test_rmsnorm_with_residual(self):
        """测试 RMSNorm 带 residual"""
        hidden_size = 128
        norm = RMSNorm(hidden_size, eps=1e-6).cuda()
        
        x = torch.randn(2, 10, hidden_size, device='cuda')
        residual = torch.randn(2, 10, hidden_size, device='cuda')
        
        output, new_residual = norm(x, residual)
        
        assert output.shape == x.shape
        assert new_residual.shape == x.shape
        print("✓ RMSNorm with residual works")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestQwen3Attention:
    """测试 Qwen3Attention 层"""
    
    def test_attention_module_creation(self):
        """测试 Attention 模块创建"""
        attn = Qwen3Attention(
            hidden_size=2048,
            num_heads=32,
            num_kv_heads=4,
            head_dim=128,
            max_position_embeddings=32768,
            rms_norm_eps=1e-6,
            qkv_bias=False,
            rope_theta=1000000.0,
            layer_idx=0,
        ).cuda()
        
        assert attn.num_heads == 32
        assert attn.num_kv_heads == 4
        assert attn.head_dim == 128
        assert attn.q_norm is not None  # Should have QK norm
        assert attn.k_norm is not None
        
        print("✓ Qwen3Attention creation works")
    
    def test_attention_forward_shape(self):
        """测试 Attention 前向传播的输出形状"""
        config = MoEConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            num_experts=8,
            num_experts_per_token=2,
            vocab_size=1000
        )
        
        attn = Qwen3Attention(
            hidden_size=128,
            num_heads=4,
            num_kv_heads=2,
            head_dim=32,
            layer_idx=0,
        ).cuda().to(torch.bfloat16)
        
        # Create KV cache with bf16
        kv_cache = PagedKVCache(config=config, block_size=16, dtype=torch.bfloat16)
        kv_cache.add_sequence(seq_id=0, prompt_len=10)
        
        # Create dummy input (must be fp16 or bf16 for flash_attn)
        hidden_states = torch.randn(10, 128, device='cuda', dtype=torch.bfloat16)
        positions = torch.arange(10, device='cuda')
        
        # Forward
        output = attn(
            hidden_states=hidden_states,
            positions=positions,
            kv_cache=kv_cache,
            seq_ids=[0],
            is_prefill=True,
        )
        
        assert output.shape == (10, 128)
        print("✓ Attention forward shape correct")
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH),
        reason="Qwen3 model not found"
    )
    def test_attention_with_real_weights(self):
        """测试使用真实权重的 Attention"""
        # Load config and weights
        model_config = MoEModelConfig(QWEN3_MODEL_PATH)
        param_loader = ParameterLoader(QWEN3_MODEL_PATH)
        param_loader._load_static_parameters()
        
        # Create attention layer (convert to bfloat16 to match Qwen3)
        attn = Qwen3Attention(
            hidden_size=model_config.hidden_size,
            num_heads=model_config.num_attention_heads,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            max_position_embeddings=model_config.max_position_embeddings,
            rms_norm_eps=model_config.rms_norm_eps,
            qkv_bias=False,
            rope_theta=model_config.rope_theta,
            layer_idx=0,
        ).cuda().to(torch.bfloat16)
        
        # Load weights
        layer_prefix = "layer_0.self_attn"
        attn.load_weights(
            q_weight=param_loader.static_params_gpu[f"{layer_prefix}.q_proj"],
            k_weight=param_loader.static_params_gpu[f"{layer_prefix}.k_proj"],
            v_weight=param_loader.static_params_gpu[f"{layer_prefix}.v_proj"],
            o_weight=param_loader.static_params_gpu[f"{layer_prefix}.o_proj"],
            q_norm_weight=param_loader.static_params_gpu.get(f"{layer_prefix}.q_norm"),
            k_norm_weight=param_loader.static_params_gpu.get(f"{layer_prefix}.k_norm"),
        )
        
        # Create KV cache (use bfloat16 to match Qwen3)
        config = MoEConfig.from_pretrained(QWEN3_MODEL_PATH)
        kv_cache = PagedKVCache(config=config, block_size=256, dtype=torch.bfloat16)
        
        # Add sequence
        seq_len = 20
        kv_cache.add_sequence(seq_id=0, prompt_len=seq_len)
        
        # Create dummy input
        hidden_states = torch.randn(seq_len, model_config.hidden_size, device='cuda', dtype=torch.bfloat16)
        positions = torch.arange(seq_len, device='cuda')
        
        # Forward
        output = attn(
            hidden_states=hidden_states,
            positions=positions,
            kv_cache=kv_cache,
            seq_ids=[0],
            is_prefill=True,
        )
        
        assert output.shape == (seq_len, model_config.hidden_size)
        assert output.dtype == torch.bfloat16
        
        print(f"✓ Attention with real weights works")
        print(f"  Input shape: {hidden_states.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output dtype: {output.dtype}")
        
        param_loader.close()


class TestQwen3AttentionWithWeights:
    """测试 Qwen3AttentionWithWeights（使用外部权重）"""
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH),
        reason="Qwen3 model not found"
    )
    def test_attention_with_weights_forward(self):
        """测试使用外部权重的 Attention 前向传播"""
        model_config = MoEModelConfig(QWEN3_MODEL_PATH)
        param_loader = ParameterLoader(QWEN3_MODEL_PATH)
        param_loader._load_static_parameters()
        
        # Create attention wrapper
        attn = Qwen3AttentionWithWeights(
            hidden_size=model_config.hidden_size,
            num_heads=model_config.num_attention_heads,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            max_position_embeddings=model_config.max_position_embeddings,
            rms_norm_eps=model_config.rms_norm_eps,
            qkv_bias=False,
            rope_theta=model_config.rope_theta,
            layer_idx=0,
        )
        
        # Create KV cache
        config = MoEConfig.from_pretrained(QWEN3_MODEL_PATH)
        kv_cache = PagedKVCache(config=config, block_size=256)
        
        seq_len = 20
        kv_cache.add_sequence(seq_id=0, prompt_len=seq_len)
        
        # Create input
        hidden_states = torch.randn(seq_len, model_config.hidden_size, device='cuda', dtype=torch.bfloat16)
        positions = torch.arange(seq_len, device='cuda')
        
        # Create KV cache (use bfloat16)
        config = MoEConfig.from_pretrained(QWEN3_MODEL_PATH)
        kv_cache = PagedKVCache(config=config, block_size=256, dtype=torch.bfloat16)
        kv_cache.add_sequence(seq_id=0, prompt_len=seq_len)
        
        # Get weights
        layer_prefix = "layer_0.self_attn"
        
        # Forward
        output = attn.forward(
            hidden_states=hidden_states,
            positions=positions,
            q_weight=param_loader.static_params_gpu[f"{layer_prefix}.q_proj"],
            k_weight=param_loader.static_params_gpu[f"{layer_prefix}.k_proj"],
            v_weight=param_loader.static_params_gpu[f"{layer_prefix}.v_proj"],
            o_weight=param_loader.static_params_gpu[f"{layer_prefix}.o_proj"],
            q_norm_weight=param_loader.static_params_gpu.get(f"{layer_prefix}.q_norm"),
            k_norm_weight=param_loader.static_params_gpu.get(f"{layer_prefix}.k_norm"),
            kv_cache=kv_cache,
            seq_ids=[0],
            is_prefill=True,
        )
        
        assert output.shape == (seq_len, model_config.hidden_size)
        print("✓ Qwen3AttentionWithWeights forward works")
        
        param_loader.close()


@pytest.mark.skipif(
    not os.path.exists(QWEN3_MODEL_PATH),
    reason="Qwen3 model not found"
)
class TestPrecisionAlignment:
    """测试所有算子与 transformers 的精度对齐"""
    
    def test_rmsnorm_vs_transformers(self):
        """测试 RMSNorm 与 transformers 的精度对齐"""
        print("\n" + "="*60)
        print("Testing RMSNorm precision alignment")
        print("="*60)
        
        # Load transformers model
        print("\nLoading transformers model...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            QWEN3_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        
        # Get config
        model_config = MoEModelConfig(QWEN3_MODEL_PATH)
        param_loader = ParameterLoader(QWEN3_MODEL_PATH)
        param_loader._load_static_parameters()
        
        # Create our RMSNorm
        our_norm = RMSNorm(
            hidden_size=model_config.hidden_size,
            eps=model_config.rms_norm_eps,
        ).cuda().to(torch.bfloat16)
        
        # Load weight from layer 0 input_layernorm
        our_norm.weight.data = param_loader.static_params_gpu["layer_0.input_layernorm"]
        
        # Get transformers norm
        hf_norm = hf_model.model.layers[0].input_layernorm
        
        # Test input
        batch_size = 2
        seq_len = 10
        x = torch.randn(
            batch_size, seq_len, model_config.hidden_size,
            device='cuda', dtype=torch.bfloat16
        )
        
        # Forward
        our_output = our_norm(x)
        
        with torch.no_grad():
            hf_output = hf_norm(x)
        
        # Compare
        max_diff = (our_output - hf_output).abs().max().item()
        mean_diff = (our_output - hf_output).abs().mean().item()
        
        print(f"\nRMSNorm precision comparison:")
        print(f"  Max difference:  {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Relative error:  {mean_diff / hf_output.abs().mean().item():.2e}")
        
        # For bfloat16, we expect very close alignment
        assert max_diff < 1e-3, f"RMSNorm precision mismatch: max_diff={max_diff}"
        
        print("\n✓ RMSNorm precision aligned with transformers!")
        
        param_loader.close()
        del hf_model
        torch.cuda.empty_cache()
    
    def test_rope_vs_transformers(self):
        """测试 RoPE 与 transformers 的精度对齐"""
        print("\n" + "="*60)
        print("Testing RoPE precision alignment")
        print("="*60)
        
        # Get config
        model_config = MoEModelConfig(QWEN3_MODEL_PATH)
        
        # Create our RoPE
        our_rope = get_rope(
            head_size=model_config.head_dim,
            rotary_dim=model_config.head_dim,
            max_position=model_config.max_position_embeddings,
            base=model_config.rope_theta,
        )
        our_rope = our_rope.cuda()
        
        # Import transformers RoPE implementation
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding, apply_rotary_pos_emb
        from transformers import AutoConfig
        
        # Load transformers config
        hf_config = AutoConfig.from_pretrained(QWEN3_MODEL_PATH)
        
        # Create transformers RoPE
        hf_rope = Qwen3MoeRotaryEmbedding(config=hf_config, device='cuda')
        
        # Test input
        num_tokens = 10
        num_heads = model_config.num_attention_heads
        head_dim = model_config.head_dim
        
        positions = torch.arange(num_tokens, device='cuda')
        q = torch.randn(num_tokens, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(num_tokens, model_config.num_key_value_heads, head_dim, device='cuda', dtype=torch.bfloat16)
        
        # Our implementation
        q_our, k_our = our_rope(positions, q, k)
        
        # Transformers implementation
        # Note: transformers uses different shape convention [batch, heads, seq_len, head_dim]
        q_hf = q.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]
        k_hf = k.transpose(0, 1).unsqueeze(0)  # [1, num_kv_heads, seq_len, head_dim]
        
        with torch.no_grad():
            cos, sin = hf_rope(k_hf, position_ids=positions.unsqueeze(0))
            # Apply RoPE
            q_hf, k_hf = apply_rotary_pos_emb(q_hf, k_hf, cos, sin)
        
        # Convert back to our shape
        q_hf = q_hf.squeeze(0).transpose(0, 1)  # [seq_len, num_heads, head_dim]
        k_hf = k_hf.squeeze(0).transpose(0, 1)  # [seq_len, num_kv_heads, head_dim]
        
        # Compare
        q_max_diff = (q_our - q_hf).abs().max().item()
        k_max_diff = (k_our - k_hf).abs().max().item()
        q_mean_diff = (q_our - q_hf).abs().mean().item()
        k_mean_diff = (k_our - k_hf).abs().mean().item()
        
        print(f"\nRoPE precision comparison:")
        print(f"  Q max difference:  {q_max_diff:.2e}")
        print(f"  Q mean difference: {q_mean_diff:.2e}")
        print(f"  K max difference:  {k_max_diff:.2e}")
        print(f"  K mean difference: {k_mean_diff:.2e}")
        print(f"  bfloat16 epsilon:  {torch.finfo(torch.bfloat16).eps:.2e}")
        
        # For bfloat16, max_diff up to 2*epsilon is acceptable
        # Mean diff should be much smaller
        bf16_eps = torch.finfo(torch.bfloat16).eps
        assert q_max_diff < 3 * bf16_eps, f"RoPE Q precision mismatch: max_diff={q_max_diff}"
        assert k_max_diff < 3 * bf16_eps, f"RoPE K precision mismatch: max_diff={k_max_diff}"
        assert q_mean_diff < bf16_eps, f"RoPE Q mean diff too large: {q_mean_diff}"
        assert k_mean_diff < bf16_eps, f"RoPE K mean diff too large: {k_mean_diff}"
        
        print("\n✓ RoPE precision aligned with transformers!")
        
        torch.cuda.empty_cache()
    
    def test_attention_vs_transformers(self):
        """测试 Attention 层与 transformers 的精度对齐"""
        print("\n" + "="*60)
        print("Testing Attention precision alignment")
        print("="*60)
        
        # Load transformers model
        print("\nLoading transformers model...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            QWEN3_MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        
        # Get config
        model_config = MoEModelConfig(QWEN3_MODEL_PATH)
        param_loader = ParameterLoader(QWEN3_MODEL_PATH)
        param_loader._load_static_parameters()
        
        # Create our Attention
        our_attn = Qwen3Attention(
            hidden_size=model_config.hidden_size,
            num_heads=model_config.num_attention_heads,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            max_position_embeddings=model_config.max_position_embeddings,
            rms_norm_eps=model_config.rms_norm_eps,
            qkv_bias=False,
            rope_theta=model_config.rope_theta,
            layer_idx=0,
        ).cuda().to(torch.bfloat16)
        
        # Load weights from layer 0
        layer_prefix = "layer_0.self_attn"
        our_attn.load_weights(
            q_weight=param_loader.static_params_gpu[f"{layer_prefix}.q_proj"],
            k_weight=param_loader.static_params_gpu[f"{layer_prefix}.k_proj"],
            v_weight=param_loader.static_params_gpu[f"{layer_prefix}.v_proj"],
            o_weight=param_loader.static_params_gpu[f"{layer_prefix}.o_proj"],
            q_norm_weight=param_loader.static_params_gpu.get(f"{layer_prefix}.q_norm"),
            k_norm_weight=param_loader.static_params_gpu.get(f"{layer_prefix}.k_norm"),
        )
        
        # Get transformers attention
        hf_attn = hf_model.model.layers[0].self_attn
        
        # Create test input
        batch_size = 2
        seq_len = 16
        hidden_states = torch.randn(
            batch_size, seq_len, model_config.hidden_size,
            device='cuda', dtype=torch.bfloat16
        )
        
        # Transformers forward (without KV cache for simplicity)
        print("\nRunning transformers forward...")
        with torch.no_grad():
            position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
            
            # Get position embeddings from rotary_emb
            from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(QWEN3_MODEL_PATH)
            rope = Qwen3MoeRotaryEmbedding(config=hf_config, device='cuda')
            
            # Dummy input for rope
            dummy_states = torch.zeros(batch_size, seq_len, model_config.head_dim, device='cuda', dtype=torch.bfloat16)
            position_embeddings = rope(dummy_states, position_ids=position_ids)
            
            # No attention_mask means causal attention
            hf_result = hf_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,  # Causal by default
                output_attentions=False,
                use_cache=False,
            )
            # Handle different return formats
            if isinstance(hf_result, tuple):
                hf_output = hf_result[0]
            else:
                hf_output = hf_result
        
        # Our forward (with KV cache)
        print("Running our forward...")
        from src.core.model import MoEConfig
        config = MoEConfig.from_pretrained(QWEN3_MODEL_PATH)
        kv_cache = PagedKVCache(config=config, block_size=256, dtype=torch.bfloat16)
        
        # Add sequences
        for i in range(batch_size):
            kv_cache.add_sequence(seq_id=i, prompt_len=seq_len)
        
        # Flatten input for our implementation
        hidden_states_flat = hidden_states.view(-1, model_config.hidden_size)
        positions_flat = torch.arange(seq_len, device='cuda').repeat(batch_size)
        
        our_output_flat = our_attn(
            hidden_states=hidden_states_flat,
            positions=positions_flat,
            kv_cache=kv_cache,
            seq_ids=list(range(batch_size)),
            is_prefill=True,
        )
        
        # Reshape our output
        our_output = our_output_flat.view(batch_size, seq_len, model_config.hidden_size)
        
        # Compare
        max_diff = (our_output - hf_output).abs().max().item()
        mean_diff = (our_output - hf_output).abs().mean().item()
        rel_error = (mean_diff / hf_output.abs().mean().item())
        
        print(f"\nAttention precision comparison:")
        print(f"  Max difference:  {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Relative error:  {rel_error:.2e}")
        print(f"  bfloat16 epsilon: {torch.finfo(torch.bfloat16).eps:.2e}")
        
        # For attention, we expect slightly larger differences due to:
        # 1. Flash attention numerical differences
        # 2. Accumulated errors through multiple operations
        bf16_eps = torch.finfo(torch.bfloat16).eps
        assert max_diff < 0.5, f"Attention max diff too large: {max_diff}"
        assert mean_diff < 0.01, f"Attention mean diff too large: {mean_diff}"
        assert rel_error < 0.05, f"Attention relative error too large: {rel_error}"
        
        print("\n✓ Attention precision aligned with transformers!")
        
        param_loader.close()
        del hf_model
        torch.cuda.empty_cache()


def run_quick_tests():
    """运行快速测试"""
    print("=" * 60)
    print("Testing Attention Layer Implementation")
    print("=" * 60)
    
    # Test 1: RoPE
    print("\n--- Test 1: RoPE ---")
    rope = get_rope(128, 128, 1024, 10000.0)
    positions = torch.arange(10, device='cuda')
    q = torch.randn(10, 4, 128, device='cuda')
    k = torch.randn(10, 4, 128, device='cuda')
    q_rot, k_rot = rope(positions, q, k)
    assert q_rot.shape == q.shape
    print("  ✓ Pass")
    
    # Test 2: RMSNorm
    print("\n--- Test 2: RMSNorm ---")
    norm = RMSNorm(128, eps=1e-6).cuda()
    x = torch.randn(2, 10, 128, device='cuda')
    output = norm(x)
    assert output.shape == x.shape
    print("  ✓ Pass")
    
    if not torch.cuda.is_available():
        print("\n⚠ CUDA not available, skipping GPU tests")
        return
    
    # Test 3: Attention Module
    print("\n--- Test 3: Qwen3Attention ---")
    attn = Qwen3Attention(
        hidden_size=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        layer_idx=0,
    ).cuda()
    print(f"  num_heads: {attn.num_heads}")
    print(f"  num_kv_heads: {attn.num_kv_heads}")
    print(f"  head_dim: {attn.head_dim}")
    print("  ✓ Pass")
    
    print("\n" + "=" * 60)
    print("All quick tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run full pytest")
    args = parser.parse_args()
    
    if args.full:
        pytest.main([__file__, "-v", "-s"])
    else:
        run_quick_tests()
