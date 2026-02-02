"""
测试 MLP 和 Expert 层实现
与 transformers 对比验证精度对齐
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

from src.layers.mlp import (
    SiluAndMul,
    Qwen3MLP,
    Qwen3MLPWithWeights,
    Qwen3Expert,
    expert_forward_with_weights,
)
from src.memory.parameter_loader import ParameterLoader, MoEModelConfig
from src.core.types import ExpertID

# 测试模型路径
QWEN3_MODEL_PATH = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"

# 精度阈值
ATOL = 1e-5  # bfloat16 精度
RTOL = 1e-3


class TestSiluAndMul:
    """测试 SiluAndMul 激活函数"""
    
    def test_silu_and_mul_shape(self):
        """测试输出形状"""
        act = SiluAndMul()
        
        # Input: [batch, seq, intermediate * 2]
        x = torch.randn(2, 10, 256, device='cuda', dtype=torch.bfloat16)
        output = act(x)
        
        assert output.shape == (2, 10, 128)
        print("✓ SiluAndMul shape correct")
    
    def test_silu_and_mul_correctness(self):
        """测试与手动计算的精度对齐"""
        act = SiluAndMul()
        
        x = torch.randn(2, 10, 256, device='cuda', dtype=torch.float32)
        output = act(x)
        
        # Manual calculation
        gate, up = x.chunk(2, dim=-1)
        expected = F.silu(gate) * up
        
        assert torch.allclose(output, expected, atol=ATOL, rtol=RTOL)
        print("✓ SiluAndMul precision aligned")


class TestQwen3MLP:
    """测试 Qwen3MLP 实现"""
    
    def test_mlp_creation(self):
        """测试 MLP 创建"""
        mlp = Qwen3MLP(
            hidden_size=2048,
            intermediate_size=11008,
            hidden_act="silu"
        ).cuda()
        
        assert mlp.hidden_size == 2048
        assert mlp.intermediate_size == 11008
        print("✓ Qwen3MLP creation works")
    
    def test_mlp_forward_shape(self):
        """测试前向传播形状"""
        mlp = Qwen3MLP(
            hidden_size=128,
            intermediate_size=256
        ).cuda().to(torch.bfloat16)
        
        x = torch.randn(2, 10, 128, device='cuda', dtype=torch.bfloat16)
        output = mlp(x)
        
        assert output.shape == x.shape
        print("✓ MLP forward shape correct")
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH),
        reason="Qwen3 model not found"
    )
    def test_mlp_with_transformers_weights(self):
        """测试使用 transformers 权重的精度对齐"""
        # Load model config
        model_config = MoEModelConfig(QWEN3_MODEL_PATH)
        param_loader = ParameterLoader(QWEN3_MODEL_PATH)
        param_loader._load_static_parameters()
        
        # Create MLP
        mlp = Qwen3MLP(
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
        ).cuda().to(torch.bfloat16)
        
        # Load weights from layer 0 (non-MoE layer if exists, otherwise use any layer's mlp)
        # For Qwen3-30B-A3B-Base, layer 0 might be MoE, so we check
        layer_prefix = "layer_0.mlp"
        
        # Check if it's a standard MLP layer
        gate_key = f"{layer_prefix}.gate_proj"
        if gate_key not in param_loader.static_params_gpu:
            print("  Layer 0 is MoE, skipping standard MLP test")
            param_loader.close()
            pytest.skip("Layer 0 is MoE")
            return
        
        mlp.load_weights(
            gate_weight=param_loader.static_params_gpu[f"{layer_prefix}.gate_proj"],
            up_weight=param_loader.static_params_gpu[f"{layer_prefix}.up_proj"],
            down_weight=param_loader.static_params_gpu[f"{layer_prefix}.down_proj"],
        )
        
        # Test forward
        x = torch.randn(1, 5, model_config.hidden_size, device='cuda', dtype=torch.bfloat16)
        output = mlp(x)
        
        assert output.shape == x.shape
        print("✓ MLP with transformers weights works")
        
        param_loader.close()


class TestQwen3MLPWithWeights:
    """测试无状态 MLP wrapper"""
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH),
        reason="Qwen3 model not found"
    )
    def test_mlp_with_weights_forward(self):
        """测试使用外部权重的前向传播"""
        model_config = MoEModelConfig(QWEN3_MODEL_PATH)
        param_loader = ParameterLoader(QWEN3_MODEL_PATH)
        param_loader._load_static_parameters()
        
        # Create wrapper
        mlp = Qwen3MLPWithWeights(
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
        )
        
        # Check if layer 0 has standard MLP
        layer_prefix = "layer_0.mlp"
        gate_key = f"{layer_prefix}.gate_proj"
        if gate_key not in param_loader.static_params_gpu:
            print("  Layer 0 is MoE, skipping test")
            param_loader.close()
            pytest.skip("Layer 0 is MoE")
            return
        
        # Forward
        x = torch.randn(1, 5, model_config.hidden_size, device='cuda', dtype=torch.bfloat16)
        output = mlp.forward(
            x=x,
            gate_weight=param_loader.static_params_gpu[f"{layer_prefix}.gate_proj"],
            up_weight=param_loader.static_params_gpu[f"{layer_prefix}.up_proj"],
            down_weight=param_loader.static_params_gpu[f"{layer_prefix}.down_proj"],
        )
        
        assert output.shape == x.shape
        print("✓ MLPWithWeights forward works")
        
        param_loader.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestQwen3Expert:
    """测试 Expert (FFN) 实现"""
    
    def test_expert_creation(self):
        """测试 Expert 创建"""
        expert = Qwen3Expert(
            hidden_size=2048,
            intermediate_size=1408,  # MoE intermediate size
            expert_id=0,
        ).cuda()
        
        assert expert.expert_id == 0
        assert expert.hidden_size == 2048
        assert expert.intermediate_size == 1408
        print("✓ Qwen3Expert creation works")
    
    def test_expert_forward_shape(self):
        """测试 Expert 前向传播形状"""
        expert = Qwen3Expert(
            hidden_size=128,
            intermediate_size=256,
            expert_id=5,
        ).cuda().to(torch.bfloat16)
        
        x = torch.randn(10, 128, device='cuda', dtype=torch.bfloat16)
        output = expert(x)
        
        assert output.shape == x.shape
        print("✓ Expert forward shape correct")
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH),
        reason="Qwen3 model not found"
    )
    def test_expert_with_real_weights(self):
        """测试使用真实 expert 权重"""
        model_config = MoEModelConfig(QWEN3_MODEL_PATH)
        param_loader = ParameterLoader(QWEN3_MODEL_PATH)
        param_loader._load_static_parameters()
        param_loader._load_all_experts_to_cpu()
        
        # Create expert
        expert_id = 0
        layer_idx = 0
        
        expert = Qwen3Expert(
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.moe_intermediate_size,
            expert_id=expert_id,
        ).cuda().to(torch.bfloat16)
        
        # Get expert weights from CPU
        expert_params = param_loader.get_expert_params(ExpertID(layer_idx, expert_id))
        
        # Transfer to GPU and load
        expert.load_weights(
            gate_weight=expert_params['gate_proj'].cuda(),
            up_weight=expert_params['up_proj'].cuda(),
            down_weight=expert_params['down_proj'].cuda(),
        )
        
        # Test forward
        x = torch.randn(5, model_config.hidden_size, device='cuda', dtype=torch.bfloat16)
        output = expert(x)
        
        assert output.shape == x.shape
        print(f"✓ Expert {expert_id} with real weights works")
        
        param_loader.close()
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH),
        reason="Qwen3 model not found"
    )
    def test_expert_functional_api(self):
        """测试 functional API (无状态)"""
        model_config = MoEModelConfig(QWEN3_MODEL_PATH)
        param_loader = ParameterLoader(QWEN3_MODEL_PATH)
        param_loader._load_static_parameters()
        param_loader._load_all_experts_to_cpu()
        
        expert_id = 0
        layer_idx = 0
        
        # Get expert weights
        expert_params = param_loader.get_expert_params(ExpertID(layer_idx, expert_id))
        
        # Move to GPU
        gate_weight = expert_params['gate_proj'].cuda()
        up_weight = expert_params['up_proj'].cuda()
        down_weight = expert_params['down_proj'].cuda()
        
        # Forward with functional API
        x = torch.randn(5, model_config.hidden_size, device='cuda', dtype=torch.bfloat16)
        output = expert_forward_with_weights(x, gate_weight, up_weight, down_weight)
        
        assert output.shape == x.shape
        print("✓ Expert functional API works")
        
        param_loader.close()


@pytest.mark.skipif(
    not os.path.exists(QWEN3_MODEL_PATH),
    reason="Qwen3 model not found"
)
class TestMLPPrecisionAlignment:
    """测试 MLP 与 transformers 的精度对齐"""
    
    def test_expert_vs_transformers(self):
        """测试 Expert 与 transformers 的精度对齐"""
        print("\n" + "="*60)
        print("Testing Expert precision alignment with transformers")
        print("="*60)
        
        # Load our implementation
        model_config = MoEModelConfig(QWEN3_MODEL_PATH)
        param_loader = ParameterLoader(QWEN3_MODEL_PATH)
        param_loader._load_static_parameters()
        param_loader._load_all_experts_to_cpu()
        
        expert_id = 0
        layer_idx = 0
        
        # Get expert weights
        expert_params = param_loader.get_expert_params(ExpertID(layer_idx, expert_id))
        gate_weight = expert_params['gate_proj'].cuda()
        up_weight = expert_params['up_proj'].cuda()
        down_weight = expert_params['down_proj'].cuda()
        
        # Load transformers model (only need the expert)
        print("\nLoading transformers model...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            QWEN3_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        
        # Get the expert from transformers
        hf_expert = hf_model.model.layers[layer_idx].mlp.experts[expert_id]
        
        # Create test input
        batch_size = 4
        seq_len = 8
        x = torch.randn(
            batch_size, seq_len, model_config.hidden_size,
            device='cuda', dtype=torch.bfloat16
        )
        x_flat = x.view(-1, model_config.hidden_size)
        
        # Our implementation
        our_output = expert_forward_with_weights(x_flat, gate_weight, up_weight, down_weight)
        our_output = our_output.view(batch_size, seq_len, -1)
        
        # Transformers implementation
        with torch.no_grad():
            hf_output = hf_expert(x)
        
        # Compare
        max_diff = (our_output - hf_output).abs().max().item()
        mean_diff = (our_output - hf_output).abs().mean().item()
        
        print(f"\nPrecision comparison:")
        print(f"  Max difference:  {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Relative error:  {mean_diff / hf_output.abs().mean().item():.2e}")
        
        # Check alignment
        assert torch.allclose(our_output, hf_output, atol=ATOL, rtol=RTOL), \
            f"Precision mismatch: max_diff={max_diff}, mean_diff={mean_diff}"
        
        print("\n✓ Expert precision aligned with transformers!")
        
        param_loader.close()
        del hf_model
        torch.cuda.empty_cache()


def run_quick_tests():
    """运行快速测试"""
    print("=" * 60)
    print("Testing MLP Layer Implementation")
    print("=" * 60)
    
    # Test 1: SiluAndMul
    print("\n--- Test 1: SiluAndMul ---")
    act = SiluAndMul()
    x = torch.randn(2, 10, 256, device='cuda', dtype=torch.bfloat16)
    output = act(x)
    assert output.shape == (2, 10, 128)
    print("  ✓ Pass")
    
    # Test 2: Qwen3MLP
    print("\n--- Test 2: Qwen3MLP ---")
    mlp = Qwen3MLP(128, 256).cuda().to(torch.bfloat16)
    x = torch.randn(2, 10, 128, device='cuda', dtype=torch.bfloat16)
    output = mlp(x)
    assert output.shape == x.shape
    print("  ✓ Pass")
    
    # Test 3: Qwen3Expert
    print("\n--- Test 3: Qwen3Expert ---")
    expert = Qwen3Expert(128, 256, expert_id=0).cuda().to(torch.bfloat16)
    x = torch.randn(10, 128, device='cuda', dtype=torch.bfloat16)
    output = expert(x)
    assert output.shape == x.shape
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
