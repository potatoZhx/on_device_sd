"""
测试参数加载器 - 针对 Qwen3-30B-A3B-Base 模型
"""

import os
import sys
import pytest
import torch

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# 使用绝对导入
from src.core.model import MoEConfig
from src.core.types import ExpertID, DeviceType, ExpertLocation
from src.memory.parameter_loader import (
    MoEModelConfig, 
    SafetensorsWeightLoader, 
    ParameterLoader
)


# 测试模型路径
QWEN3_MODEL_PATH = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"


class TestMoEModelConfig:
    """测试 MoEModelConfig 从 HuggingFace 配置加载"""
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH), 
        reason="Qwen3 model not found"
    )
    def test_load_qwen3_config(self):
        """测试加载 Qwen3-30B-A3B-Base 配置"""
        config = MoEModelConfig(QWEN3_MODEL_PATH)
        
        # 验证基本配置
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 48
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 4
        assert config.head_dim == 128
        assert config.vocab_size == 151936
        
        # 验证 MoE 配置
        assert config.num_experts == 128
        assert config.num_experts_per_token == 8
        assert config.moe_intermediate_size == 768
        assert config.is_moe == True
        
        # Qwen3-30B-A3B-Base 没有 shared experts
        assert config.num_shared_experts == 0
        
        print(f"✓ MoE Config loaded: {config.num_hidden_layers} layers, "
              f"{config.num_experts} experts, top-{config.num_experts_per_token}, "
              f"shared_experts={config.num_shared_experts}")
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH), 
        reason="Qwen3 model not found"
    )
    def test_expert_weight_size(self):
        """测试计算 expert 权重大小"""
        config = MoEModelConfig(QWEN3_MODEL_PATH)
        
        # 计算预期大小
        # gate_proj: [768, 2048], up_proj: [768, 2048], down_proj: [2048, 768]
        # 共 3 * 768 * 2048 = 4,718,592 参数
        # bfloat16: 2 bytes per param
        expected_bytes = 3 * 768 * 2048 * 2
        
        actual_bytes = config.get_expert_weight_size_bytes()
        assert actual_bytes == expected_bytes, f"Expected {expected_bytes}, got {actual_bytes}"
        
        print(f"✓ Expert weight size: {actual_bytes / (1024*1024):.2f} MB")


class TestMoEConfigFromPretrained:
    """测试 MoEConfig.from_pretrained"""
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH), 
        reason="Qwen3 model not found"
    )
    def test_from_pretrained(self):
        """测试从预训练模型路径加载配置"""
        config = MoEConfig.from_pretrained(QWEN3_MODEL_PATH)
        
        # 验证配置
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 48
        assert config.num_experts == 128
        assert config.num_experts_per_token == 8
        assert config.is_moe == True
        assert config.model_type == "qwen3_moe"
        
        print(f"✓ MoEConfig.from_pretrained: {config.model_type}")


class TestSafetensorsWeightLoader:
    """测试 SafetensorsWeightLoader"""
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH), 
        reason="Qwen3 model not found"
    )
    def test_load_weight_map(self):
        """测试加载权重映射"""
        loader = SafetensorsWeightLoader(QWEN3_MODEL_PATH)
        
        # 验证权重映射已加载
        assert len(loader.weight_map) > 0
        print(f"✓ Weight map loaded with {len(loader.weight_map)} entries")
        
        # 验证一些关键权重存在
        assert loader.has_weight("model.embed_tokens.weight")
        assert loader.has_weight("model.norm.weight")
        assert loader.has_weight("lm_head.weight")
        assert loader.has_weight("model.layers.0.self_attn.q_proj.weight")
        assert loader.has_weight("model.layers.0.mlp.experts.0.gate_proj.weight")
        assert loader.has_weight("model.layers.0.mlp.gate.weight")  # router
        
        loader.close()
        print("✓ All expected weights found")
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH), 
        reason="Qwen3 model not found"
    )
    def test_get_tensor(self):
        """测试加载单个张量"""
        loader = SafetensorsWeightLoader(QWEN3_MODEL_PATH)
        
        # 加载 embedding
        embed_weight = loader.get_tensor("model.embed_tokens.weight")
        assert embed_weight.shape == (151936, 2048)  # vocab_size x hidden_size
        print(f"✓ Embedding weight shape: {embed_weight.shape}")
        
        # 加载 expert 权重
        expert_weight = loader.get_tensor("model.layers.0.mlp.experts.0.gate_proj.weight")
        assert expert_weight.shape == (768, 2048)  # moe_intermediate_size x hidden_size
        print(f"✓ Expert gate_proj shape: {expert_weight.shape}")
        
        # 加载 router 权重
        router_weight = loader.get_tensor("model.layers.0.mlp.gate.weight")
        assert router_weight.shape == (128, 2048)  # num_experts x hidden_size
        print(f"✓ Router weight shape: {router_weight.shape}")
        
        loader.close()
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH), 
        reason="Qwen3 model not found"
    )
    def test_get_weight_names_with_pattern(self):
        """测试按模式过滤权重名称"""
        loader = SafetensorsWeightLoader(QWEN3_MODEL_PATH)
        
        # 获取所有 layer 0 expert 0 的权重
        expert_weights = loader.get_weight_names(r"layers\.0\.mlp\.experts\.0\.")
        assert len(expert_weights) == 3  # gate_proj, up_proj, down_proj
        print(f"✓ Expert 0 weights: {expert_weights}")
        
        # 获取所有 router 权重
        router_weights = loader.get_weight_names(r"mlp\.gate\.weight")
        assert len(router_weights) == 48  # 48 layers
        print(f"✓ Router weights count: {len(router_weights)}")
        
        loader.close()


class TestParameterLoaderStaticParams:
    """测试 ParameterLoader 的静态参数加载"""
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH), 
        reason="Qwen3 model not found"
    )
    def test_load_static_parameters(self):
        """测试加载静态参数"""
        loader = ParameterLoader(QWEN3_MODEL_PATH)
        
        # 只加载静态参数（不加载 experts）
        loader._load_static_parameters()
        
        # 验证关键静态参数已加载
        assert 'embed_tokens' in loader.static_params_gpu
        assert 'final_layernorm' in loader.static_params_gpu
        assert 'lm_head' in loader.static_params_gpu
        
        # 验证 attention 参数
        assert 'layer_0.self_attn.q_proj' in loader.static_params_gpu
        assert 'layer_0.self_attn.k_proj' in loader.static_params_gpu
        assert 'layer_0.self_attn.v_proj' in loader.static_params_gpu
        assert 'layer_0.self_attn.o_proj' in loader.static_params_gpu
        
        # 验证 layer norms
        assert 'layer_0.input_layernorm' in loader.static_params_gpu
        assert 'layer_0.post_attention_layernorm' in loader.static_params_gpu
        
        # 验证 router
        assert 'layer_0.router' in loader.static_params_gpu
        
        # 验证参数在 GPU 上
        assert loader.static_params_gpu['embed_tokens'].device.type == 'cuda'
        
        print(f"✓ Loaded {len(loader.static_params_gpu)} static parameters to GPU")
        
        loader.close()


class TestParameterLoaderSharedExperts:
    """测试 ParameterLoader 的 shared experts 加载"""
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH), 
        reason="Qwen3 model not found"
    )
    def test_shared_experts_loading_qwen3(self):
        """测试 Qwen3 模型的 shared experts 加载（该模型无 shared experts）"""
        loader = ParameterLoader(QWEN3_MODEL_PATH)
        
        # 加载静态参数和 shared experts
        loader._load_static_parameters()
        
        # Qwen3-30B-A3B-Base 没有 shared experts
        assert len(loader.shared_experts_gpu) == 0
        assert len(loader.shared_expert_ids) == 0
        
        print(f"✓ No shared experts for Qwen3-30B-A3B-Base (as expected)")
        print(f"  shared_experts_gpu: {len(loader.shared_experts_gpu)}")
        print(f"  shared_expert_ids: {len(loader.shared_expert_ids)}")
        
        loader.close()
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH), 
        reason="Qwen3 model not found"
    )
    def test_is_shared_expert(self):
        """测试 is_shared_expert 方法"""
        loader = ParameterLoader(QWEN3_MODEL_PATH)
        loader._load_static_parameters()
        
        # Qwen3 所有 experts 都是 routed experts
        expert_id = ExpertID(0, 0)
        assert not loader.is_shared_expert(expert_id)
        
        expert_id = ExpertID(0, 127)
        assert not loader.is_shared_expert(expert_id)
        
        print("✓ is_shared_expert works correctly")
        
        loader.close()
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH), 
        reason="Qwen3 model not found"
    )
    def test_get_all_shared_experts_for_layer(self):
        """测试获取指定层的所有 shared experts"""
        loader = ParameterLoader(QWEN3_MODEL_PATH)
        loader._load_static_parameters()
        
        # Qwen3 没有 shared experts
        shared_experts_layer0 = loader.get_all_shared_experts_for_layer(0)
        assert len(shared_experts_layer0) == 0
        
        print("✓ get_all_shared_experts_for_layer works correctly")
        
        loader.close()


class TestParameterLoaderRoutedExperts:
    """测试 ParameterLoader 的 routed experts 加载"""
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH), 
        reason="Qwen3 model not found"
    )
    def test_load_single_expert(self):
        """测试加载单个 expert"""
        loader = ParameterLoader(QWEN3_MODEL_PATH)
        
        # 加载 layer 0, expert 0
        expert_weights = loader._load_single_expert(0, 0, device='cpu')
        
        assert expert_weights is not None
        assert 'gate_proj' in expert_weights
        assert 'up_proj' in expert_weights
        assert 'down_proj' in expert_weights
        
        # 验证形状
        assert expert_weights['gate_proj'].shape == (768, 2048)
        assert expert_weights['up_proj'].shape == (768, 2048)
        assert expert_weights['down_proj'].shape == (2048, 768)
        
        # 验证在 CPU 上
        assert expert_weights['gate_proj'].device.type == 'cpu'
        
        print(f"✓ Expert weights loaded with correct shapes")
        
        loader.close()
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH), 
        reason="Qwen3 model not found"
    )
    def test_load_experts_to_cpu_skips_shared(self):
        """测试加载 experts 到 CPU 时跳过 shared experts"""
        loader = ParameterLoader(QWEN3_MODEL_PATH)
        
        # 先加载静态参数和 shared experts
        loader._load_static_parameters()
        
        # 临时修改配置只加载前 2 层
        original_layers = loader.config.num_hidden_layers
        loader.config.num_hidden_layers = 2
        
        try:
            loader._load_all_experts_to_cpu()
            
            # Qwen3 没有 shared experts，所以所有 128 * 2 = 256 个 experts 都应该加载到 CPU
            expected_experts = 2 * 128  # 2 layers x 128 experts
            assert len(loader.expert_params_cpu) == expected_experts
            
            # 验证没有 shared experts 被跳过
            # 因为 Qwen3 没有 shared experts，所以不应该跳过任何
            assert len(loader.shared_expert_ids) == 0
            
            print(f"✓ Loaded {len(loader.expert_params_cpu)} routed experts to CPU")
            print(f"  Shared experts skipped: {len(loader.shared_expert_ids)}")
            
        finally:
            loader.config.num_hidden_layers = original_layers
            loader.close()


class TestParameterLoaderExpertOperations:
    """测试 ParameterLoader 的 expert 操作"""
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH), 
        reason="Qwen3 model not found"
    )
    def test_expert_gpu_operations(self):
        """测试 expert 的 GPU 加载和驱逐操作"""
        loader = ParameterLoader(QWEN3_MODEL_PATH)
        loader._load_static_parameters()
        
        # 加载单个 expert 到 CPU
        expert_id = ExpertID(0, 0)
        expert_weights = loader._load_single_expert(0, 0, device='cpu')
        loader.expert_params_cpu[expert_id] = expert_weights
        loader.expert_locations[expert_id] = ExpertLocation(
            expert_id=expert_id,
            device=DeviceType.CPU,
            is_cached=False
        )
        
        # 测试加载到 GPU
        success = loader.load_expert_to_gpu(expert_id)
        assert success
        assert expert_id in loader.expert_params_gpu
        assert loader.expert_locations[expert_id].device == DeviceType.GPU
        assert loader.expert_params_gpu[expert_id]['gate_proj'].device.type == 'cuda'
        print("✓ Expert loaded to GPU")
        
        # 测试从 GPU 驱逐
        success = loader.evict_expert_from_gpu(expert_id)
        assert success
        assert expert_id not in loader.expert_params_gpu
        assert loader.expert_locations[expert_id].device == DeviceType.CPU
        # CPU 副本仍然存在
        assert expert_id in loader.expert_params_cpu
        print("✓ Expert evicted from GPU (CPU copy retained)")
        
        loader.close()
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH), 
        reason="Qwen3 model not found"
    )
    def test_get_expert_params_priority(self):
        """测试 get_expert_params 的优先级"""
        loader = ParameterLoader(QWEN3_MODEL_PATH)
        loader._load_static_parameters()
        
        # 加载单个 expert 到 CPU
        expert_id = ExpertID(0, 0)
        expert_weights = loader._load_single_expert(0, 0, device='cpu')
        loader.expert_params_cpu[expert_id] = expert_weights
        loader.expert_locations[expert_id] = ExpertLocation(
            expert_id=expert_id,
            device=DeviceType.CPU,
            is_cached=False
        )
        
        # 获取参数（应该返回 CPU 版本）
        params = loader.get_expert_params(expert_id)
        assert params is not None
        assert params['gate_proj'].device.type == 'cpu'
        
        # 加载到 GPU
        loader.load_expert_to_gpu(expert_id)
        
        # 再次获取参数（应该返回 GPU 版本，因为优先级更高）
        params = loader.get_expert_params(expert_id)
        assert params is not None
        assert params['gate_proj'].device.type == 'cuda'
        
        # 指定设备获取
        cpu_params = loader.get_expert_params(expert_id, device=DeviceType.CPU)
        assert cpu_params is not None
        assert cpu_params['gate_proj'].device.type == 'cpu'
        
        gpu_params = loader.get_expert_params(expert_id, device=DeviceType.GPU)
        assert gpu_params is not None
        assert gpu_params['gate_proj'].device.type == 'cuda'
        
        print("✓ get_expert_params priority works correctly")
        
        loader.close()


class TestParameterLoaderMemoryUsage:
    """测试 ParameterLoader 的内存使用统计"""
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH), 
        reason="Qwen3 model not found"
    )
    def test_memory_usage_stats(self):
        """测试内存使用统计"""
        loader = ParameterLoader(QWEN3_MODEL_PATH)
        loader._load_static_parameters()
        
        # 加载几个 experts 到 CPU
        for expert_idx in range(3):
            expert_id = ExpertID(0, expert_idx)
            expert_weights = loader._load_single_expert(0, expert_idx, device='cpu')
            if expert_weights:
                loader.expert_params_cpu[expert_id] = expert_weights
        
        # 复制一个到 GPU
        expert_id = ExpertID(0, 0)
        loader.expert_params_gpu[expert_id] = {
            k: v.to('cuda') for k, v in loader.expert_params_cpu[expert_id].items()
        }
        
        # 获取内存使用
        memory = loader.get_expert_memory_usage()
        
        assert memory['cpu_bytes'] > 0
        assert memory['routed_gpu_bytes'] > 0
        assert memory['shared_gpu_bytes'] == 0  # Qwen3 没有 shared experts
        
        print(f"✓ Memory usage:")
        print(f"  Shared GPU: {memory['shared_gpu_mb']:.2f} MB")
        print(f"  Routed GPU: {memory['routed_gpu_mb']:.2f} MB")
        print(f"  CPU: {memory['cpu_mb']:.2f} MB")
        
        loader.close()


class TestFullModelLoading:
    """测试完整模型加载（较慢的测试）"""
    
    @pytest.mark.skipif(
        not os.path.exists(QWEN3_MODEL_PATH), 
        reason="Qwen3 model not found"
    )
    @pytest.mark.slow
    def test_full_load_parameters(self):
        """测试完整参数加载"""
        loader = ParameterLoader(QWEN3_MODEL_PATH)
        
        # 执行完整加载
        loader.load_parameters()
        
        # 验证静态参数
        assert len(loader.static_params_gpu) > 0
        print(f"✓ Static params: {len(loader.static_params_gpu)}")
        
        # 验证 shared experts（Qwen3 没有）
        assert len(loader.shared_experts_gpu) == 0
        print(f"✓ Shared experts: {len(loader.shared_experts_gpu)}")
        
        # 验证 routed experts
        expected_experts = 48 * 128  # 48 layers x 128 experts = 6144
        assert len(loader.expert_params_cpu) == expected_experts
        print(f"✓ CPU routed experts: {len(loader.expert_params_cpu)}")
        
        # 验证完整内存使用（包括静态参数）
        memory_usage = loader.get_memory_usage()
        print(f"✓ Static GPU memory: {memory_usage['static_gpu_mb']:.2f} MB")
        print(f"✓ Shared expert GPU memory: {memory_usage['shared_expert_gpu_mb']:.2f} MB")
        print(f"✓ Routed expert GPU memory: {memory_usage['routed_expert_gpu_mb']:.2f} MB")
        print(f"✓ Total GPU memory: {memory_usage['total_gpu_mb']:.2f} MB")
        print(f"✓ CPU memory (routed experts): {memory_usage['cpu_mb']:.2f} MB")
        
        # 静态参数应该占用 GPU 内存
        assert memory_usage['static_gpu_mb'] > 0, "Static params should use GPU memory"
        
        loader.close()


def run_quick_tests():
    """运行快速测试"""
    print("=" * 60)
    print("Testing Qwen3-30B-A3B-Base Parameter Loading")
    print("=" * 60)
    
    if not os.path.exists(QWEN3_MODEL_PATH):
        print(f"❌ Model not found at {QWEN3_MODEL_PATH}")
        return False
    
    print(f"\n📁 Model path: {QWEN3_MODEL_PATH}")
    
    # Test 1: MoE Config
    print("\n--- Test 1: MoE Config ---")
    config = MoEModelConfig(QWEN3_MODEL_PATH)
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_layers: {config.num_hidden_layers}")
    print(f"  num_experts: {config.num_experts}")
    print(f"  num_shared_experts: {config.num_shared_experts}")
    print(f"  top-k: {config.num_experts_per_token}")
    print(f"  expert_size: {config.get_expert_weight_size_bytes() / (1024*1024):.2f} MB")
    print("  ✓ Pass")
    
    # Test 2: Weight Loader
    print("\n--- Test 2: Weight Loader ---")
    weight_loader = SafetensorsWeightLoader(QWEN3_MODEL_PATH)
    print(f"  Total weights: {len(weight_loader.weight_map)}")
    embed = weight_loader.get_tensor("model.embed_tokens.weight")
    print(f"  Embedding shape: {embed.shape}")
    weight_loader.close()
    print("  ✓ Pass")
    
    # Test 3: Static Parameters + Shared Experts
    print("\n--- Test 3: Static Parameters + Shared Experts ---")
    param_loader = ParameterLoader(QWEN3_MODEL_PATH)
    param_loader._load_static_parameters()
    print(f"  Static params loaded: {len(param_loader.static_params_gpu)}")
    print(f"  Shared experts loaded: {len(param_loader.shared_experts_gpu)}")
    print(f"  Shared expert IDs: {len(param_loader.shared_expert_ids)}")
    print("  ✓ Pass")
    
    # Test 4: Single Expert Loading
    print("\n--- Test 4: Single Expert Loading ---")
    expert_weights = param_loader._load_single_expert(0, 0, device='cpu')
    print(f"  gate_proj shape: {expert_weights['gate_proj'].shape}")
    print(f"  up_proj shape: {expert_weights['up_proj'].shape}")
    print(f"  down_proj shape: {expert_weights['down_proj'].shape}")
    print("  ✓ Pass")
    
    # Test 5: Expert GPU Operations
    print("\n--- Test 5: Expert GPU Operations ---")
    expert_id = ExpertID(0, 0)
    param_loader.expert_params_cpu[expert_id] = expert_weights
    param_loader.expert_locations[expert_id] = ExpertLocation(
        expert_id=expert_id,
        device=DeviceType.CPU,
        is_cached=False
    )
    param_loader.load_expert_to_gpu(expert_id)
    print(f"  Expert on GPU: {param_loader.expert_params_gpu[expert_id]['gate_proj'].device}")
    param_loader.evict_expert_from_gpu(expert_id)
    print(f"  Expert evicted, CPU copy exists: {expert_id in param_loader.expert_params_cpu}")
    print("  ✓ Pass")
    
    # Test 6: Shared Expert APIs
    print("\n--- Test 6: Shared Expert APIs ---")
    print(f"  is_shared_expert(0, 0): {param_loader.is_shared_expert(ExpertID(0, 0))}")
    print(f"  get_shared_expert_count(): {param_loader.get_shared_expert_count()}")
    shared_layer0 = param_loader.get_all_shared_experts_for_layer(0)
    print(f"  Shared experts in layer 0: {len(shared_layer0)}")
    print("  ✓ Pass")
    
    param_loader.close()
    
    print("\n" + "=" * 60)
    print("All quick tests passed! ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run full model loading test")
    args = parser.parse_args()
    
    if args.full:
        # 运行完整测试
        pytest.main([__file__, "-v", "-s"])
    else:
        # 运行快速测试
        run_quick_tests()
