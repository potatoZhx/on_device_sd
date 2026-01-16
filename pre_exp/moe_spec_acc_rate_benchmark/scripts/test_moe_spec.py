#!/usr/bin/env python3
"""
简单的MOE推测解码测试脚本
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from model.moe_spec.moe_model import MOEModelWrapper, ModifiedMOEModel
from model.moe_spec.moe_spec_decoder import MOESpecDecoder

def test_moe_spec_decoding():
    """测试MOE推测解码"""
    
    # 配置
    model_path = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"
    device = "cuda"
    dtype = "float16"
    
    # 测试输入
    test_prompt = "Hello, how are you today?"
    
    try:
        # 初始化模型
        print("加载MOE模型...")
        original_model = MOEModelWrapper(model_path, device, dtype)
        
        print("创建修改的MOE模型...")
        modified_model = ModifiedMOEModel(original_model, top_k_experts_to_remove=2)
        
        # 初始化推测解码器
        spec_decoder = MOESpecDecoder(original_model, modified_model, draft_length=1)
        
        # 编码输入
        input_ids = original_model.tokenizer.encode(test_prompt, return_tensors="pt").to(device)
        
        print(f"输入: {test_prompt}")
        print("执行推测解码...")
        
        # 执行推测解码
        result = spec_decoder.speculate_decode(input_ids, max_new_tokens=50)
        
        # 解码输出
        generated_tokens = result['output_ids'][0, input_ids.shape[1]:].tolist()
        generated_text = original_model.tokenizer.decode(generated_tokens)
        
        print(f"输出: {generated_text}")
        print(f"接收率: {result['acceptance_rate']:.4f}")
        print(f"总草稿长度: {result['total_draft_length']}")
        print(f"总接受长度: {result['total_accept_length']}")
        print(f"解码步数: {result['step']}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_moe_spec_decoding()
