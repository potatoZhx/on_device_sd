#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最终测试：验证draft_length=2的投机采样实现
"""
import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.moe_spec.moe_model import MOEModelWrapper, ModifiedMOEModel
from model.moe_spec.moe_spec_decoder import MOESpecDecoder

def main():
    # 初始化
    model_path = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"
    original_model = MOEModelWrapper(model_path, device="cuda", dtype="float16")
    modified_model = ModifiedMOEModel(
        original_model,
        num_to_modify=1,
        src_positions=[1],
        dst_positions=[20],
    )
    spec_decoder = MOESpecDecoder(original_model, modified_model, draft_length=3)
    
    print(f"✓ 模型加载完成")
    print(f"✓ Draft length: {spec_decoder.draft_length}")
    
    # 测试不同的prompts
    prompts = [
        "你好，请介绍一下北京",
        "什么是人工智能？",
        "请简单介绍一下Python编程语言"
    ]
    
    tokenizer = original_model.tokenizer
    
    for i, prompt in enumerate(prompts, 1):
        print("\n" + "=" * 80)
        print(f"测试Prompt {i}: {prompt}")
        print("=" * 80)
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        
        # 投机采样生成
        result = spec_decoder.speculate_decode(input_ids, max_new_tokens=100)
        
        generated_text = tokenizer.decode(
            result['output_ids'][0][len(input_ids[0]):],
            skip_special_tokens=True
        )
        
        print(f"生成文本: {generated_text}")
        print(f"\n统计信息:")
        print(f"  生成token数: {result['new_token']}")
        print(f"  步数: {result['step']}")
        print(f"  总draft长度: {result['total_draft_length']}")
        print(f"  总接受长度: {result['total_accept_length']}")
        print(f"  接受率: {result['acceptance_rate']:.2%}")
        print(f"  每步接受长度: {result['accept_length_list']}")
        

if __name__ == "__main__":
    main()

