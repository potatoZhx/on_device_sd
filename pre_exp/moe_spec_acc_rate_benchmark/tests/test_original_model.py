#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试原始模型的推理质量
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_original_model():
    print("加载原始模型...")
    model_path = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"
    
    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # 设置为评估模式
    model.eval()
    
    # 测试提示
    prompts = [
        "你好，请介绍一下北京的天气预报，北京的景点非常丰富的旅游资源，故宫、天坛庙、颐和园、北京动物园"
        # "你好，请介绍一下北京",
        # "什么是人工智能？",
        # "请简单介绍一下Python编程语言"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n测试提示 {i+1}/{len(prompts)}: {prompt}")
        
        # 编码输入
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        
        # 使用原始模型生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=30,
                do_sample=False,  # 使用贪婪解码
                use_cache=True
            )
        
        # 解码输出
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"原始模型输出: {output_text}")
        
        # 测试单步解码
        print("\n测试单步解码:")
        current_input = input_ids.clone()
        generated_tokens = []
        
        # 执行prefill
        with torch.no_grad():
            outputs = model(current_input, use_cache=True)
            kv_cache = outputs.past_key_values
        
        # 单步解码5个token
        for _ in range(100):
            with torch.no_grad():
                outputs = model(
                    current_input[:, -1:],  # 只使用最后一个token
                    past_key_values=kv_cache,
                    use_cache=True
                )
                
                # 获取下一个token
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
                generated_tokens.append(next_token)
                
                # 更新输入和KV cache
                current_input = torch.cat([
                    current_input,
                    torch.tensor([[next_token]], device=current_input.device)
                ], dim=1)
                kv_cache = outputs.past_key_values
        
        # 解码单步生成的tokens
        step_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"单步解码前5个token: {step_output}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_original_model()




