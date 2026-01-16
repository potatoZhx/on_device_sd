#!/usr/bin/env python


import os
# 关闭所有网络请求，强制使用本地文件
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import torch
import time

# 直接导入MixtralForCausalLM而不是通过Auto类
# 这可以避免加载所有模型实现导致的问题
print("导入所需库...")
from transformers import MixtralForCausalLM, PreTrainedTokenizerFast, AutoModelForCausalLM, AutoTokenizer

def main():
    # 模型路径
    # model_path = "/zx_data1/models/mixtral/models--mistralai--Mixtral-8x7B-v0.1"
    model_path = "/zx_data1/models/deepseek/ds-16b-moe"
    prompt = "once upon a time"
    num_tokens_to_generate = 5
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型路径 {model_path} 不存在")
        return
        
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        print(f"错误：模型配置文件 {config_path} 不存在")
        return
    
    # 加载分词器和模型
    print(f"加载模型: {model_path}")
    try:
        # 显式指定local_files_only=True确保不联网
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )
        # tokenizer = PreTrainedTokenizerFast.from_pretrained(
        #     model_path, 
        #     local_files_only=True
        # )
        # model = MixtralForCausalLM.from_pretrained(
        #     model_path,
        #     torch_dtype=torch.float16,
        #     device_map="auto",
        #     local_files_only=True
        # )
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 加载模型后打印设备信息
    if hasattr(model, "hf_device_map"):
        print("模型分布在以下设备上:")
        for layer, device in model.hf_device_map.items():
            print(f"  - {layer}: {device}")
    else:
        print(f"模型位于单一设备上: {model.device}")
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    print(f"提示词长度: {input_ids.shape[1]} tokens")

    # 模型预热
    print("模型预热中...")
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=True)
    print("模型预热完成")
    
    # 计时开始 - prefill阶段
    prefill_start = time.time()
    
    # 初始前向传播(prefill阶段)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    
    # 获取KV缓存
    past_key_values = outputs.past_key_values
    
    # 计算prefill时间
    prefill_time = time.time() - prefill_start
    
    # 准备自回归生成
    generated_ids = input_ids
    next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
    
    # 计时开始 - decode阶段
    decode_start = time.time()
    
    # 自回归生成剩余token
    for i in range(num_tokens_to_generate - 1):
        print(f"生成第 {i+2}/{num_tokens_to_generate} 个token...")
        with torch.no_grad():
            outputs = model(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                use_cache=True
            )
            
        past_key_values = outputs.past_key_values
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
    
    # 计算decode时间
    decode_time = time.time() - decode_start
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # 打印结果
    print("\n===== 结果 =====")
    print(f"输入提示: '{prompt}'")
    print(f"生成文本: '{generated_text}'")
    print(f"Prefill时间: {prefill_time:.4f}秒")
    print(f"Decode时间: {decode_time:.4f}秒")
    print(f"Decode每token平均时间: {decode_time/(num_tokens_to_generate-1):.4f}秒/token")
    print(f"总生成时间: {prefill_time + decode_time:.4f}秒")

if __name__ == "__main__":
    main()