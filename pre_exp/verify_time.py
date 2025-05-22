# 载入模型 /zx_data1/models/Llama-3-8B-Instruct，以"once upon a time"为提示词，生成5个token的文本
# 并计算prefill_time, decode_time


import os
# 设置环境变量以减少依赖
os.environ["TRANSFORMERS_OFFLINE"] = "1"  

import torch
import time
from transformers import LlamaForCausalLM, AutoTokenizer

def main():
    # 模型路径
    model_path = "/zx_data1/models/Llama-3-8B-Instruct"
    prompt = "once upon a time"
    num_tokens_to_generate = 5
    
    # 加载模型和分词器
    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids

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
    for _ in range(num_tokens_to_generate - 1):
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
    print(f"输入提示: '{prompt}'")
    print(f"生成文本: '{generated_text}'")
    print(f"Prefill时间: {prefill_time:.4f}秒")
    print(f"Decode时间: {decode_time:.4f}秒")
    print(f"Decode每token平均时间: {decode_time/(num_tokens_to_generate-1):.4f}秒")
    print(f"总生成时间: {prefill_time + decode_time:.4f}秒")

if __name__ == "__main__":
    main()