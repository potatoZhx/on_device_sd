#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试spec_sampling函数
"""
import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.moe_spec.moe_model import MOEModelWrapper, ModifiedMOEModel
from model.moe_spec.moe_spec_decoder import MOESpecDecoder
from model.moe_spec.spec_sampling import speculative_sampling

def debug_spec_sampling():
    print("初始化模型...")
    # 使用CPU以便快速测试
    model = MOEModelWrapper(
        model_path="/zx_data1/models/Qwen--Qwen3-30B-A3B-Base",
        device="cpu",
        dtype="float16"
    )
    
    print("创建修改后的MoE模型...")
    modified_model = ModifiedMOEModel(
        model,
        num_to_modify=2,
        src_positions=[7, 8],
        dst_positions=[9, 10],
    )
    
    print("创建推测解码器...")
    spec_decoder = MOESpecDecoder(model, modified_model)
    
    # 创建一个简单的输入
    tokenizer = model.tokenizer
    text = "你好，请介绍一下北京"
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    
    print(f"输入文本: {text}")
    print(f"输入形状: {input_ids.shape}")
    print(f"输入tokens: {input_ids.tolist()}")
    
    # 执行prefill获取KV cache
    print("执行prefill...")
    logits, kv_cache = model.prefill(input_ids)
    
    # 生成draft token
    print("生成draft token...")
    draft_tokens, draft_logits = spec_decoder._generate_draft(input_ids, kv_cache)
    print(f"Draft tokens: {draft_tokens}")
    print(f"Draft logits shape: {draft_logits.shape}")
    
    # 构建候选序列
    print("构建候选序列...")
    draft_tensor = torch.tensor([draft_tokens], device=input_ids.device)
    candidate_input_ids = torch.cat([input_ids, draft_tensor], dim=1)
    candidate_length = len(draft_tokens)
    print(f"Candidate input_ids: {candidate_input_ids.tolist()}")
    print(f"Candidate length: {candidate_length}")
    
    # 原模型验证
    print("原模型验证...")
    verification_logits, updated_kv_cache = model.decode(candidate_input_ids, kv_cache)
    print(f"Verification logits shape: {verification_logits.shape}")
    
    # 使用原始verification_logits
    print("使用原始verification_logits...")
    new_logits = verification_logits
    print(f"New logits shape: {new_logits.shape}")
    
    # 打印draft token和对应的概率
    print("\n分析draft token的概率...")
    draft_token = draft_tokens[0]
    draft_prob = torch.softmax(draft_logits[0, 0], dim=-1)[draft_token].item()
    original_prob = torch.softmax(verification_logits[0, 0], dim=-1)[draft_token].item()
    print(f"Draft token: {draft_token}, 对应的文本: {tokenizer.decode([draft_token])}")
    print(f"修改模型给出的概率: {draft_prob:.6f}")
    print(f"原始模型给出的概率: {original_prob:.6f}")
    print(f"概率比率: {original_prob/draft_prob:.6f}")
    
    # 执行推测采样
    print("\n执行推测采样...")
    valid_tokens, n_matches = speculative_sampling(
        candidate_input_ids=candidate_input_ids,
        candidate_logits=draft_logits,
        candidate_length=candidate_length,
        new_logits=new_logits,
        last_assistant_token_is_eos=(model.tokenizer.eos_token_id in draft_tokens),
        max_matches=candidate_length
    )
    
    print(f"有效tokens: {valid_tokens.tolist()}")
    print(f"匹配数量: {n_matches}")
    
    if n_matches > 0:
        accepted_tokens = valid_tokens[0, -n_matches:].tolist()
        print(f"接受的tokens: {accepted_tokens}")
        print(f"接受的文本: {tokenizer.decode(accepted_tokens)}")
    else:
        print("没有接受任何token")
        if valid_tokens.numel() > 0:
            new_token = valid_tokens[0].item()
            print(f"新采样的token: {new_token}")
            print(f"新采样的文本: {tokenizer.decode([new_token])}")
    
    print("\n调试完成!")

if __name__ == "__main__":
    debug_spec_sampling()
