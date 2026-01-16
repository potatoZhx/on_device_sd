#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
详细调试投机采样的每一步
"""
import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.moe_spec.moe_model import MOEModelWrapper, ModifiedMOEModel

def debug_spec_detailed():
    print("=" * 80)
    print("详细调试投机采样流程")
    print("=" * 80)
    
    # 初始化
    model_path = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"
    original_model = MOEModelWrapper(model_path, device="cuda", dtype="float16")
    modified_model = ModifiedMOEModel(
        original_model,
        num_to_modify=2,
        src_positions=[7, 8],
        dst_positions=[9, 10],
    )
    
    tokenizer = original_model.tokenizer
    prompt = "你好，请介绍一下北京"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    
    print(f"\n输入: {prompt}")
    print(f"输入tokens: {input_ids.tolist()}")
    
    # Prefill
    print("\n" + "=" * 80)
    print("Prefill阶段")
    print("=" * 80)
    prefill_logits, kv_cache = original_model.prefill(input_ids)
    first_token = torch.argmax(prefill_logits[:, -1, :], dim=-1).item()
    print(f"第一个token: {first_token} ({tokenizer.decode([first_token])})")
    print(f"KV cache长度: {kv_cache.get_seq_length() if hasattr(kv_cache, 'get_seq_length') else 'unknown'}")
    
    # 将第一个token加入KV cache
    first_token_tensor = torch.tensor([[first_token]], device="cuda")
    with torch.no_grad():
        outputs = original_model.model(first_token_tensor, past_key_values=kv_cache, use_cache=True)
        kv_cache = outputs.past_key_values
    print(f"加入first_token后KV cache长度: {kv_cache.get_seq_length() if hasattr(kv_cache, 'get_seq_length') else 'unknown'}")
    
    # 模拟第一轮投机采样
    print("\n" + "=" * 80)
    print("第1轮投机采样")
    print("=" * 80)
    
    last_token = first_token
    
    # Draft阶段
    print("\n[Draft] 生成draft tokens...")
    print(f"  输入last_token: {last_token} ({tokenizer.decode([last_token])})")
    print(f"  输入KV cache长度: {kv_cache.get_seq_length() if hasattr(kv_cache, 'get_seq_length') else 'unknown'}")
    
    # 手动执行draft
    modified_model.routing_modifier.enable_routing_modification(modified_model.model)
    draft_tokens = []
    draft_logits_list = []
    current_kv = kv_cache
    current_token = last_token
    
    for i in range(2):
        input_tensor = torch.tensor([[current_token]], device="cuda")
        with torch.no_grad():
            outputs = modified_model.model(input_tensor, past_key_values=current_kv, use_cache=True)
        logits = outputs.logits[:, -1:, :]
        draft_logits_list.append(logits)
        next_token = torch.argmax(logits[:, 0, :], dim=-1).item()
        draft_tokens.append(next_token)
        print(f"  Draft {i+1}: token={next_token} ({tokenizer.decode([next_token])})")
        current_token = next_token
        current_kv = outputs.past_key_values
    
    modified_model.routing_modifier.disable_routing_modification(modified_model.model)
    draft_logits = torch.cat(draft_logits_list, dim=1)
    print(f"  Draft tokens: {draft_tokens}")
    print(f"  Draft tokens文本: {tokenizer.decode(draft_tokens)}")
    
    # Verify阶段
    print("\n[Verify] 验证draft tokens...")
    verify_input_tokens = [last_token] + draft_tokens
    verify_input_ids = torch.tensor([verify_input_tokens], device="cuda")
    print(f"  Verify输入tokens: {verify_input_tokens}")
    print(f"  Verify输入文本: {tokenizer.decode(verify_input_tokens)}")
    print(f"  Verify输入KV cache长度: {kv_cache.get_seq_length() if hasattr(kv_cache, 'get_seq_length') else 'unknown'}")
    
    with torch.no_grad():
        verify_outputs = original_model.model(verify_input_ids, past_key_values=kv_cache, use_cache=True)
    verify_logits = verify_outputs.logits
    print(f"  Verify输出logits shape: {verify_logits.shape}")
    
    # 提取各个位置预测的token
    for i in range(verify_logits.shape[1]):
        pred_token = torch.argmax(verify_logits[:, i, :], dim=-1).item()
        print(f"  位置{i}的预测: {pred_token} ({tokenizer.decode([pred_token])})")
    
    # 对比原始模型的逐步生成
    print("\n" + "=" * 80)
    print("对比：原始模型逐步生成（从相同状态）")
    print("=" * 80)
    
    # 重新创建相同的kv_cache状态
    _, kv_cache_orig = original_model.prefill(input_ids)
    with torch.no_grad():
        outputs = original_model.model(first_token_tensor, past_key_values=kv_cache_orig, use_cache=True)
        kv_cache_orig = outputs.past_key_values
    
    print(f"起始状态：KV cache包含{kv_cache_orig.get_seq_length() if hasattr(kv_cache_orig, 'get_seq_length') else 'unknown'}个tokens")
    print(f"last_token: {last_token} ({tokenizer.decode([last_token])})")
    
    # 逐步生成3个tokens
    current_token = last_token
    for i in range(3):
        input_tensor = torch.tensor([[current_token]], device="cuda")
        with torch.no_grad():
            outputs = original_model.model(input_tensor, past_key_values=kv_cache_orig, use_cache=True)
        kv_cache_orig = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
        print(f"  Step {i+1}: 输入={current_token}({tokenizer.decode([current_token])}), 输出={next_token}({tokenizer.decode([next_token])})")
        current_token = next_token
    
    print("\n" + "=" * 80)
    print("调试完成")
    print("=" * 80)
    
    modified_model.routing_modifier.restore_model(modified_model.model)

if __name__ == "__main__":
    debug_spec_detailed()

