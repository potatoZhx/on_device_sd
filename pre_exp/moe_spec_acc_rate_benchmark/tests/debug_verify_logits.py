#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试verify阶段的logits
"""
import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.moe_spec.moe_model import MOEModelWrapper, ModifiedMOEModel

def debug_verify_logits():
    print("=" * 80)
    print("调试Verify阶段的Logits")
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
    
    # Prefill
    prefill_logits, kv_cache = original_model.prefill(input_ids)
    first_token = torch.argmax(prefill_logits[:, -1, :], dim=-1).item()
    print(f"First token: {first_token} ({tokenizer.decode([first_token])})")
    print(f"Prefill后KV cache长度: {kv_cache.get_seq_length()}")
    
    # 使用MOESpecDecoder的_generate_draft方法
    print("\n" + "=" * 80)
    print("Draft阶段")
    print("=" * 80)
    
    from model.moe_spec.moe_spec_decoder import MOESpecDecoder
    spec_decoder = MOESpecDecoder(original_model, modified_model)
    
    print(f"Draft前KV cache长度: {kv_cache.get_seq_length()}")
    draft_tokens, draft_logits = spec_decoder._generate_draft(first_token, kv_cache)
    print(f"Draft后KV cache长度: {kv_cache.get_seq_length()}")  # 应该还是4
    
    print(f"Draft tokens: {draft_tokens}")
    print(f"Draft logits shape: {draft_logits.shape}")
    
    # Verify阶段 - 我的实现
    print("\n" + "=" * 80)
    print("Verify阶段 - 当前实现")
    print("=" * 80)
    verify_input_tokens = [first_token] + draft_tokens
    verify_input_ids = torch.tensor([verify_input_tokens], device="cuda")
    print(f"Verify输入tokens: {verify_input_tokens}")
    print(f"Verify输入文本: {tokenizer.decode(verify_input_tokens)}")
    print(f"Verify使用KV cache长度: {kv_cache.get_seq_length()}")
    
    with torch.no_grad():
        verify_outputs = original_model.model(verify_input_ids, past_key_values=kv_cache, use_cache=True)
    verify_logits = verify_outputs.logits
    print(f"Verify输出logits shape: {verify_logits.shape}")
    
    # 提取new_logits
    new_logits = verify_logits[:, :3, :]  # [1, 3, vocab_size]
    print(f"new_logits shape: {new_logits.shape}")
    
    for i in range(new_logits.shape[1]):
        pred_token = torch.argmax(new_logits[:, i, :], dim=-1).item()
        print(f"  new_logits位置{i}预测: {pred_token} ({tokenizer.decode([pred_token])})")
    
    # 对比：原始模型逐步生成
    print("\n" + "=" * 80)
    print("对比：原始模型逐步生成")
    print("=" * 80)
    
    _, kv_orig = original_model.prefill(input_ids)
    print(f"起始KV cache长度: {kv_orig.get_seq_length()}")
    
    current_token = first_token
    for step in range(3):
        input_tensor = torch.tensor([[current_token]], device="cuda")
        print(f"\nStep {step+1}:")
        print(f"  输入token: {current_token} ({tokenizer.decode([current_token])})")
        print(f"  输入KV cache长度: {kv_orig.get_seq_length()}")
        
        with torch.no_grad():
            outputs = original_model.model(input_tensor, past_key_values=kv_orig, use_cache=True)
        kv_orig = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
        print(f"  输出token: {next_token} ({tokenizer.decode([next_token])})")
        print(f"  输出KV cache长度: {kv_orig.get_seq_length()}")
        
        current_token = next_token
    
    # 关键对比
    print("\n" + "=" * 80)
    print("关键对比")
    print("=" * 80)
    print("\nVerify方法（输入[e, f, g]，kv=[a,b,c,d]）：")
    print(f"  位置0（输入e后）预测: {torch.argmax(new_logits[:, 0, :], dim=-1).item()}")
    print(f"  位置1（输入f后）预测: {torch.argmax(new_logits[:, 1, :], dim=-1).item()}")
    print(f"  位置2（输入g后）预测: {torch.argmax(new_logits[:, 2, :], dim=-1).item()}")
    
    print("\n原始逐步方法（每次输入1个token）：")
    _, kv_test = original_model.prefill(input_ids)
    t1 = first_token
    with torch.no_grad():
        out1 = original_model.model(torch.tensor([[t1]], device="cuda"), past_key_values=kv_test, use_cache=True)
    kv_test = out1.past_key_values
    t2 = torch.argmax(out1.logits[:, -1, :], dim=-1).item()
    print(f"  输入e，预测: {t2}")
    
    with torch.no_grad():
        out2 = original_model.model(torch.tensor([[t2]], device="cuda"), past_key_values=kv_test, use_cache=True)
    kv_test = out2.past_key_values
    t3 = torch.argmax(out2.logits[:, -1, :], dim=-1).item()
    print(f"  输入{t2}，预测: {t3}")
    
    with torch.no_grad():
        out3 = original_model.model(torch.tensor([[t3]], device="cuda"), past_key_values=kv_test, use_cache=True)
    t4 = torch.argmax(out3.logits[:, -1, :], dim=-1).item()
    print(f"  输入{t3}，预测: {t4}")
    
    modified_model.routing_modifier.restore_model(modified_model.model)

if __name__ == "__main__":
    debug_verify_logits()

