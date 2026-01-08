#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试修复后的实现
验证：
1. 原始模型生成的输出
2. draft模型生成的输出（应该与原始模型不同）
3. 投机采样的输出（应该与原始模型一致）
"""
import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.moe_spec.moe_model import MOEModelWrapper, ModifiedMOEModel
from model.moe_spec.moe_spec_decoder import MOESpecDecoder
import time

def test_fixed_implementation():
    print("=" * 80)
    print("测试修复后的实现")
    print("=" * 80)
    
    # 初始化模型
    print("\n1. 初始化模型...")
    model_path = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"
    original_model = MOEModelWrapper(
        model_path=model_path,
        device="cuda",
        dtype="float16"
    )
    
    print("\n2. 创建修改后的MoE模型（draft模型）...")
    modified_model = ModifiedMOEModel(
        original_model,
        num_to_modify=2,
        src_positions=[7, 8],   
        dst_positions=[9, 10],  
    )
    
    # 创建输入
    tokenizer = original_model.tokenizer
    prompt = "你好，请介绍一下北京"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(original_model.device)
    
    print(f"\n输入文本: {prompt}")
    print(f"输入tokens: {input_ids.tolist()}")
    
    generate_length = 100

    # 测试1: 原始模型生成
    print("\n" + "=" * 80)
    print("测试1: 原始模型生成（使用generate方法）")
    print("=" * 80)
    with torch.no_grad():
        original_outputs = original_model.model.generate(
            input_ids,
            max_new_tokens=generate_length + input_ids.shape[1],
            do_sample=False,  # 使用贪婪解码
            use_cache=True
        )
    original_text = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
    original_tokens = original_outputs[0].tolist()
    print(f"原始模型输出tokens: {original_tokens}")
    print(f"原始模型输出文本: {original_text}")
    
    # 测试2: 原始模型逐步生成（验证prefill+decode）
    print("\n" + "=" * 80)
    print("测试2: 原始模型逐步生成（prefill + decode）")
    print("=" * 80)
    current_input = input_ids.clone()
    generated_tokens = []
    
    # Prefill
    prefill_logits, kv_cache = original_model.prefill(current_input)
    
    # 先从prefill的logits中获取首个生成token并接入到序列与KV缓存推进流程
    first_token = torch.argmax(prefill_logits[:, -1, :], dim=-1).item()
    generated_tokens.append(first_token)
    current_input = torch.cat([
        current_input,
        torch.tensor([[first_token]], device=current_input.device)
    ], dim=1)
    
    # 继续生成剩余的token（共generate_length个，已加入1个，这里再生成generate_length-1个）
    for i in range(generate_length-1):
        logits, kv_cache = original_model.decode(current_input, kv_cache)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        generated_tokens.append(next_token)
        current_input = torch.cat([
            current_input,
            torch.tensor([[next_token]], device=current_input.device)
        ], dim=1)
    
    step_by_step_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"逐步生成tokens: {generated_tokens}")
    print(f"逐步生成文本: {step_by_step_text}")
    
    # 验证一致性
    original_generated_tokens = original_tokens[len(input_ids[0]):][:generate_length]
    if generated_tokens == original_generated_tokens:
        print("\n✓ 验证通过：逐步生成与generate方法一致")
    else:
        print(f"\n✗ 验证失败：逐步生成与generate方法不一致")
        print(f"  期望: {original_generated_tokens}")
        print(f"  实际: {generated_tokens}")
    
    # 测试3: draft模型生成（应该与原始模型不同）
    print("\n" + "=" * 80)
    print("测试3: draft模型生成（应该与原始模型不同）")
    print("=" * 80)
    current_input = input_ids.clone()
    draft_tokens = []
    
    # Prefill（使用原始模型）
    logits, kv_cache = original_model.prefill(current_input)
    
    # 使用draft模型生成10个token
    for i in range(generate_length-1):
        logits, kv_cache = modified_model.decode(current_input, kv_cache)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        draft_tokens.append(next_token)
        current_input = torch.cat([
            current_input,
            torch.tensor([[next_token]], device=current_input.device)
        ], dim=1)
    
    draft_text = tokenizer.decode(draft_tokens, skip_special_tokens=True)
    print(f"Draft模型生成tokens: {draft_tokens}")
    print(f"Draft模型生成文本: {draft_text}")
    
    # 验证差异
    if draft_tokens != generated_tokens:
        print(f"\n✓ 验证通过：draft模型输出与原始模型不同")
        diff_count = sum(1 for a, b in zip(draft_tokens, generated_tokens) if a != b)
        print(f"  差异token数: {diff_count}/{len(draft_tokens)}")
    else:
        print(f"\n✗ 警告：draft模型输出与原始模型完全相同，可能路由修改未生效")
    
    # 测试4: 投机采样生成（应该与原始模型一致）
    print("\n" + "=" * 80)
    print("测试4: 投机采样生成（应该与原始模型一致）")
    print("=" * 80)
    spec_decoder = MOESpecDecoder(original_model, modified_model)
    
    start_time = time.time()
    result = spec_decoder.speculate_decode(input_ids, max_new_tokens=generate_length, deterministic=True)
    elapsed = time.time() - start_time
    
    spec_output_text = tokenizer.decode(result['output_ids'][0], skip_special_tokens=True)
    spec_generated_tokens = result['output_ids'][0].tolist()[len(input_ids[0]):]
    
    print(f"投机采样生成tokens: {spec_generated_tokens}")
    print(f"投机采样生成文本: {spec_output_text}")
    print(f"\n统计信息:")
    print(f"  生成token数: {result['new_token']}")
    print(f"  步数: {result['step']}")
    print(f"  总draft长度: {result['total_draft_length']}")
    print(f"  总接受长度: {result['total_accept_length']}")
    print(f"  接受率: {result['acceptance_rate']:.2%}")
    print(f"  接受长度列表: {result['accept_length_list']}")
    print(f"  生成速度: {result['new_token']/elapsed:.2f} tokens/s")
    
    # 验证一致性
    print("\n" + "=" * 80)
    print("最终验证：投机采样输出与原始模型的一致性")
    print("=" * 80)
    
    # 比较前10个token（如果生成的token数不足10，就比较实际数量）
    compare_length = min(generate_length, len(spec_generated_tokens), len(original_generated_tokens))
    spec_tokens_to_compare = spec_generated_tokens[:compare_length]
    original_tokens_to_compare = original_generated_tokens[:compare_length]
    
    if spec_tokens_to_compare == original_tokens_to_compare:
        print(f"\n✓✓✓ 验证成功：投机采样输出与原始模型完全一致！")
        print(f"  对比token数: {compare_length}")
    else:
        print(f"\n✗✗✗ 验证失败：投机采样输出与原始模型不一致")
        print(f"  对比token数: {compare_length}")
        diff_count = sum(1 for a, b in zip(spec_tokens_to_compare, original_tokens_to_compare) if a != b)
        print(f"  差异token数: {diff_count}/{compare_length}")
        print(f"\n  期望tokens: {original_tokens_to_compare}")
        print(f"  实际tokens: {spec_tokens_to_compare}")
        print(f"\n  期望文本: {tokenizer.decode(original_tokens_to_compare)}")
        print(f"  实际文本: {tokenizer.decode(spec_tokens_to_compare)}")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

if __name__ == "__main__":
    test_fixed_implementation()




