#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试投机采样 - 使用greedy模式（设置seed确保确定性）
"""
import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.moe_spec.moe_model import MOEModelWrapper, ModifiedMOEModel
from model.moe_spec.moe_spec_decoder import MOESpecDecoder

# 设置随机种子确保确定性
torch.manual_seed(42)
torch.cuda.manual_seed(42)

print("=" * 80)
print("测试投机采样 - Greedy模式")
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
spec_decoder = MOESpecDecoder(original_model, modified_model)

tokenizer = original_model.tokenizer
prompt = "你好，请介绍一下北京"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

generate_length = 100

print(f"\n输入: {prompt}")

# 测试1: 原始模型逐步greedy生成
print("\n" + "=" * 80)
print("原始模型逐步生成（greedy）")
print("=" * 80)

logits_prefill, kv_orig = original_model.prefill(input_ids)
tokens_orig = []
current_token = torch.argmax(logits_prefill[:, -1, :], dim=-1).item()
tokens_orig.append(current_token)

for i in range(generate_length-1):
    logits, kv_orig = original_model.decode(
        torch.tensor([[current_token]], device="cuda"),
        kv_orig
    )
    current_token = torch.argmax(logits[:, -1, :], dim=-1).item()
    tokens_orig.append(current_token)

print(f"原始模型tokens: {tokens_orig}")
print(f"原始模型文本: {tokenizer.decode(tokens_orig)}")

# 测试2: 投机采样（使用相同的seed）
print("\n" + "=" * 80)
print("投机采样生成")
print("=" * 80)

torch.manual_seed(42)  # 重置种子
torch.cuda.manual_seed(42)

result = spec_decoder.speculate_decode(input_ids, max_new_tokens=generate_length)
tokens_spec = result['output_ids'][0].tolist()[len(input_ids[0]):]

print(f"投机采样tokens: {tokens_spec}")
print(f"投机采样文本: {tokenizer.decode(tokens_spec)}")

print(f"\n统计:")
print(f"  接受率: {result['acceptance_rate']:.2%}")
print(f"  每步接受长度: {result['accept_length_list']}")

# 对比
print("\n" + "=" * 80)
print("对比")
print("=" * 80)
compare_len = min(len(tokens_orig), len(tokens_spec))
matches = sum(1 for i in range(compare_len) if tokens_orig[i] == tokens_spec[i])
print(f"前{compare_len}个tokens中，{matches}个匹配（{matches/compare_len*100:.1f}%）")

for i in range(min(compare_len, 10)):
    match = "✓" if tokens_orig[i] == tokens_spec[i] else "✗"
    print(f"  位置{i}: {match} 原={tokens_orig[i]}, 投={tokens_spec[i]}")

modified_model.routing_modifier.restore_model(modified_model.model)

