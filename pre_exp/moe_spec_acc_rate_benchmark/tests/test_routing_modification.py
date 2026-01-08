#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试路由修改是否生效
"""
import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.moe_spec.moe_model import MOEModelWrapper, ModifiedMOEModel

print("=" * 80)
print("测试路由修改")
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
_, kv = original_model.prefill(input_ids)
first_token = torch.argmax(_[:, -1, :], dim=-1).item()
print(f"First token: {first_token} ({tokenizer.decode([first_token])})")

# 测试1: 原始模型生成下一个token
print("\n" + "=" * 80)
print("原始模型")
print("=" * 80)
input_tensor = torch.tensor([[first_token]], device="cuda")
with torch.no_grad():
    outputs_orig = original_model.model(input_tensor, past_key_values=kv, use_cache=True)
orig_token = torch.argmax(outputs_orig.logits[:, -1, :], dim=-1).item()
print(f"输入: {first_token} ({tokenizer.decode([first_token])})")
print(f"输出: {orig_token} ({tokenizer.decode([orig_token])})")

# 获取top-5预测
orig_logits = outputs_orig.logits[:, -1, :]
orig_probs = torch.softmax(orig_logits, dim=-1)
orig_top5_probs, orig_top5_indices = torch.topk(orig_probs, 5, dim=-1)
print(f"Top-5预测:")
for i in range(5):
    idx = orig_top5_indices[0, i].item()
    prob = orig_top5_probs[0, i].item()
    print(f"  {i+1}. {idx} ({tokenizer.decode([idx])}) - {prob:.4f}")

# 测试2: Draft模型生成下一个token
print("\n" + "=" * 80)
print("Draft模型（排除top-2）")
print("=" * 80)
modified_model.routing_modifier.enable_routing_modification(modified_model.model)
with torch.no_grad():
    outputs_draft = modified_model.model(input_tensor, past_key_values=kv, use_cache=True)
modified_model.routing_modifier.disable_routing_modification(modified_model.model)

draft_token = torch.argmax(outputs_draft.logits[:, -1, :], dim=-1).item()
print(f"输入: {first_token} ({tokenizer.decode([first_token])})")
print(f"输出: {draft_token} ({tokenizer.decode([draft_token])})")

# 获取top-5预测
draft_logits = outputs_draft.logits[:, -1, :]
draft_probs = torch.softmax(draft_logits, dim=-1)
draft_top5_probs, draft_top5_indices = torch.topk(draft_probs, 5, dim=-1)
print(f"Top-5预测:")
for i in range(5):
    idx = draft_top5_indices[0, i].item()
    prob = draft_top5_probs[0, i].item()
    print(f"  {i+1}. {idx} ({tokenizer.decode([idx])}) - {prob:.4f}")

# 对比
print("\n" + "=" * 80)
print("对比")
print("=" * 80)
if orig_token == draft_token:
    print(f"⚠ 警告：原始模型和Draft模型输出相同！")
    print(f"  这可能意味着路由修改没有生效")
else:
    print(f"✓ 原始模型和Draft模型输出不同")
    print(f"  原始: {orig_token} ({tokenizer.decode([orig_token])})")
    print(f"  Draft: {draft_token} ({tokenizer.decode([draft_token])})")

modified_model.routing_modifier.restore_model(modified_model.model)

