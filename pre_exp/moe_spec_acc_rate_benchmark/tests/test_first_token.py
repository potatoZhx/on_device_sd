#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
专门测试第一个token的生成
验证prefill后第一个token是否一致
"""
import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.moe_spec.moe_model import MOEModelWrapper, ModifiedMOEModel

def test_first_token():
    print("=" * 80)
    print("测试第一个token的生成")
    print("=" * 80)
    
    # 初始化模型
    model_path = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"
    original_model = MOEModelWrapper(
        model_path=model_path,
        device="cuda",
        dtype="float16"
    )
    
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
    print(f"输入长度: {input_ids.shape[1]}")
    
    # 1. Prefill阶段
    print("\n" + "=" * 80)
    print("1. 原始模型Prefill")
    print("=" * 80)
    logits_prefill, kv_cache = original_model.prefill(input_ids)
    print(f"Prefill后logits shape: {logits_prefill.shape}")
    print(f"Prefill后最后一个位置的logits（用于预测第一个新token）: {logits_prefill[:, -1, :].shape}")
    
    # 从prefill的logits中预测第一个token
    first_token_from_prefill = torch.argmax(logits_prefill[:, -1, :], dim=-1).item()
    print(f"\n从Prefill直接预测第一个token: {first_token_from_prefill} ({tokenizer.decode([first_token_from_prefill])})")
    
    # 2. 使用decode方法生成第一个token
    print("\n" + "=" * 80)
    print("2. 使用原始模型decode生成第一个token")
    print("=" * 80)
    logits_decode, kv_cache_decode = original_model.decode(input_ids, kv_cache)
    first_token_from_decode = torch.argmax(logits_decode[:, -1, :], dim=-1).item()
    print(f"通过decode预测第一个token: {first_token_from_decode} ({tokenizer.decode([first_token_from_decode])})")
    
    # 3. 直接用模型前向传播
    print("\n" + "=" * 80)
    print("3. 直接用模型前向传播（不使用KV cache）")
    print("=" * 80)
    with torch.no_grad():
        outputs = original_model.model(input_ids, use_cache=False)
    first_token_direct = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
    print(f"直接前向传播预测第一个token: {first_token_direct} ({tokenizer.decode([first_token_direct])})")
    
    # 4. Draft模型生成（使用完整序列）
    print("\n" + "=" * 80)
    print("4. Draft模型生成第一个token（使用完整序列，启用路由修改）")
    print("=" * 80)
    modified_model.routing_modifier.enable_routing_modification(modified_model.model)
    with torch.no_grad():
        outputs_draft = modified_model.model(input_ids, use_cache=False)
    modified_model.routing_modifier.disable_routing_modification(modified_model.model)
    first_token_draft = torch.argmax(outputs_draft.logits[:, -1, :], dim=-1).item()
    print(f"Draft模型预测第一个token: {first_token_draft} ({tokenizer.decode([first_token_draft])})")
    
    # 5. 模拟投机采样的第一步
    print("\n" + "=" * 80)
    print("5. 模拟投机采样的第一步")
    print("=" * 80)
    
    # 5.1 生成draft
    print("\n5.1 生成draft token")
    draft_token = first_token_draft
    print(f"Draft token: {draft_token}")
    
    # 5.2 构建候选序列
    candidate_input_ids = torch.cat([
        input_ids,
        torch.tensor([[draft_token]], device=input_ids.device)
    ], dim=1)
    print(f"候选序列: {candidate_input_ids.tolist()}")
    print(f"候选序列长度: {candidate_input_ids.shape[1]}")
    
    # 5.3 验证
    print("\n5.2 验证draft token")
    with torch.no_grad():
        outputs_verify = original_model.model(candidate_input_ids, use_cache=False)
    print(f"验证输出logits shape: {outputs_verify.logits.shape}")
    
    # 获取最后两个位置的logits
    new_logits = outputs_verify.logits[:, -2:, :]
    print(f"最后两个位置的logits shape: {new_logits.shape}")
    
    # 第一个logits（索引-2）用于验证draft token
    verify_position_logits = new_logits[:, 0, :]
    verify_token = torch.argmax(verify_position_logits, dim=-1).item()
    print(f"\n从验证logits（索引-2）预测的token: {verify_token} ({tokenizer.decode([verify_token])})")
    
    # 第二个logits（索引-1）用于采样下一个token
    next_position_logits = new_logits[:, 1, :]
    next_token = torch.argmax(next_position_logits, dim=-1).item()
    print(f"从下一位置logits（索引-1）预测的token: {next_token} ({tokenizer.decode([next_token])})")
    
    # 验证一致性
    print("\n" + "=" * 80)
    print("一致性检查")
    print("=" * 80)
    print(f"Prefill预测: {first_token_from_prefill}")
    print(f"Decode预测: {first_token_from_decode}")
    print(f"直接前向预测: {first_token_direct}")
    print(f"验证logits预测: {verify_token}")
    
    if first_token_from_prefill == first_token_from_decode == first_token_direct == verify_token:
        print("\n✓ 所有方法预测的第一个token一致！")
    else:
        print("\n✗ 预测结果不一致！")
    
    print("\nDraft模型预测的token不同是预期的（因为路由被修改了）")
    print(f"Draft token: {first_token_draft}")

if __name__ == "__main__":
    test_first_token()

