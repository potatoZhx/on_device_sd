#!/usr/bin/env python
"""
检查Qwen3 MoE模型的expert激活数量
"""

import os
# 设置环境变量
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoConfig

def check_qwen3_expert_config():
    """检查Qwen3模型的expert配置"""
    model_path = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"
    
    print("="*60)
    print("Qwen3 MoE Expert配置检查")
    print("="*60)
    
    # 加载配置
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    
    print(f"模型路径: {model_path}")
    print(f"模型类型: {config.model_type}")
    print(f"架构: {config.architectures}")
    print()
    
    # MoE相关配置
    print("MoE配置信息:")
    print(f"  总层数 (num_hidden_layers): {config.num_hidden_layers}")
    print(f"  每层expert总数 (num_experts): {config.num_experts}")
    print(f"  每个token激活的expert数 (num_experts_per_tok): {config.num_experts_per_tok}")
    print(f"  MoE中间层大小 (moe_intermediate_size): {config.moe_intermediate_size}")
    print(f"  普通MLP中间层大小 (intermediate_size): {config.intermediate_size}")
    print(f"  仅MLP层 (mlp_only_layers): {config.mlp_only_layers}")
    print()
    
    # 计算激活expert的比例
    activation_ratio = config.num_experts_per_tok / config.num_experts
    print("激活统计:")
    print(f"  每个token激活expert比例: {activation_ratio:.1%} ({config.num_experts_per_tok}/{config.num_experts})")
    print(f"  未激活expert数量: {config.num_experts - config.num_experts_per_tok}")
    print()
    
    # 检查哪些层是MoE层
    if hasattr(config, 'mlp_only_layers') and config.mlp_only_layers:
        moe_layers = [i for i in range(config.num_hidden_layers) if i not in config.mlp_only_layers]
        mlp_layers = config.mlp_only_layers
        print(f"MoE层 (有experts): {len(moe_layers)} 层")
        print(f"  层索引: {moe_layers}")
        print(f"普通MLP层 (无experts): {len(mlp_layers)} 层")
        print(f"  层索引: {mlp_layers}")
    else:
        print(f"所有 {config.num_hidden_layers} 层都是MoE层")
        print(f"  每层都有 {config.num_experts} 个experts")
        print(f"  每层每个token都激活 {config.num_experts_per_tok} 个experts")
    
    print()
    print("="*60)
    print("总结:")
    print(f"• 模型总共有 {config.num_hidden_layers} 层")
    print(f"• 每个MoE层包含 {config.num_experts} 个experts")
    print(f"• 每个token在每层激活 {config.num_experts_per_tok} 个experts")
    print(f"• 激活率: {activation_ratio:.1%}")
    print("="*60)

if __name__ == "__main__":
    check_qwen3_expert_config()
