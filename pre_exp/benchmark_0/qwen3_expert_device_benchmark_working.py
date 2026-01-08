#!/usr/bin/env python
"""
Qwen3 MoE Expert设备分布性能测试实验 - 工作版本
基于正确的推理流程实现
"""

import os
# 设置环境变量
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
import torch
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F


class CustomQwen3MoELayer(torch.nn.Module):
    """自定义的Qwen3 MoE层，支持手动控制expert设备分布"""
    
    def __init__(self, original_layer, model_rotary_emb, num_experts_per_tok=8):
        super().__init__()
        self.original_layer = original_layer
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = len(original_layer.mlp.experts)
        
        # 复制原始组件
        self.input_layernorm = copy.deepcopy(original_layer.input_layernorm)
        self.self_attn = copy.deepcopy(original_layer.self_attn)
        self.post_attention_layernorm = copy.deepcopy(original_layer.post_attention_layernorm)
        
        # 设置rotary embedding引用
        self.rotary_emb = model_rotary_emb
        
        # 复制MoE组件
        self.gate = copy.deepcopy(original_layer.mlp.gate)
        self.experts = torch.nn.ModuleList([
            copy.deepcopy(expert) for expert in original_layer.mlp.experts
        ])
        
        # 设备分布控制
        self.expert_device_assignment = {}
        self.force_routing = None
        
    def set_expert_device_distribution(self, cpu_ratio: float):
        """设置expert的设备分布"""
        num_cpu_experts = int(self.num_experts_per_tok * cpu_ratio)
        num_gpu_experts = self.num_experts_per_tok - num_cpu_experts
        
        self.expert_device_assignment = {}
        
        # 将前num_cpu_experts个激活的expert放到CPU
        for i in range(num_cpu_experts):
            self.expert_device_assignment[i] = 'cpu'
        
        # 将剩余的expert放到GPU
        for i in range(num_cpu_experts, self.num_experts_per_tok):
            self.expert_device_assignment[i] = 'cuda:0'
            
        print(f"设备分布: CPU {num_cpu_experts} experts, GPU {num_gpu_experts} experts")
        
    def set_forced_routing(self, expert_indices: List[int]):
        """强制路由到指定的experts"""
        if len(expert_indices) != self.num_experts_per_tok:
            raise ValueError(f"必须指定{self.num_experts_per_tok}个expert索引")
        self.force_routing = expert_indices
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, 
                output_attentions=False, use_cache=False, cache_position=None, **kwargs):
        """前向传播"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # 生成position embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
        
        # 获取rotary embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # 注意力机制
        attn_outputs = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs
        )
        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states
        
        # MoE部分
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # 路由决策
        if self.force_routing is not None:
            # 使用强制路由
            selected_experts = self.force_routing
            routing_weights = torch.ones(self.num_experts_per_tok, 
                                       device=hidden_states.device, dtype=hidden_states.dtype) / self.num_experts_per_tok
        else:
            # 使用原始路由
            router_logits = self.gate(hidden_states)
            routing_weights, selected_experts = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)
            routing_weights = F.softmax(routing_weights, dim=-1)
            selected_experts = selected_experts.squeeze()
            if selected_experts.dim() > 0:
                selected_experts = selected_experts[0]  # 取第一个token的路由结果
            routing_weights = routing_weights.squeeze()[0] if routing_weights.dim() > 1 else routing_weights.squeeze()
        
        # Expert计算
        expert_outputs = []
        
        for i, expert_idx in enumerate(selected_experts):
            if isinstance(expert_idx, torch.Tensor):
                expert_idx = expert_idx.item()
            
            # 获取对应的expert
            expert = self.experts[expert_idx]
            
            # 根据设备分布移动expert
            if self.expert_device_assignment and i in self.expert_device_assignment:
                target_device = self.expert_device_assignment[i]
                expert = expert.to(target_device)
                
                # 移动输入到相同设备
                input_for_expert = hidden_states.to(target_device)
                expert_output = expert(input_for_expert)
                # 移回原设备
                expert_output = expert_output.to(hidden_states.device)
            else:
                expert_output = expert(hidden_states)
            
            expert_outputs.append(expert_output)
        
        # 加权组合expert输出
        final_output = torch.zeros_like(hidden_states)
        for i, expert_output in enumerate(expert_outputs):
            weight = routing_weights[i]
            final_output += weight * expert_output
            
        hidden_states = residual + final_output
        
        # 返回格式与原始layer一致
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[2],) if len(attn_outputs) > 2 else (None,)
        
        return outputs


def create_test_inputs(hidden_size: int, batch_size: int = 1, seq_len: int = 1):
    """创建测试输入"""
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    
    return {
        'hidden_states': hidden_states,
        'position_ids': position_ids
    }


def benchmark_device_distribution(custom_layer: CustomQwen3MoELayer, 
                                test_inputs: Dict,
                                cpu_ratios: List[float],
                                num_runs: int = 50) -> Dict[float, float]:
    """测试不同设备分布下的性能"""
    
    results = {}
    
    for cpu_ratio in cpu_ratios:
        print(f"\n测试CPU比例: {cpu_ratio:.1f} ({int(cpu_ratio*8)}::{int((1-cpu_ratio)*8)})")
        
        # 设置设备分布
        custom_layer.set_expert_device_distribution(cpu_ratio)
        
        # 强制使用前8个experts以保证一致性
        custom_layer.set_forced_routing(list(range(8)))
        
        # 预热
        print("  预热中...")
        with torch.no_grad():
            for _ in range(5):
                outputs = custom_layer(**test_inputs)
        
        # 同步
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 正式测试
        print(f"  执行{num_runs}次测试...")
        times = []
        
        with torch.no_grad():
            for i in range(num_runs):
                # 开始计时
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()
                
                # 执行前向传播
                outputs = custom_layer(**test_inputs)
                
                # 结束计时
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                
                times.append((end_time - start_time) * 1000)  # 转换为毫秒
                
                if (i + 1) % 10 == 0:
                    print(f"    完成 {i + 1}/{num_runs}")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        results[cpu_ratio] = avg_time
        
        print(f"  平均时间: {avg_time:.4f} ms (±{std_time:.4f} ms)")
    
    return results


def benchmark_batch_size_scaling(custom_layer: CustomQwen3MoELayer,
                                hidden_size: int,
                                batch_sizes: List[int],
                                cpu_ratios: List[float],
                                num_runs: int = 30) -> Dict[int, Dict[float, float]]:
    """测试不同批次大小和设备分布下的性能"""
    
    all_results = {}
    
    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"测试批次大小: {batch_size}")
        print(f"{'='*60}")
        
        # 创建当前批次大小的测试输入
        test_inputs = create_test_inputs(hidden_size, batch_size=batch_size, seq_len=1)
        
        # 移动输入到GPU
        for key in test_inputs:
            test_inputs[key] = test_inputs[key].to('cuda:0')
        
        # 测试不同设备分布
        batch_results = benchmark_device_distribution(custom_layer, test_inputs, cpu_ratios, num_runs)
        all_results[batch_size] = batch_results
        
        print(f"\n批次大小 {batch_size} 完成!")
        
        # 显示当前批次的简要结果
        fastest_ratio = min(batch_results.keys(), key=lambda k: batch_results[k])
        slowest_ratio = max(batch_results.keys(), key=lambda k: batch_results[k])
        print(f"  最快配置: {int(fastest_ratio*8)}::{int((1-fastest_ratio)*8)} - {batch_results[fastest_ratio]:.4f}ms")
        print(f"  最慢配置: {int(slowest_ratio*8)}::{int((1-slowest_ratio)*8)} - {batch_results[slowest_ratio]:.4f}ms")
    
    return all_results


def plot_device_distribution_results(results: Dict[float, float], save_path: str):
    """绘制设备分布性能结果"""
    cpu_ratios = sorted(results.keys())
    execution_times = [results[ratio] for ratio in cpu_ratios]
    
    # 转换为百分比显示
    cpu_percentages = [ratio * 100 for ratio in cpu_ratios]
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制折线图
    plt.plot(cpu_percentages, execution_times, 'o-', linewidth=3, markersize=10, 
             color='darkblue', alpha=0.8)
    
    # 设置图表属性
    plt.xlabel('CPU Expert Ratio (%)', fontsize=14)
    plt.ylabel('Execution Time (ms)', fontsize=14)
    plt.title('Decode Block Execution Time vs CPU/GPU Expert Distribution', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴
    plt.xlim(-5, 105)
    plt.xticks(cpu_percentages)
    
    # 添加数值标注
    for cpu_pct, exec_time in zip(cpu_percentages, execution_times):
        plt.annotate(f'{exec_time:.2f}ms', 
                    (cpu_pct, exec_time), 
                    textcoords="offset points", 
                    xytext=(0,15), 
                    ha='center', 
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # 添加设备分布标签
    ax2 = plt.twiny()
    ax2.set_xlim(-5, 105)
    ax2.set_xticks(cpu_percentages)
    device_labels = [f"{int(ratio*8)}::{int((1-ratio)*8)}" for ratio in cpu_ratios]
    ax2.set_xticklabels(device_labels)
    ax2.set_xlabel('CPU::GPU Expert Count', fontsize=12)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")
    
    return plt


def plot_batch_size_results(all_results: Dict[int, Dict[float, float]], save_dir: str):
    """绘制批次大小缩放结果"""
    
    # 为每个设备分布创建一个图表
    cpu_ratios = sorted(list(all_results.values())[0].keys())
    batch_sizes = sorted(all_results.keys())
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Expert Device Distribution Performance vs Batch Size', fontsize=16)
    
    # 选择几个代表性的配置进行绘制
    selected_ratios = [0.0, 0.25, 0.5, 1.0]  # 0:8, 2:6, 4:4, 8:0
    colors = ['blue', 'green', 'orange', 'red']
    
    for idx, (cpu_ratio, color) in enumerate(zip(selected_ratios, colors)):
        ax = axes[idx // 2, idx % 2]
        
        execution_times = [all_results[bs][cpu_ratio] for bs in batch_sizes]
        
        ax.plot(batch_sizes, execution_times, 'o-', linewidth=2, markersize=8, 
                color=color, alpha=0.8)
        
        cpu_count = int(cpu_ratio * 8)
        gpu_count = 8 - cpu_count
        ax.set_title(f'CPU:GPU = {cpu_count}:{gpu_count}', fontsize=14)
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Execution Time (ms)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(batch_sizes)
        
        # 添加数值标注
        for bs, time in zip(batch_sizes, execution_times):
            ax.annotate(f'{time:.2f}', (bs, time), 
                       textcoords="offset points", xytext=(0,10), 
                       ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = f"{save_dir}/batch_size_scaling.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"批次大小缩放图表已保存到: {save_path}")
    
    plt.close()
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 准备热力图数据
    heatmap_data = np.zeros((len(cpu_ratios), len(batch_sizes)))
    for i, cpu_ratio in enumerate(cpu_ratios):
        for j, batch_size in enumerate(batch_sizes):
            heatmap_data[i, j] = all_results[batch_size][cpu_ratio]
    
    # 绘制热力图
    im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
    
    # 设置标签
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticks(range(len(cpu_ratios)))
    ax.set_yticklabels([f"{int(r*8)}::{int((1-r)*8)}" for r in cpu_ratios])
    
    ax.set_xlabel('Batch Size', fontsize=14)
    ax.set_ylabel('CPU::GPU Expert Distribution', fontsize=14)
    ax.set_title('Execution Time Heatmap (ms)', fontsize=16)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Execution Time (ms)', fontsize=12)
    
    # 在每个单元格中添加数值
    for i in range(len(cpu_ratios)):
        for j in range(len(batch_sizes)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                          ha="center", va="center", color="white", fontweight="bold")
    
    plt.tight_layout()
    
    # 保存热力图
    heatmap_path = f"{save_dir}/performance_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"性能热力图已保存到: {heatmap_path}")
    
    plt.close()


def save_results_to_file(results: Dict[float, float], file_path: str):
    """保存结果到文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("Qwen3 MoE Expert设备分布性能测试结果 - 工作版本\n")
        f.write("="*60 + "\n\n")
        
        f.write("CPU比例\tCPU::GPU分布\t执行时间(ms)\n")
        f.write("-" * 40 + "\n")
        
        for cpu_ratio in sorted(results.keys()):
            cpu_count = int(cpu_ratio * 8)
            gpu_count = 8 - cpu_count
            exec_time = results[cpu_ratio]
            f.write(f"{cpu_ratio:.1f}\t\t{cpu_count}::{gpu_count}\t\t{exec_time:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("性能分析:\n")
        
        times = list(results.values())
        min_time = min(times)
        max_time = max(times)
        
        # 找到最快和最慢的配置
        fastest_ratio = min(results.keys(), key=lambda k: results[k])
        slowest_ratio = max(results.keys(), key=lambda k: results[k])
        
        f.write(f"最快配置: CPU {fastest_ratio:.1f} ({int(fastest_ratio*8)}::{int((1-fastest_ratio)*8)}) - {min_time:.4f}ms\n")
        f.write(f"最慢配置: CPU {slowest_ratio:.1f} ({int(slowest_ratio*8)}::{int((1-slowest_ratio)*8)}) - {max_time:.4f}ms\n")
        f.write(f"性能差异: {((max_time - min_time) / min_time * 100):.1f}%\n")


def save_batch_size_results(all_results: Dict[int, Dict[float, float]], save_dir: str):
    """保存批次大小实验结果"""
    
    # 创建目录
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存详细结果
    results_file = f"{save_dir}/batch_size_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Qwen3 MoE Expert设备分布与批次大小性能测试结果\n")
        f.write("="*80 + "\n\n")
        
        batch_sizes = sorted(all_results.keys())
        cpu_ratios = sorted(list(all_results.values())[0].keys())
        
        # 写入表头
        f.write("批次大小")
        for cpu_ratio in cpu_ratios:
            cpu_count = int(cpu_ratio * 8)
            gpu_count = 8 - cpu_count
            f.write(f"\t{cpu_count}::{gpu_count}")
        f.write("\n")
        f.write("-" * 100 + "\n")
        
        # 写入数据
        for batch_size in batch_sizes:
            f.write(f"{batch_size}")
            for cpu_ratio in cpu_ratios:
                exec_time = all_results[batch_size][cpu_ratio]
                f.write(f"\t{exec_time:.4f}")
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("性能分析:\n\n")
        
        # 分析每个批次大小的最优配置
        for batch_size in batch_sizes:
            batch_results = all_results[batch_size]
            fastest_ratio = min(batch_results.keys(), key=lambda k: batch_results[k])
            slowest_ratio = max(batch_results.keys(), key=lambda k: batch_results[k])
            
            fastest_time = batch_results[fastest_ratio]
            slowest_time = batch_results[slowest_ratio]
            
            f.write(f"批次大小 {batch_size}:\n")
            f.write(f"  最快配置: {int(fastest_ratio*8)}::{int((1-fastest_ratio)*8)} - {fastest_time:.4f}ms\n")
            f.write(f"  最慢配置: {int(slowest_ratio*8)}::{int((1-slowest_ratio)*8)} - {slowest_time:.4f}ms\n")
            f.write(f"  性能差异: {((slowest_time - fastest_time) / fastest_time * 100):.1f}%\n\n")
        
        # 分析批次大小缩放性
        f.write("批次大小缩放分析:\n")
        for cpu_ratio in cpu_ratios:
            cpu_count = int(cpu_ratio * 8)
            gpu_count = 8 - cpu_count
            f.write(f"\n配置 {cpu_count}::{gpu_count}:\n")
            
            times = [all_results[bs][cpu_ratio] for bs in batch_sizes]
            min_time = min(times)
            max_time = max(times)
            
            f.write(f"  时间范围: {min_time:.4f} - {max_time:.4f} ms\n")
            f.write(f"  缩放比: {max_time/min_time:.2f}x\n")
    
    print(f"批次大小实验结果已保存到: {results_file}")
    
    # 为每个批次大小单独保存结果
    for batch_size in batch_sizes:
        batch_file = f"{save_dir}/batch_size_{batch_size}_results.txt"
        save_results_to_file(all_results[batch_size], batch_file)
    
    print(f"单独的批次结果文件已保存到: {save_dir}/batch_size_*_results.txt")


def main():
    model_path = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"
    
    print("="*60)
    print("Qwen3 MoE Expert设备分布性能测试 - 工作版本")
    print("="*60)
    
    # 检查模型路径
    if not os.path.exists(model_path):
        print(f"错误：模型路径 {model_path} 不存在")
        return
    
    # 加载模型
    print(f"\n1. 加载模型: {model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=None,
            local_files_only=True,
            trust_remote_code=True,
        )
        print("模型加载成功!")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 提取第一层decode block
    print("\n2. 提取第一层decode block...")
    first_layer = model.model.layers[0]
    print(f"Layer类型: {type(first_layer)}")
    
    # 创建自定义MoE层
    print("\n3. 创建自定义MoE层...")
    custom_layer = CustomQwen3MoELayer(first_layer, model.model.rotary_emb)
    custom_layer = custom_layer.to('cuda:0')  # 主要在GPU上
    print("自定义层创建完成!")
    
    # 创建测试输入
    print("\n4. 创建测试输入...")
    hidden_size = model.config.hidden_size
    test_inputs = create_test_inputs(hidden_size, batch_size=1, seq_len=1)
    
    # 移动输入到GPU
    for key in test_inputs:
        test_inputs[key] = test_inputs[key].to('cuda:0')
    
    print(f"输入形状: {test_inputs['hidden_states'].shape}")
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        return
    
    # 定义测试的CPU比例 (0:8, 1:7, 2:6, ..., 8:0)
    cpu_ratios = [i/8 for i in range(9)]  # [0.0, 0.125, 0.25, ..., 1.0]
    
    # 定义测试的批次大小
    batch_sizes = list(range(1, 11))  # [1, 2, 3, ..., 10]
    
    print(f"\n5. 开始批次大小缩放实验...")
    print(f"测试批次大小: {batch_sizes}")
    print(f"测试设备分布: {len(cpu_ratios)} 种CPU/GPU配置")
    
    # 执行批次大小缩放测试
    all_results = benchmark_batch_size_scaling(custom_layer, hidden_size, batch_sizes, cpu_ratios, num_runs=20)
    
    # 保存结果
    save_dir = "/zx_data1/sparsity/on_device_sd/pre_exp/benchmarks/results/bs_expert_device_distribution"
    print(f"\n6. 保存实验结果...")
    save_batch_size_results(all_results, save_dir)
    
    # 绘制结果图表
    print(f"\n7. 生成结果图表...")
    plot_batch_size_results(all_results, save_dir)
    
    print("\n" + "="*60)
    print("批次大小缩放实验完成!")
    print("="*60)
    print(f"结果保存目录: {save_dir}")
    print("生成的文件:")
    print("  - batch_size_results.txt (汇总结果)")
    print("  - batch_size_*_results.txt (单独批次结果)")
    print("  - batch_size_scaling.png (缩放图表)")
    print("  - performance_heatmap.png (性能热力图)")
    print("="*60)


if __name__ == "__main__":
    main()
