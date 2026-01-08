#!/usr/bin/env python
"""
MoE Expert cpu/gpu 性能测试实验
测试单个expert在GPU和CPU上的计算时间
"""

import os
# 设置环境变量以减少依赖和网络请求
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
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_test_input(hidden_size: int, batch_size: int = 1, seq_len: int = 1) -> torch.Tensor:
    """创建测试输入数据"""
    # 创建随机输入，模拟实际的hidden states
    test_input = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
    return test_input


def extract_first_layer_experts(model) -> List[torch.nn.Module]:
    """从模型第一层提取MoE experts"""
    try:
        # 获取第一层
        first_layer = model.model.layers[0]
        
        # 检查是否有MoE层
        if hasattr(first_layer, 'mlp') and hasattr(first_layer.mlp, 'experts'):
            experts = first_layer.mlp.experts
            print(f"找到 {len(experts)} 个experts在第一层MoE中")
            return experts
        else:
            print("第一层没有找到MoE experts")
            return []
    except Exception as e:
        print(f"提取experts时出错: {e}")
        return []


def benchmark_expert_on_device(expert: torch.nn.Module, 
                             test_input: torch.Tensor, 
                             device: str, 
                             num_runs: int = 100) -> Dict[str, float]:
    """在指定设备上测试expert性能"""
    print(f"在 {device} 上测试expert性能...")
    
    # 创建expert的深拷贝以避免设备冲突
    expert_copy = copy.deepcopy(expert)
    
    # 移动expert和输入到指定设备
    expert_copy = expert_copy.to(device)
    test_input = test_input.to(device)
    
    # 预热
    print(f"  预热阶段...")
    with torch.no_grad():
        for _ in range(10):
            _ = expert_copy(test_input)
    
    # 同步GPU操作
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    
    # 正式测试
    print(f"  正式测试 {num_runs} 次...")
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            # 记录开始时间
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            start_time = time.time()
            
            # 执行前向传播
            output = expert_copy(test_input)
            
            # 记录结束时间
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
            
            if (i + 1) % 20 == 0:
                print(f"    完成 {i + 1}/{num_runs} 次测试")
    
    # 计算统计信息
    times = np.array(times)
    stats = {
        'mean_time_ms': float(np.mean(times)),
        'std_time_ms': float(np.std(times)),
        'min_time_ms': float(np.min(times)),
        'max_time_ms': float(np.max(times)),
        'median_time_ms': float(np.median(times)),
        'device': device,
        'num_runs': num_runs
    }
    
    return stats


def print_benchmark_results(gpu_stats: Dict, cpu_stats: Dict):
    """打印基准测试结果"""
    print("\n" + "="*60)
    print("Expert性能测试结果")
    print("="*60)
    
    print(f"\nGPU性能 ({gpu_stats['device']}):")
    print(f"  平均时间: {gpu_stats['mean_time_ms']:.4f} ms")
    print(f"  标准差:   {gpu_stats['std_time_ms']:.4f} ms")
    print(f"  最小时间: {gpu_stats['min_time_ms']:.4f} ms")
    print(f"  最大时间: {gpu_stats['max_time_ms']:.4f} ms")
    print(f"  中位数:   {gpu_stats['median_time_ms']:.4f} ms")
    
    print(f"\nCPU性能 ({cpu_stats['device']}):")
    print(f"  平均时间: {cpu_stats['mean_time_ms']:.4f} ms")
    print(f"  标准差:   {cpu_stats['std_time_ms']:.4f} ms")
    print(f"  最小时间: {cpu_stats['min_time_ms']:.4f} ms")
    print(f"  最大时间: {cpu_stats['max_time_ms']:.4f} ms")
    print(f"  中位数:   {cpu_stats['median_time_ms']:.4f} ms")
    
    # 计算加速比
    speedup = cpu_stats['mean_time_ms'] / gpu_stats['mean_time_ms']
    print(f"\n性能对比:")
    print(f"  GPU相对CPU加速比: {speedup:.2f}x")
    print(f"  GPU比CPU快: {((cpu_stats['mean_time_ms'] - gpu_stats['mean_time_ms']) / cpu_stats['mean_time_ms'] * 100):.1f}%")
    
    print("="*60)


def benchmark_expert_seq_len_scaling(expert: torch.nn.Module, 
                                    hidden_size: int,
                                    seq_lengths: List[int] = list(range(1, 11)),
                                    num_runs: int = 50) -> Dict[str, Dict[int, float]]:
    """测试不同序列长度下expert的性能"""
    print("\n开始序列长度缩放测试...")
    
    gpu_device = "cuda:0"
    cpu_device = "cpu"
    
    results = {
        'gpu_times': {},
        'cpu_times': {},
        'seq_lengths': seq_lengths
    }
    
    for seq_len in seq_lengths:
        print(f"\n测试序列长度: {seq_len}")
        
        # 创建当前序列长度的测试输入
        test_input = create_test_input(hidden_size, batch_size=1, seq_len=seq_len)
        print(f"  输入形状: {test_input.shape}")
        
        # 测试GPU性能
        print(f"  在GPU上测试...")
        gpu_stats = benchmark_expert_on_device(expert, test_input.clone(), gpu_device, num_runs=num_runs)
        results['gpu_times'][seq_len] = gpu_stats['mean_time_ms']
        
        # 测试CPU性能
        print(f"  在CPU上测试...")
        cpu_stats = benchmark_expert_on_device(expert, test_input.clone(), cpu_device, num_runs=num_runs//2)  # CPU测试次数减半
        results['cpu_times'][seq_len] = cpu_stats['mean_time_ms']
        
        print(f"  GPU: {gpu_stats['mean_time_ms']:.4f}ms, CPU: {cpu_stats['mean_time_ms']:.4f}ms")
    
    return results


def plot_seq_len_scaling_results(results: Dict[str, Dict[int, float]], save_path: str = None):
    """绘制序列长度缩放结果"""
    seq_lengths = results['seq_lengths']
    gpu_times = [results['gpu_times'][seq_len] for seq_len in seq_lengths]
    cpu_times = [results['cpu_times'][seq_len] for seq_len in seq_lengths]
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制折线图
    plt.plot(seq_lengths, gpu_times, 'o-', linewidth=2, markersize=8, 
             label='GPU', color='blue', alpha=0.8)
    plt.plot(seq_lengths, cpu_times, 's-', linewidth=2, markersize=8, 
             label='CPU', color='red', alpha=0.8)
    
    # 设置图表属性
    plt.xlabel('Sequence Length', fontsize=14)
    plt.ylabel('Computation Time (ms)', fontsize=14)
    plt.title('Expert Computation Time vs Sequence Length', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴
    plt.xticks(seq_lengths)
    plt.xlim(0.5, max(seq_lengths) + 0.5)
    
    # 添加数值标注
    for i, (seq_len, gpu_time, cpu_time) in enumerate(zip(seq_lengths, gpu_times, cpu_times)):
        plt.annotate(f'{gpu_time:.2f}', (seq_len, gpu_time), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        plt.annotate(f'{cpu_time:.2f}', (seq_len, cpu_time), 
                    textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    # 显示图表（如果在交互环境中）
    try:
        plt.show()
    except:
        pass
    
    return plt


def save_scaling_results_to_file(results: Dict[str, Dict[int, float]], file_path: str):
    """保存序列长度缩放结果到文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("Expert序列长度缩放测试结果\n")
        f.write("="*60 + "\n\n")
        
        f.write("序列长度\tGPU时间(ms)\tCPU时间(ms)\t加速比\n")
        f.write("-" * 50 + "\n")
        
        for seq_len in results['seq_lengths']:
            gpu_time = results['gpu_times'][seq_len]
            cpu_time = results['cpu_times'][seq_len]
            speedup = cpu_time / gpu_time
            f.write(f"{seq_len}\t\t{gpu_time:.4f}\t\t{cpu_time:.4f}\t\t{speedup:.2f}x\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("统计信息:\n")
        
        gpu_times = list(results['gpu_times'].values())
        cpu_times = list(results['cpu_times'].values())
        
        f.write(f"GPU时间范围: {min(gpu_times):.4f} - {max(gpu_times):.4f} ms\n")
        f.write(f"CPU时间范围: {min(cpu_times):.4f} - {max(cpu_times):.4f} ms\n")
        
        avg_speedup = np.mean([cpu_times[i] / gpu_times[i] for i in range(len(gpu_times))])
        f.write(f"平均加速比: {avg_speedup:.2f}x\n")


def main():
    # 模型路径
    model_path = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型路径 {model_path} 不存在")
        return
    
    print("="*60)
    print("Qwen3 MoE Expert性能测试实验")
    print("="*60)
    
    # 加载模型
    print(f"\n1. 加载模型: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        
        # 只加载模型结构，不需要分配到GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=None,  # 不自动分配设备
            local_files_only=True,
            trust_remote_code=True,
        )
        print("模型加载成功!")
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 提取第一层的experts
    print("\n2. 提取第一层MoE experts...")
    experts = extract_first_layer_experts(model)
    
    if not experts:
        print("未找到experts，退出实验")
        return
    
    # 选择第一个expert进行测试
    expert = experts[0]
    print(f"选择第一个expert进行测试")
    print(f"Expert类型: {type(expert)}")
    
    # 获取hidden_size
    hidden_size = model.config.hidden_size
    print(f"Hidden size: {hidden_size}")
    
    # 创建测试输入
    print("\n3. 创建测试输入...")
    test_input = create_test_input(hidden_size, batch_size=1, seq_len=1)
    print(f"测试输入形状: {test_input.shape}")
    print(f"测试输入数据类型: {test_input.dtype}")
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，跳过GPU测试")
        return
    
    gpu_device = "cuda:0"
    cpu_device = "cpu"
    
    # 测试GPU性能
    print("\n4. 测试GPU性能...")
    gpu_stats = benchmark_expert_on_device(expert, test_input.clone(), gpu_device, num_runs=20)
    
    # 测试CPU性能
    print("\n5. 测试CPU性能...")
    cpu_stats = benchmark_expert_on_device(expert, test_input.clone(), cpu_device, num_runs=20)  # CPU测试次数少一些
    
    # 打印结果
    print_benchmark_results(gpu_stats, cpu_stats)
    
    # 保存结果到文件
    results_file = "/zx_data1/sparsity/on_device_sd/pre_exp/qwen3_expert_benchmark_results.txt"
    print(f"\n保存结果到: {results_file}")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Qwen3 MoE Expert性能测试结果\n")
        f.write("="*60 + "\n\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"测试输入形状: {test_input.shape}\n")
        f.write(f"Expert数量: {len(experts)}\n\n")
        
        f.write(f"GPU性能 ({gpu_stats['device']}):\n")
        f.write(f"  平均时间: {gpu_stats['mean_time_ms']:.4f} ms\n")
        f.write(f"  标准差:   {gpu_stats['std_time_ms']:.4f} ms\n")
        f.write(f"  最小时间: {gpu_stats['min_time_ms']:.4f} ms\n")
        f.write(f"  最大时间: {gpu_stats['max_time_ms']:.4f} ms\n")
        f.write(f"  中位数:   {gpu_stats['median_time_ms']:.4f} ms\n\n")
        
        f.write(f"CPU性能 ({cpu_stats['device']}):\n")
        f.write(f"  平均时间: {cpu_stats['mean_time_ms']:.4f} ms\n")
        f.write(f"  标准差:   {cpu_stats['std_time_ms']:.4f} ms\n")
        f.write(f"  最小时间: {cpu_stats['min_time_ms']:.4f} ms\n")
        f.write(f"  最大时间: {cpu_stats['max_time_ms']:.4f} ms\n")
        f.write(f"  中位数:   {cpu_stats['median_time_ms']:.4f} ms\n\n")
        
        speedup = cpu_stats['mean_time_ms'] / gpu_stats['mean_time_ms']
        f.write(f"性能对比:\n")
        f.write(f"  GPU相对CPU加速比: {speedup:.2f}x\n")
        f.write(f"  GPU比CPU快: {((cpu_stats['mean_time_ms'] - gpu_stats['mean_time_ms']) / cpu_stats['mean_time_ms'] * 100):.1f}%\n")
    
    print("基础性能测试完成!")
    
    # 序列长度缩放测试
    print("\n" + "="*60)
    print("开始序列长度缩放测试 (seq_len: 1-10)")
    print("="*60)
    
    # 进行序列长度缩放测试
    seq_len_results = benchmark_expert_seq_len_scaling(
        expert, 
        hidden_size, 
        seq_lengths=list(range(1, 11)),
        num_runs=30
    )
    
    # 绘制结果图表
    plot_file = "/zx_data1/sparsity/on_device_sd/pre_exp/qwen3_expert_seq_len_scaling.png"
    print(f"\n绘制序列长度缩放图表...")
    plot_seq_len_scaling_results(seq_len_results, save_path=plot_file)
    
    # 保存详细结果
    scaling_results_file = "/zx_data1/sparsity/on_device_sd/pre_exp/qwen3_expert_seq_len_results.txt"
    print(f"保存序列长度缩放结果到: {scaling_results_file}")
    save_scaling_results_to_file(seq_len_results, scaling_results_file)
    
    print("\n" + "="*60)
    print("所有实验完成!")
    print("="*60)
    print(f"基础性能结果: {results_file}")
    print(f"序列长度缩放结果: {scaling_results_file}")
    print(f"序列长度缩放图表: {plot_file}")
    print("="*60)


if __name__ == "__main__":
    main()
