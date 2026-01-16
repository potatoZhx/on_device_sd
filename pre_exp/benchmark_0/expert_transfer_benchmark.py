#!/usr/bin/env python
"""
Expert传输时间测试实验
测试将1-8个expert从CPU传输到GPU的时间
"""

import os
# 设置环境变量
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM


def get_expert_memory_size(expert: torch.nn.Module) -> float:
    """获取expert的内存大小（MB）"""
    param_size = 0
    buffer_size = 0
    
    for param in expert.parameters():
        param_size += param.numel() * param.element_size()
    
    for buffer in expert.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    total_size_mb = (param_size + buffer_size) / (1024 * 1024)
    return total_size_mb


def benchmark_expert_transfer(experts: List[torch.nn.Module], 
                            num_experts_list: List[int],
                            num_runs: int = 50) -> Dict[int, Dict[str, float]]:
    """测试不同数量expert的传输时间"""
    
    results = {}
    
    for num_experts in num_experts_list:
        print(f"\n{'='*60}")
        print(f"测试传输 {num_experts} 个experts")
        print(f"{'='*60}")
        
        # 选择前num_experts个experts进行测试
        selected_experts = experts[:num_experts]
        
        # 计算总内存大小
        total_memory_mb = sum(get_expert_memory_size(expert) for expert in selected_experts)
        print(f"总内存大小: {total_memory_mb:.2f} MB")
        
        # 确保所有experts都在CPU上
        cpu_experts = []
        for expert in selected_experts:
            cpu_expert = copy.deepcopy(expert).to('cpu')
            cpu_experts.append(cpu_expert)
        
        # 预热GPU
        print("预热GPU...")
        dummy = torch.randn(1000, 1000, device='cuda:0')
        del dummy
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 测试CPU到GPU传输时间
        print(f"测试CPU->GPU传输时间 ({num_runs}次测试)...")
        cpu_to_gpu_times = []
        
        for run in range(num_runs):
            # 确保experts在CPU上
            test_experts = [copy.deepcopy(expert).to('cpu') for expert in selected_experts]
            
            # 同步并开始计时
            torch.cuda.synchronize()
            start_time = time.time()
            
            # 传输到GPU
            gpu_experts = []
            for expert in test_experts:
                gpu_expert = expert.to('cuda:0')
                gpu_experts.append(gpu_expert)
            
            # 同步并结束计时
            torch.cuda.synchronize()
            end_time = time.time()
            
            transfer_time = (end_time - start_time) * 1000  # 转换为毫秒
            cpu_to_gpu_times.append(transfer_time)
            
            # 清理GPU内存
            del gpu_experts
            torch.cuda.empty_cache()
            
            if (run + 1) % 10 == 0:
                print(f"  完成 {run + 1}/{num_runs} 次测试")
        
        # 测试GPU到CPU传输时间
        print(f"测试GPU->CPU传输时间 ({num_runs}次测试)...")
        gpu_to_cpu_times = []
        
        for run in range(num_runs):
            # 确保experts在GPU上
            test_experts = [copy.deepcopy(expert).to('cuda:0') for expert in selected_experts]
            
            # 同步并开始计时
            torch.cuda.synchronize()
            start_time = time.time()
            
            # 传输到CPU
            cpu_experts_test = []
            for expert in test_experts:
                cpu_expert = expert.to('cpu')
                cpu_experts_test.append(cpu_expert)
            
            # 同步并结束计时
            torch.cuda.synchronize()
            end_time = time.time()
            
            transfer_time = (end_time - start_time) * 1000  # 转换为毫秒
            gpu_to_cpu_times.append(transfer_time)
            
            # 清理
            del test_experts, cpu_experts_test
            torch.cuda.empty_cache()
            
            if (run + 1) % 10 == 0:
                print(f"  完成 {run + 1}/{num_runs} 次测试")
        
        # 计算统计信息
        cpu_to_gpu_stats = {
            'mean': np.mean(cpu_to_gpu_times),
            'std': np.std(cpu_to_gpu_times),
            'min': np.min(cpu_to_gpu_times),
            'max': np.max(cpu_to_gpu_times),
            'median': np.median(cpu_to_gpu_times)
        }
        
        gpu_to_cpu_stats = {
            'mean': np.mean(gpu_to_cpu_times),
            'std': np.std(gpu_to_cpu_times),
            'min': np.min(gpu_to_cpu_times),
            'max': np.max(gpu_to_cpu_times),
            'median': np.median(gpu_to_cpu_times)
        }
        
        results[num_experts] = {
            'memory_mb': total_memory_mb,
            'cpu_to_gpu': cpu_to_gpu_stats,
            'gpu_to_cpu': gpu_to_cpu_stats
        }
        
        # 计算传输带宽
        cpu_to_gpu_bandwidth = total_memory_mb / (cpu_to_gpu_stats['mean'] / 1000)  # MB/s
        gpu_to_cpu_bandwidth = total_memory_mb / (gpu_to_cpu_stats['mean'] / 1000)  # MB/s
        
        print(f"\n结果总结:")
        print(f"  CPU->GPU: {cpu_to_gpu_stats['mean']:.4f} ms (±{cpu_to_gpu_stats['std']:.4f})")
        print(f"  GPU->CPU: {gpu_to_cpu_stats['mean']:.4f} ms (±{gpu_to_cpu_stats['std']:.4f})")
        print(f"  CPU->GPU 带宽: {cpu_to_gpu_bandwidth:.2f} MB/s")
        print(f"  GPU->CPU 带宽: {gpu_to_cpu_bandwidth:.2f} MB/s")
        
        # 清理内存
        del cpu_experts
        torch.cuda.empty_cache()
    
    return results


def plot_transfer_results(results: Dict[int, Dict], save_dir: str):
    """绘制传输时间结果"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    num_experts_list = sorted(results.keys())
    
    # 提取数据
    cpu_to_gpu_times = [results[n]['cpu_to_gpu']['mean'] for n in num_experts_list]
    gpu_to_cpu_times = [results[n]['gpu_to_cpu']['mean'] for n in num_experts_list]
    memory_sizes = [results[n]['memory_mb'] for n in num_experts_list]
    
    cpu_to_gpu_stds = [results[n]['cpu_to_gpu']['std'] for n in num_experts_list]
    gpu_to_cpu_stds = [results[n]['gpu_to_cpu']['std'] for n in num_experts_list]
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Expert Transfer Time Analysis', fontsize=16)
    
    # 1. 传输时间对比
    ax1 = axes[0, 0]
    ax1.errorbar(num_experts_list, cpu_to_gpu_times, yerr=cpu_to_gpu_stds, 
                 fmt='o-', linewidth=2, markersize=8, label='CPU→GPU', color='blue', alpha=0.8)
    ax1.errorbar(num_experts_list, gpu_to_cpu_times, yerr=gpu_to_cpu_stds,
                 fmt='s-', linewidth=2, markersize=8, label='GPU→CPU', color='red', alpha=0.8)
    ax1.set_xlabel('Number of Experts', fontsize=12)
    ax1.set_ylabel('Transfer Time (ms)', fontsize=12)
    ax1.set_title('Transfer Time vs Number of Experts', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(num_experts_list)
    
    # 2. 内存大小 vs 传输时间
    ax2 = axes[0, 1]
    ax2.scatter(memory_sizes, cpu_to_gpu_times, s=100, alpha=0.7, label='CPU→GPU', color='blue')
    ax2.scatter(memory_sizes, gpu_to_cpu_times, s=100, alpha=0.7, label='GPU→CPU', color='red')
    ax2.set_xlabel('Memory Size (MB)', fontsize=12)
    ax2.set_ylabel('Transfer Time (ms)', fontsize=12)
    ax2.set_title('Transfer Time vs Memory Size', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 传输带宽
    ax3 = axes[1, 0]
    cpu_to_gpu_bandwidth = [results[n]['memory_mb'] / (results[n]['cpu_to_gpu']['mean'] / 1000) 
                           for n in num_experts_list]
    gpu_to_cpu_bandwidth = [results[n]['memory_mb'] / (results[n]['gpu_to_cpu']['mean'] / 1000) 
                           for n in num_experts_list]
    
    ax3.plot(num_experts_list, cpu_to_gpu_bandwidth, 'o-', linewidth=2, markersize=8, 
             label='CPU→GPU', color='blue', alpha=0.8)
    ax3.plot(num_experts_list, gpu_to_cpu_bandwidth, 's-', linewidth=2, markersize=8, 
             label='GPU→CPU', color='red', alpha=0.8)
    ax3.set_xlabel('Number of Experts', fontsize=12)
    ax3.set_ylabel('Bandwidth (MB/s)', fontsize=12)
    ax3.set_title('Transfer Bandwidth vs Number of Experts', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(num_experts_list)
    
    # 4. 每个expert的平均传输时间
    ax4 = axes[1, 1]
    cpu_to_gpu_per_expert = [cpu_to_gpu_times[i] / num_experts_list[i] for i in range(len(num_experts_list))]
    gpu_to_cpu_per_expert = [gpu_to_cpu_times[i] / num_experts_list[i] for i in range(len(num_experts_list))]
    
    ax4.plot(num_experts_list, cpu_to_gpu_per_expert, 'o-', linewidth=2, markersize=8, 
             label='CPU→GPU', color='blue', alpha=0.8)
    ax4.plot(num_experts_list, gpu_to_cpu_per_expert, 's-', linewidth=2, markersize=8, 
             label='GPU→CPU', color='red', alpha=0.8)
    ax4.set_xlabel('Number of Experts', fontsize=12)
    ax4.set_ylabel('Time per Expert (ms)', fontsize=12)
    ax4.set_title('Transfer Time per Expert', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(num_experts_list)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = f"{save_dir}/expert_transfer_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"传输分析图表已保存到: {plot_path}")
    
    plt.close()


def save_transfer_results(results: Dict[int, Dict], save_dir: str):
    """保存传输时间结果"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    results_file = f"{save_dir}/expert_transfer_results.txt"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Expert传输时间测试结果\n")
        f.write("="*80 + "\n\n")
        
        # 写入详细结果
        f.write("Expert数量\t内存(MB)\tCPU→GPU(ms)\tGPU→CPU(ms)\tCPU→GPU带宽(MB/s)\tGPU→CPU带宽(MB/s)\n")
        f.write("-" * 100 + "\n")
        
        for num_experts in sorted(results.keys()):
            data = results[num_experts]
            cpu_to_gpu_time = data['cpu_to_gpu']['mean']
            gpu_to_cpu_time = data['gpu_to_cpu']['mean']
            memory_mb = data['memory_mb']
            
            cpu_to_gpu_bandwidth = memory_mb / (cpu_to_gpu_time / 1000)
            gpu_to_cpu_bandwidth = memory_mb / (gpu_to_cpu_time / 1000)
            
            f.write(f"{num_experts}\t\t{memory_mb:.2f}\t\t{cpu_to_gpu_time:.4f}\t\t{gpu_to_cpu_time:.4f}\t\t{cpu_to_gpu_bandwidth:.2f}\t\t{gpu_to_cpu_bandwidth:.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("详细统计信息:\n\n")
        
        for num_experts in sorted(results.keys()):
            data = results[num_experts]
            f.write(f"{num_experts} 个Experts:\n")
            f.write(f"  内存大小: {data['memory_mb']:.2f} MB\n")
            f.write(f"  CPU→GPU: {data['cpu_to_gpu']['mean']:.4f}±{data['cpu_to_gpu']['std']:.4f} ms\n")
            f.write(f"    范围: {data['cpu_to_gpu']['min']:.4f} - {data['cpu_to_gpu']['max']:.4f} ms\n")
            f.write(f"  GPU→CPU: {data['gpu_to_cpu']['mean']:.4f}±{data['gpu_to_cpu']['std']:.4f} ms\n")
            f.write(f"    范围: {data['gpu_to_cpu']['min']:.4f} - {data['gpu_to_cpu']['max']:.4f} ms\n\n")
        
        # 分析总结
        f.write("="*80 + "\n")
        f.write("性能分析:\n\n")
        
        # 线性度分析
        num_experts_list = sorted(results.keys())
        cpu_to_gpu_times = [results[n]['cpu_to_gpu']['mean'] for n in num_experts_list]
        
        # 计算单个expert的平均传输时间
        single_expert_time = cpu_to_gpu_times[0]  # 1个expert的时间
        max_experts_time = cpu_to_gpu_times[-1]   # 8个experts的时间
        
        f.write(f"传输时间缩放性:\n")
        f.write(f"  单个expert传输时间: {single_expert_time:.4f} ms\n")
        f.write(f"  8个experts传输时间: {max_experts_time:.4f} ms\n")
        f.write(f"  理论线性缩放时间: {single_expert_time * 8:.4f} ms\n")
        f.write(f"  实际缩放比: {max_experts_time / single_expert_time:.2f}x\n")
        f.write(f"  线性度: {(single_expert_time * 8) / max_experts_time:.2%}\n")
    
    print(f"传输时间结果已保存到: {results_file}")


def main():
    model_path = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"
    
    print("="*60)
    print("Expert传输时间测试实验")
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
    
    # 提取experts
    print("\n2. 提取experts...")
    first_layer = model.model.layers[0]
    experts = list(first_layer.mlp.experts)
    print(f"提取到 {len(experts)} 个experts")
    
    # 显示单个expert的信息
    single_expert_size = get_expert_memory_size(experts[0])
    print(f"单个expert内存大小: {single_expert_size:.2f} MB")
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        return
    
    # 定义测试的expert数量
    num_experts_list = list(range(1, 9))  # 1-8个experts
    
    print(f"\n3. 开始传输时间测试...")
    print(f"测试expert数量: {num_experts_list}")
    
    # 执行传输时间测试
    results = benchmark_expert_transfer(experts, num_experts_list, num_runs=30)
    
    # 保存结果
    save_dir = "/zx_data1/sparsity/on_device_sd/pre_exp/benchmarks/results/expert_transfer"
    print(f"\n4. 保存实验结果...")
    save_transfer_results(results, save_dir)
    
    # 绘制结果图表
    print(f"\n5. 生成结果图表...")
    plot_transfer_results(results, save_dir)
    
    print("\n" + "="*60)
    print("Expert传输时间测试完成!")
    print("="*60)
    print(f"结果保存目录: {save_dir}")
    print("生成的文件:")
    print("  - expert_transfer_results.txt (详细结果)")
    print("  - expert_transfer_analysis.png (分析图表)")
    print("="*60)


if __name__ == "__main__":
    main()
