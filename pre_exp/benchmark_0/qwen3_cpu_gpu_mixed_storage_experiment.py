#!/usr/bin/env python
"""
Qwen3-30B CPU-GPU混合存储实验
将50%的expert权重存储在CPU中，其余在GPU中
记录prefill和decode时间，以及每层激活的GPU权重比例
"""

import os
# 设置环境变量
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import torch
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from datetime import datetime


class MixedStorageQwen3MoELayer(torch.nn.Module):
    """支持CPU-GPU混合存储的Qwen3 MoE层"""
    
    def __init__(self, original_layer, model_rotary_emb, layer_idx, cpu_ratio=0.5, num_experts_per_tok=8):
        super().__init__()
        self.layer_idx = layer_idx
        self.original_layer = original_layer
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = len(original_layer.mlp.experts)
        self.cpu_ratio = cpu_ratio
        
        # 复制原始组件到GPU
        self.input_layernorm = copy.deepcopy(original_layer.input_layernorm).cuda()
        self.self_attn = copy.deepcopy(original_layer.self_attn).cuda()
        self.post_attention_layernorm = copy.deepcopy(original_layer.post_attention_layernorm).cuda()
        
        # 设置rotary embedding引用
        self.rotary_emb = model_rotary_emb
        
        # 复制MoE组件
        self.gate = copy.deepcopy(original_layer.mlp.gate).cuda()
        self.experts = torch.nn.ModuleList()
        
        # 根据CPU比例初始化expert设备分布
        self.expert_devices = {}
        cpu_expert_count = int(self.num_experts * cpu_ratio)
        
        for i, expert in enumerate(original_layer.mlp.experts):
            expert_copy = copy.deepcopy(expert)
            if i < cpu_expert_count:
                # 前cpu_ratio比例的experts放在CPU
                expert_copy = expert_copy.cpu()
                self.expert_devices[i] = 'cpu'
            else:
                # 剩余的experts放在GPU
                expert_copy = expert_copy.cuda()
                self.expert_devices[i] = 'cuda'
            
            self.experts.append(expert_copy)
        
        print(f"Layer {layer_idx}: {cpu_expert_count} experts on CPU ({cpu_ratio*100:.0f}%), {self.num_experts - cpu_expert_count} experts on GPU")
        
        # 追踪信息
        self.activation_stats = {
            'gpu_weight_ratio': 0.0,
            'activated_experts': [],
            'expert_devices_used': {},
            'transfer_count': 0
        }
        
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
        router_logits = self.gate(hidden_states)
        routing_weights, selected_experts = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # 处理多维度情况：期望选取第0个样本、序列最后一个位置的路由结果 => 形状[K]
        # 输入一般为 [batch, seq_len, hidden]
        if selected_experts.dim() == 3:
            # [B, S, K] -> 选 batch 0、最后一个时间步 S-1
            selected_experts = selected_experts[0, -1, :]
            routing_weights = routing_weights[0, -1, :]
        elif selected_experts.dim() == 2:
            # [S, K] -> 选最后一个时间步
            selected_experts = selected_experts[-1, :]
            routing_weights = routing_weights[-1, :]
        elif selected_experts.dim() == 1:
            # [K] -> 已是期望形状
            pass
        
        # 更新激活统计
        if isinstance(selected_experts, torch.Tensor):
            if selected_experts.dim() == 0:
                # 0维张量，单个expert
                activated_expert_indices = [selected_experts.item()]
            else:
                # 多维张量
                activated_expert_indices = selected_experts.cpu().numpy().tolist()
                if not isinstance(activated_expert_indices, list):
                    activated_expert_indices = [activated_expert_indices]
        else:
            # 已经是标量或列表
            activated_expert_indices = [selected_experts] if not isinstance(selected_experts, list) else selected_experts
        
        self.activation_stats['activated_experts'] = activated_expert_indices
        
        # 计算GPU权重比例
        gpu_experts = sum(1 for idx in activated_expert_indices if self.expert_devices[int(idx)] == 'cuda')
        self.activation_stats['gpu_weight_ratio'] = gpu_experts / len(activated_expert_indices)
        
        # 记录使用的设备
        self.activation_stats['expert_devices_used'] = {
            int(idx): self.expert_devices[int(idx)] for idx in activated_expert_indices
        }
        
        # Expert计算
        expert_outputs = []
        transfer_count = 0
        
        # 确保selected_experts是可迭代的
        if isinstance(selected_experts, torch.Tensor):
            if selected_experts.dim() == 0:
                expert_indices = [selected_experts.item()]
            else:
                expert_indices = selected_experts.tolist()
        else:
            expert_indices = selected_experts if isinstance(selected_experts, list) else [selected_experts]
        
        for i, expert_idx in enumerate(expert_indices):
            if isinstance(expert_idx, torch.Tensor):
                expert_idx = expert_idx.item()
            
            expert = self.experts[expert_idx]
            expert_device = self.expert_devices[expert_idx]
            
            if expert_device == 'cpu':
                # Expert在CPU上，需要数据传输
                input_cpu = hidden_states.cpu()
                expert_output = expert(input_cpu)
                expert_output = expert_output.cuda()
                transfer_count += 1
            else:
                # Expert在GPU上，直接计算
                expert_output = expert(hidden_states)
            
            expert_outputs.append(expert_output)
        
        self.activation_stats['transfer_count'] = transfer_count
        
        # 加权组合expert输出
        final_output = torch.zeros_like(hidden_states)
        
        # 确保routing_weights是可索引的
        if isinstance(routing_weights, torch.Tensor):
            if routing_weights.dim() == 0:
                # 0维张量，只有一个权重
                weights = [routing_weights.item()]
            else:
                weights = routing_weights.tolist()
        else:
            weights = routing_weights if isinstance(routing_weights, list) else [routing_weights]
        
        for i, expert_output in enumerate(expert_outputs):
            weight = weights[i] if i < len(weights) else 1.0 / len(expert_outputs)
            final_output += weight * expert_output
            
        hidden_states = residual + final_output
        
        # 返回格式与原始layer一致
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[2],) if len(attn_outputs) > 2 else (None,)
        
        return outputs


class MixedStorageQwen3Model(torch.nn.Module):
    """支持CPU-GPU混合存储的Qwen3模型"""
    
    def __init__(self, original_model, cpu_ratio=0.5):
        super().__init__()
        self.original_model = original_model
        self.config = original_model.config
        self.cpu_ratio = cpu_ratio
        
        # 复制基础组件到GPU
        self.embed_tokens = copy.deepcopy(original_model.model.embed_tokens).cuda()
        self.norm = copy.deepcopy(original_model.model.norm).cuda()
        self.rotary_emb = copy.deepcopy(original_model.model.rotary_emb).cuda()
        
        # 创建混合存储的MoE层
        self.layers = torch.nn.ModuleList()
        for i, layer in enumerate(original_model.model.layers):
            mixed_layer = MixedStorageQwen3MoELayer(layer, self.rotary_emb, i, cpu_ratio)
            self.layers.append(mixed_layer)
        
        # 复制输出层
        self.lm_head = copy.deepcopy(original_model.lm_head).cuda()
        
        print(f"创建混合存储模型完成，共 {len(self.layers)} 层，CPU比例: {cpu_ratio*100:.0f}%")
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=False, output_attentions=False, **kwargs):
        """前向传播"""
        
        # Token embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # 通过每一层
        for i, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[i] if past_key_values is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs
            )
            hidden_states = layer_outputs[0]
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        # 输出层
        logits = self.lm_head(hidden_states)
        
        return type('ModelOutput', (), {
            'logits': logits,
            'past_key_values': None,
            'hidden_states': None,
            'attentions': None
        })()
    
    def get_activation_stats(self):
        """获取所有层的激活统计"""
        stats = {}
        for i, layer in enumerate(self.layers):
            stats[f'layer_{i}'] = copy.deepcopy(layer.activation_stats)
        return stats


def run_cpu_ratio_experiment(original_model, tokenizer, cpu_ratios, test_prompt="The future of artificial intelligence is", batch_size=3):
    """测试不同CPU比例下的性能"""
    
    print(f"测试提示: '{test_prompt}'")
    print(f"测试CPU比例: {[f'{r*100:.0f}%' for r in cpu_ratios]}")
    
    all_results = {}
    
    for cpu_ratio in cpu_ratios:
        print(f"\n{'='*60}")
        print(f"测试CPU比例: {cpu_ratio*100:.0f}%")
        print(f"{'='*60}")
        
        # 创建当前CPU比例的模型
        print("创建混合存储模型...")
        try:
            mixed_model = MixedStorageQwen3Model(original_model, cpu_ratio)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"OOM during model construction at CPU ratio {cpu_ratio*100:.0f}%. 跳过该比例。")
                torch.cuda.empty_cache()
                all_results[cpu_ratio] = {
                    'error': 'oom_during_model_construction'
                }
                continue
            else:
                raise
        
        # 预热一次（不计入统计）
        print("预热中 (1 次)...")
        try:
            _ = run_inference_experiment(mixed_model, tokenizer, test_prompt, batch_size=batch_size)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"OOM during warmup at CPU ratio {cpu_ratio*100:.0f}%. 跳过该比例。")
                del mixed_model
                torch.cuda.empty_cache()
                all_results[cpu_ratio] = {
                    'error': 'oom_during_warmup'
                }
                continue
            else:
                del mixed_model
                torch.cuda.empty_cache()
                raise
        
        # 运行推理实验
        print("运行推理实验...")
        try:
            results = run_inference_experiment(mixed_model, tokenizer, test_prompt, batch_size=batch_size)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"OOM during inference at CPU ratio {cpu_ratio*100:.0f}%. 跳过该比例。")
                del mixed_model
                torch.cuda.empty_cache()
                all_results[cpu_ratio] = {
                    'error': 'oom_during_inference'
                }
                continue
            else:
                del mixed_model
                torch.cuda.empty_cache()
                raise
        
        # 添加CPU比例信息
        results['cpu_ratio'] = cpu_ratio
        results['cpu_percentage'] = f"{cpu_ratio*100:.0f}%"
        
        all_results[cpu_ratio] = results
        
        # 清理内存
        del mixed_model
        torch.cuda.empty_cache()
        
        print(f"CPU比例 {cpu_ratio*100:.0f}% 测试完成!")
        print(f"  Prefill时间: {results['prefill_time']:.4f} ms")
        print(f"  平均Decode时间: {np.mean(results['decode_times']):.4f} ms")
    
    return all_results


def run_inference_experiment(model, tokenizer, test_prompt="Hello, how are you today?", batch_size=3):
    """运行推理实验，记录prefill和decode时间"""
    
    print(f"测试提示: '{test_prompt}'")
    
    # Tokenize输入
    inputs = tokenizer([test_prompt]*batch_size, return_tensors="pt", padding=True).to('cuda')
    input_ids = inputs['input_ids']
    
    print(f"输入token数量: {input_ids.shape[1]}  | batch size: {input_ids.shape[0]}")
    
    results = {
        'prefill_time': 0.0,
        'decode_times': [],
        'total_tokens_generated': 0,
        'layer_activation_stats': {},
        'timestamp': datetime.now().isoformat()
    }
    
    # Prefill阶段
    print("\n开始Prefill阶段...")
    torch.cuda.synchronize()
    prefill_start = time.time()
    
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    
    torch.cuda.synchronize()
    prefill_end = time.time()
    
    prefill_time = (prefill_end - prefill_start) * 1000  # 转换为毫秒
    results['prefill_time'] = prefill_time
    
    print(f"Prefill时间: {prefill_time:.4f} ms")
    
    # 记录prefill阶段的激活统计
    results['layer_activation_stats']['prefill'] = model.get_activation_stats()
    
    # Decode阶段 (生成5个token)
    print("\n开始Decode阶段...")
    current_ids = torch.cat([input_ids, next_token], dim=1)
    max_new_tokens = 5
    
    for i in range(max_new_tokens):
        print(f"  生成token {i+1}/{max_new_tokens}...")
        
        torch.cuda.synchronize()
        decode_start = time.time()
        
        with torch.no_grad():
            outputs = model(current_ids[:, -1:])  # 按批处理最后一个token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        torch.cuda.synchronize()
        decode_end = time.time()
        
        decode_time = (decode_end - decode_start) * 1000
        results['decode_times'].append(decode_time)
        
        print(f"    Decode时间: {decode_time:.4f} ms")
        
        # 记录当前decode步骤的激活统计
        results['layer_activation_stats'][f'decode_step_{i}'] = model.get_activation_stats()
        
        current_ids = torch.cat([current_ids, next_token], dim=1)
        
        # 解码生成的token
        generated_texts = [tokenizer.decode(next_token[b], skip_special_tokens=True) for b in range(next_token.shape[0])]
        print(f"    生成token(batch): {generated_texts}")
    
    results['total_tokens_generated'] = max_new_tokens
    
    # 生成完整文本
    full_texts = [tokenizer.decode(current_ids[b], skip_special_tokens=True) for b in range(current_ids.shape[0])]
    results['generated_text'] = full_texts
    
    print(f"\n完整生成文本(batch): {full_texts}")
    
    return results


def analyze_cpu_ratio_results(all_results):
    """分析不同CPU比例的实验结果"""
    analysis = {
        'cpu_ratio_comparison': {},
        'timing_trends': {},
        'summary': {}
    }
    
    # 收集所有CPU比例的数据
    cpu_ratios = sorted(all_results.keys())
    prefill_times = []
    avg_decode_times = []
    total_inference_times = []
    avg_gpu_ratios = []
    total_transfers = []
    
    for cpu_ratio in cpu_ratios:
        results = all_results[cpu_ratio]
        
        prefill_time = results['prefill_time']
        decode_times = results['decode_times']
        avg_decode_time = np.mean(decode_times)
        total_decode_time = sum(decode_times)
        total_inference_time = prefill_time + total_decode_time
        
        prefill_times.append(prefill_time)
        avg_decode_times.append(avg_decode_time)
        total_inference_times.append(total_inference_time)
        
        # 计算平均GPU权重比例
        gpu_ratios = []
        transfer_counts = []
        
        for phase, layer_stats in results['layer_activation_stats'].items():
            for layer_name, stats in layer_stats.items():
                gpu_ratios.append(stats['gpu_weight_ratio'])
                transfer_counts.append(stats['transfer_count'])
        
        avg_gpu_ratio = np.mean(gpu_ratios) if gpu_ratios else 0
        total_transfer = sum(transfer_counts)
        
        avg_gpu_ratios.append(avg_gpu_ratio)
        total_transfers.append(total_transfer)
        
        # 存储每个CPU比例的详细分析
        analysis['cpu_ratio_comparison'][cpu_ratio] = {
            'prefill_time_ms': prefill_time,
            'avg_decode_time_ms': avg_decode_time,
            'total_inference_time_ms': total_inference_time,
            'avg_gpu_weight_ratio': avg_gpu_ratio,
            'total_transfers': total_transfer,
            'decode_times_ms': decode_times
        }
    
    # 时间趋势分析
    analysis['timing_trends'] = {
        'cpu_ratios': cpu_ratios,
        'prefill_times_ms': prefill_times,
        'avg_decode_times_ms': avg_decode_times,
        'total_inference_times_ms': total_inference_times,
        'avg_gpu_weight_ratios': avg_gpu_ratios,
        'total_transfers': total_transfers
    }
    
    # 总结分析
    analysis['summary'] = {
        'fastest_prefill': {
            'cpu_ratio': cpu_ratios[np.argmin(prefill_times)],
            'time_ms': min(prefill_times)
        },
        'fastest_decode': {
            'cpu_ratio': cpu_ratios[np.argmin(avg_decode_times)],
            'time_ms': min(avg_decode_times)
        },
        'fastest_total': {
            'cpu_ratio': cpu_ratios[np.argmin(total_inference_times)],
            'time_ms': min(total_inference_times)
        },
        'slowest_prefill': {
            'cpu_ratio': cpu_ratios[np.argmax(prefill_times)],
            'time_ms': max(prefill_times)
        },
        'slowest_decode': {
            'cpu_ratio': cpu_ratios[np.argmax(avg_decode_times)],
            'time_ms': max(avg_decode_times)
        },
        'slowest_total': {
            'cpu_ratio': cpu_ratios[np.argmax(total_inference_times)],
            'time_ms': max(total_inference_times)
        }
    }
    
    return analysis


def analyze_results(results):
    """分析单个实验结果"""
    analysis = {
        'timing_analysis': {},
        'gpu_ratio_analysis': {},
        'transfer_analysis': {}
    }
    
    # 时间分析
    prefill_time = results['prefill_time']
    decode_times = results['decode_times']
    avg_decode_time = np.mean(decode_times)
    total_decode_time = sum(decode_times)
    
    analysis['timing_analysis'] = {
        'prefill_time_ms': prefill_time,
        'avg_decode_time_ms': avg_decode_time,
        'total_decode_time_ms': total_decode_time,
        'total_inference_time_ms': prefill_time + total_decode_time,
        'decode_times_ms': decode_times
    }
    
    # GPU权重比例分析
    gpu_ratios_by_layer = {}
    transfer_counts_by_layer = {}
    
    for phase, layer_stats in results['layer_activation_stats'].items():
        gpu_ratios_by_layer[phase] = {}
        transfer_counts_by_layer[phase] = {}
        
        for layer_name, stats in layer_stats.items():
            gpu_ratios_by_layer[phase][layer_name] = stats['gpu_weight_ratio']
            transfer_counts_by_layer[phase][layer_name] = stats['transfer_count']
    
    analysis['gpu_ratio_analysis'] = gpu_ratios_by_layer
    analysis['transfer_analysis'] = transfer_counts_by_layer
    
    # 计算平均GPU权重比例
    all_gpu_ratios = []
    all_transfer_counts = []
    
    for phase_ratios in gpu_ratios_by_layer.values():
        for ratio in phase_ratios.values():
            all_gpu_ratios.append(ratio)
    
    for phase_transfers in transfer_counts_by_layer.values():
        for count in phase_transfers.values():
            all_transfer_counts.append(count)
    
    analysis['summary'] = {
        'avg_gpu_weight_ratio': np.mean(all_gpu_ratios),
        'avg_transfer_count_per_layer': np.mean(all_transfer_counts),
        'total_transfers': sum(all_transfer_counts)
    }
    
    return analysis


def save_cpu_ratio_results(all_results, analysis, save_dir):
    """保存CPU比例实验结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存详细结果
    results_file = os.path.join(save_dir, 'cpu_ratio_experiment_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 保存分析结果
    analysis_file = os.path.join(save_dir, 'cpu_ratio_analysis_results.json')
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    # 创建可读的文本报告
    report_file = os.path.join(save_dir, 'cpu_ratio_experiment_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Qwen3-30B CPU Expert比例性能测试报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"实验时间: {datetime.now().isoformat()}\n")
        f.write(f"测试CPU比例: 0%, 10%, 20%, ..., 100%\n\n")
        
        # 性能总结
        f.write("性能总结:\n")
        f.write("-" * 30 + "\n")
        summary = analysis['summary']
        f.write(f"最快Prefill: {summary['fastest_prefill']['cpu_ratio']*100:.0f}% CPU - {summary['fastest_prefill']['time_ms']:.4f} ms\n")
        f.write(f"最快Decode: {summary['fastest_decode']['cpu_ratio']*100:.0f}% CPU - {summary['fastest_decode']['time_ms']:.4f} ms\n")
        f.write(f"最快总时间: {summary['fastest_total']['cpu_ratio']*100:.0f}% CPU - {summary['fastest_total']['time_ms']:.4f} ms\n\n")
        
        f.write(f"最慢Prefill: {summary['slowest_prefill']['cpu_ratio']*100:.0f}% CPU - {summary['slowest_prefill']['time_ms']:.4f} ms\n")
        f.write(f"最慢Decode: {summary['slowest_decode']['cpu_ratio']*100:.0f}% CPU - {summary['slowest_decode']['time_ms']:.4f} ms\n")
        f.write(f"最慢总时间: {summary['slowest_total']['cpu_ratio']*100:.0f}% CPU - {summary['slowest_total']['time_ms']:.4f} ms\n\n")
        
        # 详细数据表
        f.write("详细性能数据:\n")
        f.write("-" * 30 + "\n")
        f.write("CPU比例\tPrefill(ms)\tDecode(ms)\t总时间(ms)\tGPU比例\t传输次数\n")
        f.write("-" * 80 + "\n")
        
        trends = analysis['timing_trends']
        for i, cpu_ratio in enumerate(trends['cpu_ratios']):
            prefill_time = trends['prefill_times_ms'][i]
            decode_time = trends['avg_decode_times_ms'][i]
            total_time = trends['total_inference_times_ms'][i]
            gpu_ratio = trends['avg_gpu_weight_ratios'][i]
            transfers = trends['total_transfers'][i]
            
            f.write(f"{cpu_ratio*100:.0f}%\t\t{prefill_time:.2f}\t\t{decode_time:.2f}\t\t{total_time:.2f}\t\t{gpu_ratio:.3f}\t\t{transfers}\n")
    
    print(f"CPU比例实验结果已保存到: {save_dir}")
    print(f"  - cpu_ratio_experiment_results.json (原始数据)")
    print(f"  - cpu_ratio_analysis_results.json (分析数据)")
    print(f"  - cpu_ratio_experiment_report.txt (可读报告)")


def save_results(results, analysis, save_dir):
    """保存实验结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存详细结果
    results_file = os.path.join(save_dir, 'experiment_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存分析结果
    analysis_file = os.path.join(save_dir, 'analysis_results.json')
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    # 创建可读的文本报告
    report_file = os.path.join(save_dir, 'experiment_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Qwen3-30B CPU-GPU混合存储实验报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"实验时间: {results['timestamp']}\n")
        f.write(f"生成文本: {results['generated_text']}\n\n")
        
        # 时间分析
        f.write("时间性能分析:\n")
        f.write("-" * 30 + "\n")
        timing = analysis['timing_analysis']
        f.write(f"Prefill时间: {timing['prefill_time_ms']:.4f} ms\n")
        f.write(f"平均Decode时间: {timing['avg_decode_time_ms']:.4f} ms\n")
        f.write(f"总Decode时间: {timing['total_decode_time_ms']:.4f} ms\n")
        f.write(f"总推理时间: {timing['total_inference_time_ms']:.4f} ms\n\n")
        
        # Decode时间详情
        f.write("各步Decode时间:\n")
        for i, decode_time in enumerate(timing['decode_times_ms']):
            f.write(f"  步骤 {i+1}: {decode_time:.4f} ms\n")
        f.write("\n")
        
        # GPU权重比例分析
        f.write("GPU权重比例分析:\n")
        f.write("-" * 30 + "\n")
        summary = analysis['summary']
        f.write(f"平均GPU权重比例: {summary['avg_gpu_weight_ratio']:.4f} ({summary['avg_gpu_weight_ratio']*100:.1f}%)\n")
        f.write(f"平均每层传输次数: {summary['avg_transfer_count_per_layer']:.2f}\n")
        f.write(f"总传输次数: {summary['total_transfers']}\n\n")
        
        # 各阶段详细分析
        f.write("各阶段详细分析:\n")
        f.write("-" * 30 + "\n")
        
        gpu_ratios = analysis['gpu_ratio_analysis']
        transfers = analysis['transfer_analysis']
        
        for phase in ['prefill'] + [f'decode_step_{i}' for i in range(results['total_tokens_generated'])]:
            if phase in gpu_ratios:
                f.write(f"\n{phase.upper()}阶段:\n")
                phase_gpu_ratios = list(gpu_ratios[phase].values())
                phase_transfers = list(transfers[phase].values())
                
                f.write(f"  平均GPU权重比例: {np.mean(phase_gpu_ratios):.4f}\n")
                f.write(f"  总传输次数: {sum(phase_transfers)}\n")
                
                # 显示前5层的详细信息
                f.write("  各层详情 (前5层):\n")
                for i in range(min(5, len(phase_gpu_ratios))):
                    layer_name = f'layer_{i}'
                    if layer_name in gpu_ratios[phase]:
                        gpu_ratio = gpu_ratios[phase][layer_name]
                        transfer_count = transfers[phase][layer_name]
                        f.write(f"    Layer {i}: GPU比例={gpu_ratio:.4f}, 传输={transfer_count}次\n")
    
    print(f"实验结果已保存到: {save_dir}")
    print(f"  - experiment_results.json (原始数据)")
    print(f"  - analysis_results.json (分析数据)")
    print(f"  - experiment_report.txt (可读报告)")


def create_cpu_ratio_visualization(analysis, save_dir):
    """创建CPU比例对比可视化图表"""
    
    # Use English labels in figures
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    trends = analysis['timing_trends']
    cpu_ratios = trends['cpu_ratios']
    cpu_percentages = [r * 100 for r in cpu_ratios]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Qwen3-30B CPU Expert Ratio Performance', fontsize=16)
    
    # 1. Prefill时间对比
    ax1 = axes[0, 0]
    ax1.plot(cpu_percentages, trends['prefill_times_ms'], 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_title('Prefill Time vs CPU Expert Ratio')
    ax1.set_xlabel('CPU Expert Ratio (%)')
    ax1.set_ylabel('Prefill Time (ms)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(cpu_percentages)
    
    # 添加数值标注
    for i, (x, y) in enumerate(zip(cpu_percentages, trends['prefill_times_ms'])):
        ax1.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    # 2. Decode时间对比
    ax2 = axes[0, 1]
    ax2.plot(cpu_percentages, trends['avg_decode_times_ms'], 'o-', linewidth=2, markersize=8, color='orange')
    ax2.set_title('Average Decode Time vs CPU Expert Ratio')
    ax2.set_xlabel('CPU Expert Ratio (%)')
    ax2.set_ylabel('Average Decode Time (ms)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(cpu_percentages)
    
    # 添加数值标注
    for i, (x, y) in enumerate(zip(cpu_percentages, trends['avg_decode_times_ms'])):
        ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    # 3. 总推理时间对比
    ax3 = axes[1, 0]
    ax3.plot(cpu_percentages, trends['total_inference_times_ms'], 'o-', linewidth=2, markersize=8, color='green')
    ax3.set_title('Total Inference Time vs CPU Expert Ratio')
    ax3.set_xlabel('CPU Expert Ratio (%)')
    ax3.set_ylabel('Total Inference Time (ms)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(cpu_percentages)
    
    # 添加数值标注
    for i, (x, y) in enumerate(zip(cpu_percentages, trends['total_inference_times_ms'])):
        ax3.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    # 4. GPU权重比例和传输次数对比
    ax4 = axes[1, 1]
    
    # 绘制GPU权重比例
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(cpu_percentages, trends['avg_gpu_weight_ratios'], 'o-', linewidth=2, markersize=8, color='purple', label='GPU weight ratio')
    line2 = ax4_twin.plot(cpu_percentages, trends['total_transfers'], 's-', linewidth=2, markersize=8, color='red', label='Transfer count')
    
    ax4.set_title('GPU Weight Ratio and Transfer Count vs CPU Expert Ratio')
    ax4.set_xlabel('CPU Expert Ratio (%)')
    ax4.set_ylabel('GPU Weight Ratio', color='purple')
    ax4_twin.set_ylabel('Transfer Count', color='red')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(cpu_percentages)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(save_dir, 'cpu_ratio_performance_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"CPU比例性能对比图表已保存到: {plot_path}")
    
    plt.close()


def create_visualization(analysis, save_dir):
    """创建可视化图表"""
    
    # Use English labels in figures
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Qwen3-30B CPU-GPU Mixed Storage Results', fontsize=16)
    
    # 1. 时间性能图表
    ax1 = axes[0, 0]
    timing = analysis['timing_analysis']
    phases = ['Prefill'] + [f'Decode {i+1}' for i in range(len(timing['decode_times_ms']))]
    times = [timing['prefill_time_ms']] + timing['decode_times_ms']
    
    bars = ax1.bar(phases, times, color=['blue'] + ['orange'] * len(timing['decode_times_ms']))
    ax1.set_title('Stage Execution Time')
    ax1.set_ylabel('Time (ms)')
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标注
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time:.2f}ms', ha='center', va='bottom', fontsize=9)
    
    # 2. GPU权重比例图表
    ax2 = axes[0, 1]
    gpu_ratios = analysis['gpu_ratio_analysis']
    
    # 收集各阶段的平均GPU权重比例
    phase_names = []
    phase_gpu_ratios = []
    
    for phase, layer_ratios in gpu_ratios.items():
        phase_names.append(phase.replace('_', ' ').title())
        phase_gpu_ratios.append(np.mean(list(layer_ratios.values())))
    
    bars = ax2.bar(phase_names, phase_gpu_ratios, color='green', alpha=0.7)
    ax2.set_title('Average GPU Weight Ratio by Stage')
    ax2.set_ylabel('GPU Weight Ratio')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加百分比标注
    for bar, ratio in zip(bars, phase_gpu_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{ratio*100:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. 传输次数图表
    ax3 = axes[1, 0]
    transfers = analysis['transfer_analysis']
    
    phase_transfer_counts = []
    for phase in phase_names:
        phase_key = phase.lower().replace(' ', '_')
        if phase_key in transfers:
            phase_transfer_counts.append(sum(transfers[phase_key].values()))
        else:
            phase_transfer_counts.append(0)
    
    bars = ax3.bar(phase_names, phase_transfer_counts, color='red', alpha=0.7)
    ax3.set_title('CPU-GPU Transfer Count by Stage')
    ax3.set_ylabel('Transfer Count')
    ax3.tick_params(axis='x', rotation=45)
    
    # 添加数值标注
    for bar, count in zip(bars, phase_transfer_counts):
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count}', ha='center', va='bottom', fontsize=9)
    
    # 4. 层级GPU权重比例热力图
    ax4 = axes[1, 1]
    
    # 准备热力图数据 (只显示前10层)
    num_layers_to_show = 10
    num_phases = len(phase_names)
    heatmap_data = np.zeros((num_layers_to_show, num_phases))
    
    for j, phase in enumerate(phase_names):
        phase_key = phase.lower().replace(' ', '_')
        if phase_key in gpu_ratios:
            for i in range(num_layers_to_show):
                layer_key = f'layer_{i}'
                if layer_key in gpu_ratios[phase_key]:
                    heatmap_data[i, j] = gpu_ratios[phase_key][layer_key]
    
    im = ax4.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    # 设置标签
    ax4.set_xticks(range(num_phases))
    ax4.set_xticklabels(phase_names, rotation=45)
    ax4.set_yticks(range(num_layers_to_show))
    ax4.set_yticklabels([f'Layer {i}' for i in range(num_layers_to_show)])
    ax4.set_title('GPU Weight Ratio Heatmap by Layer')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label('GPU权重比例')
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(save_dir, 'experiment_visualization.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存到: {plot_path}")
    
    plt.close()


def main():
    model_path = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"
    
    print("=" * 60)
    print("Qwen3-30B CPU Expert Ratio Performance Benchmark")
    print("=" * 60)
    
    # 检查模型路径
    if not os.path.exists(model_path):
        print(f"错误：模型路径 {model_path} 不存在")
        return
    
    # 加载模型和tokenizer
    print(f"\n1. 加载模型: {model_path}")
    try:
        original_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=None,
            local_files_only=True,
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        
        print("模型和tokenizer加载成功!")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 定义测试的CPU比例 (0%, 10%, 20%, ..., 100%)
    cpu_ratios = [i/10 for i in range(11)]  # [0.0, 0.1, 0.2, ..., 1.0]
    
    print(f"\n2. Start CPU ratio performance test...")
    print(f"CPU ratios: {[f'{r*100:.0f}%' for r in cpu_ratios]}")
    
    # 运行CPU比例实验
    test_prompt = "The future of artificial intelligence is"
    
    try:
        all_results = run_cpu_ratio_experiment(original_model, tokenizer, cpu_ratios, test_prompt, batch_size=5)
        print("CPU比例实验完成!")
    except Exception as e:
        print(f"CPU比例实验失败: {e}")
        return
    
    # 分析结果
    print("\n3. Analyze results...")
    analysis = analyze_cpu_ratio_results(all_results)
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"/zx_data1/sparsity/on_device_sd/pre_exp/benchmarks/results/cpu_ratio_experiment_{timestamp}"
    
    # 保存结果
    print("\n4. Save results...")
    save_cpu_ratio_results(all_results, analysis, save_dir)
    
    # 创建可视化
    print("\n5. Create visualizations...")
    create_cpu_ratio_visualization(analysis, save_dir)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("CPU ratio performance benchmark finished!")
    print("=" * 60)
    
    summary = analysis['summary']
    
    print(f"Results saved to: {save_dir}")
    print(f"\nPer-CPU ratio timings:")
    trends = analysis['timing_trends']
    for i, r in enumerate(trends['cpu_ratios']):
        prefill = trends['prefill_times_ms'][i]
        avg_decode = trends['avg_decode_times_ms'][i]
        total_time = trends['total_inference_times_ms'][i]
        print(f"  - {int(r*100)}% CPU -> Prefill: {prefill:.2f} ms, Avg Decode: {avg_decode:.2f} ms, Total: {total_time:.2f} ms")
    
    print(f"\nSummary:")
    print(f"  - Fastest prefill: {summary['fastest_prefill']['cpu_ratio']*100:.0f}% CPU - {summary['fastest_prefill']['time_ms']:.4f} ms")
    print(f"  - Fastest decode: {summary['fastest_decode']['cpu_ratio']*100:.0f}% CPU - {summary['fastest_decode']['time_ms']:.4f} ms")
    print(f"  - Fastest total: {summary['fastest_total']['cpu_ratio']*100:.0f}% CPU - {summary['fastest_total']['time_ms']:.4f} ms")
    print(f"  - Slowest prefill: {summary['slowest_prefill']['cpu_ratio']*100:.0f}% CPU - {summary['slowest_prefill']['time_ms']:.4f} ms")
    print(f"  - Slowest decode: {summary['slowest_decode']['cpu_ratio']*100:.0f}% CPU - {summary['slowest_decode']['time_ms']:.4f} ms")
    print(f"  - Slowest total: {summary['slowest_total']['cpu_ratio']*100:.0f}% CPU - {summary['slowest_total']['time_ms']:.4f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
