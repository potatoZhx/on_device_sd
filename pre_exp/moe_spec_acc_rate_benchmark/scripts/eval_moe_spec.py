#!/usr/bin/env python3
"""
MOE推测解码接收率评估主脚本
"""
import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.moe_spec_config import MOESpecConfig

def run_moe_spec_evaluation():
    """运行MOE推测解码评估"""
    
    config = MOESpecConfig()
    
    # 创建结果目录
    results_dir = project_root / "results" / "model_answer"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查数据集是否存在
    dataset_path = project_root / "data" / "spec_bench" / "question.jsonl"
    if not dataset_path.exists():
        # 创建软链接到Spec-Bench数据集
        spec_bench_path = Path("/zx_data1/sparsity/Spec-Bench/data/spec_bench/question.jsonl")
        if spec_bench_path.exists():
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ln", "-sf", str(spec_bench_path), str(dataset_path)])
        else:
            print(f"Error: Dataset not found at {spec_bench_path}")
            return
    
    # 构建评估命令
    cmd = [
        "python", "evaluation/inference_moe_spec.py",
        "--model-path", config.model_path,
        "--model-id", config.model_id,
        "--bench-name", config.bench_name,
        "--max-new-tokens", str(config.max_new_tokens),
        "--draft-length", str(config.draft_length),
        "--top-k-experts-to-remove", str(config.top_k_experts_to_remove),
        "--dtype", config.dtype,
        "--num-choices", str(config.num_choices),
        "--num-gpus-per-model", str(config.num_gpus_per_model),
        "--num-gpus-total", str(config.num_gpus_total)
    ]
    
    if config.answer_file:
        cmd.extend(["--answer-file", config.answer_file])
    
    print("Running MOE Speculative Decoding Evaluation...")
    print(f"Command: {' '.join(cmd)}")
    
    # 执行评估
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode == 0:
        print("Evaluation completed successfully!")
        
        # 计算接收率统计
        calculate_acceptance_rate_stats()
    else:
        print("Evaluation failed!")

def calculate_acceptance_rate_stats():
    """计算接收率统计"""
    import json
    
    results_dir = project_root / "results" / "model_answer"
    answer_files = list(results_dir.glob("*.jsonl"))
    
    if not answer_files:
        print("No answer files found!")
        return
    
    total_draft_length = 0
    total_accept_length = 0
    
    for answer_file in answer_files:
        print(f"Processing {answer_file.name}...")
        
        with open(answer_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                for choice in data.get('choices', []):
                    accept_lengths = choice.get('accept_lengths', [])
                    if accept_lengths:
                        total_accept_length += sum(accept_lengths)
                        # draft_length = accept_lengths的数量 * draft_length
                        # 这里需要根据实际实现调整
                        total_draft_length += len(accept_lengths)
    
    if total_draft_length > 0:
        overall_acceptance_rate = total_accept_length / total_draft_length
        print(f"\nOverall Acceptance Rate: {overall_acceptance_rate:.4f}")
        print(f"Total Draft Length: {total_draft_length}")
        print(f"Total Accept Length: {total_accept_length}")
    else:
        print("No valid data found!")

if __name__ == "__main__":
    run_moe_spec_evaluation()
