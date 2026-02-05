import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

# ================= 配置区域 =================
DIR = "/data2/group_谈海生/lagin/data/Sd_Data/data/"

TRAIN_DIRS = [
    "wiki_results_1_with_Qwen3-30B-A3B-Base",
    "wiki_results_2_with_Qwen3-30B-A3B-Base",
    "wiki_results_3_with_Qwen3-30B-A3B-Base",
    "wiki_results_4_with_Qwen3-30B-A3B-Base"
]

TEST_DIRS = [
    "mtbench_results_1_with_Qwen3-30B-A3B-Base",
    "mtbench_results_2_with_Qwen3-30B-A3B-Base",
    "mtbench_results_3_with_Qwen3-30B-A3B-Base",
    "mtbench_results_4_with_Qwen3-30B-A3B-Base"
]
# ===========================================

def find_latest_summary_file(directory):
    if not os.path.exists(directory): return None
    files = [f for f in os.listdir(directory) if f.endswith('.jsonl') and 'summary' in f]
    if not files: return None
    files.sort(reverse=True)
    return os.path.join(directory, files[0])

def print_group_stats(name, scores):
    scores = np.array(scores)
    count = len(scores)
    if count == 0:
        print(f"  🔹 {name:<25} | Count: 0")
        return
    
    mean_val = np.mean(scores)
    std_val = np.std(scores)
    min_val = np.min(scores)
    max_val = np.max(scores)
    
    print(f"  🔹 {name:<25} | Count: {count:<6} | Mean: {mean_val:.4f} | Std: {std_val:.4f} | Min/Max: {min_val:.2f}/{max_val:.2f}")

def process_strict_stats(dirs, dataset_name):
    print(f"\n{'='*20} 分析数据集: {dataset_name} {'='*20}")
    
    file_list = []
    for d in dirs:
        f = find_latest_summary_file(DIR + d)
        if f: file_list.append(f)
        
    if not file_list:
        print("❌ 未找到源文件")
        return

    # 存储分数的列表
    match_scores = []    # 正样本 (Token 一致) 的分数
    mismatch_scores = [] # 负样本 (Token 不一致) 的分数
    
    for f_path in file_list:
        with open(f_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Reading {os.path.basename(f_path)}", leave=False):
                try:
                    record = json.loads(line)
                    int_out = record["intervention"]["output"]
                    base_out = record["baseline"]["output"]
                    steps_data = record["intervention"]["steps"]
                    steps_base = record["baseline"]["steps"]
                    
                    # 1. 确定截断点 (第一个不一致的位置)
                    cutoff_idx = len(int_out)
                    for i in range(min(len(int_out), len(base_out))):
                        if int_out[i] != base_out[i]:
                            cutoff_idx = i
                            break
                    
                    for i, step_data in enumerate(steps_data):
                        # 超过截断点的数据是无效的，直接跳过
                        if i > cutoff_idx: break
                        
                        q_logits = torch.tensor(step_data["full_logits"])
                        p_logits = torch.tensor(steps_base[i]["full_logits"])
                        
                        # 2. 获取 Draft 和 Target 的 Top-1 Token ID
                        draft_token_id = torch.argmax(q_logits).item()
                        target_token_id = torch.argmax(p_logits).item()
                        
                        # 3. 计算软标签分数 (Soft Label Score)
                        q_probs = F.softmax(q_logits, dim=-1)
                        p_probs = F.softmax(p_logits, dim=-1)
                        q_x = q_probs[draft_token_id].item()
                        p_x = p_probs[draft_token_id].item()
                        score = min(1.0, p_x / (q_x + 1e-10))
                        
                        # 4. 严格分组：根据 Token 是否一致来归类
                        if draft_token_id == target_token_id:
                            match_scores.append(score) # 正样本
                        else:
                            mismatch_scores.append(score) # 负样本
                            
                        # 如果当前步就是截断点 (Mismatch)，记录完负样本后，停止处理后续序列
                        if i == cutoff_idx: break
                        
                except Exception:
                    continue

    # === 打印统计结果 ===
    print(f"📊 统计结果 (基于严格 Token 匹配):")
    
    # 1. 正样本 (Token Match)
    # 我们期望这里的均值非常高 (接近 1.0)
    print_group_stats("正样本 (Token Match)", match_scores)
    
    # 2. 负样本 (Token Mismatch)
    # 我们期望这里的均值比较低。
    # 如果均值不是 0，说明存在“Draft 猜错了，但 Target 觉得也能接受”的情况 (Soft Label 的价值)
    print_group_stats("负样本 (Token Mismatch)", mismatch_scores)
    
    # 3. 全样本
    all_scores = match_scores + mismatch_scores
    print_group_stats("全样本 (All Samples)", all_scores)

def main():
    process_strict_stats(TRAIN_DIRS, "训练集 (Train)")
    process_strict_stats(TEST_DIRS, "测试集 (Test)")

if __name__ == "__main__":
    main()