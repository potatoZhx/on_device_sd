import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

# =================配置区域=================
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

OUTPUT_FILE = "srdp_processed_data.pt"
# =========================================

def find_latest_summary_file(directory):
    """根据用户逻辑寻找最新的 summary jsonl 文件"""
    if not os.path.exists(directory):
        print(f"⚠️ 目录不存在: {directory}")
        return None
    
    files = [f for f in os.listdir(directory) if f.endswith('.jsonl') and 'summary' in f]
    if not files:
        print(f"⚠️ 目录 {directory} 下没有找到 summary jsonl 文件")
        return None
    
    # 按时间排序取最新的 (文件名通常包含时间戳)
    files.sort(reverse=True)
    return os.path.join(directory, files[0])

class SRDPFeatureExtractor:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        """重置序列历史状态"""
        self.state = {
            "min_top1_prob": 1.0,
            "avg_entropy": 0.0,
            "step_count": 0,
            "accum_score_loss": 0.0
        }

    def compute_entropy(self, logits):
        """计算 Top-K Logits 熵"""
        # 注意：这里 logits 是 list，转 tensor
        probs = F.softmax(torch.tensor(logits), dim=-1)
        # 为了速度和数值稳定，取 Top-10
        topk_probs, _ = torch.topk(probs, 10)
        topk_probs = topk_probs / topk_probs.sum()
        entropy = -(topk_probs * torch.log(topk_probs + 1e-9)).sum()
        return entropy.item(), probs.max().item(), probs

    def extract(self, step_data, step_idx, prev_mlp_pred=1.0):
        """
        提取 14 维特征向量
        """
        # 1. 基础 Logits 特征
        logits = step_data["full_logits"]
        entropy, top1_prob, all_probs = self.compute_entropy(logits)
        
        # 计算 Margin (Top1 - Top2)
        top2_vals, _ = torch.topk(all_probs, 2)
        margin = (top2_vals[0] - top2_vals[1]).item()

        # 2. 路由特征 (核心)
        orig_ids = np.array(step_data["router_original"]["ids"]) # [Layers, TopK]
        orig_weights = np.array(step_data["router_original"]["weights"])
        mod_ids = np.array(step_data["router_modified"]["ids"])
        mod_weights = np.array(step_data["router_modified"]["weights"])
        
        # 掩码：哪些位置发生了替换
        mask = (orig_ids != mod_ids)
        num_replaced = np.sum(mask)
        total_slots = orig_ids.size
        
        # Replacement Rate
        replace_rate = num_replaced / (total_slots + 1e-9)
        
        # Score Loss (Current): 损失了多少原始权重
        # 简化计算：只计算被替换位置的原始权重之和 (假设这些权重被浪费了)
        # 更精细的计算是: sum(orig_weight - mod_weight) at mismatch positions
        # 这里用简化的 sum(orig_weight * mask) 代表 "Loss of Original Intent"
        curr_score_loss = np.sum(orig_weights * mask)
        
        # Max Layer Loss (木桶效应)
        layer_losses = np.sum(orig_weights * mask, axis=1) # Sum over TopK dimension
        max_layer_loss = np.max(layer_losses) if layer_losses.size > 0 else 0.0

        # 3. 隐状态特征
        emb = torch.tensor(step_data["final_embedding"])
        hidden_norm = torch.norm(emb, p=2).item()

        # 4. 历史状态更新
        s = self.state
        s["step_count"] += 1
        s["min_top1_prob"] = min(s["min_top1_prob"], top1_prob)
        s["avg_entropy"] = (s["avg_entropy"] * (s["step_count"] - 1) + entropy) / s["step_count"]
        s["accum_score_loss"] += curr_score_loss

        # 5. 组装特征 (14 dims)
        features = [
            # --- Pain Signals (路由降级) ---
            curr_score_loss,            # 0. 当前步分数损失
            replace_rate,               # 1. 替换率
            max_layer_loss,             # 2. 最大单层损失
            s["accum_score_loss"] / 10.0, # 3. 累积损失 (简单归一化)
            0.0,                        # 4. Sim Loss (预留位，暂无矩阵)
            
            # --- Confusion Signals (不确定性) ---
            top1_prob,                  # 5. Top1 概率
            entropy,                    # 6. 熵
            margin,                     # 7. 概率差 Margin
            
            # --- History Signals (历史状态) ---
            step_idx / 10.0,            # 8. 归一化步数
            s["min_top1_prob"],         # 9. 历史最低置信度
            s["avg_entropy"],           # 10. 历史平均熵
            hidden_norm / 100.0,        # 11. 隐状态范数 (简单归一化)
            prev_mlp_pred,              # 12. 上一步 MLP 预测值 (Label Shift)
            
            # 1.0 if top1_prob > 0.9 else 0.0 # 13. Is High Confidence? (Binary Feature)
        ]
        
        return torch.tensor(features, dtype=torch.float32)

def process_files(file_paths, extractor):
    """处理一组文件并返回样本"""
    all_features = []
    all_labels = []
    
    total_files = len(file_paths)
    print(f"正在处理 {total_files} 个文件...")

    for f_path in file_paths:
        if not f_path: continue
        print(f"读取: {f_path}")
        
        with open(f_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Parsing"):
                try:
                    record = json.loads(line)
                    
                    int_out = record["intervention"]["output"]
                    base_out = record["baseline"]["output"]
                    steps_data = record["intervention"]["steps"]
                    
                    # === 截断式清洗逻辑 ===
                    # 找到第一个 mismatch 的位置
                    cutoff_idx = len(int_out)
                    for i in range(min(len(int_out), len(base_out))):
                        if int_out[i] != base_out[i]:
                            cutoff_idx = i
                            break
                    
                    # 重置序列状态
                    extractor.reset_state()
                    
                    # 遍历每一步
                    for i, step_data in enumerate(steps_data):
                        # 如果当前步超过了 cutoff_idx，说明已经是前缀错误的无效数据，丢弃
                        if i > cutoff_idx:
                            break
                        
                        # Label Logic:
                        # i < cutoff_idx -> 1 (Accept)
                        # i == cutoff_idx -> 0 (Reject, 第一个错误)
                        label = 1.0 if i < cutoff_idx else 0.0
                        
                        # 获取特征 (对于第一步，prev_pred 设为 1.0；后续用上一步的 Label 近似)
                        prev_pred = 1.0 if i == 0 else (1.0 if (i-1) < cutoff_idx else 0.0)
                        
                        feat = extractor.extract(step_data, i + 1, prev_mlp_pred=prev_pred)
                        
                        all_features.append(feat)
                        all_labels.append(label)
                        
                        # 只要遇到 Reject (0)，该序列后续数据全部丢弃，跳出当前样本循环
                        if label == 0.0:
                            break
                            
                except Exception as e:
                    # print(f"Error skipping line: {e}")
                    continue
                    
    return torch.stack(all_features), torch.tensor(all_labels).unsqueeze(1)

def main():
    extractor = SRDPFeatureExtractor()
    
    # 1. 寻找训练文件
    train_files = []
    for d in TRAIN_DIRS:
        f = find_latest_summary_file(DIR+d)
        if f: train_files.append(f)
        
    # 2. 寻找测试文件
    test_files = []
    for d in TEST_DIRS:
        f = find_latest_summary_file(DIR+d)
        if f: test_files.append(f)
        
    if not train_files:
        print("❌ 没有找到训练数据")
        return

    # 3. 处理数据
    print("\n=== Processing Train Data ===")
    X_train, y_train = process_files(train_files, extractor)
    
    print("\n=== Processing Test Data ===")
    X_test, y_test = process_files(test_files, extractor)
    
    # 4. 保存
    print(f"\n保存数据到 {DIR+OUTPUT_FILE} ...")
    data_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }
    torch.save(data_dict, DIR+OUTPUT_FILE)
    
    print("\n=== 数据统计 ===")
    print(f"Train Samples: {len(y_train)} | Pos Rate: {y_train.mean():.2%}") # 正样本率89.5
    print(f"Test Samples : {len(y_test)}  | Pos Rate: {y_test.mean():.2%}") # 正样本率94.23
    print(f"Feature Dim  : {X_train.shape[1]}")
    print("✅ 数据处理完成！")

if __name__ == "__main__":
    main()