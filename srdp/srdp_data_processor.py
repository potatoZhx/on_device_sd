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

# 改名为 filtered，代表经过了 Filtered Soft Label 清洗
OUTPUT_FILE = "srdp_processed_filtered.pt"
# =========================================

def find_latest_summary_file(directory):
    if not os.path.exists(directory):
        print(f"⚠️ 目录不存在: {directory}")
        return None
    files = [f for f in os.listdir(directory) if f.endswith('.jsonl') and 'summary' in f]
    if not files:
        print(f"⚠️ 目录 {directory} 下没有找到 summary jsonl 文件")
        return None
    files.sort(reverse=True)
    return os.path.join(directory, files[0])

class SRDPFeatureExtractor:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.state = {
            "min_top1_prob": 1.0,
            "avg_entropy": 0.0,
            "step_count": 0,
            "accum_score_loss": 0.0
        }

    def compute_entropy(self, logits):
        probs = F.softmax(torch.tensor(logits), dim=-1)
        topk_probs, _ = torch.topk(probs, 10)
        topk_probs = topk_probs / topk_probs.sum()
        entropy = -(topk_probs * torch.log(topk_probs + 1e-9)).sum()
        return entropy.item(), probs.max().item(), probs

    def extract(self, step_data, step_idx, prev_mlp_pred=1.0):
        logits = step_data["full_logits"]
        entropy, top1_prob, all_probs = self.compute_entropy(logits)
        
        top2_vals, _ = torch.topk(all_probs, 2)
        margin = (top2_vals[0] - top2_vals[1]).item()

        orig_ids = np.array(step_data["router_original"]["ids"])
        orig_weights = np.array(step_data["router_original"]["weights"])
        mod_ids = np.array(step_data["router_modified"]["ids"])
        mod_weights = np.array(step_data["router_modified"]["weights"])
        
        mask = (orig_ids != mod_ids)
        num_replaced = np.sum(mask)
        total_slots = orig_ids.size
        
        replace_rate = num_replaced / (total_slots + 1e-9)
        curr_score_loss = np.sum(orig_weights * mask)
        
        layer_losses = np.sum(orig_weights * mask, axis=1)
        max_layer_loss = np.max(layer_losses) if layer_losses.size > 0 else 0.0

        emb = torch.tensor(step_data["final_embedding"])
        hidden_norm = torch.norm(emb, p=2).item()

        s = self.state
        s["step_count"] += 1
        s["min_top1_prob"] = min(s["min_top1_prob"], top1_prob)
        s["avg_entropy"] = (s["avg_entropy"] * (s["step_count"] - 1) + entropy) / s["step_count"]
        s["accum_score_loss"] += curr_score_loss

        features = [
            curr_score_loss, replace_rate, max_layer_loss, s["accum_score_loss"] / 10.0, 0.0,
            top1_prob, entropy, margin,
            step_idx / 10.0, s["min_top1_prob"], s["avg_entropy"], hidden_norm / 100.0, prev_mlp_pred,
        ]
        return torch.tensor(features, dtype=torch.float32)

def process_files(file_paths, extractor):
    all_features = []
    all_labels = []
    all_weights = []
    
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
                    steps_base = record["baseline"]["steps"]
                    
                    cutoff_idx = len(int_out)
                    for i in range(min(len(int_out), len(base_out))):
                        if int_out[i] != base_out[i]:
                            cutoff_idx = i
                            break
                    
                    extractor.reset_state()
                    prev_label = 1.0
                    
                    for i, step_data in enumerate(steps_data):
                        if i > cutoff_idx: break
                        
                        q_logits = torch.tensor(step_data["full_logits"])
                        p_logits = torch.tensor(steps_base[i]["full_logits"])
                        
                        q_probs = F.softmax(q_logits, dim=-1)
                        p_probs = F.softmax(p_logits, dim=-1)
                        
                        draft_token_id = torch.argmax(q_probs).item()
                        target_token_id = torch.argmax(p_probs).item()
                        
                        # === [核心修改] Filtered Soft Label ===
                        is_match = (draft_token_id == target_token_id)
                        
                        if is_match:
                            # 正样本：Match
                            # 保留软标签，告诉模型"这个词有多好"
                            q_x = q_probs[draft_token_id].item()
                            p_x = p_probs[draft_token_id].item()
                            label = min(1.0, p_x / (q_x + 1e-10))
                            
                            # 权重设为 1.0 (标准 MSE)
                            weight = 1.0
                        else:
                            # 负样本：Mismatch
                            # [关键修改] 既然 Token 不对，强制 Label = 0.0
                            # 这是对模型最清晰的指导：只要猜错，就是零分
                            label = 0.0 
                            
                            # [关键修改] 权重恢复为 1.0
                            # 理由：Label=0 本身已经是极强的惩罚信号 (MSE会产生巨大的Loss)，
                            # 如果再乘 10 倍，会导致模型过度恐慌，全盘输出 0。
                            weight = 1.0 
                        
                        feat = extractor.extract(step_data, i + 1, prev_mlp_pred=prev_label)
                        
                        all_features.append(feat)
                        all_labels.append(label)
                        all_weights.append(weight)
                        
                        prev_label = label
                        
                        if i == cutoff_idx: break
                            
                except Exception as e:
                    continue
    
    return (torch.stack(all_features), 
            torch.tensor(all_labels).unsqueeze(1), 
            torch.tensor(all_weights).unsqueeze(1))

def main():
    extractor = SRDPFeatureExtractor()
    
    train_files = []
    for d in TRAIN_DIRS:
        f = find_latest_summary_file(DIR+d)
        if f: train_files.append(f)
        
    test_files = []
    for d in TEST_DIRS:
        f = find_latest_summary_file(DIR+d)
        if f: test_files.append(f)
        
    if not train_files:
        print("❌ 没有找到训练数据")
        return

    print("\n=== Processing Train Data ===")
    X_train, y_train, w_train = process_files(train_files, extractor)
    
    print("\n=== Processing Test Data ===")
    X_test, y_test, w_test = process_files(test_files, extractor)
    
    print(f"\n保存数据到 {DIR+OUTPUT_FILE} ...")
    data_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "w_train": w_train,
        "X_test": X_test,
        "y_test": y_test,
        "w_test": w_test
    }
    torch.save(data_dict, DIR+OUTPUT_FILE)
    
    print("\n=== 数据统计 ===")
    print(f"Train Samples: {len(y_train)}")
    print(f"  - Match (Pos) Count: {torch.sum(y_train > 0.0).item()}")
    print(f"  - Mismatch (Neg, Label=0) Count: {torch.sum(y_train == 0.0).item()}")
    print("✅ 数据处理完成！")

if __name__ == "__main__":
    main()