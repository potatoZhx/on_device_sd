import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error
import numpy as np
import os
import time

# ================= 配置区域 =================
# 必须与 data_processor 输出的文件名一致
DATA_FILE = "/data2/group_谈海生/lagin/data/Sd_Data/data/srdp_processed_filtered.pt"
BASE_MODEL_DIR = "/data2/group_谈海生/lagin/models/SRDP_Experiments"
BATCH_SIZE = 512
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================================

class SRDP_Predictor(nn.Module):
    def __init__(self, input_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # 输出 0~1，用于拟合 Soft Label
        )

    def forward(self, x):
        return self.net(x)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        # Loader 吐出 X, y, w (虽然 w 在 MSE 评估中不直接使用，但占位符需要接住)
        for X, y, w in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # [核心指标] 
    # MSE: 均方误差 (主要优化目标)
    # MAE: 平均绝对误差 (更直观的误差幅度)
    val_mse = mean_squared_error(all_targets, all_preds)
    val_mae = mean_absolute_error(all_targets, all_preds)
    
    # [辅助指标] 
    # 为了看 AUC/Acc，我们需要一个临时的二分类标准。
    # 现在的 Label 定义：Mismatch=0, Match=Score(>0)
    # 我们认为只要 Label > 0.01 (即非 Mismatch) 就算正类
    all_targets_binary = (all_targets > 0.01).astype(int)

    print("\n" + "="*85)
    print(f" [Objective] MSE: {val_mse:.6f} (📉 Target) | MAE: {val_mae:.6f}")
    print("-" * 85)
    print(f" {'Threshold':<10} | {'Recall':<20} | {'Specificity':<22} | {'Accuracy':<10}")
    print("-" * 85)

    best_th = 0.5
    target_spec = 0.65  
    best_recall_at_spec = 0.0

    # 遍历阈值寻找最佳观察点 (仅供分析，不影响模型保存)
    for th in np.arange(0.05, 0.96, 0.05):
        preds_binary = (all_preds > th).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_targets_binary, preds_binary).ravel()
        
        recall = tp / (tp + fn + 1e-9)          
        specificity = tn / (tn + fp + 1e-9)     
        acc = (tp + tn) / (tp + tn + fp + fn)

        marker = ""
        if specificity >= target_spec:
            marker = "✅"
            if recall > best_recall_at_spec:
                best_recall_at_spec = recall
                best_th = th
        
        print(f" {th:.2f}       | {recall:.2%}             | {specificity:.2%}               | {acc:.2%} {marker}")

    print("="*85)
    
    # 使用 best_th 计算最终的分类指标
    final_preds = (all_preds > best_th).astype(int)
    
    metrics = {
        "mse": val_mse,
        "mae": val_mae,
        "auc": roc_auc_score(all_targets_binary, all_preds) if len(set(all_targets_binary)) > 1 else 0.5,
        "acc": accuracy_score(all_targets_binary, final_preds),
        "prec": precision_score(all_targets_binary, final_preds, zero_division=0),
        "rec": recall_score(all_targets_binary, final_preds, zero_division=0),
        "f1": f1_score(all_targets_binary, final_preds, zero_division=0)
    }
    return metrics

def train():
    if not os.path.exists(DATA_FILE):
        print(f"❌ 数据文件未找到: {DATA_FILE}")
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(BASE_MODEL_DIR, f"run_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"📂 本次实验目录: {exp_dir}")

    print(f"Loading data...")
    data = torch.load(DATA_FILE)
    
    X_train_full, y_train_full, w_train_full = data["X_train"], data["y_train"], data["w_train"]
    X_test, y_test, w_test = data["X_test"], data["y_test"], data["w_test"]
    
    # 划分验证集
    total_train = len(X_train_full)
    val_size = int(total_train * 0.1)
    train_size = total_train - val_size
    
    # Dataset 现在包含 (X, y, w)
    dataset_full = TensorDataset(X_train_full, y_train_full, w_train_full)
    train_subset, val_subset = random_split(dataset_full, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test, w_test), batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Data Split: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(X_test)}")
    
    input_dim = X_train_full.shape[1]
    model = SRDP_Predictor(input_dim=input_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # [核心策略] 使用标准 MSE Loss，回归任务
    criterion = nn.MSELoss()
    
    print("\n=== Start Training (Standard MSE) ===")
    log_file = os.path.join(exp_dir, "training_log.txt")
    
    # [核心参考指标] 初始化最佳 MSE 为无穷大
    best_val_mse = float('inf') 
    
    with open(log_file, "w") as f:
        f.write("Epoch,Train_Loss,Val_MSE,Val_MAE,Val_AUC,Val_Acc\n")
        
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for X, y, w in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            # w 不参与 MSE Loss 计算，因为 Label 0.0 已经是强信号
            
            optimizer.zero_grad()
            preds = model(X)
            
            # 回归拟合
            loss = criterion(preds, y) 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, DEVICE)
        
        log_str = (f"Epoch {epoch+1:02d} | Loss: {avg_loss:.6f} | "
                   f"MSE: {val_metrics['mse']:.6f} | MAE: {val_metrics['mae']:.6f} | "
                   f"AUC: {val_metrics['auc']:.4f}")
        print(log_str)
        
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.6f},{val_metrics['mse']:.6f},{val_metrics['mae']:.6f},{val_metrics['auc']:.4f},{val_metrics['acc']:.4f}\n")

        # [模型保存逻辑]：只要 MSE 创新低，就保存
        if val_metrics['mse'] < best_val_mse:
            print(f"🔥 New Best MSE: {val_metrics['mse']:.6f} (Was: {best_val_mse:.6f}) -> Saving Model...")
            best_val_mse = val_metrics['mse']
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))

    print("\n=== Final Evaluation on Test Set (MT-Bench) ===")
    # 加载 MSE 最小的那个最佳模型
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pth")))
    test_metrics = evaluate(model, test_loader, DEVICE)
    
    report = (
        f"Final Test Results (MT-Bench):\n"
        f"-----------------------------\n"
        f"MSE (Target): {test_metrics['mse']:.6f}\n"
        f"MAE (Error) : {test_metrics['mae']:.6f}\n"
        f"AUC         : {test_metrics['auc']:.4f}\n"
        f"Accuracy    : {test_metrics['acc']:.2%}\n"
        f"Precision   : {test_metrics['prec']:.2%}\n"
        f"Recall      : {test_metrics['rec']:.2%}\n"
        f"F1 Score    : {test_metrics['f1']:.4f}\n"
    )
    print(report)
    
    with open(os.path.join(exp_dir, "final_test_report.txt"), "w") as f:
        f.write(report)
        
    print(f"✅ 实验全部完成！所有结果已保存至: {exp_dir}")

if __name__ == "__main__":
    train()