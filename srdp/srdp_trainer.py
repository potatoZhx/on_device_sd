import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch.nn.functional as F
import numpy as np
import os
import time
import json

# ================= 配置区域 =================
DATA_FILE = "/data2/group_谈海生/lagin/data/Sd_Data/data/srdp_processed_data.pt"
BASE_MODEL_DIR = "/data2/group_谈海生/lagin/models/SRDP_Experiments"
BATCH_SIZE = 512
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs 已经是 Sigmoid 后的概率
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss

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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    print("\n" + "="*85)
    print(f" {'Threshold':<10} | {'Recall (保留好Token)':<20} | {'Specificity (拦截坏Token)':<22} | {'Accuracy':<10}")
    print("-" * 85)

    best_th = 0.5
    target_spec = 0.65  # 我们的目标：至少拦截 65% 的错误
    best_recall_at_spec = 0.0

    # 遍历阈值，寻找最佳平衡点
    for th in np.arange(0.05, 0.96, 0.05):
        preds_binary = (all_preds > th).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_targets, preds_binary).ravel()
        
        # 计算关键指标
        recall = tp / (tp + fn + 1e-9)          # 越高越好：不打断正确的生成
        specificity = tn / (tn + fp + 1e-9)     # 核心指标：能抓住多少错误
        acc = (tp + tn) / (tp + tn + fp + fn)

        # 标记出符合我们策略的行
        marker = ""
        if specificity >= target_spec:
            marker = "✅ (达标)"
            # 在达标的情况下，找 Recall 最高的
            if recall > best_recall_at_spec:
                best_recall_at_spec = recall
                best_th = th
        
        print(f" {th:.2f}       | {recall:.2%}             | {specificity:.2%}               | {acc:.2%} {marker}")

    print("="*85)
    print(f"🚀 推荐最佳阈值: {best_th} (在拦截率 >={target_spec:.0%} 的前提下，召回率最高)")
    
    # 用推荐阈值计算最终 Return 指标
    final_preds = (all_preds > best_th).astype(int)
    metrics = {
        "acc": accuracy_score(all_targets, final_preds),
        "auc": roc_auc_score(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.5,
        "prec": precision_score(all_targets, final_preds, zero_division=0),
        "rec": recall_score(all_targets, final_preds, zero_division=0),
        "f1": f1_score(all_targets, final_preds, zero_division=0)
    }
    return metrics

def train():
    if not os.path.exists(DATA_FILE):
        print(f"❌ 数据文件未找到: {DATA_FILE}")
        return

    # 1. 创建实验目录 (按时间戳)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(BASE_MODEL_DIR, f"run_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"📂 本次实验目录: {exp_dir}")

    # 2. 加载数据
    print(f"Loading data...")
    data = torch.load(DATA_FILE)
    X_train_full, y_train_full = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    
    # 3. 划分 训练集 / 验证集 (9:1)
    # 注意：验证集必须来自训练分布(Wiki)，不能碰测试分布(MT-Bench)
    total_train = len(X_train_full)
    val_size = int(total_train * 0.1)
    train_size = total_train - val_size
    
    dataset_full = TensorDataset(X_train_full, y_train_full)
    train_subset, val_subset = random_split(dataset_full, [train_size, val_size])
    
    # 构建 Loader
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Data Split: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(X_test)}")
    
    # 4. 初始化模型
    input_dim = X_train_full.shape[1]
    model = SRDP_Predictor(input_dim=input_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.8, gamma=2)
    
    # 5. 训练循环
    print("\n=== Start Training ===")
    log_file = os.path.join(exp_dir, "training_log.txt")
    best_val_auc = 0.0
    
    with open(log_file, "w") as f:
        f.write("Epoch,Train_Loss,Val_Acc,Val_AUC,Val_F1\n")
        
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # 验证集评估
        val_metrics = evaluate(model, val_loader, DEVICE)
        
        # 记录日志
        log_str = (f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | "
                   f"Val Acc: {val_metrics['acc']:.2%} | AUC: {val_metrics['auc']:.4f}")
        print(log_str)
        
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.4f},{val_metrics['acc']:.4f},{val_metrics['auc']:.4f},{val_metrics['f1']:.4f}\n")

        # 保存最佳模型 (基于 Val AUC)
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))

    # 6. 最终测试 (Test Set - MT Bench)
    print("\n=== Final Evaluation on Test Set (MT-Bench) ===")
    # 加载最佳权重
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pth")))
    test_metrics = evaluate(model, test_loader, DEVICE)
    
    report = (
        f"Final Test Results (MT-Bench):\n"
        f"-----------------------------\n"
        f"Accuracy : {test_metrics['acc']:.2%}\n"
        f"AUC      : {test_metrics['auc']:.4f}\n"
        f"Precision: {test_metrics['prec']:.2%}\n"
        f"Recall   : {test_metrics['rec']:.2%}\n"
        f"F1 Score : {test_metrics['f1']:.4f}\n"
    )
    print(report)
    
    # 保存最终报告
    with open(os.path.join(exp_dir, "final_test_report.txt"), "w") as f:
        f.write(report)
        
    print(f"✅ 实验全部完成！所有结果已保存至: {exp_dir}")

if __name__ == "__main__":
    train()