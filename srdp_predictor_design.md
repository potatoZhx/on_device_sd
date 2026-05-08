基于我们之前的深入探讨，特别是关于**“GPU计算必要性”**、**“截断式数据清洗”**以及**“条件概率 vs 累计概率”**的共识，我为您重新拟写了最终的 MLP 预测器方案。

我们将此方案命名为 **SRDP (Self-Referential Degradation Perception，自指型降级感知)** 预测器。

---

# SRDP 预测器方案设计文档

## 1. 核心设计哲学

* **无需 Target 参与**：利用 Draft 模型“原本想选的专家”与“实际被迫选的专家”之间的差异（自指性）来量化生成质量。
* **GPU 原位计算**：所有特征提取、推理、决策逻辑全在 GPU 显存内完成，杜绝 `Device-to-Host` 拷贝，将延迟压至微秒级。
* **截断式学习**：只学习有效前缀后的第一个错误，防止模型拟合“基于垃圾输入的垃圾输出”。
* **双重决策机制**：MLP 负责输出**单步条件概率**，外部 Controller 负责维护**全局累计概率**。

---

## 2. 特征工程 (Input Features, )

特征向量  旨在回答三个问题：**我现在多痛？（路由损失）**，**我现在多晕？（输出熵）**，**我之前表现如何？（历史状态）**。

### A. 路由降级特征 (The "Pain" Signals) - 核心驱动力

*这些特征衡量“强行替换专家”造成的内部损伤。*

1. **`curr_score_loss`**: 当前步所有层中，被替换掉的 Original Expert 的原始权重之和。

* *含义*：损失了多少“注意力质量”。


2. **`curr_sim_loss`**: 当前步所有层中，(1 - CosineSim(原专家, 替补专家)) 的加权平均。

* *实现*：依赖离线预计算的 `[Layers, Experts, Experts]` 相似度矩阵查表。


3. **`replacement_rate`**: 当前步发生替换的层数占比。
4. **`max_layer_loss`**: 单层最大的 Score Loss。

* *含义*：捕捉“木桶效应”，某一层彻底崩坏可能导致全局崩坏。


5. **`accum_score_loss`**: 历史所有步的 Score Loss 累加（归一化）。

### B. 输出不确定性 (The "Confusion" Signals)

1. **`top1_prob`**: 当前 Token 的原始 Top-1 概率。
2. **`logit_entropy`**: 当前 Token 的 Top-K Logits 熵。
3. **`prob_margin`**: Top-1 概率减去 Top-2 概率。

### C. 历史状态 (The "History" Signals)

*用于捕捉误差累积和隐状态漂移。*

1. **`step_idx_norm`**: 当前步数 / Max_Steps。
2. **`min_prev_prob`**: 历史各步中最低的 `top1_prob`。
3. **`avg_prev_entropy`**: 历史平均熵。
4. **`hidden_state_norm`**: 最后一层 Hidden State 的 L2 范数（异常检测）。
5. **`prev_mlp_pred`**: 上一步 MLP 预测出的接受率（递归特征）。

---

## 3. 模型架构 (Model Architecture)

为了配合 PyTorch 的 `torch.compile` 或 CUDA Graphs 优化，模型保持极简的全连接结构。

```python
class SRDP_Predictor(nn.Module):
    def __init__(self, input_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: 特征融合
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64), # 只有第一层加 Norm 稳定分布
            
            # Layer 2: 压缩
            nn.Linear(64, 32),
            nn.ReLU(),
            
            # Layer 3: 输出概率
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

```

---

## 4. 数据构建与清洗 (Data Pipeline)

### 清洗策略：截断式 (Truncated)

对于一个生成序列 `[Token_1, Token_2, ..., Token_N]`，与 Target 对比后发现 `Token_k` 是第一个不匹配的词。

* **样本 1 ~ k-1**: Label `1` (Accept)
* **样本 k**: Label `0` (Reject) —— **这是最有价值的样本，教会模型识别临界点。**
* **样本 k+1 ~ N**: **丢弃**。

### 训练配置

* **Loss**: `BCELoss`。
* **Resampling**: 由于正样本通常多于负样本（假设 Draft 准确率 > 50%），需对 Label `0` 的样本进行过采样（Oversampling）或在 Loss 中设置 `pos_weight`。

---

## 5. 在线推理逻辑 (Inference Logic)

这是集成到 Speculative Decoding `while` 循环中的逻辑。

**关键点：**

1. **MLP 预测单步概率** ().
2. **Controller 维护累计概率** ().
3. **双阈值停止**。

```python
# 初始化 (GPU Tensors)
accum_prob = torch.tensor(1.0, device='cuda')
prev_mlp_pred = torch.tensor(1.0, device='cuda')
history_state = init_history_state() # 维护 min_conf, avg_entropy 等

# 预热过的极速 MLP
fast_mlp = torch.compile(model, mode="reduce-overhead")

for step in range(MAX_DRAFT_LEN):
    # 1. Draft Model Forward (产生 logits, hidden_states, metadata)
    # ...
    
    # 2. 特征提取 (全 GPU 操作，无 .cpu())
    # 包含查表相似度矩阵、计算 Norm 等
    feats = extract_features_on_gpu(
        logits, hidden_states, metadata, history_state, prev_mlp_pred
    )
    
    # 3. MLP 推理 (微秒级)
    p_current = fast_mlp(feats) # Scalar Tensor on GPU
    
    # 4. 更新状态
    accum_prob *= p_current
    prev_mlp_pred = p_current
    update_history_state(history_state, ...)
    
    # 5. 停止决策 (唯一的 CPU 同步点)
    # 逻辑：
    # A. 累计概率太低：说明生成的这串序列整体可信度低，不值得继续算了。
    # B. 单步概率极低：说明这一步肯定是错的，没必要往后生了。
    if accum_prob.item() < GLOBAL_THRESHOLD (e.g., 0.6) or \
       p_current.item() < LOCAL_THRESHOLD (e.g., 0.4):
        break 

```

---

## 6. 预期效果与优势

1. **精准止损**：相比于固定的 Top-K 熵阈值，MLP 能理解“虽然我很自信（熵低），但我知道我换了一个很差的专家（Score Loss 高），所以我可能在瞎说”。
2. **速度无损**：在 A100/A800 上，配合 CUDA Graphs，MLP 推理耗时可忽略不计（< 5us），完全被 Attention 计算掩盖。
3. **鲁棒性**：引入历史特征（前置 Token 信息）后，能有效抵抗 Speculative Decoding 中的“盲目自信”陷阱。

### 下一步行动建议

1. **预计算相似度矩阵**：写脚本提取 Qwen3-30B 所有 MoE 层的权重，计算 Expert-to-Expert 余弦相似度并保存。
2. **生成数据**：利用您现有的脚本跑出带 `final_embedding` 的 jsonl 数据。
3. **训练验证**：先训练模型，在 MT-Bench 测试集上观察 AUC 指标和 Accuracy。


# 为什么截断

正是因为它的特征像正样本（低熵、高自信），但标签是负样本（Label 0），它构成了机器学习中最讨厌的Label Noise（标签噪声）。所以，必须通过截断，把这些“剧毒”样本从训练集中剔除。