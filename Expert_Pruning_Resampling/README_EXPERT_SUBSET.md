# 专家子集推理的PPL检测

本目录包含用于分析MoE模型专家敏感性的脚本，实现了**专家子集推理**的困惑度（PPL）检测功能。

## 功能说明

### 专家子集推理（Expert Subset Inference）
- **功能**：对于原本要使用的top_k个专家，只保留前m个专家进行推理（m < top_k）
  - 例如：如果模型配置为top_k=4，但只使用前2个专家进行推理
  - 这相当于"删除"排名较低的专家，只使用排名靠前的专家
- **脚本**：
  - `ExpertSubsetInference.py` - 核心实现类
  - `run_expert_subset_eval.py` - 基本评估脚本
  - `run_expert_subset_analysis.py` - 高级分析和图表生成
- **评估指标**：Wikitext数据集上的困惑度（PPL）

## 实现原理

### ExpertSubsetBlockWrapper类
```python
class ExpertSubsetBlockWrapper(nn.Module):
    def __init__(self, original_block, use_top_m):
        # 初始化包装器
        # use_top_m: 实际使用的top_m专家数量 (m < top_k)
    
    def forward(self, hidden_states):
        # 1. 计算原始路由logits
        # 2. 计算路由权重并选择top_k专家
        # 3. 只保留前m个专家
        # 4. 计算这些专家的贡献
        # 5. 返回最终结果
```

### 核心逻辑
1. 保持原始的路由计算不变（选择top_k专家）
2. 但在实际计算时，只使用排名前m的专家
3. 权重重新归一化（如果需要）
4. 只计算选中的专家的输出

## 配置参数

在各脚本中可以修改以下主要参数：

```python
# 模型配置
MODELS = [
    {"id": "模型路径", "name": "模型名称", "color": "颜色", "marker": "标记"},
    # 可以添加更多模型
]

SEQ_LEN = 1024      # WikiText 评测的上下文长度
MAX_USE_TOP_M = 7   # 最大使用的top_m专家数量
BATCH_SIZE = 8      # 批量大小
```

## 使用方法

### 方法一：基本评估

```bash
python run_expert_subset_eval.py
```

### 方法二：高级分析（推荐）

```bash
python run_expert_subset_analysis.py
```

该脚本会自动运行评估并生成与论文相似的图表。

## 输出文件

运行后会生成以下文件：

1. `expert_subset_eval_results.txt` - 专家子集推理评估结果
2. `expert_subset_analysis_results.txt` - 高级分析结果
3. `expert_subset_inference_curve.png` - 基本PPL曲线图
4. `expert_pruning_subset_curve.png` - 专家删除风格曲线图
5. `expert_resampling_subset_curve.png` - 专家重采样风格曲线图
6. `combined_expert_sensitivity_analysis.png` - 组合分析图表

## 结果解读

### 专家删除曲线
- X轴：删除专家的排名（保留前m个专家）
- Y轴：Wikitext困惑度（值越低表示性能越好）
- 曲线趋势：随着m值增加（使用更多专家），PPL通常会下降
- 虚线：表示不同性能水平的基线困惑度

### 与直接删除专家的区别

| 方法 | 实现方式 | 特点 |
|------|----------|------|
| 直接删除专家 | 从模型中物理移除专家 | 1. 改变模型结构<br>2. 影响路由计算<br>3. 可能导致模型配置不一致 |
| 专家子集推理 | 只使用top_k中的前m个专家 | 1. 保持模型结构不变<br>2. 路由计算正常进行<br>3. 只限制实际参与计算的专家 |

## 优势

1. **保持模型一致性**：不改变模型结构，只是限制实际使用的专家
2. **模拟真实场景**：更准确地模拟"删除"排名较低的专家的效果
3. **灵活配置**：可以轻松调整使用的专家数量
4. **可重复性**：每次运行的结果更加稳定

## 依赖要求

- PyTorch
- Transformers
- Matplotlib
- NumPy
- Datasets

## 注意事项

1. 确保模型路径正确，并且有足够的GPU内存
2. 可以根据需要调整批量大小以适应不同的硬件配置
3. 评估过程可能需要较长时间，尤其是对于较大的模型
4. 可以通过修改`MAX_USE_TOP_M`来控制评估的范围

## 示例图表

生成的图表与论文中的图表风格一致，包含：
- 多个模型的比较曲线
- 清晰的标题和坐标轴标签
- 网格线和基线虚线
- 图例说明

这些图表可以用于分析不同MoE模型的专家敏感性，了解使用不同数量的专家对模型性能的影响。
