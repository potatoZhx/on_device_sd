# 实验数据总结报告：Wiki 与 MTBench

本报告深入分析了 `step_empirical_rates_*_avg_summary.json` 和 `expert_coverage_*.json` 文件中的核心指标，对比了不同替换策略（Replace Count）在两个数据集上的表现。

## 1. Wiki 实验数据总结报告

**实验配置：**

*   **数据集**: Wiki (Wikipedia)
*   **模型**: Qwen3-30B-A3B-Base
*   **对比策略**: Replace Count (替换专家数量) = 1, 2, 3, 4

### 1.1 核心指标趋势分析

我们对比了不同替换数量下的平均接受率（Mean Alpha）和理论加速比（Theoretical Speedup）。**Wiki 数据集展现出较高的预测难度，导致指标随 Step 衰减较快。**

| 实验组 (Replace Count) | Step 1 Mean Alpha | Step 10 Mean Alpha |
| :--------------------- | :---------------- | :----------------- |
| **Replace 1**          | **0.979**         | **0.625**          |
| **Replace 2**          | 0.960             | 0.510              |
| **Replace 3**          | 0.936             | 0.400              |
| **Replace 4**          | 0.894             | 0.254              |

**关键发现：**

1.  **显著的性能衰减**：Wiki 数据的长序列预测难度较大。即使是表现最好的 **Replace 1** 策略，Alpha 值也从 Step 1 的 0.979 大幅滑落至 Step 10 的 0.625，加速比损失近 73% (9.90x -> 2.65x)。
2.  **激进策略的失效**：**Replace 4** 在 Wiki 数据上几乎不可用。在 Step 10 时，其 Alpha 值仅为 0.254，理论加速比降至 1.34x，几乎退化为无加速状态。这表明在复杂文本（Wiki）上，过多的专家替换会严重破坏模型的预测能力。
3.  **最优策略**：**Replace 1** 依然是 Wiki 数据集上的稳健选择，尽管后期衰减明显，但仍保持了相对最高的接受率。

### 1.2 专家覆盖率 (Expert Coverage) 分析

以 **Replace 4** (Wiki 实验中最激进策略) 的 Step 1 数据 (`expert_coverage_232739.json`) 为例：

*   **层级分布**：
    *   **Layer 0-2**：覆盖率较高，分别为 100%, 95.0%, 94.3%。
    *   **Layer 3-13**：覆盖率迅速下降并稳定在 **91% - 94%** 区间（如 Layer 11 为 91.9%）。这比 MTBench 同策略下的覆盖率波动稍大，说明 Wiki 数据激活了更多样化的专家组合。
*   **平均相交数 (Intersection)**：
    *   Layer 1 的平均交集为 **7.60** (总数 8)，Layer 10 为 **7.39**。这意味着即使在激进替换下，Top-P 选出的专家与 Base 模型仍有约 92%-95% 的重合度。

---

## 2. MTBench 实验数据总结报告

**实验配置：**

*   **数据集**: MTBench
*   **模型**: Qwen3-30B-A3B-Base
*   **对比策略**: Replace Count = 1, 2, 3, 4 及 Test Run (Topp Test)

### 2.1 核心指标趋势分析

MTBench 数据相对更规律，整体指标优于 Wiki。特别加入了 `replace_last_one_with_topp_test` (Test Run) 进行对比。

| 实验组 (Replace Count) | Step 1 Mean Alpha | Step 10 Mean Alpha |
| :--------------------- | :---------------- | :----------------- |
| **Replace 1**          | 0.980             | 0.767              |
| **Replace 2**          | 0.972             | 0.670              |
| **Replace 3**          | 0.965             | 0.565              |
| **Replace 4**          | 0.949             | 0.481              |
| **Test Run (Ref)**     | **0.990**         | **0.893**          |

**关键发现：**

1.  **高稳定性的 Test Run**：`replace_last_one_with_topp_test` 展现了惊人的稳定性。在 Step 10，其 Alpha 值仍高达 **0.893**，加速比保持在 **6.64x**。这显著优于标准的 Replace 1 策略 (Step 10 Alpha 0.767)，暗示该测试组使用了更优的参数（如动态阈值或更精确的 Top-P 截断）。
2.  **优于 Wiki 的表现**：对比 Wiki 数据，MTBench 在相同策略下的 Alpha 衰减更缓。例如 **Replace 1** 在 MTBench 的 Step 10 Alpha 为 0.767，而在 Wiki 仅为 0.625。
3.  **策略分水岭**：Replace 3 和 Replace 4 在 Step 10 的加速比均低于 2.5x，收益有限。**Replace 1** 是标准策略中的最佳平衡点。

### 2.2 专家覆盖率 (Expert Coverage) 分析

以 **Replace 4** 的 Step 1 数据 (`expert_coverage_221546.json`) 为例：

*   **层级分布**：
    *   **Layer 0-2**：覆盖率极高 (1.0, 0.953, 0.957)。
    *   **Layer 3-13**：覆盖率非常稳定，基本维持在 **93% ± 1%** 的窄区间内。
*   **平均相交数**：
    *   Layer 1 交集为 **7.63**，Layer 10 为 **7.44**。MTBench 的专家选择模式比 Wiki 更为集中和确定。

---

## 3. 综合结论与建议

1.  **数据敏感性差异**：
    *   **结论**：模型在 **Wiki 数据集上的表现显著弱于 MTBench**。Wiki 的 Step 10 平均 Alpha 比 MTBench 低约 **0.14 - 0.23**。
    *   **建议**：在处理 Wiki 类复杂知识文本时，应采取更保守的策略（严格限制 Replace Count = 1，甚至考虑动态降低 Step 长度），以避免加速比崩塌。

2.  **Test Run 的异常优越性**：
    *   **结论**：MTBench 的 `Test Run` 组数据异常出色（Step 10 加速比 6.64x vs 标准组 4.06x）。
    *   **建议**：**极高优先级**去分析 `mtbench_results_replace_last_one_with_topp_test` 对应的代码逻辑或参数配置。复现该配置是提升整体系统性能的关键突破口。

3.  **通用推荐策略**：
    *   无论数据集如何，**Replace Count = 1** 均是当前最稳健的基准策略。对于 Replace 2/3/4，除非在 Step 前期（Step < 3），否则带来的 Alpha 损失远超其理论上的计算量节省。