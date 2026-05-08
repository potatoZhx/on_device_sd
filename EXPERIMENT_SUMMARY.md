# MOE_SD 实验总结

本文档总结当前仓库中已经实现或执行过的实验，以及可从脚本输出、结果文件、日志和仓库内 md/txt 记录中确认的结果。

## SRDP 是什么

SRDP 是 Self-Referential Degradation Perception 的缩写，本文档中译为“自指型降级感知”。它是本仓库为 MoE 推测解码设计的轻量接受率预测器，用来判断当前 draft token 是否还值得继续生成，或者应该提前停止 draft 并交给 target 模型验证。

“自指型”的含义是：SRDP 不依赖 target 模型实时参与判断，而是只观察 draft 模型自身的退化信号。例如，原始 router 本来想选择哪些专家，实际由于低显存、CPU/GPU 异构调度或专家替换后又选择了哪些专家；这些差异会形成 `curr_score_loss`、`replace_rate`、`max_layer_loss`、`accum_score_loss` 等路由降级特征。SRDP 还结合 logits 置信度、entropy、top1/top2 margin、hidden state norm 和历史预测值，输出当前 step 的接受概率或 soft label。

在系统中的作用可以概括为：

1. 训练阶段：用 intervention/draft 与 baseline/target 的解码日志构造样本，采用截断式清洗，只保留第一个分歧点及其之前的数据。
2. 推理阶段：SRDP 对每个 draft step 输出单步接受概率，controller 维护累计可靠度。
3. 决策阶段：当单步概率或累计概率低于阈值时停止 draft，避免继续生成低质量 token。

因此，SRDP 的目标不是提升模型本身的精度，而是在异构 MoE 和专家替换造成 draft 退化时，用很低的额外开销预测“还能不能继续草稿生成”，从而在接受率和端到端加速比之间取得更好的平衡。

## 参考的 md/txt 文件

本次总结综合了以下本地文档和结果文本：

- `README.md`：仓库结构、运行方式和已知注意事项。
- `Expert_Pruning_Resampling/README_EXPERT_SUBSET.md`：专家子集推理实验设计、PPL 指标和图表说明。
- `Cache_Prior_Moe/Cache_Prior_out/mmlu_eval_results.txt`：MMLU Cache-Prior 数值结果。
- `srdp/data.md`：SRDP 训练/测试集 soft label 统计。
- `draft_stopping_acceptance_prediction.md`：draft 停止策略、接受率预测特征和启发式判定思路。
- `srdp_predictor_design.md`：SRDP 预测器方案设计，包括特征、截断式清洗、MLP 架构和在线推理逻辑。
- `srdp_experiment_results.md`：SRDP 硬标签与软标签多轮实验结果对比。
- `wiki_mtbench_results_analysis.md`：Wiki 与 MTBench 不同专家替换数量下的 mean alpha、理论加速比和专家覆盖率分析。
- `0317.txt`：低内存异构调度方案、n=2 CPU 专家限制算法和 A800 物理常数测试。
- `.trae/documents/Create MTBench Experiment and Update Inference Logic.md`：新增 MTBench 实验与 Qwen wrapper 的实现计划。

## 1. Cache-Prior MoE 专家缓存实验

### 实验目的

验证在 MoE 路由中加入专家缓存先验后，能否降低专家 cache miss rate，并观察 PPL/accuracy 的代价。

### 方法

- 模型：`/data2/group_谈海生/lagin/models/Qwen1.5-MoE-A2.7B`
- 缓存策略：`ExpertCache` 使用 LRU 维护每层专家缓存。
- Cache-Prior：对已经在缓存内的专家和原始 top-j 专家增加 bias，形式为 `logits + lambda * avg_range * mask`。
- 主要参数：`CACHE_RATIO=0.5`、`TOP_J=2`、WikiText `SEQ_LEN=1024`、MMLU 5-shot、GSM8K 8-shot。

### 已有结果

MMLU 结果来自 `Cache_Prior_Moe/Cache_Prior_out/mmlu_eval_results.txt`：

| Lambda | Accuracy | Miss Rate |
| --- | ---: | ---: |
| 0.0 | 59.87% | 32.57% |
| 0.5 | 54.77% | 13.75% |

结论：`lambda=0.5` 将 miss rate 从 32.57% 降到 13.75%，但 accuracy 从 59.87% 降到 54.77%，说明缓存命中优化有效，但当前配置对任务精度有明显损伤。

WikiText 相关结果：

- `Cache_Prior_Moe/Cache_Prior_out/wikitext_eval_results.txt` 当前为空。
- 根目录日志中存在一次 `lambda=0.5` 的 WikiText/PPL 结果：PPL 8.0566、miss rate 0.17%，日志标注“不强制卸载”。该结果显示 miss rate 极低，但因为未强制卸载，和真实容量约束下的 offload 策略不完全等价。
- 另一个早期日志出现 PPL 14.6543、miss rate 1.80%，并伴随大量“选择专家数量超过缓存限制”警告，更像是旧实现/异常配置下的结果。

GSM8K：

- `Cache_Prior_Moe/Cache_Prior_out/gsm8k_tradeoff_curve.png` 存在。
- 当前没有对应文本结果文件，需从日志或重新运行 `run_gsm8k_eval.py` 补齐准确率和 miss rate 表。

### 输出文件

- `Cache_Prior_Moe/Cache_Prior_out/mmlu_tradeoff_curve.png`
- `Cache_Prior_Moe/Cache_Prior_out/gsm8k_tradeoff_curve.png`
- `Cache_Prior_Moe/Cache_Prior_out/wikitext_tradeoff_curve*.png`
- `Cache_Prior_Moe/Cache_Prior_out/cache_performance.png`

## 2. 专家剪枝与专家重采样敏感性实验

### 实验目的

分析 MoE 中不同 rank 专家的重要性：只保留 top-m 专家或随机替换某个 rank 的专家后，WikiText PPL 如何变化。

### 方法

- 数据集：WikiText-2。
- 指标：PPL，越低越好。
- 模型：`Qwen1.5-MoE-A2.7B`、`DeepSeek-V2-Lite`、`Phi-3.5-MoE-instruct`、`Mixtral-8x7B-v0.1`。
- `remove`：只保留路由 top-m 专家。
- `replace`：随机替换 rank-k 专家，用于观察某个 rank 的敏感性。

### Remove 结果

结果来自 `Expert_Pruning_Resampling/expert_remove_results.json`。

| Model | 最差/最少专家 | 最好/最多专家 | 现象 |
| --- | ---: | ---: | --- |
| Qwen1.5-MoE-A2.7B | m=1, PPL 10.0458 | m=4, PPL 7.9714 | 增加保留专家数后 PPL 稳定下降 |
| DeepSeek-V2-Lite | m=1, PPL 12.6697 | m=6, PPL 7.0110 | top-1 单独不足，m>=4 后接近稳定 |
| Phi-3.5-MoE-instruct | m=1, PPL 5.8689 | m=2, PPL 4.6149 | 第二个专家仍有明显贡献 |
| Mixtral-8x7B-v0.1 | m=1, PPL 6.6476 | m=2, PPL 4.8541 | 第二个专家贡献明显 |

结论：只保留 top-1 会显著升高 PPL；随着保留更多 top-rank 专家，性能恢复并趋于稳定。

### Replace 结果

结果来自 `Expert_Pruning_Resampling/expert_replace_results.json`。

| Model | 替换高 rank 专家时 | 接近原始配置时 | 现象 |
| --- | ---: | ---: | --- |
| Qwen1.5-MoE-A2.7B | m=1, PPL 79.6456 | m=5, PPL 7.9714 | rank-1 专家极敏感 |
| DeepSeek-V2-Lite | m=1, PPL 92.7215 | m=7, PPL 7.0110 | rank-1 专家极敏感 |
| Phi-3.5-MoE-instruct | m=1, PPL 2805.8430 | m=3, PPL 4.6149 | 替换首要专家会崩溃 |
| Mixtral-8x7B-v0.1 | m=1, PPL 258.5952 | m=3, PPL 4.8541 | 替换首要专家会崩溃 |

结论：rank-1 专家承担关键贡献，随机替换 rank-1 会导致 PPL 大幅恶化；替换靠后的专家损害明显减小。

### 输出文件

- `Expert_Pruning_Resampling/expert_remove_results.json`
- `Expert_Pruning_Resampling/expert_replace_results.json`
- `Expert_Pruning_Resampling/combined_expert_sensitivity_analysis.png`

## 3. 推测解码数据采集实验

### 实验目的

构造 target/baseline 和 intervention/draft 的逐步解码对比数据，用于后续估计接受率、路由保真度和训练 SRDP 预测器。

### 方法

- `get_sd_data/wiki_experiment.py`：DeepSeek-V2-Lite 上采集 WikiText。
- `get_sd_data/mtbench101_experiment.py`：Qwen3-30B-A3B-Base 上采集 Wiki 或 MTBench；当前脚本配置为 `DATASET_NAME="wiki"`、`replace_count=2`。
- 每条样本包含输入 token、prefill token、10 步 intervention 输出、10 步 baseline 输出、每步路由专家、路由权重、full logits、最终 embedding 和 `match_rate`。

### 仓库内已有数据

文件：

- `get_sd_data/data/wiki_results_1_with_Qwen3-30B-A3B-Base/experiment_summary_20260122_234314.jsonl`

统计：

- 文件大小约 7.9GB。
- 共 292 条样本。
- 平均 `match_rate` 约 75.62%。
- 平均输入长度约 545 token。
- 参数字段显示 `topm=2`、`p=0.9`、`replace_count=1`。

### 外部 SD 数据结果

外部目录 `/data2/group_谈海生/lagin/data/Sd_Data/data/` 下还有多组 Wiki/MTBench 结果，每组 summary JSONL 通常约 7-8GB，并已生成经验接受率汇总。


### 不同 Replace Count 的整体趋势

`wiki_mtbench_results_analysis.md` 对 `step_empirical_rates_*_avg_summary.json` 和 `expert_coverage_*.json` 做了进一步汇总。这里的 Replace Count 表示每步解码中替换专家数量，数值越大，草稿模型越激进、偏离 target 越多。

Wiki 数据集：

| Replace Count | Step 1 Mean Alpha | Step 10 Mean Alpha | 结论 |
| ---: | ---: | ---: | --- |
| 1 | 0.979 | 0.625 | 最稳健，但长步数仍明显衰减 |
| 2 | 0.960 | 0.510 | 第 10 步已接近低收益区域 |
| 3 | 0.936 | 0.400 | 后期接受率不足 |
| 4 | 0.894 | 0.254 | 基本失去加速价值 |

MTBench 数据集：

| Replace Count | Step 1 Mean Alpha | Step 10 Mean Alpha | 结论 |
| ---: | ---: | ---: | --- |
| 1 | 0.980 | 0.767 | 标准策略中最稳健 |
| 2 | 0.972 | 0.670 | 可用，但后期损失明显 |
| 3 | 0.965 | 0.565 | 第 10 步收益有限 |
| 4 | 0.949 | 0.481 | 后期加速比不足 |
| Test Run | 0.990 | 0.893 | 异常优越，值得单独复现参数 |

结论：

- Wiki 更难，alpha 从第 1 步到第 10 步衰减更快；MTBench 的专家替换容忍度更高。
- Replace Count = 1 是当前最稳健的通用策略。
- Replace Count = 3/4 只适合 very short draft；在第 10 步附近，接受率损失通常超过潜在计算收益。
- `mtbench_results_replace_last_one_with_topp_test` 在 `wiki_mtbench_results_analysis.md` 中表现异常好，Step 10 alpha 仍有 0.893，后续应优先复现实验参数。

### 专家覆盖率结果

`wiki_mtbench_results_analysis.md` 记录了 Replace Count = 4 时的专家覆盖率：

- Wiki：Layer 0-2 覆盖率分别约 100%、95.0%、94.3%，中间层稳定在约 91%-94%。说明 Wiki 激活的专家组合更分散，激进替换下更容易偏离 target。
- MTBench：Layer 0-2 覆盖率约 100%、95.3%、95.7%，Layer 3-13 基本稳定在 93% 左右。说明 MTBench 的专家选择更集中、更稳定。
- 即便 Replace Count = 4，Top-P 替换后的专家集合与 baseline 仍有约 92%-95% 的重合，但这种小比例偏移仍会在多步 decode 中积累成明显 alpha 衰减。

## 4. SRDP 接受率预测器实验

### 实验目的

用采集到的 logits、路由变化、embedding 等特征训练一个轻量 MLP，预测 draft token 被 target 接受的概率或 soft label。

### 数据处理

`srdp/srdp_data_processor.py`：

- 训练集：外部 `wiki_results_1/2/3/4_with_Qwen3-30B-A3B-Base`。
- 测试集：外部 `mtbench_results_1/2/3/4_with_Qwen3-30B-A3B-Base`。
- 输出：`/data2/group_谈海生/lagin/data/Sd_Data/data/srdp_processed_filtered.pt`。
- 特征包括 score loss、replace rate、max layer loss、累积 score loss、top1 probability、entropy、margin、step index、历史最小 top1、平均 entropy、hidden norm、上一轮预测等。

`srdp/data.md` 中记录的严格 token match 统计：

| Split | 正样本数 | 正样本均值 | 负样本数 | 负样本均值 | 全样本均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 6614 | 0.9722 | 776 | 0.6979 | 0.9434 |
| Test | 8806 | 0.9801 | 539 | 0.7359 | 0.9661 |

这说明 token mismatch 不一定对应完全不可接受，负样本的 soft label 均值仍较高，因此使用 soft label 有意义。

### 方案设计

`draft_stopping_acceptance_prediction.md` 和 `srdp_predictor_design.md` 将该方向命名为 SRDP，即 Self-Referential Degradation Perception / 自指型降级感知。核心思想是：不依赖 target 参与，而用 draft 自身“原本想选的专家”和“实际被迫选的专家”的差异来预测退化程度。

主要设计点：

- 截断式清洗：找到 intervention 和 baseline 第一个 token 分歧点；分歧前标为可接受，分歧点标为拒绝，分歧后的样本丢弃。
- 单步预测 + 全局控制：MLP 预测当前 step 的接受概率，controller 维护累计接受概率，并根据局部阈值和全局阈值决定是否停止 draft。
- 低延迟目标：特征提取和 MLP 推理应尽量在 GPU 上完成，避免频繁 device-to-host 拷贝。

特征工程主要覆盖三类信号：

| 特征组 | 代表特征 | 含义 |
| --- | --- | --- |
| 路由降级 | `curr_score_loss`、`replace_rate`、`max_layer_loss`、`accum_score_loss` | 衡量专家替换造成的内部损伤 |
| 输出不确定性 | `top1_prob`、`entropy`、`margin` | 衡量 draft 对下一个 token 的置信度 |
| 历史状态 | `step_idx`、`min_top1_prob`、`avg_entropy`、`hidden_norm`、`prev_mlp_pred` | 捕捉误差累积和隐藏状态漂移 |

`srdp_predictor_design.md` 中建议的 MLP 是轻量网络：Linear -> ReLU -> LayerNorm -> Linear -> ReLU -> Linear -> Sigmoid。当前代码中的 `srdp_trainer.py` 采用了同类结构。

### 训练结果

代表性结果来自 `/data2/group_谈海生/lagin/models/SRDP_Experiments/run_soft_20260128_231323/final_test_report.txt`：

| Metric | Value |
| --- | ---: |
| MSE | 0.019903 |
| MAE | 0.124241 |
| AUC | 0.8973 |
| Accuracy | 82.61% |
| Precision | 99.86% |
| Recall | 82.60% |
| F1 | 0.9041 |

结论：该软标签 MSE 版本是当前记录中明确标注的较好结果，AUC 接近 0.90，precision 很高，适合作为 `heterogeneous_spec_dec/run_inference.py` 中的 SRDP predictor 权重。

### 硬标签实验结果

`srdp_experiment_results.md` 记录了 6 轮硬标签实验，测试集为 MTBench。该路径的目标是直接判断 token 是否可接受。

| 实验 ID | 策略 | Accuracy | AUC | Precision | Recall | F1 | 评价 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `run_235815` | Baseline | 94.24% | 0.8995 | 94.25% | 99.99% | 0.9704 | 高召回基准 |
| `run_001026` | 负样本加权 W=20 | 74.53% | 0.9095 | 99.64% | 73.23% | 0.8442 | 过于保守，误伤多 |
| `run_003220` | Focal Loss, Acc 优先 | 94.23% | 0.9124 | 94.23% | 100.00% | 0.9703 | 接近 baseline |
| `run_003917` | Specificity 优先 | 44.49% | 0.9226 | 100.00% | 41.10% | 0.5825 | 阈值过严，不可用 |
| `run_004444` | Focal Loss, F1 优先 | 94.31% | 0.9105 | 94.46% | 99.81% | 0.9706 | 硬标签综合最佳 |
| `run_004717` | 阈值扫描 0.65 | 87.34% | 0.9171 | 98.25% | 88.13% | 0.9292 | 安全侧重方案 |

结论：硬标签路径上，`run_004444` 的 F1 和 Accuracy 最均衡；如果更在意“不放行坏 token”，`run_004717` 的 0.65 阈值用更高 Precision 换取较低 Recall。

### 软标签实验结果

`srdp_experiment_results.md` 还记录了 5 轮软标签实验。软标签目标不是简单 0/1，而是拟合连续接受率/置信度。

| 实验 ID | 策略 | Accuracy | AUC | Precision | Recall | F1 | MSE | 评价 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `run_230452` | AUC 优化 baseline | 80.19% | 0.8935 | 99.85% | 80.17% | 0.8893 | MAE 0.1255 | 高精低召 |
| `run_231323` | MSE 优化 | 82.61% | 0.8973 | 99.86% | 82.60% | 0.9041 | 0.0199 | 软标签综合最佳 |
| `run_095031` | 预测值缩放 x10 | 85.33% | 0.8821 | 99.76% | 85.43% | 0.9204 | 0.0739 | 回归误差变大 |
| `run_104549` | 负样本加权 W=10 | 59.46% | 0.6136 | 99.96% | 59.46% | 0.7456 | 0.0090 | AUC 崩塌，过拟合 |
| `run_155848` | 标准 MSE, 负样本置 0 | 83.62% | 0.8825 | 97.93% | 84.40% | 0.9066 | 0.0453 | 高召回平衡方案 |

结论：

- 软标签首选 `run_231323`，它在 AUC、Precision、F1、MSE 之间最均衡。
- 负样本强加权会让 MSE 看似更低，但排序能力明显下降，不适合作为最终方案。
- 硬标签模型在二分类边界上更强，软标签模型更适合提供连续置信度给 draft controller。

### Draft 停止策略

`draft_stopping_acceptance_prediction.md` 提出 draft 停止不应只看固定长度，而应由累计可靠度或累计接受概率控制。可用信号包括：

- 专家路由分差：原 top expert 与替代 expert 的 score gap。
- 累积替换率：当前 draft 序列中发生专家替换的比例和累积损失。
- logits 熵和 top-k 置信度：识别 draft 生成不确定性。
- hidden state norm：捕捉数值漂移。
- token 类型触发：数学、代码、数字等关键 token 可以更早验证。

建议初始参数来自 `draft_stopping_acceptance_prediction.md`：停止阈值 `tau=0.7-0.8`，专家损失权重 `alpha=0.5`，平滑系数 `gamma=0.2`，最大 draft 长度 `Max K=8-16`。后续应结合真实硬件开销和任务类型调参。

## 5. 异构推测解码物理常数实验

### 实验目的

比较 Qwen3 细粒度 MoE 中专家计算和 CPU/GPU 数据搬运的物理成本，为异构调度策略提供依据。

### 方法

`heterogeneous_spec_dec/test_pcie_compute.py` 构造 Qwen3 形状的 dummy expert：

- `HIDDEN_SIZE=2048`
- `INTERMEDIATE_SIZE=768`
- `SEQ_LEN=5`
- 单专家 FP16 权重约 9MB
- 单次激活值约 20KB

`0317.txt` 还定义了低内存异构调度实验的目标形态：

- 模型：`Qwen3-30B-A3B-Base`。
- 每层一半专家常驻 GPU，一半专家存放 CPU，用于模拟低显存场景。
- Target 模型严格使用原始 router top-k，保证精度。
- Draft 模型使用 n=2 延迟保护策略：CPU 最多计算 2 个专家，剩余名额由 GPU 上 ranking 更靠后的专家补齐。

n=2 调度算法：

1. Router 选出 top-8 专家。
2. 若 top-8 中 CPU 专家数不超过 2，则按原 top-8 执行。
3. 若 CPU 专家数超过 2，只保留概率最高的 2 个 CPU 专家。
4. 被丢弃的 CPU 名额由 rank 9、10、11... 中概率最高且位于 GPU 的专家补齐，直到仍有 8 个专家。
5. 对最终 8 个专家重新 softmax 归一化。
6. GPU 执行 GPU expert，CPU 异步接收 hidden state 并执行最多 2 个 CPU expert，最后回传 GPU 做加权合并。

该设计明确区分 target 和 draft：target 为保精度可以等待 CPU；draft 为保延迟会截断 CPU 专家并用 GPU 专家替补。

### 结果

结果来自 `heterogeneous_spec_dec/logs/test_pice-10851.out`：

| n experts | GPU compute | CPU compute | PCIe weight swap | Activation swap |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.116 ms | 1.240 ms | 0.499 ms | 0.028 ms |
| 2 | 0.208 ms | 2.574 ms | 0.982 ms | 0.028 ms |
| 4 | 0.395 ms | 4.904 ms | 1.962 ms | 0.028 ms |
| 8 | 0.787 ms | 9.761 ms | 3.920 ms | 0.027 ms |

结论：

- GPU expert 计算最快。
- CPU expert 计算随专家数近似线性增长，显著慢于 GPU。
- PCIe 搬运权重比 CPU 计算快，但仍远慢于搬运激活值。
- 对 target verification 阶段，更值得优先考虑减少 CPU expert 计算，或通过权重/激活搬运策略控制异构代价。

### 端到端异构推测解码状态

`heterogeneous_spec_dec/run_inference.py` 已写出目标模型自回归和异构 draft 推测解码的对比框架，但当前没有发现完整成功日志。该文件还存在需要修复的实现问题：

- 使用 `np.array`、`np.sum`，但没有 `import numpy as np`。
- `heterogeneous_spec_dec/srdp_predictor.py` 和 `heterogeneous_spec_dec/feature_extractor.py` 为空。
- 当前 SRDP predictor 类和特征提取器内嵌在 `run_inference.py` 中，后续应拆分或补齐空文件。

## 总体结论

1. Cache-Prior 能明显降低 miss rate，但当前 MMLU 配置带来精度下降，需要继续搜索 lambda、cache ratio、decode-only 策略和真实 offload 约束。
2. 专家敏感性实验结果很清楚：rank-1 专家最关键，替换首要专家会让 PPL 急剧恶化；低 rank 专家影响较小。
3. 推测解码数据采集已形成可训练数据，Wiki/MTBench 外部结果规模完整，前几步接受率高，支持继续做动态 draft length。
4. SRDP 软标签预测器已有可用结果，AUC 0.8973、F1 0.9041，可以作为异构推测解码的停止判据。
5. 异构物理常数测试显示 CPU 计算 expert 成本高，PCIe 权重搬运成本低于 CPU 计算但高于激活搬运，后续调度策略应围绕这三个成本做权衡。
