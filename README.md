# MOE_SD 实验说明

本仓库围绕 MoE 模型的专家路由、专家缓存、专家剪枝/重采样、推测解码数据采集和 SRDP 接受率预测器做实验。代码默认依赖本机已有模型和数据，路径大多硬编码在脚本中：

- 模型目录：`/data2/group_谈海生/lagin/models/`
- 数据目录：`/data2/group_谈海生/lagin/data/`
- SD 数据外部汇总：`/data2/group_谈海生/lagin/data/Sd_Data/data/`
- SRDP 训练输出：`/data2/group_谈海生/lagin/models/SRDP_Experiments/`

运行前请先确认这些路径在当前机器上存在，且 GPU 显存足够。

## 环境

项目使用 Python 3.11，依赖写在 `pyproject.toml` 中，主要包括：

- `torch==2.7`
- `transformers>=4.57.3`
- `datasets>=4.4.1`
- `accelerate`
- `matplotlib`
- `scikit-learn`

已有虚拟环境时：

```bash
source .venv/bin/activate
```

需要重新创建环境时：

```bash
uv sync
source .venv/bin/activate
```

SLURM 脚本默认加载 `cuda/12.9.1` 和 `Anaconda3/2025.06`，并使用 A800 分区、`gpu5` 节点。换机器时需要修改对应 `#SBATCH` 配置。

## 目录和文件

### 根目录

- `README.md`：当前文件，解释仓库结构和运行方式。
- `EXPERIMENT_SUMMARY.md`：实验总结和已有结果汇总。
- `draft_stopping_acceptance_prediction.md`：draft 停止策略、接受率预测特征和启发式方案记录。
- `srdp_predictor_design.md`：SRDP 预测器设计文档，包含特征、截断式清洗、MLP 架构和在线推理逻辑。
- `srdp_experiment_results.md`：SRDP 硬标签与软标签训练结果对比。
- `wiki_mtbench_results_analysis.md`：Wiki 与 MTBench 不同专家替换数量下的接受率、理论加速比和覆盖率分析。
- `pyproject.toml` / `uv.lock`：Python 依赖和锁定文件。
- `main.py`：占位入口，只打印 `Hello from moe-sd!`。
- `test.py`：扫描 JSONL 中是否存在 `NaN` 的辅助脚本，目标文件路径需手动修改。
- `logs/`：根目录 SLURM 日志，本地存在但被 `.gitignore` 忽略。

### `Cache_Prior_Moe/`

专家缓存和 Cache-Prior 路由实验。

- `Moe_LRU.py`：`ExpertCache`，使用 `OrderedDict` 模拟 LRU 专家缓存，统计 hit/miss。
- `Cache_Prior.py`：早期逐 token Cache-Prior wrapper。
- `Cache_Prior_batch.py`：当前工具函数默认使用的 batch/padding 版本 wrapper，支持只在 decode 阶段启用 cache 逻辑。
- `moe_utils.py`：加载 WikiText/MMLU/GSM8K、本地 parquet 数据，注入 Cache-Prior wrapper，计算 PPL、accuracy、miss rate。
- `run_wikitext_eval.py`：WikiText-2 PPL 和 cache miss rate 实验。
- `run_mmlu_eval.py`：MMLU 5-shot accuracy 和 miss rate 实验。
- `run_gsm8k_eval.py`：GSM8K 8-shot 生成式 accuracy 和 miss rate 实验。
- `run_moe.sh`：SLURM 启动脚本，当前指向 `run_gsm8k_eval.py`。
- `Cache_Prior_out/`：已有结果图和部分文本结果。

### `Expert_Pruning_Resampling/`

专家剪枝和专家重采样敏感性分析。

- `ExpertSubsetInference.py`：核心 wrapper，支持 `remove`、`replace`、`replace_remaining`、`replace_with_topp` 等策略。
- `run_expert_subset_eval.py`：基础 top-m 专家子集 PPL 评测。
- `run_expert_subset_analysis.py`：多模型 remove/replace 实验，支持断点保存 JSON。
- `plot_expert_analysis.py`：读取 JSON 结果并绘制组合图。
- `expert_remove_results.json`：专家删除/只保留 top-m 的结果。
- `expert_replace_results.json`：随机替换 rank-k 专家的结果。
- `combined_expert_sensitivity_analysis.png`：专家剪枝和重采样组合图。
- `README_EXPERT_SUBSET.md`：该子模块的原始说明。
- `run_top_m.sh`：SLURM 脚本。注意：当前脚本中的 `PYTHON_SCRIPT="./top_m/run_expert_subset_analysis.py"` 与实际目录不一致，应改为 `./Expert_Pruning_Resampling/run_expert_subset_analysis.py` 后再提交作业。

### `get_sd_data/`

推测解码/SRDP 数据采集和离线分析。

- `ExpertSubsetInference.py`：面向 Qwen/DeepSeek 的专家替换 wrapper，记录原始路由、修改后路由、logits、embedding。
- `wiki_experiment.py`：DeepSeek-V2-Lite 上的 WikiText 干预/基准采集。
- `mtbench101_experiment.py`：Qwen3-30B-A3B-Base 上的 Wiki 或 MTBench 采集，当前配置为 `DATASET_NAME="wiki"`、`replace_count=2`。
- `data_plot.py`：计算路由保真度并绘图。
- `exactly_data.py`：计算经验接受率、理论加速比、专家覆盖率。
- `wiki_experiment.sh`：SLURM 脚本，当前实际运行 `mtbench101_experiment.py`。
- `data/`：仓库内保留了一份大型 Wiki summary JSONL；多数大型数据在外部 `/data2/.../Sd_Data/data/`。
- `logs/`：数据采集日志。

### `srdp/`

SRDP 接受率预测器的数据处理、训练和统计。

- `srdp_data_processor.py`：从 Wiki 结果构造训练集、从 MTBench 结果构造测试集；输出 `srdp_processed_filtered.pt`。
- `srdp_trainer.py`：训练 MLP 回归器拟合 filtered soft label，保存 `best_model.pth`、训练日志和测试报告。
- `test.py`：按严格 token match/mismatch 统计 soft label 分布。
- `data.md`：已有训练集/测试集 soft label 统计结果。

### `heterogeneous_spec_dec/`

异构推测解码和 CPU/GPU 物理常数实验。

- `test_pcie_compute.py`：模拟 Qwen3 细粒度 MoE expert，测 GPU 计算、CPU 计算、PCIe 权重搬运、激活值搬运耗时。
- `test_pcie_compute.sh`：对应 SLURM 脚本。
- `run_inference.py`：目标模型 vs 异构 draft 模型的推测解码端到端评测草稿。
- `srdp_predictor.py`、`feature_extractor.py`：当前为空文件。

`run_inference.py` 当前还不是稳定入口：代码中使用了 `np`，但文件未导入 `numpy as np`；同时该目录中的 predictor/feature extractor 文件为空，实际逻辑写在 `run_inference.py` 内部。

## 常用运行方式

所有命令默认从仓库根目录 `/home/lagin/MOE_SD` 执行。

### 1. Cache-Prior MoE

直接运行单个评测：

```bash
python -u Cache_Prior_Moe/run_wikitext_eval.py
python -u Cache_Prior_Moe/run_mmlu_eval.py
python -u Cache_Prior_Moe/run_gsm8k_eval.py
```

提交 SLURM：

```bash
sbatch Cache_Prior_Moe/run_moe.sh
```

说明：这些 Python 脚本中的输出文件名是相对路径，写入位置取决于启动时的工作目录。若希望输出留在 `Cache_Prior_Moe/` 内，可先 `cd Cache_Prior_Moe` 后运行。

### 2. 专家剪枝/重采样

推荐直接运行：

```bash
python -u Expert_Pruning_Resampling/run_expert_subset_analysis.py
python -u Expert_Pruning_Resampling/plot_expert_analysis.py
```

如果使用 SLURM，请先修正 `Expert_Pruning_Resampling/run_top_m.sh` 中的脚本路径。

### 3. 推测解码数据采集

提交作业：

```bash
sbatch get_sd_data/wiki_experiment.sh
```

或直接运行：

```bash
python -u get_sd_data/mtbench101_experiment.py
python -u get_sd_data/wiki_experiment.py
```

采集结果会非常大，单个 summary JSONL 通常约 7-8GB。`data/` 和 `logs/` 已被 `.gitignore` 忽略，不建议提交这些文件。

### 4. 经验接受率、覆盖率和 SRDP 训练

```bash
python -u get_sd_data/exactly_data.py
python -u srdp/srdp_data_processor.py
python -u srdp/srdp_trainer.py
python -u srdp/test.py
```

`srdp_data_processor.py` 和 `exactly_data.py` 读取的是外部 `/data2/.../Sd_Data/data/` 路径。若只想分析仓库内数据，需要改脚本中的 `DIR`。

### 5. 异构推测解码物理常数测试

```bash
sbatch heterogeneous_spec_dec/test_pcie_compute.sh
```

直接运行：

```bash
python -u heterogeneous_spec_dec/test_pcie_compute.py
```

## 注意事项

- 多数脚本硬编码本地模型和数据路径，换环境时优先检查路径。
- 大模型实验需要 A800 级别显存；日志中出现过 CUDA OOM、磁盘配额不足、路径写错和旧接口不匹配错误。
- `.gitignore` 忽略 `logs/`、`data/`、`.venv/` 和缓存文件。当前工作区仍能看到这些本地结果，但它们一般不会被提交。
- `Cache_Prior_Moe/Cache_Prior_out/wikitext_eval_results.txt` 当前为空，WikiText 结果主要需要参考日志和图片。
- `get_sd_data/data/wiki_results_1_with_Qwen3-30B-A3B-Base/experiment_summary_20260122_234314.jsonl` 是 7.9GB 大文件，不要用 `cat`、`head` 直接展示完整内容。
