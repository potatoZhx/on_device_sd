# 对比实验说明

本文档说明 `comparison_experiments` 中两组对比实验的目标、算法逻辑、参数含义、输出指标和运行方式。实验均以 `实验.md` 为需求来源：在 `Qwen3-30B-A3B-Base` 与 `mtbench101` 上，统计不同专家缓存率、不同草稿长度下的 token 接受率。

## 共同设置

两组实验都使用 speculative decoding 的验证口径：

1. 给定 prompt，当前上下文为 `context`。
2. draft 阶段生成长度为 `draft_len` 的草稿 token：`c_1 ... c_n`。
3. verify 阶段用目标模型对 `context + c_1 ... c_n` 做一次验证，得到每个位置的目标 token：`d_1 ... d_n`。
4. 比较 `c_i` 和 `d_i`，统计接受率。

脚本会输出两类接受率：

- `prefix_acceptance_rate`：speculative decoding 的前缀接受率。只有连续前缀匹配才算接受，例如 `c_1=d_1`、`c_2=d_2`、`c_3!=d_3`，则本轮接受 2 个 token。
- `position_match_rate`：逐位置精确匹配率。每个位置单独比较 `c_i == d_i`，不要求前缀连续。

主指标建议使用 `prefix_acceptance_rate`，因为它更接近真实 speculative decoding 的回滚逻辑。

正式实验将 `draft_len` 固定为 10，并额外输出第 1 到第 10 个草稿 token 的位置级统计。`summary.jsonl` 中包含数组字段：

- `per_position_totals`：每个草稿位置被统计的轮次数。
- `per_position_position_matches`：每个草稿位置逐位置匹配的次数。
- `per_position_position_match_rates`：每个草稿位置的逐位置匹配率。
- `per_position_prefix_accepts`：每个草稿位置按前缀规则真正被接受的次数。
- `per_position_prefix_acceptance_rates`：每个草稿位置按前缀规则真正被接受的比例。

`summary.csv` 会展开为 `draft_pos_1_*` 到 `draft_pos_10_*` 列，便于直接画图。例如 `draft_pos_2_prefix_acceptance_rate` 表示第二个草稿 token 在前一个草稿 token 也被接受的前提下，按 speculative 前缀规则实际保留下来的比例；`draft_pos_2_position_match_rate` 则只看第二个 token 本身是否等于目标 token。

## 实验一：方法 M / Cache-Prior

脚本：

- `cache_prior_acceptance.py`
- `run_cache_prior_acceptance.sh`

对应 `实验.md` 中的 “Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference 方法（方法 M）”。

### 算法逻辑

实验一在同一个 Qwen3 模型中切换两种路由模式：

- draft 模式：启用 cache-prior 路由。
- target 模式：关闭 cache-prior，恢复原始全精度路由。

每个 MoE 层维护一个专家缓存，缓存大小由 `cache_rate` 决定。例如 Qwen3 某层有 128 个专家，`cache_rate=0.5` 时该层缓存 64 个专家。draft 阶段对当前 decode token 的 router logits 加 bias：

```text
logits' = logits + lambda_val * avg_range * mask
```

其中：

- `avg_range` 是当前层 router logits 最大值和最小值差值的滑动平均。
- `mask` 标记两类专家：当前缓存内专家，以及当前 token 原始 logits 的 top-j 专家。
- `lambda_val` 控制缓存专家优先级强度。

draft 模式下，模型基于 `logits'` 选择专家并生成草稿 token；target 模式下，模型用原始 `logits` 选择专家，生成验证 token。

### 关键参数

- `--cache-rates`：专家缓存率列表。逗号分隔，例如 `0.25,0.5,0.75,1.0`。
- `--draft-lengths`：草稿长度列表。逗号分隔，例如 `1,2,4,8`。
- `--lambda-val`：方法 M 的 cache-prior bias 强度。`0` 表示不加缓存优先 bias；值越大，路由越偏向缓存专家。当前默认 `0.5` 是沿用 `Cache_Prior_Moe` 中的默认值。
- `--top-j`：即使专家不在缓存中，也会加入 bias mask 的当前 token top-j 专家数量。
- `--initial-cache-policy`：每层初始缓存专家选择策略，支持 `head`、`tail`、`even`、`random`。
- `--max-samples`：最多测试多少条 mtbench101 样本。
- `--max-new-tokens`：每条样本最多通过 speculative 流程生成多少 token。
- `--max-prompt-tokens`：prompt 最大 token 数，过长时截取尾部。
- `--dtype`：加载模型的数据类型，支持 `bf16`、`fp16`、`fp32`。

### 输出指标

- `prefix_acceptance_rate`：前缀 token 接受率。
- `position_match_rate`：逐位置 token 匹配率。
- `full_round_acceptance_rate`：整段 draft 全部被接受的轮次比例。
- `avg_prefix_accepted_per_round`：平均每轮 verify 接受多少前缀 token。
- `cache_hit_rate`：draft 阶段 cache-prior 模拟缓存命中率。
- `elapsed_sec`：该组参数的运行耗时。

## 实验二：MoE-SpeQ 风格 INT4 专家量化 draft

脚本：

- `quantize_qwen3_moe_experts.py`
- `run_quantize_qwen3_experts.sh`
- `quantized_activation_acceptance.py`
- `run_quantized_activation_acceptance.sh`

对应 `实验.md` 中 “用量化模型的激活来预测，参考论文 moe-speq”。

### 算法逻辑

MoE-SpeQ 的核心观察是：量化 MoE draft 模型能够高保真预测全精度模型的专家激活，从而提前得到未来 token 的专家 lookahead，用于 prefetch/offload 调度。

实验二不再使用运行时伪量化。需要先生成一个量化 draft 权重目录，然后实验脚本加载该目录进行 draft：

1. 量化阶段：按论文比例把 routed expert 内的 `gate_proj`、`up_proj`、`down_proj` 权重做对称 INT4 groupwise 量化，`group_size=128`。
2. 保持全精度的部分：router/gate、attention、normalization、embedding、lm_head、shared expert、其他非 expert 参数。
3. draft 阶段：MoE 层使用预生成的 INT4 routed expert 权重；router/gate 与 shared expert 保持全精度。
4. verify 阶段：切回原始全精度 expert 权重，比较目标输出 token 与草稿 token。

当前实现为算法验证，INT4 权重会在专家计算前反量化后调用 PyTorch `linear`；因此它验证的是“量化权重对接受率和专家预测的影响”，不是低比特 kernel 的吞吐性能。

因此当前实验二的 wall-clock 时间会比论文系统设计更慢。论文中的量化 draft 模型是为了降低显存和计算压力，并配合专家预取/卸载调度；当前脚本没有 fused INT4/GPTQ kernel，也没有异步 prefetch/offload pipeline，而是在 Python/PyTorch 中逐次把 packed INT4 权重反量化后再做 `linear`。把 `--quantized-weight-device` 设为 `cuda` 可以去掉 CPU 到 GPU 的权重搬运，但仍然不是论文中的高性能量化推理内核。

draft 阶段每生成一个 token，脚本记录该 token 在每个 MoE 层预测出的 top-k 专家，形成 expert lookahead。verify 阶段对相同位置记录目标模型实际 top-k 专家，然后统计：

- token 是否被接受。
- draft 预测专家和 target 实际专家是否 hard match。
- draft 预测专家和 target 实际专家是否 soft match。
- 按 draft lookahead 管理的专家缓存是否覆盖 target 需要的专家。

### 量化模型生成

先提交量化作业：

```bash
sbatch comparison_experiments/run_quantize_qwen3_experts.sh
```

默认输出目录：

```text
/data2/group_谈海生/lagin/models/Qwen3-30B-A3B-Base-experts-int4-g128/
```

目录中包含：

- `quantization_config.json`：记录源模型、量化方法、bit 数、group size、每层 shard。
- `layers/layer_XXX.pt`：每个 MoE 层的 packed INT4 expert 权重和 scale。

论文中的 draft 比例是 routed expert 线性层 INT4、`group_size=128`，其他模块保留 FP16；脚本按这个比例生成权重。若量化目录不存在，实验二 Slurm 脚本会直接退出并提示先提交量化作业。

### 关键参数

- `--cache-rates`：专家缓存率列表。
- `--draft-lengths`：草稿长度列表。
- `--quantized-model-dir`：预生成 INT4 expert draft 权重目录。
- `--quantized-weight-device`：packed INT4 权重存放位置，`cpu` 更省显存但更慢，`cuda` 小样本更快但可能额外占显存。
- `--initial-cache-policy`：初始缓存策略，支持 `head`、`tail`、`even`、`random`。
- `--max-samples`：最多测试样本数。
- `--max-new-tokens`：每条样本最多生成 token 数。
- `--max-prompt-tokens`：prompt 最大 token 数。
- `--dtype`：加载模型的数据类型。

### 输出指标

- `prefix_acceptance_rate`：前缀 token 接受率。
- `position_match_rate`：逐位置 token 匹配率。
- `target_cache_hit_rate`：target verify 阶段实际需要的专家，在 draft lookahead 管理的缓存中命中的比例。这个指标更接近 MoE-SpeQ prefetch 目标。
- `draft_cache_hit_rate`：draft 阶段自身访问缓存的命中率。
- `expert_hard_match_rate`：draft 预测专家列表与 target 专家列表完全相同且顺序相同的比例。
- `expert_soft_match_rate`：draft 和 target 专家集合相同但允许顺序不同的比例。
- `elapsed_sec`：该组参数运行耗时。

## 小样本烟测

烟测脚本：

- `run_smoke_tests.sh`

烟测只跑极小配置，用于验证代码路径是否可执行：

- `cache_rates=0.5`
- `draft_lengths=1,2`
- `max_samples=1`
- `max_new_tokens=2`
- `max_prompt_tokens=256`

提交方式：

```bash
sbatch comparison_experiments/run_smoke_tests.sh
```

烟测输出目录形如：

```text
comparison_experiments/results/smoke_<timestamp>_<job_id>/
```

其中包含：

- `cache_prior/summary.csv`
- `cache_prior/summary.jsonl`
- `cache_prior/round_details.jsonl`
- `speq_int4/summary.csv`
- `speq_int4/summary.jsonl`
- `speq_int4/round_details.jsonl`

## 扩展验证

扩展验证脚本：

- `run_validation_tests.sh`

它比烟测覆盖更多组合，但仍小于正式实验：

- `cache_rates=0.25,0.5,1.0`
- `draft_lengths=1,2,4`
- `max_samples=3`
- `max_new_tokens=8`
- `max_prompt_tokens=512`

提交方式：

```bash
sbatch comparison_experiments/run_validation_tests.sh
```

输出目录形如：

```text
comparison_experiments/results/validation_<timestamp>_<job_id>/
```

## 正式运行

登录节点不要直接执行 Python。通过 Slurm 提交：

```bash
sbatch comparison_experiments/run_quantize_qwen3_experts.sh
sbatch comparison_experiments/run_cache_prior_acceptance.sh
sbatch comparison_experiments/run_quantized_activation_acceptance.sh
```

如需调整实验矩阵，修改对应 `.sh` 文件中的参数即可。
