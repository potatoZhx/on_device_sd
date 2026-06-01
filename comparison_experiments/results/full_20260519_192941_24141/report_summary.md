# MoE 对比实验汇报摘要

## 实验配置

- 模型：`Qwen3-30B-A3B-Base`
- 数据集：`mtbench101`
- 样本数：50
- 每条样本最大生成长度：100 tokens
- 草稿长度：10
- 专家缓存率：0.25、0.5、0.75、1.0
- 实验一：Cache-Prior，`lambda_val=0.5`，`top_j=2`
- 实验二：MoE-SpeQ-style INT4 expert draft，量化专家权重加载到 GPU，INT4 group size 128

## 核心指标表

| 方法 | cache_rate | Prefix 接受率 | 逐位置匹配率 | 平均每轮接受 token | Cache 命中 / 覆盖 |
|---|---:|---:|---:|---:|---:|
| Cache-Prior | 0.25 | 79.81% | 95.50% | 7.73 / 10 | cache_hit 94.61% |
| Cache-Prior | 0.50 | 87.02% | 97.26% | 8.47 / 10 | cache_hit 97.19% |
| Cache-Prior | 0.75 | 89.28% | 98.18% | 8.67 / 10 | cache_hit 98.56% |
| Cache-Prior | 1.00 | 94.42% | 99.03% | 9.28 / 10 | cache_hit 100.00% |
| MoE-SpeQ INT4 | 0.25 | 84.08% | 96.75% | 8.15 / 10 | target_cache_hit 78.28% |
| MoE-SpeQ INT4 | 0.50 | 84.08% | 96.75% | 8.15 / 10 | target_cache_hit 99.15% |
| MoE-SpeQ INT4 | 0.75 | 84.08% | 96.75% | 8.15 / 10 | target_cache_hit 99.66% |
| MoE-SpeQ INT4 | 1.00 | 84.08% | 96.75% | 8.15 / 10 | target_cache_hit 100.00% |

## 指标解释

- `prefix_acceptance_rate`：按 speculative decoding 前缀规则统计的真实接受率。第 i 个草稿 token 只有在第 1 到第 i 个 token 都匹配目标模型时才算被接受。
- `position_match_rate`：逐位置匹配率。只看草稿 token 和目标 token 在同一位置是否相同，不要求前面 token 连续匹配。
- `full_round_acceptance_rate`：整轮草稿 10 个 token 全部被接受的比例。
- `avg_prefix_accepted_per_round`：每轮平均接受多少个草稿 token。草稿长度固定为 10，因此该值越接近 10 越好。
- `cache_hit_rate`：实验一中，方法 M 动态专家缓存的命中率。
- `draft_cache_hit_rate`：实验二中，INT4 draft 阶段预测专家访问动态缓存时的命中率。
- `target_cache_hit_rate`：实验二中，target verify 实际需要的专家是否已被 draft lookahead/cache 覆盖。该指标更接近 MoE-SpeQ 的专家预取效果。
- `expert_hard_match_rate`：实验二中，draft 与 target 的专家选择完全一致且顺序一致的比例。
- `expert_soft_match_rate`：实验二中，draft 与 target 选择到相同专家集合但允许顺序不同的比例。
- `draft_pos_i_prefix_acceptance_rate`：第 i 个草稿 token 按前缀规则真正被接受的比例。该指标可用于观察草稿越往后接受率是否下降。

## 初步结论

1. Cache-Prior 的 token 接受率随 cache_rate 提升明显增加。`cache_rate=0.25` 时 prefix 接受率为 79.81%，提升到 `cache_rate=1.0` 后达到 94.42%。

2. Cache-Prior 在高缓存率下表现最好。`cache_rate=1.0` 时平均每轮接受 9.28 个 token，整轮 10 个 token 全接受比例为 91.76%。

3. MoE-SpeQ INT4 的 token 接受率不随 cache_rate 变化，这是预期现象。cache_rate 只影响专家缓存覆盖率，不改变 INT4 draft 生成的 token，因此四个 cache_rate 下 prefix 接受率都为 84.08%。

4. MoE-SpeQ INT4 的专家缓存覆盖效果很强。`cache_rate=0.5` 时 target_cache_hit 已达到 99.15%，说明只缓存 50% 专家时，draft lookahead 已能覆盖绝大多数 target verify 需要的专家。

5. MoE-SpeQ INT4 在低缓存率下仍有一定预取价值。`cache_rate=0.25` 时 target_cache_hit 为 78.28%，说明 25% 专家缓存已经覆盖了大部分实际需要的 target 专家。

6. 从 token 接受率看，Cache-Prior 在 `cache_rate>=0.5` 时优于当前 INT4 draft；从专家预取覆盖看，MoE-SpeQ INT4 在 `cache_rate>=0.5` 时已经接近满覆盖。

7. 实验二的专家集合匹配率为 51.32%，低于论文中 Qwen1.5-MoE 报告的 90.9%。可能原因包括：本实验使用 Qwen3-30B-A3B-Base，数据集为 mtbench101，当前量化为对称 groupwise INT4 权重量化而非完整 GPTQ + fused INT4 kernel 系统。
