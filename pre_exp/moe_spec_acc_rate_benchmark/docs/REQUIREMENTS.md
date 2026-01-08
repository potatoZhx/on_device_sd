# MoE投机采样测试框架 - 需求文档

## 1. 项目概述

### 1.1 项目目标
实现一个测试框架，用于评估MoE（Mixture of Experts）模型的精度对路由专家选择的敏感度。通过修改路由策略的draft模型与原始verify模型进行投机解码（Speculative Decoding），计算并分析接受率（Acceptance Rate）。

### 1.2 核心思想
- **Draft模型**：使用修改后的路由策略（排除top-2专家）生成候选tokens
- **Verify模型**：使用原始路由策略验证候选tokens并执行投机采样
- **评估指标**：通过接受率来评估路由修改对模型输出质量的影响

---

## 2. Draft模型定义

### 2.1 路由修改策略
对于模型的每个MoE层，在路由时应用以下策略：

1. **排除阶段**：
   - 计算所有专家的路由分数
   - 识别分数最高的前2个专家（top-2）
   - 将这2个专家排除在外

2. **重新路由阶段**：
   - 从剩余的专家中使用模型原本的路由策略（如top-k）重新选择
   - 使用原始的路由权重分布（不修改分数本身）

### 2.2 关键要求
- ✅ 修改的是**专家选择逻辑**，而不是路由分数
- ✅ 保持原始模型的top-k等路由参数不变
- ✅ 对所有MoE层应用相同的修改策略

---

## 3. 执行流程

### 3.1 Prefill阶段
**目标**：生成初始KV cache

**步骤**：
1. 使用原始模型（未修改路由）对输入prompt进行prefill
2. 生成初始KV cache
3. 从prefill的logits中采样第一个token

**重要约束**：
- ❗ Prefill阶段**必须**使用原始路由逻辑
- ❗ 不能使用draft模式

### 3.2 投机解码主循环
**循环条件**：直到生成n个token（或遇到EOS）

每轮包含以下步骤：

#### 步骤1：Draft生成
- **输入**：
  - `last_token`：上一轮输出的最后一个token
  - `verify_kv_cache`：Verify模型的KV cache
- **过程**：
  - Draft模型生成`draft_length`个tokens（当前设定为2）
  - 使用修改后的路由策略
  - 管理独立的临时KV cache
- **输出**：
  - `draft_tokens`：生成的候选tokens列表
  - `draft_logits`：对应的logits

#### 步骤2：Verify验证
- **输入**：
  - `last_token`：与draft输入相同
  - `draft_tokens`：draft生成的候选
  - `draft_logits`：draft的logits
  - `verify_kv_cache`：Verify模型的KV cache
- **过程**：
  - 使用原始模型验证draft tokens
  - 执行投机采样算法
  - 决定接受/拒绝draft tokens
  - 更新KV cache
- **输出**：
  - `accepted_tokens`：实际接受的所有tokens
  - `n_matches`：被接受的draft token数量
  - `updated_kv_cache`：更新后的KV cache

#### 步骤3：状态更新
- 累加统计量：
  - `all_draft_length += draft_length`
  - `all_accept_length += n_matches`
- 更新生成状态：
  - 将`accepted_tokens`添加到输出序列
  - 更新`last_token`为本轮最后一个token
  - 更新`current_length`

### 3.3 输出结果
**打印内容**：
1. 使用投机采样生成的完整文本
2. 接受率：`all_accept_length / all_draft_length`
3. 其他统计指标：
   - 总生成token数
   - 总步数
   - 每步的接受长度列表

---

## 4. 参数配置

### 4.1 固定参数
| 参数 | 值 | 说明 |
|------|-----|------|
| `draft_length` | 2 | 每次draft生成的token数（benchmark测试值） |
| `max_new_tokens` | 10 | 总生成token数（测试阶段的小值） |
| `top_k_experts_to_remove` | 2 | 排除的top专家数量 |

### 4.2 模型配置
| 参数 | 值 | 说明 |
|------|-----|------|
| 模型路径 | `/zx_data1/models/Qwen--Qwen3-30B-A3B-Base` | Qwen3系列MoE模型 |
| 设备 | CUDA | GPU加速 |
| 数据类型 | float16 | 半精度推理 |

---

## 5. 技术约束

### 5.1 KV Cache管理
1. **Prefill后的KV cache**：
   - 包含输入prompt的信息
   - 不包含第一个生成的token

2. **Draft阶段的KV cache**：
   - 使用verify KV cache的**深拷贝**
   - Draft内部累积临时cache
   - Draft结束后**丢弃**所有临时cache

3. **Verify阶段的KV cache**：
   - 使用原始的verify KV cache
   - 根据接受的tokens更新cache
   - 传递到下一轮

### 5.2 模型状态管理
1. **Draft模型与Verify模型共享底层模型实例**
2. **通过标志位控制路由行为**：
   - Draft调用时：临时启用路由修改
   - Verify调用时：使用原始路由
3. **确保线程安全**：使用try-finally确保状态正确恢复

### 5.3 数值一致性
1. **Greedy采样**：使用argmax确保确定性
2. **投机采样**：使用torch.multinomial（随机性）
3. **精度控制**：使用float16节省显存

---

## 6. 验证要求

### 6.1 正确性验证
1. ✅ **第一个token一致性**：
   - 所有方法的第一个token必须相同
   - 因为都使用原始模型的prefill

2. ✅ **Draft长度正确性**：
   - `total_draft_length = step × draft_length`
   - 每步draft长度固定为2

3. ✅ **接受长度范围**：
   - 每步接受长度：0 ≤ n_matches ≤ draft_length
   - 即：[0, 2]

4. ✅ **路由修改生效**：
   - Draft模型输出与原始模型不同
   - Draft模型的top预测应排除原始模型的top-2

### 6.2 性能指标
1. **接受率**：
   - 计算公式：`acceptance_rate = total_accept_length / total_draft_length`
   - 理论范围：0% ~ 100%
   - 预期：由于排除top-2，接受率应 < 100%

2. **生成速度**：
   - 记录总耗时
   - 计算tokens/second

---

## 7. 使用场景

### 7.1 主要用途
- 评估MoE模型对路由策略的敏感度
- 分析top专家的重要性
- 优化路由策略设计
- 研究投机采样在MoE模型上的效果

### 7.2 扩展可能
- 调整`top_k_experts_to_remove`参数
- 测试不同的draft_length值
- 应用于其他MoE模型（Mixtral等）
- 结合其他路由修改策略

---

## 8. 关键设计决策

### 8.1 为什么draft_length=2？
- **权衡因素**：
  - 太小（=1）：draft开销大，加速效果不明显
  - 太大（>2）：接受率下降，waste增加
- **benchmark测试**：从1改为2，作为性能评估基准

### 8.2 为什么排除top-2？
- **灵敏度测试**：排除最重要的专家，观察影响
- **可配置性**：参数化设计，可调整为top-1, top-3等

### 8.3 为什么共享模型实例？
- **内存效率**：避免加载两个大模型
- **实现简洁**：通过标志位控制行为
- **性能优化**：减少模型切换开销

---

## 9. 已知限制

### 9.1 模型支持
- ✅ 目前仅支持Qwen3模型
- ⚠️ 其他MoE模型需要适配

### 9.2 采样策略
- ✅ Verify阶段使用投机采样（带随机性）
- ⚠️ 输出与原始greedy不完全一致是正常现象

### 9.3 性能考虑
- ⚠️ Draft阶段使用deepcopy，有一定开销
- ⚠️ float16可能导致数值误差

---

## 10. 测试用例

### 10.1 基础测试
```python
prompt = "你好，请介绍一下北京"
max_new_tokens = 10
draft_length = 2
```

**预期结果**：
- 生成10个tokens
- 步数约5步（可能更少，因为接受率>0）
- 每步draft长度=2
- 接受率 > 0%

### 10.2 路由修改测试
**验证方法**：
- 输入相同的token
- 对比原始模型和draft模型的输出
- 原始模型top-1和top-2应被draft模型排除

**预期结果**：
- Draft模型的输出 ≠ 原始模型的输出
- Draft模型的top-5预测不包含原始top-2

### 10.3 KV Cache正确性测试
**验证方法**：
- 在draft前后检查verify KV cache长度
- 应保持不变

**预期结果**：
- Draft前后，verify KV cache长度相同
- 说明draft没有污染verify cache

---

## 11. 文档版本

- **版本**：v2.0
- **最后更新**：2025-10-21
- **状态**：已实现并测试通过
- **draft_length**：2（从v1.0的1更新）

