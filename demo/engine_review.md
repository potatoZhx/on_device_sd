# on_device_sd/demo Engine 设计实现 Review（基于 docs/engine_design.md）

## 1. Review 范围与方法

- 设计基线：docs/engine_design.md
- 重点核查实现：
  - src/execution/continuous_batch_engine.py
  - src/execution/orchestrator.py
  - src/memory/paged_kv_cache.py
  - src/layers/attention.py
  - src/core/model_runner.py
  - src/model/qwen3_runner.py
  - src/memory/expert_cache.py
- 本次为代码静态评审（未执行全量端到端性能回归）。

---

## 2. 总体结论

**整体完成度：约 88%（核心链路已落地，剩余主要为 scheduler 策略项）**

已实现：
- Continuous batching 主干（`Sequence`/`CBScheduler`/`CBExecutor`/`ContinuousBatchEngine`）
- Standard + Speculative 双模式统一执行入口
- Verify 复用 KV 的关键接口与调用链（`is_verify`、`get_verify_context`、`accept_draft`）
- ModelRunner 接口扩展（`is_verify`）
- CPU/GPU 混合 expert 执行并发化（线程并行版本）

仍存在：
- 调度策略与文档不一致（`prefill-first` 未严格实现）

---

## 3. 设计对齐矩阵

| 设计项 | 设计要求 | 当前实现 | 结论 |
|---|---|---|---|
| Continuous batching 主框架 | 新增 scheduler/executor/engine 统一循环 | 已在 `continuous_batch_engine.py` 实现 | ✅ |
| `Sequence` 状态机 | WAITING/RUNNING/DRAFTING/FINISHED/ERROR | 已实现并用于执行流 | ✅ |
| Scheduler 不感知 decode mode | 仅做 waiting/running + prefill/decode 切换 | 已实现 | ✅ |
| prefill-first 策略 | waiting 有请求时优先 prefill | `schedule()` 仅在 `running` 为空时才 prefill | ❌ |
| 无 padding 计算 | varlen prefill + paged decode | 已走 varlen/paged 路径 | ✅ |
| speculative batched draft | batch 内同步 draft step | 已实现 | ✅ |
| verify 复用 prompt KV | 不再新建 verify_seq，直接复用原 seq KV | 已实现 `get_verify_context` + `accept_draft` + `is_verify` | ✅ |
| draft 异步 prefetch | step 间异步预取 + 独立组件 | 统一走 `select_experts_to_prefetch` + `expert_cache.prefetch_async` 入口 | ✅ |
| CPU/GPU expert 并行 | stream/event 级并行 | 已有线程并行版本，但非文档所述 stream 方案 | 🟡 |
| 模块替换清理 | 旧 BatchManager/旧 orchestrator 逐步退场 | 旧单请求 draft/verify 路径已冻结，入口统一委托 continuous engine | ✅ |
| 文件拆分 | `sequence.py/cb_scheduler.py/cb_executor.py/...` | 已拆分到独立文件，`continuous_batch_engine.py` 保留兼容导出层 | ✅ |

---

## 4. 关键偏差与影响

### 4.1 调度策略偏差：`prefill-first` 未严格执行（高优先级）

- 现状：`CBScheduler.schedule()` 仅在 `running` 为空时执行 prefill。
- 影响：新请求无法在 decode 持续期间及时插入，连续批处理收益下降；文档目标“动态插入”打折。
- 建议：移除 `if not self.running` 限制；在每步先尝试 prefill（受 token budget/blocks 限制），再调度 decode。

### 4.2 迁移收敛：旧路径已冻结（已完成）

- 状态：`_generate_speculative`/`_generate_batch_speculative` 已改为兼容入口并直接委托 continuous engine；旧 draft/verify 辅助路径已清理。
- 备注：`BatchManager` 仍用于请求入队与异步处理，不影响 continuous 执行主链路。

### 4.3 prefetch 决策入口统一（已完成）

- 状态：draft 阶段统一使用 `select_experts_to_prefetch()` 选择集合，再通过 `expert_cache.prefetch_async()` 执行传输。
- 备注：`AsyncExpertTransfer` 仍保留在 `expert_cache.py` 作为可选工具类，不影响主路径。

### 4.4 CPU/GPU 并行执行形态与设计不一致（中优先级）

- 现状：`Qwen3ModelRunner` 采用 `ThreadPoolExecutor` 并行 CPU 任务与 GPU 任务，不是 stream/event pipeline。
- 影响：可获得部分并行收益，但对传输与计算重叠控制较弱，性能上限受限。
- 建议：若目标是峰值性能，后续切到 CUDA stream + event 方案；短期可先补 benchmark 对比证明线程方案收益。

---

## 5. 已实现亮点

1. **Verify 复用 KV 主链路已打通**：`ModelRunner.forward_attention(is_verify)`、attention verify 分支、`PagedKVCache.get_verify_context/accept_draft` 已连通。
2. **Speculative 批执行主流程可用**：draft→verify→accept 已在 `CBExecutor` 一体实现。
3. **接口演进方向正确**：`ModelRunner` 抽象面向 execution 层解耦，支持后续更换 runner。
4. **测试骨架较完整**：tests 下已覆盖 unit/integration/performance 目录，结构与设计计划基本一致。

---

## 6. 设计中可优化/不合理点（建议同步更新设计文档）

### 6.1 纯 prefill-first 可能导致 decode 饥饿

当请求持续涌入时，严格 prefill-first 可能让 running 序列长时间得不到 decode 配额。

建议：采用配额式调度，例如每步预算按 $B=B_{prefill}+B_{decode}$，或每 $N$ 次 prefill 强制插入一次 decode。

### 6.2 `max_num_batched_tokens = max_batch_size * max_position_embeddings` 过于激进

该上限在大模型上容易导致不现实的峰值预算，调度粒度失真。

建议：改为显式配置并基于 GPU 可用内存/历史 profile 自适应。

### 6.3 设计文档与实现文件组织存在漂移

文档给出多文件拆分，但实现采用单文件聚合。长期会影响可维护性。

建议：要么按文档拆分，要么更新文档承认“单文件阶段”并给出后续重构里程碑。

### 6.4 Verify 仍逐序列执行

当前 verify 为 per-seq，正确性优先合理，但吞吐上限受限。

建议：下一阶段推进 batched varlen verify，统一多序列 verify 的张量拼接与索引管理。

---

## 7. 落地进展（截至当前）

1. ⏳ 修正 scheduler 为真正 prefill-first（并加入 anti-starvation 配额）。
2. ✅ 清理/冻结旧执行路径，统一以 continuous engine 为主入口。
3. ✅ 统一 prefetch 策略入口，去除未使用主路径。
4. ✅ 已补 3 组基准脚本：
  - `tests/performance/bench_legacy_vs_continuous.py`
  - `tests/performance/bench_verify_before_after.py`
  - `tests/performance/bench_moe_thread_vs_sequential.py`
5. ✅ 已完成 continuous 相关文件拆分重构：
  - `src/core/sequence.py`
  - `src/execution/cb_scheduler.py`
  - `src/execution/cb_executor.py`
  - `src/execution/cb_engine.py`
  - `src/execution/prefetch_selector.py`
  - `src/execution/continuous_batch_engine.py`（兼容导出层）

---

## 8. 最终判断

该项目在 `engine_design.md` 的核心方向上已经进入“**可运行 + 可优化**”阶段，不再是纯设计稿状态；但要达到文档目标中的“高吞吐连续批处理 + 清晰可维护架构”，还需要完成调度策略修正、路径收敛与优化模块统一三件事。