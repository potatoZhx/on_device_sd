# Continuous Batching 重构设计

## 1. 背景与问题

### 1.1 现状分析

当前 demo 的批处理链路由三个组件串联：

1. **`BatchManager`**：基于 `PriorityQueue` 收集请求，按 `max_batch_size` + 超时组批，对不等长 prompt 做 **padding 对齐**，产出 `BatchedRequest`（含 padded `input_ids` / `attention_mask`）。
2. **`StandardDecodeEngine`**：接收 `BatchedRequest`，按 prompt 长度分组（`requests_by_len`），同长度组内做 batched prefill + batched decode；不同长度组之间串行处理。
3. **`Orchestrator`**：`_execute_batch` 将整个 batch 交给 `StandardDecodeEngine.generate_batch()`（标准模式）或逐请求串行调用 `_generate_speculative()`（speculative 模式）。

面对真实离线推理负载（prompt 长度分布离散、请求量大）时存在以下问题：

| 问题 | 原因 | 影响 |
|------|------|------|
| **padding 浪费** | `BatchManager._form_batch` 将所有请求 pad 到 `max_seq_len` | 短 prompt 产生大量无效计算 |
| **批次不连续** | 一个 batch 内所有请求必须全部 decode 完成后才能处理下一批 | 先完成的请求空等，GPU 利用率随 batch 内请求逐步完成而递减 |
| **无法动态插入** | decode 过程中无法加入新请求 | 吞吐受限于最慢请求 |
| **KV cache 利用率低** | `PagedKVCache` 已支持 block 管理和变长 attention，但调度层未充分利用 | 内存浪费 |
| **Speculative batching 缺失** | `_generate_batch_speculative` 逐请求串行执行 speculative decoding | 完全无法利用 batch 并行，speculative 模式吞吐极差 |

### 1.2 目标

引入参考 nano-vllm 的 continuous batching 机制，同时兼容现有 speculative decoding（draft-verify）流程：

1. **连续批处理**：每个 step 动态构造 batch，已完成的序列立即移出，新请求可随时加入。
2. **无 padding 计算**：prefill 使用 varlen attention（`flash_attn_varlen_func`），decode 使用 block_table（`flash_attn_with_kvcache`）。
3. **Token budget 控制**：通过 `max_num_seqs` 和 `max_num_batched_tokens` 限制单步工作量。
4. **Speculative decoding 兼容**：batch 内所有序列同步执行 draft-verify 循环，共享同一套 KV cache 管理和 MoE 调度。
5. **与现有 MoE 调度兼容**：routing / placement / expert prefetch 逻辑不变，只改变 batch 构造与执行编排方式。

### 1.3 Non-Goals（本阶段不处理）

- Prefix caching（hash 去重共享 KV block）
- Preemption（KV cache 不足时回退低优先级序列）
- 在线推理（流式返回、异步请求队列）
- Batch 内不同序列使用不同 `GenerationConfig`

### 1.4 调度策略

采用 **prefill-first** 策略，与 nano-vllm 一致：

- 每个 step 优先调度 `waiting` 队列中的 prefill 请求。
- 仅当没有可调度的 prefill 时，才执行 `running` 队列中的 decode。
- prefill 和 decode **不混合**在同一个 step 中执行。

---

## 2. 参考实现分析（nano-vllm）

nano-vllm 的核心执行循环：

```
LLMEngine.generate()
  for prompt in prompts:
    scheduler.add(Sequence(prompt))
  while not scheduler.is_finished():
    seqs, is_prefill = scheduler.schedule()      # 调度
    token_ids = model_runner.run(seqs, is_prefill) # 执行
    scheduler.postprocess(seqs, token_ids)        # 后处理
```

### 2.1 Scheduler

维护 `waiting` / `running` 双队列（`deque`），`schedule()` 返回 `(seqs, is_prefill)`：

- **Prefill 阶段**：从 `waiting` FIFO 取序列，受 `max_num_seqs` 和 `max_num_batched_tokens` 约束，为每个序列分配 KV block（`block_manager.allocate`），移入 `running`。
- **Decode 阶段**：从 `running` 取序列，每序列 1 token，受 `max_num_seqs` 约束，按需追加 block（`block_manager.may_append`）。

### 2.2 BlockManager

纯逻辑层，管理 block 分配/释放，不持有物理 KV 存储：

- `can_allocate(seq)` / `allocate(seq)`：prefill 时分配所需 block。
- `can_append(seq)` / `may_append(seq)`：decode 时按需追加新 block。
- `deallocate(seq)`：序列完成时释放所有 block。

### 2.3 关键差异点（nano-vllm vs demo 项目）

| 维度 | nano-vllm | demo 项目 |
|------|-----------|-----------|
| 模型执行 | 标准 dense model | MoE + CPU/GPU 异构执行 |
| Decode 模式 | 仅标准自回归 | 标准 + speculative（draft-verify） |
| ModelRunner | 直接调用 model forward | 三步 API（attention → route → moe） |
| KV cache | 物理存储在 model 内部 | `PagedKVCache` 独立管理，支持 draft-verify |
| Block 管理 | `Sequence.block_table` + `BlockManager` | `PagedKVCache.sequences[seq_id].block_table` + `BlockManager` |

---

## 3. Speculative Decoding 与 Expert 调度分析

### 3.1 Expert Placement 机制

demo 项目的 MoE 推理核心在于 **routing 与执行分离**：先计算路由（`route_experts`），再由引擎决定每个 expert 在哪里执行（`ExpertPlacement`），最后 `ModelRunner` 根据 placement 执行。

两种 placement 构建函数（`model_runner_utils.py`）的行为截然不同：

#### `build_prefill_placement`（用于 prefill / standard decode / verify）

精确执行，无替换：

1. 遍历所有激活的 expert
2. 尝试从 GPU 获取参数：shared expert 直接在 GPU；GPU cache 中有则从 cache 取；`parameter_loader` 能直接提供 GPU 参数则用
3. GPU 上拿不到的，从 CPU 获取参数在 **CPU 上精确执行**
4. **不做任何替换**（`substitution_map` 为空）

结果：**GPU cached expert（精确）+ CPU expert（精确）= 全精确执行，CPU/GPU 混合**

#### `build_draft_placement`（用于 draft）

快速近似执行，含替换和 CPU 限额：

1. 计算路由，得到所有激活 expert
2. `draft_scheduler.select_cpu_experts(activations, top_c)` —— 从激活 expert 中选出 **top-c 个**在 CPU 上精确执行（按 activation score 排序，选最重要的 c 个）
3. 其余激活 expert 中，GPU cache 有参数的在 **GPU 精确执行**
4. 剩余（不在 GPU cache 也不在 top-c CPU 中的）由 `draft_scheduler.select_gpu_substitutes()` 找一个 GPU cache 中的 expert **替换**（近似执行）
5. 同时 `DraftEngine._schedule_expert_transfers()` 根据 draft 激活历史，将高频 expert **从 CPU 预取到 GPU cache**，为后续 verify 做准备

结果：**top-c CPU expert（精确）+ GPU cached expert（精确）+ GPU substitute（近似替换）+ 并行 prefetch**

### 3.2 现有 Speculative 流程（单序列）

```
1. Prefill: prefill_engine.forward(input_ids, kv_cache, seq_ids=[0])
   Expert: build_prefill_placement（精确，CPU/GPU 混合）
   → 填充 KV cache，采样第一个 token

2. Decode loop:
   2a. Draft phase:
       - kv_cache.start_draft(seq_id=0)
       - 自回归 N 步，每步：
           - kv_cache.append_token(0)
           - forward_attention (decode mode)
           - route_experts → build_draft_placement
             → top-c CPU 精确 + GPU cache 精确 + 替换近似
           - forward_moe
           - 采样 → drafted_tokens
       - _schedule_expert_transfers()
         → 根据 draft 激活，将高频 expert 预取到 GPU cache

   2b. Verify phase:
       - verify_engine.forward(all_ids, kv_cache, seq_ids=[1])
         → 创建新 verify 序列，以 prefill 模式从头做完整前向
         Expert: build_prefill_placement（精确，CPU/GPU 混合，
           得益于 draft 阶段的 prefetch，更多 expert 已在 GPU cache 中）
         → 产出 verify_logits

   2c. Accept:
       - acceptance_strategy.accept(draft_token_ids, verify_logits)
         → 返回 num_accepted, accepted_tokens

   2d. KV cache reconciliation:
       - kv_cache.replace_draft_with_verify(seq_id=0, verify_seq_id=1, num_accepted)
```

### 3.3 Verify 阶段的 KV cache 策略

现有实现中 verify 使用 **`is_prefill=True` 对整个序列从头重新计算 KV**（prompt + output + draft tokens 拼接为完整序列，创建新的 `verify_seq_id`）。

这在语义上是正确的——verify 需要全精度模型产出每个 draft 位置的 logits，prefill 模式能一次性计算所有位置。但效率低：prompt 部分的 KV 被重复计算了。

更高效的方式是让 verify 只计算 draft tokens 的 KV，同时从 KV cache 读取 prompt 部分的历史 KV。`flash_attn_with_kvcache` 支持同时传入新 K/V 和读取 paged cache 中的历史 K/V。详见 **第 12 节 Verify 复用 Prompt KV 优化设计**。

### 3.4 Batched Speculative Decoding 的挑战

将 speculative decoding 从单序列扩展到 batch 需要解决：

1. **Draft 长度统一**：batch 内所有序列执行相同步数的 draft，每步一起做 batched decode。
2. **Verify 同步**：所有序列的 draft 完成后，一起做 batched verify（prefill 模式）。
3. **Accept 独立**：每个序列独立判断接受数量，接受数量可能不同。
4. **KV cache 分别 reconcile**：每个序列独立执行 `replace_draft_with_verify`。
5. **序列状态分化**：accept 后部分序列可能提前结束，需要从 batch 中移除。

---

## 4. 整体架构

### 4.1 组件关系

```
                    ┌─────────────────────────────────────────────────────┐
                    │                ContinuousBatchEngine                 │
                    │                                                     │
  add_requests() ──>│  ┌──────────┐    ┌──────────────────────────────┐  │
                    │  │Scheduler │───>│         Executor              │  │
                    │  │          │    │                              │  │
                    │  │ waiting  │    │ ┌──────────────────────────┐ │  │
                    │  │ running  │    │ │  Standard Decode Step    │ │  │
                    │  │          │    │ │  (单 token 自回归)        │ │  │
                    │  └────┬─────┘    │ ├──────────────────────────┤ │  │
                    │       │          │ │  Speculative Step        │ │  │
                    │       │          │ │  (draft → verify → accept)│ │  │
                    │       │          │ └──────────────────────────┘ │  │
                    │       │          └──────────┬───────────────────┘  │
                    │       │                     │                      │
                    │       ▼                     ▼                      │
                    │  ┌────────────────────────────────┐                │
                    │  │        PagedKVCache             │                │
                    │  │  (BlockManager + Storage +      │                │
                    │  │   draft-verify state)           │                │
                    │  └────────────────────────────────┘                │
                    │                     │                              │
                    │                     ▼                              │
                    │  ┌──────────────────────────────┐                  │
                    │  │        ModelRunner             │                  │
                    │  │  (embed → attn → route        │                  │
                    │  │   → moe → logits)             │                  │
                    │  └──────────────────────────────┘                  │
                    └─────────────────────────────────────────────────────┘
```

### 4.2 设计原则

1. **单一 Engine 入口**：用 `ContinuousBatchEngine` 替代 `BatchManager` + `StandardDecodeEngine` + `Orchestrator` 的三层组合。
2. **Scheduler 只做逻辑调度**：不持有物理存储，通过 `PagedKVCache` 的接口查询/操作 block。
3. **Executor 统一 standard 和 speculative**：通过 decode mode 控制是执行单步自回归还是 draft-verify 循环。
4. **Sequence 持有完整状态**：token_ids、生成参数、draft 状态，与 `PagedKVCache.sequences` 一一对应。
5. **复用现有 ModelRunner 三步 API**：保留 MoE 异构调度能力。
6. **复用现有 DraftSchedulingStrategy 和 AcceptanceStrategy**：调度策略和接受策略不变。

---

## 5. 核心数据结构

### 5.1 SequenceStatus

```python
class SequenceStatus(Enum):
    WAITING = auto()    # 在 waiting 队列，等待 prefill
    RUNNING = auto()    # 在 running 队列，正在 decode
    DRAFTING = auto()   # 正在执行 draft phase（speculative 模式）
    FINISHED = auto()   # 生成完成（EOS 或达到 max_tokens）
    ERROR = auto()      # 执行出错，已从 batch 移除
```

### 5.2 Sequence

```python
class Sequence:
    _counter = count()

    def __init__(
        self,
        token_ids: list[int],
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ):
        self.seq_id: int = next(Sequence._counter)
        self.status: SequenceStatus = SequenceStatus.WAITING

        self.prompt_token_ids: list[int] = list(token_ids)
        self.output_token_ids: list[int] = []

        self.max_new_tokens: int = max_new_tokens
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.top_k: int = top_k
        self.eos_token_id: Optional[int] = eos_token_id

        # Draft 状态（speculative decoding 专用）
        self.draft_token_ids: list[int] = []
        self.num_tokens_before_draft: int = 0

        self.error_msg: Optional[str] = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.num_generated

    @property
    def last_token_id(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    @property
    def is_finished(self) -> bool:
        return self.status in (SequenceStatus.FINISHED, SequenceStatus.ERROR)

    def append_token(self, token_id: int):
        self.output_token_ids.append(token_id)

    def check_finished(self) -> bool:
        if self.status == SequenceStatus.FINISHED:
            return False
        if self.num_generated >= self.max_new_tokens:
            self.status = SequenceStatus.FINISHED
            return True
        if self.eos_token_id is not None and self.last_token_id == self.eos_token_id:
            self.status = SequenceStatus.FINISHED
            return True
        return False

    def mark_error(self, msg: str):
        self.status = SequenceStatus.ERROR
        self.error_msg = msg

    # ---- Draft-Verify 支持 ----
    def start_draft(self):
        self.status = SequenceStatus.DRAFTING
        self.draft_token_ids = []
        self.num_tokens_before_draft = self.total_len

    def append_draft_token(self, token_id: int):
        self.draft_token_ids.append(token_id)

    def accept_draft(self, num_accepted: int):
        accepted = self.draft_token_ids[:num_accepted]
        self.output_token_ids.extend(accepted)
        self.draft_token_ids = []
        self.num_tokens_before_draft = 0
        self.status = SequenceStatus.RUNNING

    @property
    def num_draft_tokens(self) -> int:
        return len(self.draft_token_ids)

    @property
    def last_draft_token_id(self) -> int:
        if self.draft_token_ids:
            return self.draft_token_ids[-1]
        return self.last_token_id
```

### 5.3 DecodeMode

```python
class DecodeMode(Enum):
    STANDARD = "standard"
    SPECULATIVE = "speculative"
```

---

## 6. Scheduler 设计

### 6.1 接口定义

```python
@dataclass
class ScheduleResult:
    sequences: List[Sequence]
    is_prefill: bool


class CBScheduler:
    def __init__(
        self,
        kv_cache: PagedKVCache,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 8192,
    ):
        self.kv_cache = kv_cache
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
```

Scheduler **不感知** decode mode（standard vs speculative），它只负责 prefill 调度和 running 队列管理。

### 6.2 schedule()

```python
def schedule(self) -> ScheduleResult:
    scheduled: List[Sequence] = []
    num_seqs = 0
    num_tokens = 0

    # ---- Phase 1: Prefill ----
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]
        prompt_len = seq.prompt_len

        if num_tokens + prompt_len > self.max_num_batched_tokens:
            break

        num_blocks_needed = (
            (prompt_len + self.kv_cache.block_size - 1) // self.kv_cache.block_size
        )
        if not self.kv_cache.block_manager.can_allocate(num_blocks_needed):
            break

        success = self.kv_cache.add_sequence(seq.seq_id, prompt_len=prompt_len)
        if not success:
            break

        self.waiting.popleft()
        seq.status = SequenceStatus.RUNNING
        self.running.append(seq)
        scheduled.append(seq)
        num_seqs += 1
        num_tokens += prompt_len

    if scheduled:
        return ScheduleResult(sequences=scheduled, is_prefill=True)

    # ---- Phase 2: Decode ----
    remaining_running: deque[Sequence] = deque()

    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()

        if not self.kv_cache.can_append_token(seq.seq_id):
            seq.mark_error("KV cache exhausted, cannot append token")
            self.kv_cache.remove_sequence(seq.seq_id)
            continue

        scheduled.append(seq)
        remaining_running.append(seq)
        num_seqs += 1

    remaining_running.extend(self.running)
    self.running = remaining_running

    if not scheduled:
        raise RuntimeError(
            "Scheduler: no sequences can be scheduled. "
            f"waiting={len(self.waiting)}, running={len(self.running)}"
        )

    return ScheduleResult(sequences=scheduled, is_prefill=False)
```

Decode 阶段 scheduler **不调用** `kv_cache.append_token()`——在 speculative 模式下一个 decode step 涉及多次 KV cache 操作（draft N 次 append + verify add_sequence + replace_draft_with_verify），时机由 Executor 控制。

### 6.3 postprocess()

```python
def postprocess(self, seqs: List[Sequence], finished_seq_ids: List[int]):
    for seq_id in finished_seq_ids:
        self.kv_cache.remove_sequence(seq_id)

    self.running = deque(
        seq for seq in self.running if not seq.is_finished
    )
```

---

## 7. Executor 设计

### 7.1 接口定义

```python
class CBExecutor:
    def __init__(
        self,
        model_runner: ModelRunner,
        kv_cache: PagedKVCache,
        expert_cache: ExpertCache,
        parameter_loader: ParameterLoader,
        prefetcher: Optional[ExpertPrefetcher] = None,
        draft_scheduler: Optional[DraftSchedulingStrategy] = None,
        acceptance_strategy: Optional[AcceptanceStrategy] = None,
        max_draft_tokens: int = 8,
    ):
        self.model_runner = model_runner
        self.kv_cache = kv_cache
        self.expert_cache = expert_cache
        self.parameter_loader = parameter_loader
        self.prefetcher = prefetcher
        self.draft_scheduler = draft_scheduler
        self.acceptance_strategy = acceptance_strategy
        self.max_draft_tokens = max_draft_tokens
        self.draft_activations: List = []   # draft 激活历史，用于 expert transfer
```

### 7.2 execute_prefill()

```python
def execute_prefill(self, seqs: List[Sequence]) -> List[int]:
    seq_ids = [seq.seq_id for seq in seqs]

    input_ids_list = []
    positions_list = []
    for seq in seqs:
        prompt = seq.prompt_token_ids
        input_ids_list.extend(prompt)
        positions_list.extend(range(len(prompt)))

    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device='cuda')
    positions = torch.tensor(positions_list, dtype=torch.long, device='cuda')

    logits = self._forward(input_ids, positions, seq_ids, is_prefill=True)

    last_indices = []
    offset = 0
    for seq in seqs:
        last_indices.append(offset + seq.prompt_len - 1)
        offset += seq.prompt_len
    last_logits = logits[last_indices]

    token_ids = self._sample(last_logits, seqs)

    finished_seq_ids = []
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        self.kv_cache.append_token(seq.seq_id)
        if seq.check_finished():
            finished_seq_ids.append(seq.seq_id)

    return finished_seq_ids
```

### 7.3 execute_decode_standard()

```python
def execute_decode_standard(self, seqs: List[Sequence]) -> List[int]:
    seq_ids = [seq.seq_id for seq in seqs]

    for seq_id in seq_ids:
        self.kv_cache.append_token(seq_id)

    input_ids = torch.tensor(
        [seq.last_token_id for seq in seqs], dtype=torch.long, device='cuda'
    )
    positions = torch.tensor(
        [seq.total_len - 1 for seq in seqs], dtype=torch.long, device='cuda'
    )

    logits = self._forward(input_ids, positions, seq_ids, is_prefill=False)
    if logits.dim() == 3:
        logits = logits[:, -1, :]

    token_ids = self._sample(logits, seqs)

    finished_seq_ids = []
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        if seq.check_finished():
            finished_seq_ids.append(seq.seq_id)

    return finished_seq_ids
```

### 7.4 execute_decode_speculative()

```python
def execute_decode_speculative(self, seqs: List[Sequence]) -> List[int]:
    draft_seqs = [seq for seq in seqs if not seq.is_finished]
    if not draft_seqs:
        return []

    # Phase 1: Draft（含 expert prefetch）
    self._execute_draft(draft_seqs)

    # Phase 2: Verify（精确执行，受益于 draft 阶段的 prefetch）
    verify_logits_map = self._execute_verify(draft_seqs)

    # Phase 3: Accept + KV Reconcile
    finished_seq_ids = self._execute_accept(draft_seqs, verify_logits_map)

    return finished_seq_ids
```

#### 7.4.1 Draft Phase

Draft 阶段：batched 自回归 N 步，使用 `build_draft_placement`。

每步的 expert 执行策略：
- **top-c CPU expert**：选出 activation score 最高的 c 个 expert 在 CPU 精确执行
- **GPU cached expert**：GPU cache 中已有参数的 expert 在 GPU 精确执行
- **GPU substitute**：既不在 CPU top-c 也不在 GPU cache 中的 expert，用 GPU cache 中的 expert 近似替换
- **Expert prefetch**：每步收集激活信息，draft 完成后根据激活历史将高频 expert 从 CPU 预取到 GPU cache，为 verify 阶段做准备

```python
def _execute_draft(self, seqs: List[Sequence]):
    seq_ids = [seq.seq_id for seq in seqs]
    self.draft_activations = []
    cache_hits = 0
    cache_misses = 0

    for seq in seqs:
        seq.start_draft()
        self.kv_cache.start_draft(seq.seq_id)

    for step in range(self.max_draft_tokens):
        active_seqs = [s for s in seqs if s.status == SequenceStatus.DRAFTING]
        if not active_seqs:
            break
        active_seq_ids = [s.seq_id for s in active_seqs]

        for seq_id in active_seq_ids:
            if not self.kv_cache.can_append_token(seq_id):
                seq = next(s for s in active_seqs if s.seq_id == seq_id)
                seq.mark_error("KV cache exhausted during draft")
                continue
            self.kv_cache.append_token(seq_id)

        active_seqs = [s for s in seqs if s.status == SequenceStatus.DRAFTING]
        if not active_seqs:
            break
        active_seq_ids = [s.seq_id for s in active_seqs]

        input_ids = torch.tensor(
            [s.last_draft_token_id for s in active_seqs],
            dtype=torch.long, device='cuda'
        )
        positions = torch.tensor(
            [s.num_tokens_before_draft + s.num_draft_tokens - 1
             for s in active_seqs],
            dtype=torch.long, device='cuda'
        )

        logits, step_activations, step_hits, step_misses = self._forward_draft(
            input_ids, positions, active_seq_ids
        )
        cache_hits += step_hits
        cache_misses += step_misses
        self.draft_activations.extend(step_activations)

        if logits.dim() == 3:
            logits = logits[:, -1, :]

        token_ids = self._sample(logits, active_seqs)

        for seq, token_id in zip(active_seqs, token_ids):
            seq.append_draft_token(token_id)

    # Draft 完成后，根据激活历史预取 expert 到 GPU cache，为 verify 做准备
    self._schedule_expert_transfers()
```

#### 7.4.2 Expert Transfer（draft → verify 之间的 prefetch）

```python
def _schedule_expert_transfers(self):
    """根据 draft 激活历史，将高频 expert 从 CPU 预取到 GPU cache。"""
    if not self.draft_activations or self.draft_scheduler is None:
        return

    cached_experts = set(self.expert_cache.cached_experts.keys())
    experts_to_transfer = self.draft_scheduler.select_experts_to_transfer(
        recent_activations=self.draft_activations,
        cached_experts=cached_experts,
        cache_capacity=self.expert_cache.max_experts,
    )

    for expert_id in experts_to_transfer:
        cpu_params = self.parameter_loader.get_expert_params(expert_id)
        if cpu_params:
            self.expert_cache.put(expert_id, cpu_params)
```

#### 7.4.3 Verify Phase

Verify 阶段：对每个序列的 draft tokens 做全精度验证。Expert 使用 `build_prefill_placement`（精确执行，无替换，CPU/GPU 混合）。由于 draft 阶段已将高频 expert 预取到 GPU cache，verify 阶段的 GPU 命中率会更高。

**基础实现**（从头 prefill，用于 Phase 1-2 验证）：

```python
def _execute_verify(
    self, seqs: List[Sequence]
) -> Dict[int, torch.Tensor]:
    verify_logits_map = {}

    for seq in seqs:
        if seq.is_finished or seq.num_draft_tokens == 0:
            continue

        verify_seq_id = seq.seq_id + 1_000_000
        all_ids = seq.prompt_token_ids + seq.output_token_ids + seq.draft_token_ids
        input_ids = torch.tensor(all_ids, dtype=torch.long, device='cuda')

        self.kv_cache.add_sequence(verify_seq_id, prompt_len=len(all_ids))

        positions = torch.arange(len(all_ids), dtype=torch.long, device='cuda')
        logits = self._forward(
            input_ids, positions, [verify_seq_id], is_prefill=True
        )

        num_draft = seq.num_draft_tokens
        draft_start = len(all_ids) - num_draft
        if logits.dim() == 3:
            logits = logits[0]
        verify_logits = logits[draft_start - 1 : draft_start - 1 + num_draft]

        verify_logits_map[seq.seq_id] = verify_logits

    return verify_logits_map
```

**优化实现**（复用 prompt KV，详见第 12 节）：使用 `_forward_verify()` 仅对 draft tokens 做增量 attention，从 paged cache 读取历史 KV，不再创建 `verify_seq_id`，不再分配额外 block。

#### 7.4.4 Accept + Reconcile

```python
def _execute_accept(
    self,
    seqs: List[Sequence],
    verify_logits_map: Dict[int, torch.Tensor],
) -> List[int]:
    finished_seq_ids = []

    for seq in seqs:
        if seq.is_finished:
            finished_seq_ids.append(seq.seq_id)
            continue

        if seq.seq_id not in verify_logits_map:
            seq.status = SequenceStatus.RUNNING
            continue

        verify_logits = verify_logits_map[seq.seq_id]
        draft_token_ids = torch.tensor(seq.draft_token_ids, dtype=torch.long)

        result = self.acceptance_strategy.accept(
            draft_token_ids=draft_token_ids,
            verify_logits=verify_logits,
            temperature=seq.temperature,
        )

        num_accepted = result['num_accepted']
        seq.accept_draft(num_accepted)

        verify_seq_id = seq.seq_id + 1_000_000
        try:
            self.kv_cache.replace_draft_with_verify(
                seq_id=seq.seq_id,
                verify_seq_id=verify_seq_id,
                num_accepted_tokens=num_accepted,
            )
        except Exception as e:
            seq.mark_error(f"KV cache reconcile failed: {e}")
            finished_seq_ids.append(seq.seq_id)
            continue

        if seq.check_finished():
            finished_seq_ids.append(seq.seq_id)

    return finished_seq_ids
```

### 7.5 _forward()（精确路径）

用于 **prefill / standard decode / verify**。使用 `build_prefill_placement`：精确执行，无替换，CPU/GPU 混合。

```python
@torch.inference_mode()
def _forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    seq_ids: List[int],
    is_prefill: bool,
) -> torch.Tensor:
    hidden_states = self.model_runner.embed(input_ids)
    if hidden_states.dim() == 1:
        hidden_states = hidden_states.unsqueeze(0)

    for layer_idx in range(self.model_runner.get_num_layers()):
        attn_output = self.model_runner.forward_attention(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            kv_cache=self.kv_cache,
            positions=positions,
            seq_ids=seq_ids,
            is_prefill=is_prefill,
        )

        routing_result = self.model_runner.route_experts(
            layer_idx=layer_idx,
            hidden_states=attn_output.post_attn_normed,
        )

        # Prefetch for next layer
        if self.prefetcher is not None:
            from .model_runner_utils import build_layer_activations
            activations = build_layer_activations(routing_result)
            self.prefetcher.on_layer_complete(layer_idx, activations)

        # 精确 placement：GPU cache 有则 GPU 执行，否则 CPU 执行，无替换
        from .model_runner_utils import build_prefill_placement
        placement = build_prefill_placement(
            routing_result=routing_result,
            expert_cache=self.expert_cache,
            parameter_loader=self.parameter_loader,
        )

        hidden_states = self.model_runner.forward_moe(
            layer_idx=layer_idx,
            attn_output=attn_output,
            expert_placement=placement,
        )

    return self.model_runner.compute_logits(hidden_states)
```

### 7.6 _forward_draft()（draft 路径）

用于 **draft 阶段**。使用 `build_draft_placement`：top-c CPU 精确 + GPU cache 精确 + 替换近似。同时收集每层的激活信息和 cache 命中统计。

```python
@torch.inference_mode()
def _forward_draft(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    seq_ids: List[int],
) -> Tuple[torch.Tensor, List, int, int]:
    """
    Returns:
        logits, layer_activations, cache_hits, cache_misses
    """
    hidden_states = self.model_runner.embed(input_ids)
    if hidden_states.dim() == 1:
        hidden_states = hidden_states.unsqueeze(0)

    config = self.model_runner.get_config()
    layer_activations = []
    cache_hits = 0
    cache_misses = 0

    for layer_idx in range(self.model_runner.get_num_layers()):
        attn_output = self.model_runner.forward_attention(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            kv_cache=self.kv_cache,
            positions=positions,
            seq_ids=seq_ids,
            is_prefill=False,
        )

        routing_result = self.model_runner.route_experts(
            layer_idx=layer_idx,
            hidden_states=attn_output.post_attn_normed,
        )

        # Draft placement：top-c CPU + GPU cache + 替换
        from .model_runner_utils import build_draft_placement, build_layer_activations
        placement = build_draft_placement(
            routing_result=routing_result,
            expert_cache=self.expert_cache,
            parameter_loader=self.parameter_loader,
            draft_scheduler=self.draft_scheduler,
            top_c=config.draft_top_c,
            num_experts=config.num_experts,
        )

        cache_hits += len(placement.gpu_expert_params)
        cache_misses += len(placement.cpu_expert_params)

        acts = build_layer_activations(routing_result, config.num_experts)
        layer_activations.append(acts)

        hidden_states = self.model_runner.forward_moe(
            layer_idx=layer_idx,
            attn_output=attn_output,
            expert_placement=placement,
        )

    logits = self.model_runner.compute_logits(hidden_states)
    return logits, layer_activations, cache_hits, cache_misses
```

### 7.7 _sample()

```python
def _sample(self, logits: torch.Tensor, seqs: List[Sequence]) -> List[int]:
    temperature = seqs[0].temperature
    top_k = seqs[0].top_k
    top_p = seqs[0].top_p

    logits = logits / temperature

    if top_k > 0:
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, top_k_indices, top_k_logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1
        )
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')

    probs = torch.softmax(logits, dim=-1)
    token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return token_ids.tolist()
```

---

## 8. ContinuousBatchEngine 设计

### 8.1 接口定义

```python
class ContinuousBatchEngine:
    def __init__(
        self,
        model_runner: ModelRunner,
        kv_cache: PagedKVCache,
        expert_cache: ExpertCache,
        parameter_loader: ParameterLoader,
        prefetcher: Optional[ExpertPrefetcher] = None,
        draft_scheduler: Optional[DraftSchedulingStrategy] = None,
        acceptance_strategy: Optional[AcceptanceStrategy] = None,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 8192,
        decode_mode: DecodeMode = DecodeMode.STANDARD,
        max_draft_tokens: int = 8,
    ):
        self.decode_mode = decode_mode

        self.scheduler = CBScheduler(
            kv_cache=kv_cache,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )

        self.executor = CBExecutor(
            model_runner=model_runner,
            kv_cache=kv_cache,
            expert_cache=expert_cache,
            parameter_loader=parameter_loader,
            prefetcher=prefetcher,
            draft_scheduler=draft_scheduler,
            acceptance_strategy=acceptance_strategy,
            max_draft_tokens=max_draft_tokens,
        )
```

### 8.2 generate()

```python
def generate(
    self,
    prompts: List[List[int]],
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    eos_token_id: Optional[int] = None,
) -> List[dict]:
    seq_id_to_idx = {}
    for idx, prompt in enumerate(prompts):
        seq = Sequence(
            token_ids=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=eos_token_id,
        )
        self.scheduler.add(seq)
        seq_id_to_idx[seq.seq_id] = idx

    results = {}
    while not self.scheduler.is_finished():
        schedule_result = self.scheduler.schedule()

        try:
            if schedule_result.is_prefill:
                finished_ids = self.executor.execute_prefill(
                    schedule_result.sequences
                )
            else:
                if self.decode_mode == DecodeMode.SPECULATIVE:
                    finished_ids = self.executor.execute_decode_speculative(
                        schedule_result.sequences
                    )
                else:
                    finished_ids = self.executor.execute_decode_standard(
                        schedule_result.sequences
                    )
        except Exception as e:
            for seq in schedule_result.sequences:
                seq.mark_error(f"Execution error: {e}")
            finished_ids = [seq.seq_id for seq in schedule_result.sequences]

        self.scheduler.postprocess(schedule_result.sequences, finished_ids)

        for seq in schedule_result.sequences:
            if seq.is_finished and seq.seq_id not in results:
                idx = seq_id_to_idx.get(seq.seq_id)
                if idx is not None:
                    results[idx] = {
                        "seq_id": seq.seq_id,
                        "prompt_token_ids": seq.prompt_token_ids,
                        "output_token_ids": seq.output_token_ids,
                        "error": seq.error_msg,
                    }

    return [results.get(i, {"error": "not found"}) for i in range(len(prompts))]
```

### 8.3 Speculative 模式执行流程

```
generate(prompts)
  │
  └── while not scheduler.is_finished():
        │
        ├── schedule → (seqs, is_prefill)
        │
        ├── [is_prefill] → executor.execute_prefill(seqs)
        │     Expert: build_prefill_placement（精确，CPU/GPU 混合）
        │
        └── [decode, speculative] → executor.execute_decode_speculative(seqs)
              │
              ├── ===== Draft Phase =====
              │   ├── seq.start_draft() + kv_cache.start_draft()
              │   └── for step in range(max_draft_tokens):
              │         ├── kv_cache.append_token
              │         ├── 收集 last_draft_token → input_ids + positions
              │         ├── _forward_draft → logits
              │         │     Expert: build_draft_placement
              │         │       top-c CPU 精确
              │         │       + GPU cache 精确
              │         │       + GPU substitute 近似替换
              │         │     收集激活信息 + cache hit/miss 统计
              │         ├── _sample → token_ids
              │         └── seq.append_draft_token
              │
              ├── ===== Expert Prefetch =====
              │   └── _schedule_expert_transfers()
              │         根据 draft 激活历史，将高频 expert
              │         从 CPU 预取到 GPU cache（为 verify 做准备）
              │
              ├── ===== Verify Phase =====
              │   └── for seq in seqs:
              │         ├── 构造 verify_input = [prompt + output + draft]
              │         ├── kv_cache.add_sequence(verify_seq_id)
              │         ├── _forward(is_prefill=True) → logits
              │         │     Expert: build_prefill_placement
              │         │       精确执行，无替换，CPU/GPU 混合
              │         │       （受益于 prefetch，GPU 命中率更高）
              │         └── 提取 draft 位置的 logits → verify_logits_map
              │
              └── ===== Accept + Reconcile =====
                  └── for seq in seqs:
                        ├── acceptance_strategy.accept()
                        ├── seq.accept_draft(num_accepted)
                        ├── kv_cache.replace_draft_with_verify()
                        └── seq.check_finished
```

---

## 9. Expert Placement 与 KV Cache 交互总览

### 9.1 各阶段 Expert 执行策略

| 阶段 | Placement 函数 | GPU expert | CPU expert | 替换 | Prefetch |
|------|---------------|-----------|-----------|------|----------|
| **Prefill** | `build_prefill_placement` | cache 中/shared/loader 可提供 GPU 参数的 | GPU 上拿不到参数的 | 无 | 逐层 prefetcher |
| **Standard decode** | `build_prefill_placement` | 同上 | 同上 | 无 | 逐层 prefetcher |
| **Draft** | `build_draft_placement` | cache 中 + substitute 替换的 | 固定 top-c 个（按 score 选） | 有（`substitution_map`） | draft 结束后批量 transfer |
| **Verify** | `build_prefill_placement` | 同 Prefill（因 draft prefetch，命中率更高） | GPU 上拿不到的 | 无 | 逐层 prefetcher |

### 9.2 KV Cache 交互

**基础版本（从头 prefill verify）：**

| 操作 | 调用方 | PagedKVCache 方法 | 说明 |
|------|--------|-------------------|------|
| Prefill 分配 | Scheduler | `add_sequence(seq_id, prompt_len)` | 为 prompt 分配 KV block |
| Standard decode | Executor | `append_token(seq_id)` | 追加 1 token slot |
| Draft 开始 | Executor | `start_draft(seq_id)` | 记录 draft 起始位置 |
| Draft 每步 | Executor | `append_token(seq_id)` | 为 draft token 追加 slot |
| Verify 创建 | Executor | `add_sequence(verify_seq_id, len)` | 创建独立 verify 序列（从头 prefill） |
| Accept 合并 | Executor | `replace_draft_with_verify(seq_id, verify_seq_id, n)` | 用 verify KV 替换 draft KV |
| 序列完成 | Scheduler | `remove_sequence(seq_id)` | 释放所有 KV block |

**优化版本（复用 prompt KV，详见第 12 节）：**

| 操作 | 调用方 | PagedKVCache 方法 | 说明 |
|------|--------|-------------------|------|
| Prefill 分配 | Scheduler | `add_sequence(seq_id, prompt_len)` | 不变 |
| Draft 开始/每步 | Executor | `start_draft` / `append_token` | 不变 |
| Verify 上下文 | Executor | `get_verify_context(seq_id, n)` | **复用原序列 block_table** |
| Accept 截断 | Executor | `accept_draft(seq_id, n)` | **仅截断多余 block，无需 verify_seq_id** |
| 序列完成 | Scheduler | `remove_sequence(seq_id)` | 不变 |

### 9.3 Sequence 状态机

```
                add()
  ┌───────┐  ────────>  ┌─────────┐
  │ (new) │             │ WAITING │
  └───────┘             └────┬────┘
                             │ schedule (prefill)
                             ▼
                        ┌─────────┐
                  ┌────>│ RUNNING │<────────────┐
                  │     └────┬────┘             │
                  │          │                  │
                  │    ┌─────┴──────┐           │
                  │    │            │           │
                  │    ▼            ▼           │
                  │ [standard]  [speculative]   │
                  │ decode      start_draft     │
                  │    │            │           │
                  │    │            ▼           │
                  │    │     ┌──────────┐       │
                  │    │     │ DRAFTING │       │
                  │    │     └────┬─────┘       │
                  │    │          │ accept_draft │
                  │    │          └──────────────┘
                  │    │
                  │    ├──── check_finished ──> ┌──────────┐
                  │    │                       │ FINISHED │
                  │    │                       └──────────┘
                  │    │
                  └────┘ (next step)
                  
         mark_error (任何状态) ──────> ┌─────────┐
                                       │  ERROR  │
                                       └─────────┘
```

---

## 10. 与现有模块的对接

### 10.1 完全复用（不修改）

| 模块 | 说明 |
|------|------|
| `src/core/model_runner.py` | ModelRunner 抽象接口 |
| `src/model/qwen3_runner.py` | Qwen3ModelRunner 实现 |
| `src/memory/paged_kv_cache.py` | PagedKVCache（含 draft-verify 支持） |
| `src/execution/model_runner_utils.py` | `build_prefill_placement` / `build_draft_placement` / `build_layer_activations` |
| `src/execution/acceptance_strategy.py` | AcceptanceStrategy |
| `src/scheduling/draft_schduler.py` | DraftSchedulingStrategy |
| `src/scheduling/prefetcher.py` | ExpertPrefetcher |
| `src/memory/expert_cache.py` | ExpertCache |
| `src/memory/parameter_loader.py` | ParameterLoader |

### 10.2 被替代的模块

| 现有模块 | 原因 |
|----------|------|
| `BatchManager` | padding 组批被 Scheduler 的 varlen 调度替代 |
| `StandardDecodeEngine` | 固定 batch decode 被 Executor 的 step-by-step 执行替代 |
| `EnhancedInferenceOrchestrator` | standard + speculative 路径统一到 `ContinuousBatchEngine` |
| `InferenceOrchestrator` | 单请求 speculative 路径统一到 `ContinuousBatchEngine` |
| `PrefillEngine` | 前向逻辑内联到 `CBExecutor._forward()` |
| `DraftEngine` | draft 循环 + expert transfer 逻辑内联到 `CBExecutor._execute_draft()` |
| `VerifyEngine` | verify 逻辑内联到 `CBExecutor._execute_verify()` |

---

## 11. 关键设计决策

### 11.1 为什么 Scheduler 不感知 decode mode？

Scheduler 只关心 prefill 调度和 running 队列管理。decode 的具体执行方式（standard vs speculative）由 Engine 层决策、Executor 层执行。Scheduler 保持简单。

### 11.2 为什么 decode 阶段 Scheduler 不调用 append_token？

在 speculative 模式下：
- Draft 阶段每序列产生 N 个 token，每步都需 `append_token`
- Verify 阶段需要 `add_sequence` 创建 verify 序列
- Accept 阶段需要 `replace_draft_with_verify`

这些 KV cache 操作与执行逻辑深度耦合，由 Executor 统一处理。

### 11.3 为什么 Sequence 不持有 block_table？

`PagedKVCache` 已通过 `SequenceState` 管理 block_table 且包含 draft-verify 的 block 逻辑。两份状态会产生同步问题。

### 11.4 Verify 阶段的 KV cache 策略

Verify 需要全精度 logits。现有 `VerifyEngine` 从头 prefill 整个序列，语义正确但浪费计算。第 12 节设计了优化方案：通过 `flash_attn_with_kvcache` 的多 token Q 模式，verify 仅计算 draft tokens 的 attention，同时从 paged cache 读取 prompt/output 部分的历史 KV，消除重复计算并避免额外 block 分配。

### 11.5 Draft 与 Verify 的 expert 协同

Draft 阶段不仅仅是近似推理——它同时起到**探测 expert 访问模式**的作用。通过收集每步每层的激活信息（`build_layer_activations`），draft 完成后 `_schedule_expert_transfers()` 将 verify 可能需要的 expert 预取到 GPU cache。这使得 verify 阶段能以更高的 GPU 命中率执行，减少 CPU 计算比例，降低 verify 延迟。

### 11.6 为什么 verify 当前是逐序列的？

各序列的 verify 长度（prompt + output + draft）不同，且需要独立的 verify_seq_id。合并为 batched varlen prefill 需要额外的张量拼接和索引管理。当前逐序列 verify 降低实现复杂度，后续可优化。

---

## 12. 优化设计：Verify 复用 Prompt KV Cache

### 12.1 问题分析

当前 `_execute_verify()` 对每个序列创建 `verify_seq_id`，以 `is_prefill=True` 从头对完整序列（prompt + output + draft）做 prefill。prompt 部分的 KV 在 `seq_id` 的 block_table 中已经存在，从头 prefill 重复计算了 prompt 的 attention，计算浪费与 prompt 长度成正比。

### 12.2 目标

Verify 只计算 **draft tokens 的新 KV**，同时从原序列的 paged KV cache 中读取 **prompt + output 部分的历史 KV**。

### 12.3 技术方案

核心思路：verify 时不创建独立的 `verify_seq_id`，而是复用原序列 `seq_id` 的 block_table，仅对 draft tokens 做增量 prefill。

#### 12.3.1 PagedKVCache 新增方法

```python
class PagedKVCache:
    def get_verify_context(
        self,
        seq_id: int,
        num_new_tokens: int,
    ) -> Dict:
        """
        为 verify 阶段构造 attention context。

        Verify 只需要计算 num_new_tokens 个 draft token 的 Q/K/V，
        但需要 attend 到整个序列（历史 KV 从 cache 读取）。

        使用 flash_attn_with_kvcache：
        - Q: [num_new_tokens, num_heads, head_dim]
        - K/V cache: 从 block_table 读取
        - cache_seqlens: 历史 token 数（不含 draft tokens）
        - 新的 K/V 通过 k/v 参数传入

        Args:
            seq_id: 序列 ID（使用原序列，不创建新序列）
            num_new_tokens: draft token 数量

        Returns:
            dict:
                - slot_mapping: [num_new_tokens] draft tokens 写入的 slot
                - block_tables: [1, max_num_blocks] 原序列的 block table
                - context_lens: [1] 包含 draft 的完整序列长度
        """
        seq_state = self.sequences[seq_id]

        # Draft tokens 的 slot mapping（从 draft_start_num_tokens 开始）
        draft_start = seq_state.draft_start_num_tokens
        slot_mapping = seq_state.get_slot_mapping(draft_start, num_new_tokens)

        # Block table 包含原序列 + draft 已分配的所有 block
        max_num_blocks = len(seq_state.block_table)
        block_table = seq_state.get_block_table_tensor(max_num_blocks)

        # context_lens 是完整序列长度（含 draft tokens）
        context_lens = seq_state.num_tokens

        return {
            'slot_mapping': torch.tensor(slot_mapping, dtype=torch.int32, device='cuda'),
            'block_tables': block_table.unsqueeze(0),
            'context_lens': torch.tensor([context_lens], dtype=torch.int32, device='cuda'),
        }
```

#### 12.3.2 Qwen3Attention 新增 verify 路径

当前 `Qwen3Attention.forward()` 的 prefill 路径使用 `flash_attn_varlen_func`（仅用 Q/K/V 本身做 attention，不读 cache），decode 路径使用 `flash_attn_with_kvcache`（读 cache）。Verify 需要一个新路径：**写入新 K/V 到 cache，同时从 cache 读取历史 KV 做 attention**。

`flash_attn_with_kvcache` 原生支持这个模式——它可以接受额外的 `k`/`v` 参数，这些新 K/V 会先写入 cache，然后与 cache 中的历史 K/V 一起做 attention。但需要 Q 为 `[batch, seqlen_q, heads, head_dim]` 格式，其中 `seqlen_q > 1`。

```python
# 在 Qwen3Attention.forward 中新增 verify 路径
def forward(
    self,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    kv_cache: PagedKVCache,
    seq_ids: list[int],
    is_prefill: bool,
    is_verify: bool = False,    # 新参数
) -> torch.Tensor:
    # ... QKV projection, reshape, QK norm, RoPE ...

    context = kv_cache.get_attention_context(seq_ids, is_prefill)
    k_cache_layer, v_cache_layer = kv_cache.get_kv_cache_for_layer(self.layer_idx)

    if is_verify:
        # Verify 路径：draft tokens 的 Q/K/V + 从 cache 读取历史 KV
        # 使用 flash_attn_with_kvcache 的 k/v 参数传入新 K/V
        verify_ctx = kv_cache.get_verify_context(seq_ids[0], q.shape[0])

        # 写入 draft tokens 的 K/V 到 cache
        if verify_ctx['slot_mapping'].numel() > 0:
            store_kvcache(k, v, k_cache_layer, v_cache_layer, verify_ctx['slot_mapping'])

        k_cache_view = k_cache_layer.view(
            k_cache_layer.shape[0], k_cache_layer.shape[1],
            self.num_kv_heads, self.head_dim,
        )
        v_cache_view = v_cache_layer.view(
            v_cache_layer.shape[0], v_cache_layer.shape[1],
            self.num_kv_heads, self.head_dim,
        )

        # Q shape: [num_draft_tokens, num_heads, head_dim] → [1, num_draft_tokens, num_heads, head_dim]
        attn_output = flash_attn_with_kvcache(
            q.unsqueeze(0),
            k_cache_view,
            v_cache_view,
            cache_seqlens=verify_ctx['context_lens'],
            block_table=verify_ctx['block_tables'],
            softmax_scale=self.scaling,
            causal=True,
        )
        attn_output = attn_output.squeeze(0)
    elif is_prefill:
        # ... 现有 prefill 路径不变 ...
    else:
        # ... 现有 decode 路径不变 ...
```

#### 12.3.3 ModelRunner 接口扩展

```python
class ModelRunner(ABC):
    @abstractmethod
    def forward_attention(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        kv_cache: Any,
        positions: Optional[torch.Tensor],
        *,
        seq_ids: Optional[List[int]] = None,
        is_prefill: bool = False,
        is_verify: bool = False,    # 新参数
    ) -> AttentionOutput:
        ...
```

#### 12.3.4 更新后的 `_execute_verify()`

```python
def _execute_verify(
    self, seqs: List[Sequence]
) -> Dict[int, torch.Tensor]:
    verify_logits_map = {}

    for seq in seqs:
        if seq.is_finished or seq.num_draft_tokens == 0:
            continue

        num_draft = seq.num_draft_tokens

        # 仅用 draft tokens 作为输入（不重复 prompt + output）
        draft_ids = seq.draft_token_ids
        input_ids = torch.tensor(draft_ids, dtype=torch.long, device='cuda')

        # position 从 draft 起始位置开始
        draft_start_pos = seq.num_tokens_before_draft
        positions = torch.arange(
            draft_start_pos,
            draft_start_pos + num_draft,
            dtype=torch.long, device='cuda'
        )

        # 使用 verify 路径前向推理（复用原序列的 KV cache）
        logits = self._forward_verify(
            input_ids, positions, [seq.seq_id]
        )

        if logits.dim() == 3:
            logits = logits[0]
        # logits[i] 对应 draft_start_pos + i 位置，
        # 其 target token 是 draft_token_ids[i]
        # 需要取 logits[0:num_draft]（对应 draft_start-1 到 draft_start+num_draft-2 位置的 next-token logits）
        # 但此处输入仅是 draft tokens，logits 已经是 [num_draft, vocab]
        # 注意：verify 需要位置 draft_start-1 的 logits 来验证第一个 draft token
        # 所以输入需要包含 draft 前一个 token

        verify_logits_map[seq.seq_id] = logits

    return verify_logits_map
```

#### 12.3.5 Verify 输入的边界处理

Verify 需要验证每个 draft token 是否与全精度模型的预测一致。位置 `draft_start - 1` 处的 logits 用于验证第一个 draft token。所以 verify 输入需要包含 **draft 前一个 token + 所有 draft tokens**：

```python
# 实际输入: [last_output_token] + draft_token_ids
verify_input_ids = [seq.output_token_ids[-1]] + seq.draft_token_ids
positions = range(draft_start_pos - 1, draft_start_pos + num_draft)
# logits 形状: [num_draft + 1, vocab]
# verify_logits = logits[0:num_draft]  # logits[i] 的 target 是 draft_token_ids[i]
```

#### 12.3.6 `_forward_verify()`

```python
@torch.inference_mode()
def _forward_verify(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    seq_ids: List[int],
) -> torch.Tensor:
    """
    Verify 专用前向路径。
    复用原序列的 KV cache，仅计算 draft tokens 的 attention。
    Expert 使用 build_prefill_placement（精确，无替换）。
    """
    hidden_states = self.model_runner.embed(input_ids)
    if hidden_states.dim() == 1:
        hidden_states = hidden_states.unsqueeze(0)

    for layer_idx in range(self.model_runner.get_num_layers()):
        attn_output = self.model_runner.forward_attention(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            kv_cache=self.kv_cache,
            positions=positions,
            seq_ids=seq_ids,
            is_prefill=False,
            is_verify=True,
        )

        routing_result = self.model_runner.route_experts(
            layer_idx=layer_idx,
            hidden_states=attn_output.post_attn_normed,
        )

        if self.prefetcher is not None:
            from .model_runner_utils import build_layer_activations
            activations = build_layer_activations(routing_result)
            self.prefetcher.on_layer_complete(layer_idx, activations)

        from .model_runner_utils import build_prefill_placement
        placement = build_prefill_placement(
            routing_result=routing_result,
            expert_cache=self.expert_cache,
            parameter_loader=self.parameter_loader,
        )

        hidden_states = self.model_runner.forward_moe(
            layer_idx=layer_idx,
            attn_output=attn_output,
            expert_placement=placement,
        )

    return self.model_runner.compute_logits(hidden_states)
```

### 12.4 KV Cache 生命周期变化

优化前后对比：

| 操作 | 优化前 | 优化后 |
|------|--------|--------|
| Draft 开始 | `start_draft(seq_id)` | 不变 |
| Draft 每步 | `append_token(seq_id)` | 不变 |
| Verify 创建 | `add_sequence(verify_seq_id, total_len)` 分配大量新 block | **不需要**，复用 `seq_id` 的 block |
| Verify attention | 从头 prefill，`flash_attn_varlen_func` | `flash_attn_with_kvcache` 读历史 KV + 写 draft KV |
| Accept 合并 | `replace_draft_with_verify(seq_id, verify_seq_id, n)` | 简化为 `accept_draft_tokens(seq_id, n)`，只截断多余 block |
| 内存消耗 | 需要额外分配 verify 序列的全部 block | 零额外分配 |

### 12.5 `replace_draft_with_verify` 简化

由于 verify 不再创建独立序列，accept 阶段只需截断 draft 阶段多余分配的 block：

```python
class PagedKVCache:
    def accept_draft(self, seq_id: int, num_accepted: int):
        """
        接受 num_accepted 个 draft tokens，释放多余的 block。
        不再需要 verify_seq_id（verify 直接写在原序列的 block 中）。
        """
        seq_state = self.sequences[seq_id]
        final_num_tokens = seq_state.draft_start_num_tokens + num_accepted
        final_num_blocks = (final_num_tokens + self.block_size - 1) // self.block_size

        # 释放多余 block
        for i in range(final_num_blocks, len(seq_state.block_table)):
            block_id = seq_state.block_table[i]
            if self.block_manager.dec_ref(block_id):
                self.block_manager.deallocate_block(block_id)

        seq_state.block_table = seq_state.block_table[:final_num_blocks]
        seq_state.accept_draft_tokens(num_accepted)
```

### 12.6 `_execute_accept()` 更新

```python
def _execute_accept(
    self,
    seqs: List[Sequence],
    verify_logits_map: Dict[int, torch.Tensor],
) -> List[int]:
    finished_seq_ids = []

    for seq in seqs:
        if seq.is_finished:
            finished_seq_ids.append(seq.seq_id)
            continue

        if seq.seq_id not in verify_logits_map:
            seq.status = SequenceStatus.RUNNING
            continue

        verify_logits = verify_logits_map[seq.seq_id]
        draft_token_ids = torch.tensor(seq.draft_token_ids, dtype=torch.long)

        result = self.acceptance_strategy.accept(
            draft_token_ids=draft_token_ids,
            verify_logits=verify_logits,
            temperature=seq.temperature,
        )

        num_accepted = result['num_accepted']
        seq.accept_draft(num_accepted)

        # 简化：直接在原序列上截断，不需要 replace_draft_with_verify
        try:
            self.kv_cache.accept_draft(seq.seq_id, num_accepted)
        except Exception as e:
            seq.mark_error(f"KV cache accept_draft failed: {e}")
            finished_seq_ids.append(seq.seq_id)
            continue

        if seq.check_finished():
            finished_seq_ids.append(seq.seq_id)

    return finished_seq_ids
```

---

## 13. 优化设计：Draft 阶段异步 Expert Prefetch

### 13.1 问题分析

当前 `_execute_draft()` 中 expert prefetch 发生在所有 draft steps 完成**之后**（`_schedule_expert_transfers()`），是同步阻塞操作。这意味着：

1. Draft 执行期间 CPU→GPU 通信带宽空闲
2. Expert 预取与 draft GPU 计算串行执行，增加了 draft→verify 之间的延迟
3. Prefetch 决策基于全部 draft 激活历史，无法在 draft 早期就开始传输

### 13.2 目标

在 draft 的**每个 step 之间**利用 GPU 计算与 CPU→GPU 通信可重叠的特性，异步预取后续可能需要的 expert，使得 draft 完成时大部分 verify 所需 expert 已在 GPU cache 中。

### 13.3 异步传输基础设施

#### 13.3.1 CUDA Stream 异步传输

```python
class AsyncExpertTransfer:
    """基于 CUDA stream 的异步 expert 参数传输。"""

    def __init__(self, max_concurrent: int = 2):
        self.transfer_stream = torch.cuda.Stream()
        self.max_concurrent = max_concurrent
        self.pending_events: Dict[ExpertID, torch.cuda.Event] = {}

    def start_transfer(
        self,
        expert_id: ExpertID,
        cpu_params: Dict[str, torch.Tensor],
        expert_cache: ExpertCache,
    ):
        """
        在 transfer_stream 上发起异步 CPU→GPU 传输。
        不阻塞 default stream 上的 GPU 计算。
        """
        if len(self.pending_events) >= self.max_concurrent:
            self._wait_oldest()

        with torch.cuda.stream(self.transfer_stream):
            gpu_params = {
                k: v.to('cuda', non_blocking=True)
                for k, v in cpu_params.items()
            }
            event = torch.cuda.Event()
            event.record(self.transfer_stream)

        self.pending_events[expert_id] = (event, gpu_params, expert_cache)

    def poll_completed(self) -> List[ExpertID]:
        """检查并完成已传输完毕的 expert，写入 cache。"""
        completed = []
        for expert_id, (event, gpu_params, cache) in list(self.pending_events.items()):
            if event.query():
                cache.put(expert_id, gpu_params)
                completed.append(expert_id)

        for eid in completed:
            del self.pending_events[eid]

        return completed

    def wait_all(self):
        """等待所有进行中的传输完成。"""
        for expert_id, (event, gpu_params, cache) in self.pending_events.items():
            event.synchronize()
            cache.put(expert_id, gpu_params)
        self.pending_events.clear()

    def _wait_oldest(self):
        """等待最早的传输完成，腾出并发槽位。"""
        if not self.pending_events:
            return
        oldest_id = next(iter(self.pending_events))
        event, gpu_params, cache = self.pending_events.pop(oldest_id)
        event.synchronize()
        cache.put(oldest_id, gpu_params)
```

### 13.4 Prefetch 集合选择函数

预取 expert 集合的选择策略是独立函数，方便后续替换为更复杂的预测模型。

```python
def select_experts_to_prefetch(
    current_step: int,
    max_draft_tokens: int,
    step_activations: List[LayerExpertActivations],
    cached_experts: Set[ExpertID],
    pending_transfers: Set[ExpertID],
    cache_capacity: int,
    num_experts_per_layer: int,
    *,
    max_prefetch_per_step: int = 4,
) -> List[ExpertID]:
    """
    在 draft step 之间选择要预取的 expert 集合。

    策略：基于已观察到的 draft 激活模式，预测 verify 阶段可能需要的 expert。
    Verify 使用 build_prefill_placement，对所有激活 expert 做精确执行，
    所以需要预取那些「被激活但不在 GPU cache 中」的 expert。

    选择逻辑：
    1. 统计已完成 draft steps 中所有层的 expert 激活频次和分数
    2. 过滤掉已在 GPU cache 中和正在传输中的 expert
    3. 按 (频次 × 平均分数) 降序排列
    4. 取 top-max_prefetch_per_step 个

    Args:
        current_step: 当前 draft step 编号（0-based）
        max_draft_tokens: 总 draft step 数
        step_activations: 到目前为止所有 step 的激活信息
        cached_experts: 当前 GPU cache 中的 expert 集合
        pending_transfers: 正在异步传输中的 expert 集合
        cache_capacity: GPU cache 最大 expert 容量
        num_experts_per_layer: 每层 expert 总数
        max_prefetch_per_step: 每步最多预取的 expert 数

    Returns:
        要预取的 expert ID 列表（按优先级降序）
    """
    if not step_activations:
        return []

    # 统计激活频次和分数
    expert_stats: Dict[ExpertID, Tuple[int, float]] = {}
    for layer_acts in step_activations:
        for act in layer_acts.activations:
            eid = act.expert_id
            count, total_score = expert_stats.get(eid, (0, 0.0))
            expert_stats[eid] = (count + 1, total_score + act.scores.max().item())

    # 过滤
    already_available = cached_experts | pending_transfers
    candidates = {
        eid: (count, total_score)
        for eid, (count, total_score) in expert_stats.items()
        if eid not in already_available
    }

    if not candidates:
        return []

    # 检查 cache 容量
    available_slots = cache_capacity - len(cached_experts) - len(pending_transfers)
    if available_slots <= 0:
        return []

    # 按 (频次 × 平均分数) 排序
    scored = [
        (eid, count * (total_score / count))
        for eid, (count, total_score) in candidates.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    num_to_prefetch = min(max_prefetch_per_step, available_slots, len(scored))
    return [eid for eid, _ in scored[:num_to_prefetch]]
```

### 13.5 更新后的 `_execute_draft()`

```python
def _execute_draft(self, seqs: List[Sequence]):
    seq_ids = [seq.seq_id for seq in seqs]
    self.draft_activations = []
    async_transfer = AsyncExpertTransfer(max_concurrent=2)

    for seq in seqs:
        seq.start_draft()
        self.kv_cache.start_draft(seq.seq_id)

    for step in range(self.max_draft_tokens):
        active_seqs = [s for s in seqs if s.status == SequenceStatus.DRAFTING]
        if not active_seqs:
            break
        active_seq_ids = [s.seq_id for s in active_seqs]

        for seq_id in active_seq_ids:
            if not self.kv_cache.can_append_token(seq_id):
                seq = next(s for s in active_seqs if s.seq_id == seq_id)
                seq.mark_error("KV cache exhausted during draft")
                continue
            self.kv_cache.append_token(seq_id)

        active_seqs = [s for s in seqs if s.status == SequenceStatus.DRAFTING]
        if not active_seqs:
            break
        active_seq_ids = [s.seq_id for s in active_seqs]

        input_ids = torch.tensor(
            [s.last_draft_token_id for s in active_seqs],
            dtype=torch.long, device='cuda'
        )
        positions = torch.tensor(
            [s.num_tokens_before_draft + s.num_draft_tokens - 1
             for s in active_seqs],
            dtype=torch.long, device='cuda'
        )

        logits, step_acts, _, _ = self._forward_draft(
            input_ids, positions, active_seq_ids
        )
        self.draft_activations.extend(step_acts)

        if logits.dim() == 3:
            logits = logits[:, -1, :]

        token_ids = self._sample(logits, active_seqs)
        for seq, token_id in zip(active_seqs, token_ids):
            seq.append_draft_token(token_id)

        # ---- 在 draft step 之间发起异步 expert prefetch ----
        async_transfer.poll_completed()

        experts_to_prefetch = select_experts_to_prefetch(
            current_step=step,
            max_draft_tokens=self.max_draft_tokens,
            step_activations=self.draft_activations,
            cached_experts=set(self.expert_cache.cached_experts.keys()),
            pending_transfers=set(async_transfer.pending_events.keys()),
            cache_capacity=self.expert_cache.max_experts,
            num_experts_per_layer=self.model_runner.get_config().num_experts,
        )

        for expert_id in experts_to_prefetch:
            cpu_params = self.parameter_loader.get_expert_params(expert_id)
            if cpu_params:
                async_transfer.start_transfer(
                    expert_id, cpu_params, self.expert_cache
                )

    # Draft 完成，等待所有异步传输完成
    async_transfer.wait_all()
```

### 13.6 与现有 `_schedule_expert_transfers()` 的关系

异步 prefetch 完全替代了原来的 `_schedule_expert_transfers()`。区别：

| 维度 | 原方案 | 新方案 |
|------|--------|--------|
| 时机 | Draft 全部完成后同步传输 | 每个 draft step 后异步传输 |
| 阻塞 | 阻塞 default stream | 不阻塞 GPU 计算 |
| 传输并发 | 串行逐个传输 | CUDA stream 并发（max_concurrent=2） |
| 选择策略 | `draft_scheduler.select_experts_to_transfer()` | `select_experts_to_prefetch()`（独立函数） |
| 决策依据 | 全部 draft 激活历史 | 逐步累积的部分激活历史 |
| 覆盖率 | 一次性决策 | 逐步补充，后续 step 可修正 |

---

## 14. 优化设计：CPU/GPU Expert 并行执行

### 14.1 问题分析

当前 `Qwen3ModelRunner._execute_moe_with_placement()` 中 CPU expert 和 GPU expert 串行执行：

```python
# 当前实现（串行）
for expert_id in activated_expert_ids:
    if expert_idx in gpu_expert_params:
        expert_output = expert_forward_with_weights(expert_input, ...)          # GPU
    elif expert_idx in cpu_expert_params:
        expert_input_cpu = expert_input.cpu()                                    # GPU→CPU
        expert_output_cpu = expert_forward_with_weights(expert_input_cpu, ...)   # CPU
        expert_output = expert_output_cpu.to(flat.device)                        # CPU→GPU
    elif expert_idx in sub_map:
        expert_output = expert_forward_with_weights(expert_input, ...)          # GPU (substitute)
```

CPU expert 执行的流程是：`input GPU→CPU` → `CPU 计算` → `output CPU→GPU`。在此期间 GPU 空闲。对于 verify/prefill/standard decode 等使用 `build_prefill_placement` 的场景，CPU expert 数量可能较多（所有不在 GPU cache 中的 expert），串行执行的性能损失显著。

### 14.2 目标

GPU expert 和 CPU expert 并行执行：GPU expert 在 default stream 上执行，CPU expert 的数据传输和计算在独立的 CUDA stream 上与 GPU 计算重叠。

### 14.3 技术方案

#### 14.3.1 并行执行架构

```
Timeline:
                                         
default stream:  [GPU expert 0] [GPU expert 1] ... [GPU expert N] [wait event] [reduce]
                                                                       ↑
cpu stream:      [D2H input] [CPU compute] [H2D output] [record event] ┘
```

- GPU expert 在 default stream 上逐个执行
- 所有 CPU expert 的 input 批量传输到 CPU，在 CPU 上执行后批量传回 GPU
- CPU 相关操作在单独的 `cpu_stream` 上执行
- GPU expert 完成后等待 CPU event，然后合并输出

#### 14.3.2 `_execute_moe_with_placement()` 重写

```python
def _execute_moe_with_placement(
    self,
    hidden_states: torch.Tensor,
    expert_placement: ExpertPlacement,
) -> torch.Tensor:
    routing = expert_placement.routing_result
    if routing is None:
        return torch.zeros_like(hidden_states)

    if hidden_states.dim() == 3:
        b, s, h = hidden_states.shape
        flat = hidden_states.view(-1, h)
        need_reshape = True
    else:
        flat = hidden_states
        h = flat.shape[-1]
        need_reshape = False

    topk_indices = routing.topk_indices
    topk_scores = routing.topk_scores
    sub_map = expert_placement.substitution_map

    final_output = torch.zeros(flat.shape[0], h, device=flat.device, dtype=flat.dtype)

    # ---- 分组：GPU experts vs CPU experts ----
    gpu_tasks = []  # [(expert_idx, token_indices, k_indices, weights, params)]
    cpu_tasks = []

    for expert_id in routing.activated_expert_ids:
        expert_idx = expert_id.expert_idx
        expert_mask = (topk_indices == expert_idx)
        token_expert_pairs = torch.where(expert_mask)
        token_indices = token_expert_pairs[0]
        k_indices = token_expert_pairs[1]

        if len(token_indices) == 0:
            continue

        weights = topk_scores[token_indices, k_indices]
        expert_input = flat[token_indices]

        if expert_idx in expert_placement.gpu_expert_params:
            params = expert_placement.gpu_expert_params[expert_idx]
            gpu_tasks.append((expert_idx, token_indices, weights, expert_input, params))
        elif expert_idx in expert_placement.cpu_expert_params:
            params = expert_placement.cpu_expert_params[expert_idx]
            cpu_tasks.append((expert_idx, token_indices, weights, expert_input, params))
        elif expert_idx in sub_map:
            sub_idx = sub_map[expert_idx]
            if sub_idx in expert_placement.gpu_expert_params:
                params = expert_placement.gpu_expert_params[sub_idx]
                gpu_tasks.append((expert_idx, token_indices, weights, expert_input, params))

    # ---- 并行执行 ----
    cpu_stream = torch.cuda.Stream()
    cpu_results = {}   # expert_idx → (token_indices, weights, output_gpu)

    if cpu_tasks:
        # 在 cpu_stream 上执行 CPU expert
        with torch.cuda.stream(cpu_stream):
            for expert_idx, token_indices, weights, expert_input, params in cpu_tasks:
                input_cpu = expert_input.to('cpu', non_blocking=True)
            # 同步确保传输完成
            cpu_stream.synchronize()

            for expert_idx, token_indices, weights, expert_input, params in cpu_tasks:
                input_cpu = expert_input.cpu()
                output_cpu = expert_forward_with_weights(
                    input_cpu,
                    params['gate_proj'], params['up_proj'], params['down_proj'],
                )
                output_gpu = output_cpu.to(flat.device, non_blocking=True)
                cpu_results[expert_idx] = (token_indices, weights, output_gpu)

            cpu_event = torch.cuda.Event()
            cpu_event.record(cpu_stream)

    # 在 default stream 上执行 GPU expert
    for expert_idx, token_indices, weights, expert_input, params in gpu_tasks:
        expert_output = expert_forward_with_weights(
            expert_input,
            params['gate_proj'], params['up_proj'], params['down_proj'],
        )
        final_output[token_indices] += expert_output * weights.unsqueeze(1)

    # 等待 CPU expert 完成，合并结果
    if cpu_tasks:
        torch.cuda.current_stream().wait_event(cpu_event)
        for expert_idx, (token_indices, weights, output_gpu) in cpu_results.items():
            final_output[token_indices] += output_gpu * weights.unsqueeze(1)

    if need_reshape:
        final_output = final_output.view(b, s, h)

    return final_output
```

### 14.4 性能分析

假设一个层有 8 个 activated expert，其中 5 个在 GPU cache，3 个需要 CPU 执行：

| 维度 | 串行（现有） | 并行（优化后） |
|------|-------------|---------------|
| GPU expert 时间 | 5 × T_gpu | 5 × T_gpu |
| CPU expert 时间 | 3 × (T_d2h + T_cpu + T_h2d) | 3 × (T_d2h + T_cpu + T_h2d) |
| 总时间 | 5×T_gpu + 3×(T_d2h+T_cpu+T_h2d) | max(5×T_gpu, 3×(T_d2h+T_cpu+T_h2d)) |

当 CPU expert 执行时间被 GPU expert 执行时间覆盖时，CPU expert 的延迟为零。

### 14.5 适用范围

| 阶段 | 是否生效 | 原因 |
|------|---------|------|
| **Prefill** | 是 | `build_prefill_placement` 可能有 CPU expert |
| **Standard decode** | 是 | 同上 |
| **Verify** | 是 | 同上，且受益于 prefetch 后 CPU expert 数量减少 |
| **Draft** | 是（效果小） | `build_draft_placement` 的 CPU expert 固定为 top-c 个（通常 2 个），剩余被替换为 GPU expert |

### 14.6 对接方式

此优化仅修改 `Qwen3ModelRunner._execute_moe_with_placement()`，对 `CBExecutor` 和其他组件透明。`ExpertPlacement` 数据结构不变，`ModelRunner` 接口不变。

---

## 15. 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| **新增** | `src/core/sequence.py` | `Sequence`、`SequenceStatus`、`DecodeMode` |
| **新增** | `src/execution/cb_scheduler.py` | `CBScheduler`、`ScheduleResult` |
| **新增** | `src/execution/cb_executor.py` | `CBExecutor`（含 standard + speculative 双路径 + verify 优化 + 异步 prefetch） |
| **新增** | `src/execution/cb_engine.py` | `ContinuousBatchEngine` |
| **新增** | `src/execution/async_transfer.py` | `AsyncExpertTransfer` |
| **新增** | `src/execution/prefetch_selector.py` | `select_experts_to_prefetch()` |
| **修改** | `src/layers/attention.py` | `Qwen3Attention.forward()` 新增 `is_verify` 路径 |
| **修改** | `src/core/model_runner.py` | `forward_attention()` 新增 `is_verify` 参数 |
| **修改** | `src/model/qwen3_runner.py` | `forward_attention()` 传递 `is_verify`；`_execute_moe_with_placement()` 改为 CPU/GPU 并行 |
| **修改** | `src/memory/paged_kv_cache.py` | 新增 `get_verify_context()` 和 `accept_draft()` |
| **不修改** | `src/execution/model_runner_utils.py` | 完全复用 |
| **不修改** | `src/execution/acceptance_strategy.py` | 完全复用 |
| **不修改** | `src/scheduling/draft_schduler.py` | 完全复用 |
| **不修改** | `src/memory/expert_cache.py` | 完全复用 |
| **不修改** | `src/memory/parameter_loader.py` | 完全复用 |

---

## 16. 迁移计划

### Phase 1：新增 Continuous Batching 基础组件

1. 新增 `src/core/sequence.py`
2. 新增 `src/execution/cb_scheduler.py`
3. 新增 `src/execution/cb_executor.py`（含基础 verify 从头 prefill 版本）
4. 新增 `src/execution/cb_engine.py`
5. 单元测试通过

### Phase 2：验证基础功能

1. **Standard 模式**：对比 `ContinuousBatchEngine(mode=STANDARD)` 与 `StandardDecodeEngine` 的输出一致性和吞吐
2. **Speculative 模式**：对比 `ContinuousBatchEngine(mode=SPECULATIVE)` 与 `Orchestrator._generate_speculative` 的输出一致性
3. 性能基准：tokens/sec、KV cache 利用率、speculative acceptance rate

### Phase 3：Verify 复用 Prompt KV 优化

1. `PagedKVCache` 新增 `get_verify_context()` 和 `accept_draft()`
2. `Qwen3Attention.forward()` 新增 `is_verify` 路径
3. `ModelRunner.forward_attention()` 新增 `is_verify` 参数
4. `CBExecutor` 新增 `_forward_verify()`，更新 `_execute_verify()` 和 `_execute_accept()`
5. 验证 verify 优化的输出与从头 prefill 一致

### Phase 4：异步 Expert Prefetch

1. 新增 `src/execution/async_transfer.py`
2. 新增 `src/execution/prefetch_selector.py`
3. 更新 `_execute_draft()` 集成异步 prefetch
4. 验证 prefetch 不影响输出正确性

### Phase 5：CPU/GPU Expert 并行执行

1. 修改 `Qwen3ModelRunner._execute_moe_with_placement()` 为并行版本
2. 验证并行执行的输出与串行执行一致

### Phase 6：切换默认路径 + 清理

1. 将离线推理入口切换到 `ContinuousBatchEngine`
2. 清理被替代的模块

---

## 17. 测试计划

### 17.1 测试策略总览

所有测试分为三层：**单元测试**（组件级）、**集成测试**（端到端流程）、**性能测试**（吞吐/延迟/资源利用率）。核心原则是**每个优化都有与未优化版本的 bit-level 或 tolerance 级一致性对比**。

### 17.2 单元测试

#### T1. Sequence 状态管理

| 编号 | 用例 | 验证内容 |
|------|------|---------|
| T1.1 | 创建 Sequence，检查初始状态 | `status==WAITING`，`prompt_len` 正确，`output_token_ids` 为空 |
| T1.2 | `append_token` + `check_finished` | `num_generated` 递增，达到 `max_new_tokens` 时状态变为 `FINISHED` |
| T1.3 | EOS 检测 | 追加 `eos_token_id` 后 `check_finished()` 返回 `True` |
| T1.4 | `start_draft` → `append_draft_token` → `accept_draft` | draft 状态正确转换，`accept_draft(n)` 后 `output_token_ids` 包含前 n 个 draft token |
| T1.5 | `accept_draft(0)` | 全部拒绝，`output_token_ids` 不变，状态回到 `RUNNING` |
| T1.6 | `mark_error` | 任何状态都能转为 `ERROR`，`is_finished` 为 `True` |

#### T2. CBScheduler

| 编号 | 用例 | 验证内容 |
|------|------|---------|
| T2.1 | 单序列 prefill 调度 | `schedule()` 返回 `is_prefill=True`，序列从 waiting 移入 running |
| T2.2 | 多序列 prefill 受 `max_num_batched_tokens` 约束 | token 总数超限时停止调度 |
| T2.3 | 多序列 prefill 受 `max_num_seqs` 约束 | 序列数超限时停止调度 |
| T2.4 | KV block 不足时 prefill 停止 | `can_allocate` 返回 False 时不再调度 |
| T2.5 | 无 waiting 时调度 decode | `schedule()` 返回 `is_prefill=False`，running 中的序列被调度 |
| T2.6 | decode 中 `can_append_token` 失败 | 序列标记为 `ERROR` 并从 running 移除 |
| T2.7 | `postprocess` 清理 | 已完成序列的 KV cache 被释放，从 running 移除 |
| T2.8 | prefill-first 优先级 | waiting 有序列时即使 running 也有序列，仍然先 prefill |

#### T3. PagedKVCache 新增接口

| 编号 | 用例 | 验证内容 |
|------|------|---------|
| T3.1 | `get_verify_context` 基础 | slot_mapping 对应 draft 区间，block_tables 包含原序列所有 block |
| T3.2 | `get_verify_context` 跨 block 边界 | draft tokens 跨 block 时 slot mapping 正确 |
| T3.3 | `accept_draft` 释放多余 block | accept 后 block_table 截断，多余 block 归还到 free pool |
| T3.4 | `accept_draft(0)` | 回退到 draft 开始前的状态，所有 draft block 释放 |
| T3.5 | `accept_draft(max_draft)` | 全部接受，block_table 不截断 |

#### T4. Attention Verify 路径

| 编号 | 用例 | 验证内容 |
|------|------|---------|
| T4.1 | `is_verify=True` 输出与 `is_prefill=True`（从头）一致 | 给定相同序列，verify 路径输出的 logits 与从头 prefill 在 tolerance 内一致 |
| T4.2 | Verify 路径 KV 写入正确 | verify 后 cache 中 draft 位置的 K/V 与 prefill 写入的 K/V 一致 |
| T4.3 | 多层 verify | 逐层 verify 后累积误差在 tolerance 内 |

#### T5. AsyncExpertTransfer

| 编号 | 用例 | 验证内容 |
|------|------|---------|
| T5.1 | 单次异步传输 | `start_transfer` 后 `wait_all`，expert 出现在 cache 中 |
| T5.2 | 并发传输限制 | 超过 `max_concurrent` 时自动等待最早的传输完成 |
| T5.3 | `poll_completed` | 已完成的传输被正确写入 cache，pending 列表更新 |
| T5.4 | 空传输 | 无 pending 时 `wait_all` 和 `poll_completed` 不报错 |

#### T6. `select_experts_to_prefetch`

| 编号 | 用例 | 验证内容 |
|------|------|---------|
| T6.1 | 无激活历史 | 返回空列表 |
| T6.2 | 所有激活 expert 已在 cache | 返回空列表 |
| T6.3 | 正常选择 | 返回的 expert 不在 cache 且不在 pending 中，数量不超过 `max_prefetch_per_step` |
| T6.4 | 优先级排序 | 高频高分 expert 排在前面 |
| T6.5 | cache 容量限制 | 不超过 `cache_capacity - len(cached) - len(pending)` |

#### T7. CPU/GPU Expert 并行执行

| 编号 | 用例 | 验证内容 |
|------|------|---------|
| T7.1 | 全 GPU expert | 输出与串行版本一致 |
| T7.2 | 全 CPU expert | 输出与串行版本一致 |
| T7.3 | CPU + GPU 混合 | 输出与串行版本一致（tolerance < 1e-5） |
| T7.4 | 含 substitution | 替换 expert 的输出正确累加 |
| T7.5 | 空输入 | 无 activated expert 时返回零张量 |

### 17.3 集成测试

#### T8. ContinuousBatchEngine Standard 模式

| 编号 | 用例 | 验证内容 |
|------|------|---------|
| T8.1 | 单序列生成 | 输出 token 数 ≤ max_new_tokens，EOS 时提前停止 |
| T8.2 | 多序列不等长 prompt | 所有序列都生成完成，无 padding |
| T8.3 | 与旧 `StandardDecodeEngine` 对比 | 相同输入 + 相同随机种子 → 输出一致 |
| T8.4 | KV cache 不足 | 部分序列标记为 ERROR，其余正常完成 |
| T8.5 | max_num_seqs 约束 | 序列数超限时分批 prefill |

#### T9. ContinuousBatchEngine Speculative 模式

| 编号 | 用例 | 验证内容 |
|------|------|---------|
| T9.1 | 单序列 speculative | 输出合理（acceptance rate > 0） |
| T9.2 | 与旧 `Orchestrator._generate_speculative` 对比 | 相同输入 + 相同种子 → 输出一致（容许 acceptance 策略的随机性） |
| T9.3 | 多序列 speculative | 所有序列完成，每个序列的 accept 独立 |
| T9.4 | Draft 中序列出错 | 出错序列从 batch 移除，其余继续 |
| T9.5 | 全拒绝场景 | `num_accepted=0` 时序列仍继续生成（从 verify 采样 bonus token） |

#### T10. Verify 优化端到端

| 编号 | 用例 | 验证内容 |
|------|------|---------|
| T10.1 | Verify 复用 KV vs 从头 prefill | 最终生成的 token 序列一致 |
| T10.2 | Verify 后 KV cache 一致性 | verify 路径写入的 KV 与从头 prefill 写入的 KV 数值一致 |
| T10.3 | Accept 后序列状态 | `accept_draft()` 后 block_table 和 num_tokens 正确 |
| T10.4 | 连续多轮 draft-verify | 多轮 draft-verify 循环后序列状态和 KV cache 一致 |

#### T11. 异步 Prefetch 端到端

| 编号 | 用例 | 验证内容 |
|------|------|---------|
| T11.1 | 开启 vs 关闭异步 prefetch | 生成结果一致 |
| T11.2 | Prefetch 覆盖率 | draft 完成时 verify 所需 expert 的 GPU 命中率 > 不使用 prefetch 时的命中率 |
| T11.3 | 多序列 prefetch 互不干扰 | batch 内多个序列同时 draft 时 prefetch 正常工作 |

#### T12. CPU/GPU 并行端到端

| 编号 | 用例 | 验证内容 |
|------|------|---------|
| T12.1 | 并行 vs 串行输出一致 | 相同输入 + 相同种子 → 输出在 tolerance 内一致 |
| T12.2 | 多层并行 | 每层都正确并行执行，最终 logits 一致 |

### 17.4 性能测试

| 编号 | 测试项 | 指标 | 基线 |
|------|--------|------|------|
| P1 | Continuous batching vs 旧 batch 引擎 | tokens/sec | `StandardDecodeEngine` |
| P2 | Verify 复用 KV vs 从头 prefill | verify 耗时 (ms) | 从头 prefill 的 verify 耗时 |
| P3 | Verify 复用 KV | KV block 峰值使用量 | 从头 prefill 的 KV block 使用量 |
| P4 | 异步 prefetch vs 同步 transfer | draft→verify 总耗时 (ms) | 同步 transfer 的耗时 |
| P5 | 异步 prefetch | verify 阶段 GPU expert 命中率 | 不使用 prefetch 的命中率 |
| P6 | CPU/GPU 并行 vs 串行 | 单层 MoE 执行耗时 (ms) | 串行 MoE 耗时 |
| P7 | 综合 | 端到端吞吐 tokens/sec（16 序列 batch） | 旧 orchestrator |

### 17.5 测试环境与工具

```python
# 测试 fixture 示例
@pytest.fixture
def mock_kv_cache():
    """创建小规模 KV cache 用于测试"""
    config = MoEConfig(
        num_hidden_layers=2,
        num_key_value_heads=4,
        head_dim=64,
        # ... 其他最小配置
    )
    return PagedKVCache(config, block_size=16, dtype=torch.float16)


@pytest.fixture
def mock_model_runner():
    """Mock ModelRunner，forward 返回确定性输出"""
    runner = Mock(spec=ModelRunner)
    runner.get_num_layers.return_value = 2
    runner.get_config.return_value = MoEConfig(...)
    # embed, forward_attention, route_experts, forward_moe, compute_logits
    # 均返回确定性张量
    return runner


def assert_logits_close(actual, expected, atol=1e-4, rtol=1e-3):
    """验证 logits 在浮点 tolerance 内一致"""
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
```

### 17.6 测试文件结构

```
tests/
├── unit/
│   ├── test_sequence.py           # T1.*
│   ├── test_cb_scheduler.py       # T2.*
│   ├── test_paged_kv_cache.py     # T3.*（新增接口）
│   ├── test_attention_verify.py   # T4.*
│   ├── test_async_transfer.py     # T5.*
│   ├── test_prefetch_selector.py  # T6.*
│   └── test_parallel_moe.py       # T7.*
├── integration/
│   ├── test_cb_engine_standard.py # T8.*
│   ├── test_cb_engine_speculative.py  # T9.*
│   ├── test_verify_optimization.py    # T10.*
│   ├── test_async_prefetch_e2e.py     # T11.*
│   └── test_parallel_moe_e2e.py       # T12.*
└── performance/
    ├── bench_throughput.py         # P1, P7
    ├── bench_verify.py             # P2, P3
    ├── bench_prefetch.py           # P4, P5
    └── bench_parallel_moe.py       # P6
```
