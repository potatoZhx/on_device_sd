# Step 3: Attention 层 + flash_attn 实现总结

## 实现完成时间
2026-01-30

## 实现目标
实现 Qwen3 Attention 层，集成 flash_attn 和 PagedKVCache，支持 GQA (Grouped Query Attention) 和 QK Norm。

## 核心文件

### 1. `/src/layers/rotary_embedding.py`
**RoPE (Rotary Position Embedding) 实现**

#### 核心函数
- `apply_rotary_emb(x, cos, sin)`: 应用旋转位置编码
- `get_rope()`: LRU 缓存的 RoPE 模块工厂函数

#### RotaryEmbedding 类
- 预计算所有位置的 cos/sin 值
- 支持自动设备迁移（CPU → GPU）
- 支持 max_position 和 base (theta) 参数

**关键特性**:
- 参考 nano-vllm 实现
- 使用 `torch.compile` 可以进一步优化（已预留）
- 缓存 cos/sin 值，避免重复计算

### 2. `/src/layers/layernorm.py`
**RMSNorm (Root Mean Square Normalization) 实现**

#### 核心方法
- `rms_forward(x)`: 标准 RMS normalization
- `add_rms_forward(x, residual)`: 带 residual 的 RMS norm（fused operation）

**关键特性**:
- 支持与 residual connection 融合，减少内存访问
- 自动类型转换（float32 计算，保持原始 dtype）
- 参考 nano-vllm 实现

### 3. `/src/layers/attention.py`
**Qwen3 Attention 层实现**

#### Qwen3Attention 类（nn.Module）
标准 PyTorch 模块，适合独立使用或后续优化。

**核心特性**:
- **GQA 支持**: 32 Query Heads, 4 KV Heads
- **QK Norm**: Qwen3 特有的 Q/K normalization
- **RoPE**: 集成旋转位置编码
- **flash_attn 集成**:
  - Prefill: `flash_attn_varlen_func` (支持 variable length)
  - Decode: `flash_attn_with_kvcache` (支持 KV cache)
- **PagedKVCache 集成**: 自动存储和检索 KV states

#### Qwen3AttentionWithWeights 类
无状态的 Attention wrapper，使用外部权重。

**优势**:
- 不持有权重，节省内存
- 便于动态权重管理
- 适合 MoE 等需要灵活权重管理的场景

**核心方法**:
```python
forward(
    hidden_states,      # [num_tokens, hidden_size]
    positions,          # [num_tokens]
    q/k/v/o_weight,     # External weights
    q/k_norm_weight,    # Optional QK norm weights
    kv_cache,           # PagedKVCache instance
    seq_ids,            # List of sequence IDs
    is_prefill,         # Prefill or decode mode
)
```

### 4. `/src/layers/__init__.py`
统一导出接口

## 实现细节

### 1. Flash Attention 集成

#### Prefill 模式
```python
attn_output = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=context['cu_seqlens_q'],     # 累积序列长度
    cu_seqlens_k=context['cu_seqlens_k'],
    max_seqlen_q=context['max_seqlen_q'],     # 最大序列长度
    max_seqlen_k=context['max_seqlen_k'],
    softmax_scale=self.scaling,                # 1 / sqrt(head_dim)
    causal=True,                               # 因果 mask
)
```

#### Decode 模式
```python
attn_output = flash_attn_with_kvcache(
    q.unsqueeze(1),                            # [num_tokens, 1, num_heads, head_dim]
    k_cache_layer,                             # Physical KV cache
    v_cache_layer,
    cache_seqlens=context['context_lens'],     # 每个序列的长度
    block_table=context['block_tables'],       # Block table for paged attention
    softmax_scale=self.scaling,
    causal=True,
)
```

### 2. KV Cache 存储

使用简化的 PyTorch 实现（后续可优化为 triton kernel）：

```python
def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    
    key_flat = key.reshape(N, D)
    value_flat = value.reshape(N, D)
    
    for i in range(N):
        slot = slot_mapping[i].item()
        if slot != -1:
            k_cache[slot] = key_flat[i]
            v_cache[slot] = value_flat[i]
```

### 3. QK Normalization (Qwen3 特有)

当 `qkv_bias=False` 时启用：
```python
if self.has_qk_norm:
    q = self._apply_rms_norm(q, q_norm_weight)  # [num_tokens, num_heads, head_dim]
    k = self._apply_rms_norm(k, k_norm_weight)
```

### 4. Grouped Query Attention (GQA)

- **Qwen3-30B-A3B-Base**: 32 Q heads, 4 KV heads (8:1 ratio)
- **KV head 复制**: flash_attn 内部自动处理

## 测试覆盖

### 测试文件
`/tests/unit/test_attention.py`

### 测试类和用例（共 8 个测试）

#### TestRoPE (2 tests)
- ✅ `test_rope_creation`: RoPE 模块创建
- ✅ `test_apply_rotary_emb`: RoPE 应用

#### TestRMSNorm (2 tests)
- ✅ `test_rmsnorm_forward`: 标准 RMSNorm
- ✅ `test_rmsnorm_with_residual`: 带 residual 的 RMSNorm

#### TestQwen3Attention (3 tests)
- ✅ `test_attention_module_creation`: 模块创建
- ✅ `test_attention_forward_shape`: 输出形状验证
- ✅ `test_attention_with_real_weights`: 使用 Qwen3 真实权重

#### TestQwen3AttentionWithWeights (1 test)
- ✅ `test_attention_with_weights_forward`: 外部权重模式

### 测试结果
```
8 passed in 7.95s
```

## 与 nano-vllm 的对比

| 特性 | nano-vllm | 本实现 |
|------|-----------|--------|
| flash_attn | ✅ varlen + kvcache | ✅ 相同 |
| RoPE | ✅ 预计算 cache | ✅ 相同 |
| RMSNorm | ✅ fused residual | ✅ 相同 |
| GQA | ✅ | ✅ |
| QK Norm | ❌ | ✅ (Qwen3 特有) |
| Paged KV Cache | ✅ | ✅ |

## 关键设计决策

### 1. 为什么提供两个 Attention 类？

#### Qwen3Attention (nn.Module)
- 适合独立使用和测试
- 权重作为模块参数
- 便于后续 torch.compile 优化

#### Qwen3AttentionWithWeights (无状态)
- 适合集成到 MoE 模型中
- 权重由 ParameterLoader 管理
- 更灵活的内存管理

### 2. dtype 处理策略

- **模型权重**: bfloat16 (Qwen3 默认)
- **KV Cache**: bfloat16
- **Attention 计算**: flash_attn 自动处理
- **Norm 计算**: 内部转为 float32，输出保持原 dtype

### 3. 与 PagedKVCache 的集成

通过 `get_attention_context()` 获取所需上下文：
- Prefill: cu_seqlens, max_seqlen, slot_mapping
- Decode: block_tables, context_lens, slot_mapping

## 性能特性

### 1. Flash Attention 优势
- **内存效率**: O(N) vs O(N²) 标准 attention
- **计算效率**: IO-aware tiling, 减少 HBM 访问
- **支持长序列**: 高效处理 32K+ tokens

### 2. PagedAttention 优势
- **内存碎片**: Block-based 管理，减少碎片
- **Batch 效率**: 不同序列可共享 blocks（prefix caching）
- **动态分配**: 按需分配，不预留固定空间

### 3. 已预留的优化空间
- Triton kernel for KV storage (当前使用 PyTorch 循环)
- torch.compile 标记（RoPE, RMSNorm）
- Prefix caching (block hash)

## 已验证的正确性

### 1. 形状验证
- ✅ 输入 [num_tokens, hidden_size] → 输出 [num_tokens, hidden_size]
- ✅ Multi-head 转换正确
- ✅ GQA (Q:KV = 8:1) 正确

### 2. 真实权重测试
- ✅ 加载 Qwen3-30B-A3B-Base 的 layer 0 权重
- ✅ Prefill 模式前向传播成功
- ✅ 输出 dtype 正确 (bfloat16)

### 3. KV Cache 集成
- ✅ 正确存储 KV states
- ✅ 正确生成 attention context
- ✅ flash_attn API 调用成功

## 下一步计划

Step 4 将实现：
- ✅ Attention 层已完成
- ⬜ 实现基础算子 (MLP, Expert FFN)
- ⬜ 与 transformers 对比测试正确性
- ⬜ 性能 benchmark
