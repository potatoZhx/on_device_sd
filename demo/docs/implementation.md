**项目概述**
- 目标是做一个“MoE CPU‑GPU 协同推理框架”，支持标准自回归与 Draft‑Verify（推测解码），并具备批处理、专家缓存、预取与异步传输等能力。架构在 [arch.md](file:///zx_data1/sparsity/on_device_sd/demo/arch.md) 中用流程图明确描述。
- 当前代码是“框架骨架 + 部分算子与流程”，关键算子和核心正确性仍未补齐。README 几乎为空（仅模型路径）[README.md](file:///zx_data1/sparsity/on_device_sd/demo/README.md)。

**现有架构与模块划分（职责与交互）**
- API 层：统一入口与批处理接口  
  [inference.py](file:///zx_data1/sparsity/on_device_sd/demo/src/api/inference.py)  
  - 初始化各子系统（参数加载、缓存、预取、调度、引擎）。  
  - `generate/submit/get_result` 负责同步/异步推理入口。  
  - Tokenizer 目前是占位实现。
- 统筹调度层（Orchestrator）：模式路由与批处理  
  [orchestrator.py](file:///zx_data1/sparsity/on_device_sd/demo/src/execution/orchestrator.py)  
  - 标准解码与推测解码分流  
  - 批处理管理、结果汇总  
- 执行引擎层  
  - Prefill：全量前向，填充 KV Cache  
    [prefill_engine.py](file:///zx_data1/sparsity/on_device_sd/demo/src/execution/prefill_engine.py)  
  - Draft：CPU‑GPU 替代专家推测解码  
    [draft_engine.py](file:///zx_data1/sparsity/on_device_sd/demo/src/execution/draft_engine.py)  
  - Verify：复用 Prefill 进行校验  
    [verify_engine.py](file:///zx_data1/sparsity/on_device_sd/demo/src/execution/verify_engine.py)  
  - Standard：标准自回归批推理  
    [standard_engine.py](file:///zx_data1/sparsity/on_device_sd/demo/src/execution/standard_engine.py)  
- 内存管理层  
  - 参数加载：支持 Safetensors、CPU/GPU 位置管理  
    [parameter_loader.py](file:///zx_data1/sparsity/on_device_sd/demo/src/memory/parameter_loader.py)  
  - 专家缓存：GPU 缓存与替换策略  
    [expert_cache.py](file:///zx_data1/sparsity/on_device_sd/demo/src/memory/expert_cache.py)  
  - KV Cache：普通与分页版本  
    [kv_cache.py](file:///zx_data1/sparsity/on_device_sd/demo/src/memory/kv_cache.py), [paged_kv_cache.py](file:///zx_data1/sparsity/on_device_sd/demo/src/memory/paged_kv_cache.py)  
- 调度与预取  
  - Draft 调度器：替代策略、触发验证条件  
    [draft_schduler.py](file:///zx_data1/sparsity/on_device_sd/demo/src/scheduling/draft_schduler.py)  
  - Cache 策略（LRU/LFU/自适应/预测）  
    [cache_strategy.py](file:///zx_data1/sparsity/on_device_sd/demo/src/scheduling/cache_strategy.py)  
  - Prefetcher  
    [prefetcher.py](file:///zx_data1/sparsity/on_device_sd/demo/src/scheduling/prefetcher.py)  
- 算子层  
  - GPU/CPU 基础算子、异步传输  
    [gpu_operators.py](file:///zx_data1/sparsity/on_device_sd/demo/src/operators/gpu_operators.py), [cpu_operators.py](file:///zx_data1/sparsity/on_device_sd/demo/src/operators/cpu_operators.py), [transfer_ops.py](file:///zx_data1/sparsity/on_device_sd/demo/src/operators/transfer_ops.py)  
- 配置与测试  
  - 模型/推理配置 [model_configs.yaml](file:///zx_data1/sparsity/on_device_sd/demo/configs/model_configs.yaml), [inference_config.yaml](file:///zx_data1/sparsity/on_device_sd/demo/configs/inference_config.yaml)  
  - 测试覆盖说明 [tests.md](file:///zx_data1/sparsity/on_device_sd/demo/tests/tests.md)

**当前已实现能力（可复用基础）**
- MoE 权重加载与 CPU/GPU 放置、shared expert 常驻 GPU  
  [parameter_loader.py](file:///zx_data1/sparsity/on_device_sd/demo/src/memory/parameter_loader.py)  
- 预取与缓存替换策略框架（策略可切换）  
  [prefetcher.py](file:///zx_data1/sparsity/on_device_sd/demo/src/scheduling/prefetcher.py), [cache_strategy.py](file:///zx_data1/sparsity/on_device_sd/demo/src/scheduling/cache_strategy.py)  
- 推理流程骨架（prefill/draft/verify/standard）  
  [prefill_engine.py](file:///zx_data1/sparsity/on_device_sd/demo/src/execution/prefill_engine.py), [draft_engine.py](file:///zx_data1/sparsity/on_device_sd/demo/src/execution/draft_engine.py), [verify_engine.py](file:///zx_data1/sparsity/on_device_sd/demo/src/execution/verify_engine.py), [standard_engine.py](file:///zx_data1/sparsity/on_device_sd/demo/src/execution/standard_engine.py)  
- Batch Manager 与异步请求通路  
  [batch_manager.py](file:///zx_data1/sparsity/on_device_sd/demo/src/execution/batch_manager.py), [inference.py](file:///zx_data1/sparsity/on_device_sd/demo/src/api/inference.py)

**关键缺口与已暴露问题（对 Qwen3‑30B‑A3B‑Base 前向推理有影响）**
- 算子正确性与完整性缺失  
  - GPU Self‑Attention 实现存在明显语法错误与 RoPE 为空 [gpu_operators.py](file:///zx_data1/sparsity/on_device_sd/demo/src/operators/gpu_operators.py)  
  - LayerNorm 与 Qwen3 实际为 RMSNorm，且权重命名/形态需对齐  
  - 注意力为 GQA（Qwen3 使用 num_key_value_heads），当前实现仅按 num_attention_heads 处理
- KV Cache 设计不匹配  
  - KVCache 使用固定 full‑seq 预分配，且 draft/verify 逻辑占位 [kv_cache.py](file:///zx_data1/sparsity/on_device_sd/demo/src/memory/kv_cache.py)  
  - PagedKVCache 更接近目标但未接入主流程 [paged_kv_cache.py](file:///zx_data1/sparsity/on_device_sd/demo/src/memory/paged_kv_cache.py)
- 标准引擎/Prefill/草案流程内部存在明显代码缺陷  
  - 多处拼写/语法错误（如 `defdef`，重复参数）  
  - 缺少 DeviceType import 等  
  - 这类问题必须先修复才能验证数值正确性  
  相关文件：[prefill_engine.py](file:///zx_data1/sparsity/on_device_sd/demo/src/execution/prefill_engine.py), [kv_cache.py](file:///zx_data1/sparsity/on_device_sd/demo/src/memory/kv_cache.py)
- Tokenizer 与 HF 对齐缺失  
  - 当前 `_tokenize` 是随机占位 [inference.py](file:///zx_data1/sparsity/on_device_sd/demo/src/api/inference.py)
- 配置文件命名不一致  
  - 代码默认 `model_config.yaml`，目录为 `model_configs.yaml` [config.py](file:///zx_data1/sparsity/on_device_sd/demo/src/utils/config.py), [configs](file:///zx_data1/sparsity/on_device_sd/demo/configs/model_configs.yaml)

---

## 详细实现方案（分阶段，先正确性、后性能）
以下方案以“先单卡 GPU 正确前向 + 对齐 transformers，再引入 CPU‑GPU 协同与优化”为主线。

**阶段 0：基线与约束对齐**
- 明确 Qwen3‑30B‑A3B‑Base 的 HF 配置字段与权重命名规则  
  - `num_hidden_layers / num_attention_heads / num_key_value_heads / rope_theta / rms_norm_eps / moe_intermediate_size / num_experts` 等  
- 明确目标对齐标准：输出 logits 与 transformers 完全对齐  
- 明确默认执行路径：先 standard/prefill 走全 GPU 的准确实现

**阶段 1：模型加载与权重映射对齐**
- 以现有 `ParameterLoader` 为核心，不重写加载逻辑  
  - 调整权重映射表：embed、qkv/o_proj、rmsnorm、router、experts gate/up/down、lm_head  
  - 核对 Qwen3 的权重命名与当前 `static_params_gpu` 索引键  
- 结果：能成功加载 Qwen3 的所有权重，并能按层访问

**阶段 2：核心算子清单与实现顺序**
必需算子（按依赖顺序）  
1) Embedding  
2) RMSNorm（替换 layernorm）  
3) Rotary Position Embedding（RoPE）  
4) Attention with GQA + KV Cache  
5) Router（线性投影 + top‑k + softmax）  
6) MoE FFN（SwiGLU: gate * up → down）  
7) Residual / Add  
8) LM Head  

实现策略建议  
- 优先以 transformers 对齐的数值为准：  
  - 先用 PyTorch 直接实现、对齐精度，再逐步替换为优化版本  
- 复用库的场景  
  - RMSNorm、RoPE、GQA Attention：可对齐 transformers 的实现逻辑  
- 参考实现  
  - KV Cache / paged attention 可以参考 nano‑vllm / vLLM 结构  
- 从头实现  
  - MoE 路由 + top‑k + token‑expert gather/scatter  

**阶段 3：算子正确性单元测试（每个算子独立对齐 transformers）**
- 对齐方法：相同随机输入，对比差值  
- 精度标准：FP32 `atol=1e-5, rtol=1e-5`，BF16/FP16 可放宽  
- 覆盖维度：  
  - 典型形状、空序列、超长序列  
  - CPU/GPU 一致性  
- 测试位置：对齐现有 tests 结构 [tests.md](file:///zx_data1/sparsity/on_device_sd/demo/tests/tests.md)

**阶段 4：模型前向集成**
- 在 Prefill/Standard 中集成算子：  
  - input_ids → embedding → N 层（attn + moe）→ final rmsnorm → lm_head  
- KV Cache 先用固定实现保证正确性  
- Draft/Verify 先跑通正确性后再性能优化

**阶段 5：端到端对齐测试**
- 使用 Qwen3‑30B‑A3B‑Base 权重  
- 与 transformers 同输入 prompt，比较 logits / next token  
- 输出一致性作为 gate

**阶段 6：CPU‑GPU 协同与优化**
- 在正确性基础上启用：  
  - 专家 CPU 执行与 GPU 缓存  
  - Prefetch + Cache 策略  
- 引入分页 KV Cache 与 FlashAttention  
- 记录性能基线（tokens/s, cache hit rate, transfer time）

---

## 算子实现策略细化（针对 Qwen3‑30B‑A3B‑Base）
- Embedding / LM Head  
  - 直接 `torch.nn.functional.embedding/linear`，要求权重布局对齐
- RMSNorm  
  - 需替换现有 LayerNorm 逻辑 [gpu_operators.py](file:///zx_data1/sparsity/on_device_sd/demo/src/operators/gpu_operators.py)  
  - 与 transformers 的 RMSNorm 实现一一对齐  
- RoPE  
  - 必须实现位置编码  
  - 对齐 Qwen3 的 `rope_theta` 与 `max_position_embeddings`  
- Attention (GQA + KV Cache)  
  - 正确拆分 Q/K/V，支持 num_key_value_heads  
  - KV cache append / read 逻辑必须与 transformers 一致  
- Router  
  - Linear → top‑k → softmax  
  - output 的 top‑k 与概率必须与 transformers 对齐  
- MoE Experts (SwiGLU)  
  - gate_proj/up_proj/down_proj  
  - 重点是 token‑expert 路由与 scatter/gather 精度  
- CPU‑GPU 协同  
  - CPU 执行部分 expert，GPU 缓存的 expert 优先  
  - 传输策略以 ExpertCache + TransferManager 为核心 [expert_cache.py](file:///zx_data1/sparsity/on_device_sd/demo/src/memory/expert_cache.py), [transfer_ops.py](file:///zx_data1/sparsity/on_device_sd/demo/src/operators/transfer_ops.py)

---

## 测试与验证计划
**单元测试（算子级）**
- 每个算子建立对齐测试  
- 输入一致性 + 输出数值误差对齐  
- 覆盖 CPU/GPU 与边界形状

**集成测试（模块级）**
- 单层前向 + KV cache  
- MoE 路由 + experts 组合输出

**端到端测试**
- 输入 prompt → logits 对齐  
- 输出 token 对齐  
- 记录时间与缓存命中率

---

## 潜在挑战与对策
- 权重命名与模型配置差异  
  - 策略：显式 mapping 表 + 断言校验  
- KV Cache 与 GQA 正确性  
  - 先对齐 transformers 的基础 KV cache，再引入 paged 版本  
- MoE 路由的 token‑expert mapping  
  - 先保证数值正确，再考虑并行/融合优化  
- CPU‑GPU 协同引入不稳定  
  - 先跑全 GPU 正确前向，再逐步引入 CPU 执行与缓存策略  

---

## 建议的实现顺序（模块级）
1) 权重映射对齐（ParameterLoader + config）  
2) RMSNorm + RoPE + GQA Attention  
3) MoE 路由 + Experts  
4) Prefill/Standard 完整前向  
5) 端到端对齐测试  
6) Draft/Verify  
7) CPU‑GPU 协同优化 + Cache/Prefetch  
8) 性能基线与 profiling  

---

如果你确认这份方案，我将按该方案推进：先修复基础算子与权重对齐，再实现完整前向与测试对齐，最后引入 CPU‑GPU 协同优化。