# MoE投机采样测试框架

> 评估MoE模型精度对路由专家选择的敏感度

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.0+-yellow.svg)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 项目简介

本项目实现了一个测试框架，用于评估MoE（Mixture of Experts）模型对路由策略的敏感度。通过修改路由策略生成draft tokens，并与原始模型进行投机解码（Speculative Decoding），计算接受率（Acceptance Rate）来量化路由修改的影响。

### 核心特性

- ✅ **Draft模型**：排除top-2专家的修改路由策略
- ✅ **Verify模型**：使用原始路由策略验证
- ✅ **投机采样**：基于拒绝采样的token选择算法
- ✅ **KV Cache管理**：正确隔离draft和verify的cache
- ✅ **灵活配置**：支持多种参数调整

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
PyTorch >= 2.0
transformers >= 4.35.0
CUDA >= 11.7
```

### 安装

```bash
# 激活conda环境
conda activate moe_spec

# 安装依赖（如已安装可跳过）
pip install torch transformers accelerate
```

### 基础使用

```python
from model.moe_spec.moe_model import MOEModelWrapper, ModifiedMOEModel
from model.moe_spec.moe_spec_decoder import MOESpecDecoder

# 1. 加载模型
model_path = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"
original_model = MOEModelWrapper(model_path, device="cuda", dtype="float16")
modified_model = ModifiedMOEModel(original_model, top_k_experts_to_remove=2)

# 2. 创建解码器
spec_decoder = MOESpecDecoder(original_model, modified_model)

# 3. 生成文本
tokenizer = original_model.tokenizer
prompt = "你好，请介绍一下北京"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

result = spec_decoder.speculate_decode(input_ids, max_new_tokens=100)

# 4. 查看结果
print(f"生成文本: {tokenizer.decode(result['output_ids'][0])}")
print(f"接受率: {result['acceptance_rate']:.2%}")
print(f"每步接受长度: {result['accept_length_list']}")
```

### 运行测试

```bash
# 进入测试目录
cd tests

# 完整功能测试
python final_test.py

# 路由修改验证
python test_routing_modification.py

# Draft length=2测试
python test_draft_length_2.py
```

## 📊 技术架构

### 系统架构

```
┌─────────────────────────────────────────┐
│         MOESpecDecoder                   │
│  ┌─────────────┐    ┌──────────────┐   │
│  │   Prefill   │───>│  Main Loop   │   │
│  └─────────────┘    └──────┬───────┘   │
│                             │            │
│                    ┌────────┴────────┐  │
│                    │                 │  │
│             ┌──────▼─────┐  ┌───────▼────┐
│             │   Draft    │  │   Verify   │
│             │ (modified) │  │ (original) │
│             └────────────┘  └────────────┘
└─────────────────────────────────────────┘
```

### 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| MOEModelWrapper | `moe_model.py` | 原始模型封装 |
| ModifiedMOEModel | `moe_model.py` | Draft模型封装 |
| QwenMOERoutingModifier | `moe_routing_modifier.py` | 路由修改实现 |
| MOESpecDecoder | `moe_spec_decoder.py` | 投机解码主控 |
| speculative_sampling | `spec_sampling.py` | 投机采样算法 |

## 🔧 参数配置

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `draft_length` | 2 | 每次draft生成的token数 |
| `top_k_experts_to_remove` | 2 | 排除的top专家数量 |
| `max_new_tokens` | 100 | 最大生成token数 |
| `dtype` | float16 | 模型精度 |

### 示例配置

```python
# 调整draft长度（需修改源码）
spec_decoder.draft_length = 3

# 调整排除专家数
modified_model = ModifiedMOEModel(
    original_model, 
    top_k_experts_to_remove=3  # 排除top-3
)

# 调整生成长度
result = spec_decoder.speculate_decode(
    input_ids, 
    max_new_tokens=200
)
```

## 📈 性能指标

### 测试结果

| Prompt | 接受率 | 每步接受长度 | 生成速度 |
|--------|--------|--------------|----------|
| "你好，请介绍一下北京" | 50.00% | [2, 0, 1] | 2.57 tokens/s |
| "什么是人工智能？" | 25.00% | [1, 1, 0, 0] | 2.95 tokens/s |
| "请简单介绍一下Python" | 100.00% | [2] | 3.98 tokens/s |

### 验证指标

- ✅ **第一个token一致性**：100%
- ✅ **Draft长度正确性**：`total_draft = step × 2`
- ✅ **接受长度范围**：`0 ≤ n ≤ 2`
- ✅ **路由修改生效**：Draft输出 ≠ 原始输出

## 📖 文档导航

- [项目结构](PROJECT_STRUCTURE.md) - 目录结构说明 📂
- [文档导航](docs/DOCS_INDEX.md) - 快速找到需要的信息 ⭐
- [需求文档](docs/REQUIREMENTS.md) - 详细的需求说明和技术约束
- [实现文档](docs/IMPLEMENTATION.md) - 完整的实现细节和代码解析
- [修改日志](docs/CHANGELOG.md) - 版本变更和升级指南
- [测试说明](tests/README.md) - 测试脚本使用指南

## 🔍 核心实现

### Draft生成（draft_length=2）

```python
def _generate_draft(self, last_token: int, verify_kv_cache: Dict):
    """生成2个draft tokens"""
    # 1. 深拷贝KV cache，避免污染verify cache
    current_kv_cache = copy.deepcopy(verify_kv_cache)
    
    # 2. 启用路由修改
    self.modified_model.routing_modifier.enable_routing_modification(...)
    
    # 3. 循环生成2个tokens
    for i in range(2):
        outputs = self.modified_model.model(
            input_token,
            past_key_values=current_kv_cache,
            use_cache=True
        )
        next_token = torch.argmax(outputs.logits, dim=-1).item()
        draft_tokens.append(next_token)
        current_kv_cache = outputs.past_key_values  # 累积
    
    # 4. 返回draft tokens，丢弃draft kv cache
    return draft_tokens, draft_logits
```

### 路由修改核心

```python
def modified_forward(hidden_states):
    # 1. 计算原始路由权重
    routing_weights = F.softmax(gate(hidden_states), dim=1)
    
    # 2. 找到top-2专家并屏蔽
    _, top_2_indices = torch.topk(routing_weights, 2, dim=-1)
    masked_weights = routing_weights.clone()
    masked_weights.scatter_(-1, top_2_indices, -1e9)
    
    # 3. 从剩余专家中选择top-k
    _, selected_experts = torch.topk(masked_weights, top_k, dim=-1)
    
    # 4. 使用原始权重执行专家计算
    routing_weights = routing_weights[batch_idx, selected_experts]
    final_hidden_states = expert_forward(hidden_states, selected_experts, routing_weights)
    
    return final_hidden_states
```

## 🧪 测试

### 测试文件

```bash
# 进入测试目录
cd tests

# 路由修改测试
python test_routing_modification.py
# 验证：Draft模型输出 ≠ 原始模型输出

# KV Cache隔离测试
python debug_verify_logits.py
# 验证：Draft前后verify cache长度不变

# 完整流程测试
python final_test.py
# 验证：所有指标正常
```

### 预期结果

```
✓ Draft length正确设置为2
✓ Draft模型使用修改后的路由（排除top-2 experts）
✓ Verify模型使用原始路由
✓ 接受率计算正确
✓ KV cache正确管理（不会被draft修改）
```

## 🐛 常见问题

### Q: 投机采样输出与原始模型不一致？
**A**: 这是正常现象。投机采样使用随机采样（`torch.multinomial`），而不是greedy，因此输出会有差异。关键是验证接受率等统计指标是否正确。

### Q: 接受率100%是否异常？
**A**: 不一定。对于某些prompt，即使排除top-2专家，剩余专家的预测也可能与原始模型一致，导致高接受率。

### Q: KV cache为什么要deepcopy？
**A**: Draft阶段会修改KV cache。如果不deepcopy，会污染verify的cache，导致错误结果。

### Q: Draft length可以改为其他值吗？
**A**: 可以。修改`moe_spec_decoder.py`中的`self.draft_length = 2`即可。但需要注意，过大的draft_length会降低接受率。

## 📝 版本历史

### v2.0 (2025-10-21)
- ✅ 将draft_length从1改为2
- ✅ 重构`_generate_draft`方法
- ✅ 重构`_verify_draft`方法
- ✅ 修复KV cache污染问题
- ✅ 添加完整的测试套件
- ✅ 完善文档

### v1.0 (初始版本)
- ✅ 实现基础投机采样框架
- ✅ draft_length=1
- ✅ 支持Qwen3模型

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 👥 联系方式

项目维护者：Claude (Anthropic)

---

**注意**：本项目专为研究和测试目的设计，请勿用于生产环境。

