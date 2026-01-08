# 测试和调试脚本目录

本目录包含所有的测试脚本和调试工具。

## 📁 目录结构

### 🎯 主要测试脚本

| 文件 | 描述 | 推荐度 |
|------|------|--------|
| **final_test.py** | 最终集成测试（3个prompt） | ⭐⭐⭐⭐⭐ |
| **test_routing_modification.py** | 验证路由修改是否生效 | ⭐⭐⭐⭐⭐ |
| **test_draft_length_2.py** | 测试draft_length=2的功能 | ⭐⭐⭐⭐ |
| **test_fixed_implementation.py** | 完整实现测试 | ⭐⭐⭐⭐ |

### 🔍 调试脚本

| 文件 | 描述 | 用途 |
|------|------|------|
| **debug_verify_logits.py** | 调试verify阶段的logits | 验证KV cache和logits正确性 |
| **debug_spec_detailed.py** | 详细调试投机采样流程 | 逐步分析每个阶段 |
| **debug_spec_sampling.py** | 调试投机采样算法 | 分析采样逻辑 |

### 🧪 单元测试

| 文件 | 描述 |
|------|------|
| **test_moe_model.py** | MOE模型测试 |
| **test_moe_model_fixed.py** | MOE模型修复后测试 |
| **test_moe_routing_modifier.py** | 路由修改器测试 |
| **test_moe_spec_decoder.py** | 投机解码器测试 |
| **test_spec_sampling.py** | 投机采样算法测试 |

### 🔬 功能测试

| 文件 | 描述 |
|------|------|
| **test_first_token.py** | 第一个token一致性测试 |
| **test_greedy_spec.py** | Greedy模式投机采样测试 |
| **test_modifications.py** | 修改功能测试 |
| **test_multi_round_sampling.py** | 多轮采样测试 |
| **test_original_model.py** | 原始模型测试 |
| **test_verify_model.py** | Verify模型测试 |
| **test_single_prompt.py** | 单个prompt测试 |

### 🔧 测试工具

| 文件 | 描述 |
|------|------|
| **run_tests.py** | 批量运行测试 |
| **conftest.py** | pytest配置文件 |
| **__init__.py** | 包初始化文件 |

## 🚀 快速开始

### 运行推荐测试

**方式1: 从tests目录运行（推荐）**
```bash
# 进入测试目录
cd tests

# 1. 最终集成测试（推荐首先运行）
conda activate moe_spec
python final_test.py

# 2. 验证路由修改
python test_routing_modification.py

# 3. 测试draft_length=2
python test_draft_length_2.py
```

**方式2: 从项目根目录运行**
```bash
# 在项目根目录
conda activate moe_spec
python tests/final_test.py
python tests/test_routing_modification.py
python tests/test_draft_length_2.py
```

**注意**: 所有测试文件已经包含了正确的路径设置，可以从tests目录直接运行 ✅

### 运行单元测试

```bash
# 运行所有单元测试
pytest test_moe_*.py test_spec_*.py

# 运行特定测试
pytest test_moe_model.py -v
```

### 调试问题

```bash
# 调试KV cache和logits
python debug_verify_logits.py

# 详细分析投机采样
python debug_spec_detailed.py
```

## 📊 测试覆盖

### 功能测试
- ✅ 路由修改生效验证
- ✅ KV cache隔离验证
- ✅ 第一个token一致性验证
- ✅ Draft长度正确性验证
- ✅ 接受率计算验证

### 性能测试
- ✅ 生成速度测试
- ✅ 接受率分析
- ✅ 多prompt测试

### 正确性测试
- ✅ Draft模型输出与原始模型对比
- ✅ Verify模型正确性验证
- ✅ 投机采样算法验证

## 🐛 问题排查

如果遇到问题，按以下顺序运行测试：

1. **test_routing_modification.py** - 验证路由修改
2. **debug_verify_logits.py** - 检查KV cache和logits
3. **test_first_token.py** - 验证第一个token
4. **final_test.py** - 完整流程测试

## 📝 添加新测试

新测试文件命名规范：
- `test_*.py` - 功能测试
- `debug_*.py` - 调试脚本
- 使用清晰的文件名描述测试内容

## 🔗 相关文档

- [主文档](../README.md)
- [实现文档](../docs/IMPLEMENTATION.md)
- [需求文档](../docs/REQUIREMENTS.md)

