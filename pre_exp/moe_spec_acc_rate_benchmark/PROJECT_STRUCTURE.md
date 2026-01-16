# 项目目录结构

```
moe_spec_acc_rate_benchmark/
│
├── 📄 README.md                    # 项目主文档（快速入门）
├── 📄 PROJECT_STRUCTURE.md         # 本文件（目录结构说明）
├── 📄 requirements.txt             # Python依赖
│
├── 📁 docs/                        # 📚 文档目录
│   ├── README.md                   # 文档目录说明
│   ├── DOCS_INDEX.md               # 文档导航索引
│   ├── REQUIREMENTS.md             # 详细需求文档
│   ├── IMPLEMENTATION.md           # 详细实现文档
│   └── CHANGELOG.md                # 版本修改日志
│
├── 📁 model/                       # 🧠 核心模型代码
│   ├── __init__.py
│   └── moe_spec/                   # MoE投机采样实现
│       ├── __init__.py
│       ├── moe_model.py            # 模型封装（原始+Draft）
│       ├── moe_routing_modifier.py # 路由修改实现
│       ├── moe_spec_decoder.py     # 投机解码主控
│       └── spec_sampling.py        # 投机采样算法
│
├── 📁 tests/                       # 🧪 测试和调试目录
│   ├── README.md                   # 测试目录说明
│   │
│   ├── # 🎯 主要测试
│   ├── final_test.py               # 最终集成测试 ⭐⭐⭐⭐⭐
│   ├── test_routing_modification.py# 路由修改验证 ⭐⭐⭐⭐⭐
│   ├── test_draft_length_2.py      # Draft length=2测试
│   ├── test_fixed_implementation.py# 完整实现测试
│   │
│   ├── # 🔍 调试脚本
│   ├── debug_verify_logits.py      # 调试verify阶段
│   ├── debug_spec_detailed.py      # 详细调试流程
│   ├── debug_spec_sampling.py      # 调试采样算法
│   │
│   ├── # 🧪 单元测试
│   ├── test_moe_model.py           # MOE模型测试
│   ├── test_moe_model_fixed.py     # MOE模型修复测试
│   ├── test_moe_routing_modifier.py# 路由修改器测试
│   ├── test_moe_spec_decoder.py    # 投机解码器测试
│   ├── test_spec_sampling.py       # 投机采样测试
│   │
│   ├── # 🔬 功能测试
│   ├── test_first_token.py         # 第一token测试
│   ├── test_greedy_spec.py         # Greedy模式测试
│   ├── test_modifications.py       # 修改功能测试
│   ├── test_multi_round_sampling.py# 多轮采样测试
│   ├── test_original_model.py      # 原始模型测试
│   ├── test_verify_model.py        # Verify模型测试
│   ├── test_single_prompt.py       # 单prompt测试
│   │
│   ├── # 🔧 测试工具
│   ├── run_tests.py                # 批量运行测试
│   ├── conftest.py                 # pytest配置
│   └── __init__.py
│
├── 📁 config/                      # ⚙️ 配置文件
│   ├── __init__.py
│   └── moe_spec_config.py          # MoE投机采样配置
│
├── 📁 evaluation/                  # 📊 评估脚本
│   ├── __init__.py
│   ├── eval.py                     # 评估主脚本
│   ├── equal.py                    # 一致性评估
│   ├── speed.py                    # 速度评估
│   ├── inference_moe_spec.py       # MoE投机推理
│   └── inference_sps.py            # SPS推理
│
├── 📁 scripts/                     # 🛠️ 工具脚本
│   ├── eval_moe_spec.py            # 评估脚本
│   └── test_moe_spec.py            # 测试脚本
│
├── 📁 data/                        # 📦 数据目录
│   └── spec_bench/                 # SpecBench数据集
│       └── question.jsonl
│
└── 📁 results/                     # 📈 结果输出
    └── model_answer/               # 模型答案
```

## 📂 目录说明

### 核心目录

#### 📚 docs/ - 文档目录
- **作用**: 存放所有详细文档
- **内容**: 需求文档、实现文档、修改日志、文档导航
- **适合**: 需要深入了解项目的开发者

#### 🧠 model/ - 核心代码
- **作用**: 项目的核心实现代码
- **内容**: 模型封装、路由修改、投机解码、投机采样
- **适合**: 需要修改或扩展功能的开发者

#### 🧪 tests/ - 测试目录
- **作用**: 所有测试和调试脚本
- **内容**: 集成测试、单元测试、调试脚本、功能测试
- **适合**: 验证功能、排查问题

### 辅助目录

#### ⚙️ config/ - 配置目录
- **作用**: 配置文件管理
- **内容**: 模型配置、参数配置

#### 📊 evaluation/ - 评估目录
- **作用**: 性能评估和对比
- **内容**: 评估脚本、推理脚本

#### 🛠️ scripts/ - 工具脚本
- **作用**: 实用工具脚本
- **内容**: 评估工具、测试工具

#### 📦 data/ - 数据目录
- **作用**: 存放数据集
- **内容**: SpecBench等数据集

#### 📈 results/ - 结果目录
- **作用**: 存放运行结果
- **内容**: 模型答案、评估结果

## 🚀 快速导航

### 我想...

#### 📖 了解项目
```
1. README.md - 快速了解（10分钟）
2. docs/REQUIREMENTS.md - 详细需求（30分钟）
3. docs/IMPLEMENTATION.md - 实现细节（60分钟）
```

#### 🏃 运行测试
```
cd tests/
python final_test.py                 # 主要测试
python test_routing_modification.py # 验证路由
```

#### 🔧 修改代码
```
1. 阅读 docs/IMPLEMENTATION.md
2. 修改 model/moe_spec/*.py
3. 运行 tests/ 中的相关测试
```

#### 🐛 排查问题
```
1. 查看 docs/DOCS_INDEX.md 快速定位
2. 运行 tests/debug_*.py 调试
3. 参考 docs/IMPLEMENTATION.md 相关章节
```

#### 📊 评估性能
```
cd evaluation/
python eval.py  # 运行评估
```

## 📝 文件命名规范

### 测试文件
- `test_*.py` - 功能测试脚本
- `debug_*.py` - 调试脚本
- `final_test.py` - 主测试脚本

### 文档文件
- `*.md` - Markdown文档
- 大写命名（如 `README.md`）

### 源码文件
- 小写+下划线（如 `moe_model.py`）
- 清晰描述功能

## 🎯 主要文件说明

### 根目录文件

| 文件 | 描述 | 重要度 |
|------|------|--------|
| `README.md` | 项目主文档 | ⭐⭐⭐⭐⭐ |
| `PROJECT_STRUCTURE.md` | 目录结构说明（本文件） | ⭐⭐⭐⭐ |
| `requirements.txt` | Python依赖 | ⭐⭐⭐⭐⭐ |

### 核心实现文件

| 文件 | 描述 | 重要度 |
|------|------|--------|
| `model/moe_spec/moe_model.py` | 模型封装 | ⭐⭐⭐⭐⭐ |
| `model/moe_spec/moe_routing_modifier.py` | 路由修改 | ⭐⭐⭐⭐⭐ |
| `model/moe_spec/moe_spec_decoder.py` | 投机解码 | ⭐⭐⭐⭐⭐ |
| `model/moe_spec/spec_sampling.py` | 投机采样 | ⭐⭐⭐⭐⭐ |

### 关键文档

| 文件 | 描述 | 重要度 |
|------|------|--------|
| `docs/DOCS_INDEX.md` | 文档导航 | ⭐⭐⭐⭐⭐ |
| `docs/REQUIREMENTS.md` | 需求文档 | ⭐⭐⭐⭐⭐ |
| `docs/IMPLEMENTATION.md` | 实现文档 | ⭐⭐⭐⭐⭐ |
| `docs/CHANGELOG.md` | 修改日志 | ⭐⭐⭐⭐ |

### 重要测试

| 文件 | 描述 | 重要度 |
|------|------|--------|
| `tests/final_test.py` | 最终集成测试 | ⭐⭐⭐⭐⭐ |
| `tests/test_routing_modification.py` | 路由修改验证 | ⭐⭐⭐⭐⭐ |
| `tests/debug_verify_logits.py` | Verify调试 | ⭐⭐⭐⭐ |

## 🔗 目录间关系

```
README.md ─────┬───> docs/ (详细文档)
               ├───> model/ (核心代码)
               ├───> tests/ (测试验证)
               └───> config/ (配置)

docs/ ─────────┬───> 指导阅读
               └───> 帮助理解

model/ ────────┬───> 被 tests/ 测试
               └───> 被 evaluation/ 评估

tests/ ────────┬───> 验证 model/ 功能
               └───> 帮助 排查问题

evaluation/ ───┬───> 使用 model/ 评估
               └───> 生成 results/
```

## 💡 使用建议

### 初学者
1. 从 `README.md` 开始
2. 运行 `tests/final_test.py`
3. 阅读 `docs/REQUIREMENTS.md`

### 开发者
1. 阅读全部文档（`docs/`）
2. 研究核心代码（`model/moe_spec/`）
3. 运行所有测试（`tests/`）
4. 修改并验证

### 维护者
1. 熟悉所有目录和文件
2. 保持文档更新（`docs/`）
3. 添加测试覆盖（`tests/`）
4. 更新 `CHANGELOG.md`

## 📮 反馈

如果目录结构需要调整或文件位置不合理，请：
- 提交Issue
- 联系项目维护者
- 提交PR改进

---

**版本**: v1.0  
**最后更新**: 2025-10-21  
**维护者**: Claude (Anthropic)


