# 文档索引

## 📚 文档导航

本项目包含以下主要文档，请根据需求选择阅读：

### 1️⃣ 快速入门

**[README.md](README.md)** - 项目总览和快速开始
- 适合：首次接触项目的用户
- 内容：
  - ✅ 项目简介和核心特性
  - ✅ 快速安装和使用示例
  - ✅ 性能指标和测试结果
  - ✅ 常见问题解答
- 阅读时间：10-15分钟

---

### 2️⃣ 需求详解

**[REQUIREMENTS.md](REQUIREMENTS.md)** - 详细需求文档
- 适合：需要了解完整需求的开发者/研究者
- 内容：
  - ✅ 项目目标和核心思想
  - ✅ Draft模型定义和路由修改策略
  - ✅ 完整执行流程（Prefill → 主循环 → 输出）
  - ✅ 参数配置和技术约束
  - ✅ 验证要求和使用场景
  - ✅ 设计决策和已知限制
- 阅读时间：30-40分钟

---

### 3️⃣ 实现详解

**[IMPLEMENTATION.md](IMPLEMENTATION.md)** - 详细实现文档
- 适合：需要深入了解实现细节的开发者
- 内容：
  - ✅ 架构概览和系统架构图
  - ✅ 核心组件详解（代码级）
  - ✅ 详细实现说明（含代码示例）
  - ✅ 完整执行流程（含数据流转）
  - ✅ 关键代码解析
  - ✅ 设计决策和测试验证
- 阅读时间：60-90分钟

---

## 🗺️ 阅读路线

### 场景1：快速上手使用
```
README.md (快速开始部分) → 运行 final_test.py
```

### 场景2：理解项目需求
```
README.md → REQUIREMENTS.md → 运行测试验证
```

### 场景3：深入研究实现
```
README.md → REQUIREMENTS.md → IMPLEMENTATION.md → 阅读源码
```

### 场景4：修改或扩展功能
```
REQUIREMENTS.md (需求) → IMPLEMENTATION.md (实现) → 
源码 (model/moe_spec/) → 测试 (test_*.py)
```

---

## 📖 文档结构对比

| 维度 | README.md | REQUIREMENTS.md | IMPLEMENTATION.md |
|------|-----------|-----------------|-------------------|
| **定位** | 快速入门 | 需求说明 | 实现细节 |
| **读者** | 所有用户 | 开发者/研究者 | 深度开发者 |
| **深度** | 浅 | 中 | 深 |
| **代码** | 少量示例 | 伪代码 | 完整代码 |
| **图表** | 简单架构图 | 流程图 | 详细架构图+数据流 |

---

## 🔍 快速查找指南

### 我想了解...

#### 📌 概念和背景
- **什么是投机采样？** → README.md 项目简介
- **为什么要修改路由？** → REQUIREMENTS.md § 1.2 核心思想
- **Draft模型和Verify模型有什么区别？** → REQUIREMENTS.md § 2 Draft模型定义

#### 📌 使用方法
- **如何安装和运行？** → README.md 快速开始
- **如何调整参数？** → README.md § 参数配置
- **有哪些测试脚本？** → README.md § 测试

#### 📌 技术细节
- **路由修改如何实现？** → IMPLEMENTATION.md § 2.2 QwenMOERoutingModifier
- **KV Cache如何管理？** → IMPLEMENTATION.md § 3.2 Draft生成
- **投机采样算法如何工作？** → IMPLEMENTATION.md § 5.2 投机采样核心逻辑
- **为什么要deepcopy KV cache？** → IMPLEMENTATION.md § 6.3

#### 📌 需求规格
- **draft_length为什么是2？** → REQUIREMENTS.md § 8.1
- **为什么排除top-2专家？** → REQUIREMENTS.md § 8.2
- **接受率如何计算？** → REQUIREMENTS.md § 6.2 性能指标

#### 📌 验证和测试
- **如何验证实现正确性？** → REQUIREMENTS.md § 6 验证要求
- **有哪些测试用例？** → REQUIREMENTS.md § 10 测试用例
- **测试结果如何？** → IMPLEMENTATION.md § 7 测试验证

---

## 📂 源码文件导航

### 核心实现
```
model/moe_spec/
├── moe_model.py              # 模型封装
│   ├── MOEModelWrapper       # 原始模型（Verify）
│   └── ModifiedMOEModel      # Draft模型
│
├── moe_routing_modifier.py  # 路由修改
│   └── QwenMOERoutingModifier
│
├── moe_spec_decoder.py       # 投机解码主控
│   └── MOESpecDecoder
│       ├── speculate_decode()    # 主流程
│       ├── _generate_draft()     # Draft生成
│       └── _verify_draft()       # Verify验证
│
└── spec_sampling.py          # 投机采样算法
    └── speculative_sampling()
```

### 测试文件
```
tests/
├── final_test.py                  # 最终集成测试 ⭐ 推荐
├── test_routing_modification.py  # 路由修改验证
├── test_draft_length_2.py         # Draft length=2测试
└── debug_verify_logits.py         # Verify阶段调试
```

---

## 🎯 常见任务快速查找

| 任务 | 查找位置 |
|------|----------|
| 运行第一个示例 | README.md § 基础使用 |
| 理解Draft生成流程 | IMPLEMENTATION.md § 3.2 + § 4.2 |
| 理解Verify验证流程 | IMPLEMENTATION.md § 3.3 + § 4.2 |
| 修改draft_length | IMPLEMENTATION.md § 8.4 + moe_spec_decoder.py:11 |
| 修改排除专家数 | README.md § 参数配置 |
| 查看测试结果 | README.md § 性能指标 |
| 理解KV Cache管理 | IMPLEMENTATION.md § 6.3 |
| 排查问题 | README.md § 常见问题 |
| 添加新模型支持 | IMPLEMENTATION.md § 2.2 + moe_routing_modifier.py |

---

## 📊 文档版本信息

| 文档 | 版本 | 最后更新 | 状态 |
|------|------|----------|------|
| README.md | v2.0 | 2025-10-21 | ✅ 最新 |
| REQUIREMENTS.md | v2.0 | 2025-10-21 | ✅ 最新 |
| IMPLEMENTATION.md | v2.0 | 2025-10-21 | ✅ 最新 |
| DOCS_INDEX.md | v1.0 | 2025-10-21 | ✅ 最新 |

---

## 💡 阅读建议

1. **初次接触**：先读README.md了解概况
2. **深入理解**：阅读REQUIREMENTS.md理解需求
3. **修改代码**：参考IMPLEMENTATION.md和源码
4. **遇到问题**：查阅常见问题或重读相关章节

---

## 📮 反馈

如果发现文档有误或需要补充，请通过以下方式反馈：
- 提交Issue
- 直接修改文档并提交PR
- 联系项目维护者

---

**最后更新**：2025-10-21

