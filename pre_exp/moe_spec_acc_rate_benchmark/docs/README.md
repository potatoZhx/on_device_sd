# 项目文档目录

本目录包含项目的所有详细文档。

## 📚 文档清单

| 文档 | 大小 | 描述 | 适合读者 |
|------|------|------|----------|
| **[DOCS_INDEX.md](DOCS_INDEX.md)** | 5.8KB | 📑 文档导航索引 | 所有人 |
| **[REQUIREMENTS.md](REQUIREMENTS.md)** | 7.6KB | 📋 详细需求文档 | 开发者/研究者 |
| **[IMPLEMENTATION.md](IMPLEMENTATION.md)** | 43KB | 🔧 详细实现文档 | 深度开发者 |
| **[CHANGELOG.md](CHANGELOG.md)** | 12KB | 📝 版本修改日志 | 维护者/开发者 |

## 🗺️ 推荐阅读路线

### 👶 初次接触
```
1. ../README.md (主文档，10分钟)
2. REQUIREMENTS.md (需求概览，20分钟)
```

### 🎓 深入学习
```
1. ../README.md (主文档)
2. REQUIREMENTS.md (完整需求，30分钟)
3. IMPLEMENTATION.md (实现细节，60分钟)
4. 源码阅读 (model/moe_spec/)
```

### 🔧 开发维护
```
1. REQUIREMENTS.md (需求规格)
2. IMPLEMENTATION.md (实现细节)
3. CHANGELOG.md (版本历史)
4. 源码修改
```

### 🐛 问题排查
```
1. DOCS_INDEX.md (快速查找)
2. ../README.md § 常见问题
3. IMPLEMENTATION.md (相关章节)
```

## 📖 各文档详解

### DOCS_INDEX.md - 文档导航
**用途**: 快速找到需要的信息

**内容**:
- ✅ 文档概览和对比
- ✅ 阅读路线建议
- ✅ 快速查找指南（按主题分类）
- ✅ 源码文件导航
- ✅ 常见任务定位表

**何时使用**: 当你不知道该看哪个文档时

---

### REQUIREMENTS.md - 需求文档
**用途**: 理解项目完整需求和技术规格

**内容**:
- ✅ 项目目标和核心思想
- ✅ Draft模型定义（排除top-2专家）
- ✅ 完整执行流程（Prefill → 主循环 → 输出）
- ✅ 参数配置说明
- ✅ 技术约束（KV Cache、模型状态）
- ✅ 验证要求（一致性、正确性）
- ✅ 设计决策理由
- ✅ 测试用例

**何时使用**: 需要了解"为什么这样设计"

---

### IMPLEMENTATION.md - 实现文档
**用途**: 理解代码实现的每个细节

**内容**:
- ✅ 系统架构图和模块关系
- ✅ 核心组件详解（含完整代码）:
  - MOEModelWrapper
  - QwenMOERoutingModifier
  - ModifiedMOEModel
  - MOESpecDecoder
- ✅ 主流程实现（`speculate_decode`）
- ✅ Draft生成实现（`_generate_draft`）
- ✅ Verify验证实现（`_verify_draft`）
- ✅ 执行流程图和数据流转示例
- ✅ 关键代码解析（KV cache、投机采样）
- ✅ 设计决策分析
- ✅ 测试验证结果

**何时使用**: 需要修改代码或深入理解实现

**特色**:
- 📊 包含详细的架构图和流程图
- 💻 完整的代码示例和注释
- 📈 数据流转的详细示例
- 🎯 设计决策的理由说明

---

### CHANGELOG.md - 修改日志
**用途**: 了解版本变更和升级指南

**内容**:
- ✅ v2.0详细修改（draft_length: 1→2）
- ✅ 代码修改前后对比
- ✅ 修复的关键问题
- ✅ 性能对比（v1.0 vs v2.0）
- ✅ 新增功能和测试
- ✅ 迁移指南
- ✅ 未来计划

**何时使用**: 
- 从旧版本升级
- 了解最新改动
- 查看修复的问题

---

## 🔍 快速查找

### 我想了解...

#### 📌 概念和原理
- **什么是投机采样?** → ../README.md, REQUIREMENTS.md § 1
- **为什么要修改路由?** → REQUIREMENTS.md § 1.2
- **Draft和Verify有什么区别?** → REQUIREMENTS.md § 2

#### 📌 实现细节
- **路由修改如何实现?** → IMPLEMENTATION.md § 2.2
- **KV cache如何管理?** → IMPLEMENTATION.md § 3.2, 6.3
- **投机采样算法怎么工作?** → IMPLEMENTATION.md § 5.2
- **数据如何流转?** → IMPLEMENTATION.md § 4.2

#### 📌 使用方法
- **如何运行测试?** → ../README.md § 测试
- **如何调整参数?** → ../README.md § 参数配置
- **如何添加新功能?** → IMPLEMENTATION.md § 2-3

#### 📌 问题排查
- **输出不一致?** → ../README.md § 常见问题 Q1
- **接受率异常?** → ../README.md § 常见问题 Q2
- **KV cache错误?** → IMPLEMENTATION.md § 6.3

---

## 📊 文档关系图

```
README.md (主文档)
    ├─── 快速入门 ──> 运行测试
    │
    ├─── 深入了解 ──> REQUIREMENTS.md
    │                    │
    │                    ├─── 需求规格
    │                    └─── 技术约束
    │
    ├─── 实现细节 ──> IMPLEMENTATION.md
    │                    │
    │                    ├─── 架构设计
    │                    ├─── 核心组件
    │                    ├─── 代码实现
    │                    └─── 测试验证
    │
    ├─── 版本历史 ──> CHANGELOG.md
    │                    │
    │                    └─── 修改日志
    │
    └─── 快速导航 ──> DOCS_INDEX.md
                        │
                        └─── 查找指南
```

---

## 💡 阅读建议

### 时间有限?
**只读**: README.md + DOCS_INDEX.md (15分钟)
- 快速了解项目和快速查找信息

### 想要使用?
**推荐**: README.md → REQUIREMENTS.md (40分钟)
- 理解需求和使用方法

### 需要修改?
**必读**: 全部文档 (2小时)
- 完整理解设计和实现

### 遇到问题?
**查阅**: DOCS_INDEX.md → 相关章节
- 针对性解决问题

---

## 🔗 相关资源

- [主README](../README.md) - 项目总览
- [测试目录](../tests/) - 测试脚本
- [源码](../model/moe_spec/) - 核心实现
- [配置](../config/) - 配置文件

---

## 📮 文档反馈

如果发现文档问题或需要补充:
- 提交Issue
- 修改文档并提交PR
- 联系项目维护者

---

## 📅 文档版本

| 文档 | 版本 | 最后更新 | 状态 |
|------|------|----------|------|
| DOCS_INDEX.md | v1.0 | 2025-10-21 | ✅ 最新 |
| REQUIREMENTS.md | v2.0 | 2025-10-21 | ✅ 最新 |
| IMPLEMENTATION.md | v2.0 | 2025-10-21 | ✅ 最新 |
| CHANGELOG.md | v1.0 | 2025-10-21 | ✅ 最新 |

---

**最后更新**: 2025-10-21


