<div align="center">

<img src="assets/banner.svg" width="100%" alt="LangChain 学习项目">

<br>

### 🦜 LangChain + LangGraph + Pydantic AI + LangSmith 学习实战

[![Stars](https://img.shields.io/github/stars/dirjaker/langchain_learning?style=flat-square&label=Stars&color=FFD700)](https://github.com/dirjaker/langchain_learning/stargazers)
[![Forks](https://img.shields.io/github/forks/dirjaker/langchain_learning?style=flat-square&label=Forks&color=4A90D9)](https://github.com/dirjaker/langchain_learning/network/members)
[![Contributors](https://img.shields.io/github/contributors/dirjaker/langchain_learning?style=flat-square&label=Contributors&color=8B4513)](https://github.com/dirjaker/langchain_learning/graphs/contributors)
[![License](https://img.shields.io/github/license/dirjaker/langchain_learning?style=flat-square&label=License&color=20B2AA)](https://github.com/dirjaker/langchain_learning/blob/dev/LICENSE)

</div>

---

## ✨ 功能特性

| 功能 | 描述 |
|------|------|
| 📚 **版本演进** | LangChain 从 v0.1 到 v1.3 的完整演进记录，旧版 API vs LCEL 对比 |
| 🔑 **核心概念** | Chain、Agent、RAG、Memory、Output Parser、Tool 等核心模块详解 |
| 💻 **实战示例** | 可运行的代码示例（7 个 Python 文件，4,400+ 行代码） |
| 🦜 **LangGraph** | State/Node/Edge、条件分支、循环图、子图、并行执行、流式输出 |
| 🤖 **Pydantic AI** | Agent 框架、依赖注入、结构化输出、流式、多 Agent 协作 |
| 🔍 **LangSmith** | Tracing 追踪、Dataset 管理、Evaluation 评估、Feedback 反馈、Hub |
| 📝 **面试准备** | 面试速查表 + 技术文档 + 设计文档，覆盖所有高频考点 |
| 📊 **对比分析** | LangChain vs LangGraph vs Pydantic AI vs LangSmith 场景选择指南 |

## 📖 在线文档

<div align="center">

**📚 [点击访问在线文档](https://dirjaker.github.io/langchain_learning/)**

</div>

| 文档 | 说明 |
|------|------|
| [面试速查表](https://dirjaker.github.io/langchain_learning/CHEATSHEET) | 面试前 30 分钟快速复习 |
| [技术文档](https://dirjaker.github.io/langchain_learning/TECHNICAL_DOC) | LangChain + LangGraph 深入讲解 |
| [技术设计文档](https://dirjaker.github.io/langchain_learning/technical-doc-cn) | 项目架构与面试问答 |
| [更新日志](https://dirjaker.github.io/langchain_learning/CHANGELOG) | 版本更新记录 |

## 📂 项目结构

```
langchain_learning/
├── README.md                        # 项目说明
├── LICENSE                          # MIT 许可证
├── requirements.txt                 # Python 依赖
├── assets/
│   └── banner.svg                   # Banner 图片
├── examples/
│   ├── 01_langchain_evolution.py    # LangChain 版本演进
│   ├── 02_langgraph_basics.py       # LangGraph 基础概念
│   ├── 03_practical_examples.py     # 实战示例（RAG/多Agent/Human-in-Loop）
│   ├── 04_core_components.py        # 核心组件详解
│   ├── 05_langgraph_advanced.py     # LangGraph 高级模式
│   ├── 06_pydantic_ai.py            # Pydantic AI 框架学习
│   └── 07_langsmith.py              # LangSmith 观测与评估
└── docs/                            # VitePress 文档站点
    ├── .vitepress/config.mts        # VitePress 配置
    ├── index.md                     # 首页
    ├── 01_langchain_evolution.md    # 版本演进章节
    ├── 02_langgraph_basics.md       # LangGraph 基础章节
    ├── 03_practical_examples.md     # 实战示例章节
    ├── 04_core_components.md        # 核心组件章节
    ├── 05_langgraph_advanced.md     # LangGraph 高级章节
    ├── 06_pydantic_ai.md            # Pydantic AI 框架章节
    ├── 07_langsmith.md              # LangSmith 观测章节
    ├── CHEATSHEET.md                # 面试速查表
    ├── TECHNICAL_DOC.md             # 技术文档
    ├── technical-doc-cn.md          # 技术设计文档
    └── CHANGELOG.md                 # 更新日志
```

## 🚀 快速开始

```bash
# 克隆项目
git clone https://github.com/dirjaker/langchain_learning.git
cd langchain_learning

# 创建虚拟环境
conda create -n langchain_learning python=3.12 -y
conda activate langchain_learning

# 安装依赖
pip install -r requirements.txt

# 运行示例
cd examples
python 01_langchain_evolution.py
python 02_langgraph_basics.py
python 03_practical_examples.py
python 04_core_components.py
python 05_langgraph_advanced.py
python 06_pydantic_ai.py
python 07_langsmith.py
```

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| **AI 框架** | LangChain 1.3.x, LangGraph 1.2.x, Pydantic AI 2.0.x |
| **观测平台** | LangSmith 0.8.x |
| **核心库** | Pydantic 2.x, TypedDict, Operator |
| **文档引擎** | VitePress |
| **部署** | GitHub Pages |
| **语言** | Python 3.12 |

## 📝 开发日志

- [x] LangChain 版本演进文档（0.1.x → 1.3.x）
- [x] LangGraph 核心概念详解（State/Node/Edge）
- [x] 实战示例代码（RAG、多 Agent、人机协作）
- [x] LangGraph 高级模式（子图、并行、流式、检查点）
- [x] 面试速查表与技术文档
- [x] VitePress 文档站点 + GitHub Pages 部署
- [x] Pydantic AI 框架学习（Agent/依赖注入/结构化输出/流式/多Agent协作）
- [x] LangSmith 观测平台（Tracing/Dataset/Evaluation/Feedback/Hub）
- [x] 更新日志（CHANGELOG）
- [ ] 视频教程
- [ ] 互动实验
- [ ] 社区讨论

## 📄 许可证

[MIT License](LICENSE)

---

<div align="center">

🔗 **GitHub**: [dirjaker/langchain_learning](https://github.com/dirjaker/langchain_learning)

⭐ 如果这个项目对你有帮助，请给一个 Star 支持一下！

</div>
