# LangChain + LangGraph 学习项目

面试准备：LangChain 版本演进和 LangGraph 核心概念的代码实现。

## 📁 项目结构

```
langchain_learning/
├── CHEATSHEET.md                    # 面试速查表
├── README.md                        # 项目文档
├── requirements.txt                 # 依赖
├── 01_langchain_evolution.py        # LangChain 版本演进
├── 02_langgraph_basics.py           # LangGraph 基础概念
└── 03_practical_examples.py         # 实战示例
```

## 🚀 运行方式

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行 LangChain 版本演进
python 01_langchain_evolution.py

# 运行 LangGraph 基础
python 02_langgraph_basics.py

# 运行实战示例
python 03_practical_examples.py
```

## 📖 内容概览

### 01_langchain_evolution.py — LangChain 版本演进
| 模块 | 面试考点 |
|------|----------|
| 旧版 API | 了解历史，说明为什么被废弃 |
| LCEL | 管道符组合，核心创新 |
| Runnable 接口 | invoke/batch/stream |
| 结构化输出 | Pydantic 集成 |

### 02_langgraph_basics.py — LangGraph 基础
| 模块 | 面试考点 |
|------|----------|
| 基础概念 | State/Node/Edge |
| 简单图 | 基本用法 |
| 条件分支 | 条件边 |
| ReAct Agent | 循环图 |

### 03_practical_examples.py — 实战示例
| 模块 | 面试考点 |
|------|----------|
| RAG 工作流 | 检索增强生成 |
| 多 Agent | Agent 协作 |
| 人机协作 | Human-in-the-Loop |

## 💡 学习建议

1. **先看 CHEATSHEET.md** — 面试前 30 分钟快速复习
2. **运行代码** — 亲手运行，观察输出
3. **理解核心概念** — LCEL、Runnable、State、Node、Edge
4. **掌握常见模式** — RAG、ReAct、多 Agent

## 📊 LangChain 版本演进

```
0.0.x (初始版本)
  ↓
0.1.x (首个稳定版)
  ↓
0.2.x (LCEL 引入 — 核心创新)
  ↓
0.3.x (模块化拆分 — langchain-core, langchain-community)
  ↓
1.0.x (正式稳定版)
  ↓
1.3.x (当前最新)
```

## 🔑 核心概念

### LCEL (LangChain Expression Language)
```python
# 管道符组合
chain = prompt | llm | output_parser

# 调用
result = chain.invoke({"input": "..."})
```

### LangGraph
```python
# 创建图
graph = StateGraph(State)
graph.add_node("node", function)
graph.add_edge(START, "node")
graph.add_edge("node", END)

# 条件边
graph.add_conditional_edges("source", router, {"path": "target"})

# 执行
app = graph.compile()
result = app.invoke(initial_state)
```

## 📚 配套文档

- 面试速查表：`CHEATSHEET.md`
- LangChain 官方文档：https://python.langchain.com/
- LangGraph 官方文档：https://langchain-ai.github.io/langgraph/

## ⚠️ 注意事项

- 代码使用模拟数据，不需要 API Key
- 重点展示架构设计，不是实际 LLM 调用
- 适合面试演示和学习理解
