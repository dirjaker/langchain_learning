# LangChain + LangGraph 学习项目 — 技术设计文档

> 项目定位：系统性梳理 LangChain 版本演进与 LangGraph 核心概念的代码级实现，覆盖面试高频考点。

---

## 一、项目背景与价值

### 为什么做

LangChain 是当前最主流的 LLM 应用开发框架，但其 API 经历了多次重大重构（从 0.0.x 到 1.3.x），面试中经常考察候选人对框架演进的理解深度。同时，LangGraph 作为 LangChain 生态中构建有状态、多步骤 AI 应用的核心框架，其 State/Node/Edge 模型、条件分支、循环图等概念是 AI 工程岗位的高频面试题。

本项目通过 **可运行的代码示例** + **结构化的面试速查表**，帮助在面试前快速掌握以下核心知识：
- LangChain 从旧版 Chain API 到 LCEL 的演进逻辑
- Runnable 统一接口的设计思想
- LangGraph 图状态机的核心概念
- RAG、ReAct Agent、多 Agent 协作等实战模式

### 面试话术

> "我系统学习了 LangChain 的版本演进，理解了为什么从 Chain 类迁移到 LCEL 管道符组合——核心原因是旧版接口不统一、难以组合和复用。LCEL 通过统一的 Runnable 接口，让所有组件都支持 invoke/batch/stream 三种调用方式。同时我深入学习了 LangGraph，用代码实现了条件分支、循环图、多 Agent 协作等模式，理解了 State/Node/Edge 的设计思想。"

---

## 二、系统架构

```
┌─────────────────────────────────────────────────────────┐
│                   langchain_learning                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────────┐    ┌───────────────────┐        │
│  │ 01_langchain_     │    │ 02_langgraph_     │        │
│  │ evolution.py      │    │ basics.py         │        │
│  │                   │    │                   │        │
│  │ · 旧版 API 对比   │    │ · State 定义      │        │
│  │ · LCEL 管道符     │    │ · Node/Edge       │        │
│  │ · Runnable 接口   │    │ · 条件分支        │        │
│  │ · 结构化输出      │    │ · ReAct Agent     │        │
│  └───────────────────┘    └───────────────────┘        │
│                                                         │
│  ┌───────────────────┐    ┌───────────────────┐        │
│  │ 03_practical_     │    │ 04_core_          │        │
│  │ examples.py       │    │ components.py     │        │
│  │                   │    │                   │        │
│  │ · RAG 工作流      │    │ · Prompt Template │        │
│  │ · 多 Agent 协作   │    │ · Output Parser   │        │
│  │ · Human-in-Loop   │    │ · Memory 管理     │        │
│  └───────────────────┘    │ · Tool 定义       │        │
│                           └───────────────────┘        │
│  ┌───────────────────┐    ┌───────────────────┐        │
│  │ 05_langgraph_     │    │ 配套文档           │        │
│  │ advanced.py       │    │                   │        │
│  │                   │    │ · CHEATSHEET.md   │        │
│  │ · 子图复用        │    │ · TECHNICAL_DOC   │        │
│  │ · 并行执行        │    │                   │        │
│  │ · 流式输出        │    └───────────────────┘        │
│  │ · 检查点持久化    │                                  │
│  │ · Map-Reduce      │                                  │
│  └───────────────────┘                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 三、核心模块设计

### 3.1 LangChain 版本演进模块 (`01_langchain_evolution.py`)

**职责**：对比 LangChain 旧版 API 与 LCEL 的差异，展示核心创新。

#### 关键接口

**旧版 API（已废弃）**：
```python
# 旧版写法 — 接口不统一，难以组合
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(template="...", input_variables=[...])
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(input="...")
```

**LCEL 新版（核心创新）**：
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 管道符组合 — 所有组件统一实现 Runnable 接口
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手"),
    ("user", "{input}")
])
output_parser = StrOutputParser()
chain = prompt | llm | output_parser
result = chain.invoke({"input": "..."})
```

**Runnable 统一接口**：
```python
from langchain_core.runnables import RunnableLambda, RunnableParallel

def add_one(x: int) -> int:
    return x + 1

add_runnable = RunnableLambda(add_one)

# 三种调用方式
result = add_runnable.invoke(5)           # 单次调用
results = add_runnable.batch([1,2,3,4,5]) # 批量调用

# 并行执行
parallel = RunnableParallel(
    added=add_runnable,
    multiplied=RunnableLambda(lambda x: x * 2),
)
result = parallel.invoke(5)  # {"added": 6, "multiplied": 10}

# 链式组合
chain = add_runnable | RunnableLambda(lambda x: x * 2)
result = chain.invoke(5)  # 12
```

**结构化输出**：
```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class Person(BaseModel):
    """人物信息"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="职业")

parser = JsonOutputParser(pydantic_object=Person)
# chain = prompt | llm | parser  → 自动验证输出为 Person 对象
```

---

### 3.2 LangGraph 基础模块 (`02_langgraph_basics.py`)

**职责**：演示 LangGraph 的图状态机核心概念——State、Node、Edge。

#### 核心模型

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
import operator

# 1. 定义状态 — TypedDict + Annotated 支持累加
class State(TypedDict):
    messages: Annotated[list, operator.add]
    current_step: str

# 2. 定义节点 — 接收状态，返回部分更新
def node_a(state: State) -> dict:
    return {
        "messages": ["Node A 处理完成"],
        "current_step": "a_done",
    }

# 3. 构建图
graph = StateGraph(State)
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_edge(START, "node_a")
graph.add_edge("node_a", "node_b")
graph.add_edge("node_b", END)

# 4. 编译 & 执行
app = graph.compile()
result = app.invoke({"messages": ["开始"], "current_step": "init"})
```

#### 条件分支

```python
from typing import Literal

def route_by_sentiment(state: State) -> Literal["positive", "negative", "neutral"]:
    """根据情感路由到不同节点"""
    return state["sentiment"]

graph.add_conditional_edges(
    "analyze",           # 源节点
    route_by_sentiment,  # 路由函数
    {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
    }
)
```

#### ReAct Agent（循环图）

```python
def should_continue(state: AgentState) -> Literal["use_tool", "respond"]:
    if state["tool_calls"]:
        return "use_tool"
    if state["iterations"] >= state["max_iterations"]:
        return "respond"
    return "respond"

graph.add_conditional_edges("think", should_continue, {
    "use_tool": "use_tool",
    "respond": "respond",
})
graph.add_edge("use_tool", "think")  # 循环: 工具执行后回到思考
```

---

### 3.3 实战示例模块 (`03_practical_examples.py`)

**职责**：用 LangGraph 实现三种典型 AI 应用模式。

#### RAG 工作流

```python
class RAGState(TypedDict):
    question: str
    documents: list
    context: str
    answer: str

# 流程: retrieve → build_context → generate
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "build_context")
graph.add_edge("build_context", "generate")
graph.add_edge("generate", END)
```

#### 多 Agent 协作（Fan-out 模式）

```python
class MultiAgentState(TypedDict):
    task: str
    results: Annotated[list, operator.add]  # 累加结果
    final_result: str

# 规划后并行执行
graph.add_edge(START, "plan")
graph.add_edge("plan", "researcher")   # Fan-out
graph.add_edge("plan", "writer")       # Fan-out
graph.add_edge("researcher", "summarize")  # Fan-in
graph.add_edge("writer", "summarize")      # Fan-in
graph.add_edge("summarize", END)
```

#### Human-in-the-Loop

```python
graph.add_conditional_edges(
    "wait_approval",
    check_approval,
    {"execute": "execute", "reject": "reject"}
)
```

---

### 3.4 LangGraph 高级模块 (`05_langgraph_advanced.py`)

**职责**：展示子图、并行、流式、检查点、Map-Reduce 五大高级模式。

#### 子图（Subgraph）

```python
# 子图 — 独立的状态和编译
inner_graph = StateGraph(InnerState)
inner_graph.add_node("process", process_data)
inner_graph.add_node("format", format_output)
inner_app = inner_graph.compile()

# 主图中调用子图
def call_subgraph(state: OuterState) -> dict:
    result = inner_app.invoke({"data": state["messages"][-1], "processed": ""})
    return {"result": result["processed"]}
```

#### 并行执行（Fan-out/Fan-in）

```python
# Fan-out: START 同时指向 A, B, C
graph.add_edge(START, "task_a")
graph.add_edge(START, "task_b")
graph.add_edge(START, "task_c")

# Fan-in: A, B, C 都指向 merge
graph.add_edge("task_a", "merge")
graph.add_edge("task_b", "merge")
graph.add_edge("task_c", "merge")
```

#### 检查点（Checkpoint）

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "session_1"}}
result = app.invoke(initial_state, config=config)

# 获取检查点历史 — 支持时间旅行
for state in app.get_state_history(config):
    print(f"step={state.values.get('step')}")
```

---

## 四、技术亮点

### 4.1 从代码层面理解框架演进

不是泛泛介绍 API 变化，而是通过对比代码展示旧版 Chain 的痛点和 LCEL 的设计优势：
- 旧版：接口不统一、缺乏流式支持、类型提示不完善
- LCEL：统一 Runnable 接口、管道符组合、自动类型推断

### 4.2 图状态机的完整演示

覆盖 LangGraph 的全部核心概念：
- **State 设计**：使用 `Annotated[list, operator.add]` 实现状态累加
- **条件分支**：`add_conditional_edges` 实现路由
- **循环图**：边指向前序节点实现循环，配合 `max_iterations` 防止无限循环
- **并行执行**：Fan-out/Fan-in 模式

### 4.3 面试导向的代码组织

每个文件开头都有面试要点注释，结尾有面试问答总结，配合 CHEATSHEET.md 可在 30 分钟内完成复习。

### 4.4 三种典型 AI 应用模式

- **RAG**：检索增强生成，最常见模式
- **ReAct Agent**：思考-行动-观察循环
- **多 Agent 协作**：规划-并行执行-汇总

---

## 五、面试常见问题

**Q1: LangChain 和 LangGraph 有什么区别？**

> LangChain 用于简单的线性链（Chain/Agent 抽象），LangGraph 用于复杂的有状态工作流（Graph/State 抽象）。LangGraph 原生支持循环、分支、并行，适合构建 Agent。LangGraph 是 LangChain 的扩展，不是替代。

**Q2: 什么是 LCEL？为什么引入？**

> LCEL（LangChain Expression Language）是 0.2.x 引入的核心创新，使用 `|` 管道符组合组件。引入原因是旧版 Chain 类接口不统一、难以组合复用。LCEL 通过统一的 Runnable 接口，让所有组件都支持 invoke/batch/stream/ainvoke/abatch/astream。

**Q3: LangGraph 的核心概念是什么？**

> State（状态）+ Node（节点）+ Edge（边）。State 是数据容器（TypedDict），Node 是处理函数（接收状态返回更新），Edge 是连接关系（支持条件边）。

**Q4: LangGraph 如何实现循环？**

> 通过边指向之前的节点形成循环图。例如 ReAct Agent 中 `graph.add_edge("use_tool", "think")` 让工具执行后回到思考节点。通常配合 `max_iterations` 参数防止无限循环。

**Q5: 什么是条件边？**

> 根据当前状态决定下一步执行哪个节点。用 `add_conditional_edges(source_node, router_function, path_map)` 定义。router 函数返回路径名，LangGraph 根据返回值选择目标节点。

**Q6: 什么时候用 LangGraph 而不是 LangChain Chain？**

> - 需要循环的 Agent（ReAct）→ LangGraph
> - 多步骤有状态工作流 → LangGraph
> - 需要并行执行 → LangGraph
> - 需要 Human-in-the-Loop → LangGraph
> - 简单的线性问答链 → LangChain Chain 足够

**Q7: LangGraph 的检查点（Checkpoint）有什么用？**

> 持久化状态到数据库，支持断点续传、时间旅行（回溯历史状态）、Human-in-the-Loop（暂停等待人类输入）。使用 `MemorySaver`（内存）或 `SqliteSaver`（SQLite）作为后端。

**Q8: `Annotated[list, operator.add]` 在 LangGraph 中是什么意思？**

> 告诉 LangGraph 该字段使用 `operator.add` 进行状态合并。当多个节点返回同一个字段时，结果会累加而非覆盖。这对并行执行和 Fan-in 模式至关重要。

---

## 六、项目数据

| 指标 | 数据 |
|------|------|
| **代码量** | 1,769 行 Python（5 个源文件） |
| **文件数** | 5 个 .py + 3 个 .md + 1 个 requirements.txt |
| **技术栈** | Python 3.11, LangChain 1.3.x, LangGraph, Pydantic, TypedDict |
| **覆盖知识点** | LCEL, Runnable, State/Node/Edge, 条件分支, 循环图, RAG, ReAct, 多 Agent, 子图, 并行, 流式, 检查点, Map-Reduce, Human-in-the-Loop |
| **适用场景** | AI 工程岗位面试准备、LangChain/LangGraph 入门学习 |
