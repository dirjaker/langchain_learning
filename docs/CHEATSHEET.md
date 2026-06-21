# LangChain + LangGraph 面试速查表

> 面试前 30 分钟快速复习

---

## 1. LangChain 版本演进

### 关键版本
| 版本 | 核心变化 |
|------|----------|
| 0.1.x | 首个稳定版 |
| 0.2.x | **LCEL 引入** — 管道符组合 |
| 0.3.x | **模块化拆分** — langchain-core, langchain-community |
| 1.0.x | 正式稳定版 |

### 包结构 (0.3.x+)
```
langchain-core        # 核心抽象 (Runnable, LCEL)
langchain-community   # 社区集成
langchain             # 主包 (Chain, Agent, Memory)
langchain-openai      # OpenAI 集成
```

---

## 2. LCEL (LangChain Expression Language)

### 核心概念
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 管道符组合
chain = prompt | llm | output_parser

# 调用
result = chain.invoke({"input": "..."})
```

### Runnable 接口
```python
# 三种调用方式
runnable.invoke(input)      # 单次
runnable.batch(inputs)      # 批量
runnable.stream(input)      # 流式

# 异步版本
await runnable.ainvoke(input)
await runnable.abatch(inputs)
async for chunk in runnable.astream(input)
```

### 常用 Runnable
```python
RunnablePassthrough()       # 直接传递
RunnableLambda(func)        # 函数转换
RunnableParallel({...})     # 并行执行
```

---

## 3. LangGraph 核心概念

### 与 LangChain 的区别
| 特性 | LangChain | LangGraph |
|------|-----------|-----------|
| 核心抽象 | Chain/Agent | Graph/State |
| 执行模型 | 线性/树形 | 图状态机 |
| 循环支持 | 有限 | 原生支持 |
| 状态管理 | Memory 类 | State 对象 |

### 基本结构
```python
from langgraph.graph import StateGraph, START, END

# 1. 定义状态
class State(TypedDict):
    messages: Annotated[list, operator.add]

# 2. 创建图
graph = StateGraph(State)

# 3. 添加节点
graph.add_node("node_name", node_function)

# 4. 添加边
graph.add_edge(START, "node_name")
graph.add_edge("node_name", END)

# 5. 条件边
graph.add_conditional_edges(
    "source_node",
    routing_function,
    {"path1": "node1", "path2": "node2"}
)

# 6. 编译执行
app = graph.compile()
result = app.invoke(initial_state)
```

---

## 4. 常见模式

### RAG 工作流
```
START → Retrieve → BuildContext → Generate → END
```

### ReAct Agent (循环)
```
START → Think → (需要工具?) → UseTool → Think → ...
                                    ↓
                               (不需要) → Respond → END
```

### 人机协作
```
START → Plan → WaitApproval → (批准?) → Execute → END
                                ↓
                           (拒绝) → Reject → END
```

---

## 5. 面试问答

### Q: LCEL 是什么？
A: LangChain Expression Language，用管道符 `|` 组合组件，统一 Runnable 接口。

### Q: LangGraph 的核心概念？
A: State (状态) + Node (节点) + Edge (边)，支持循环和条件分支。

### Q: 什么时候用 LangGraph？
A: 需要循环 (ReAct)、复杂状态管理、多 Agent 协作、人机协作。

### Q: Runnable 接口有哪些方法？
A: invoke (单次)、batch (批量)、stream (流式)，都有异步版本。

### Q: LangGraph 如何实现循环？
A: 通过边指向之前的节点，配合 max_iterations 防止无限循环。

### Q: 条件边是什么？
A: 根据当前状态决定下一步执行哪个节点，用 add_conditional_edges()。

---

*最后更新: 2026-06-22*
