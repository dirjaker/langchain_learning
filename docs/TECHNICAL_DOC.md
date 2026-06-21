# LangChain + LangGraph 技术文档

> 面试准备：深入理解 LangChain 版本演进和 LangGraph 核心概念

---

## 一、LangChain 版本演进

### 1.1 版本历史

| 版本 | 时间 | 核心变化 | 面试要点 |
|------|------|----------|----------|
| 0.0.x | 2022-2023 | 初始版本 | 了解历史 |
| 0.1.x | 2023 | 首个稳定版 | 基础 API |
| 0.2.x | 2024 | **LCEL 引入** | 核心创新 |
| 0.3.x | 2024 | **模块化拆分** | 包结构变化 |
| 1.0.x | 2024 | 正式稳定版 | API 稳定 |
| 1.3.x | 2025 | 性能优化 | 当前最新 |

### 1.2 包结构 (0.3.x+)

```
langchain-core        # 核心抽象
├── Runnable          # 统一接口
├── LCEL              # 表达式语言
├── Prompt            # 提示模板
├── OutputParser      # 输出解析
└── messages          # 消息类型

langchain-community   # 社区集成
├── llms              # LLM 集成
├── embeddings        # Embedding
├── vectorstores      # 向量数据库
└── tools             # 工具集成

langchain             # 主包
├── chains            # 链
├── agents            # Agent
├── memory            # 记忆
└── callbacks         # 回调

langchain-openai      # OpenAI 集成
langchain-anthropic   # Anthropic 集成
```

### 1.3 为什么要做模块化拆分？

**问题：**
- 旧版 langchain 包太大，依赖太多
- 安装慢，容易冲突
- 难以只使用部分功能

**解决方案：**
- `langchain-core`: 最小依赖，核心抽象
- `langchain-community`: 可选依赖，社区集成
- `langchain`: 主包，组合 core + community

**面试话术：**
> "LangChain 0.3.x 做了模块化拆分，将核心抽象（langchain-core）和社区集成（langchain-community）分离。这样用户可以只安装需要的部分，减少依赖冲突，也方便第三方开发者贡献集成。"

---

## 二、LCEL (LangChain Expression Language)

### 2.1 什么是 LCEL？

LCEL 是 LangChain 0.2.x 引入的核心创新，用管道符 `|` 组合组件。

**核心思想：**
- 所有组件都实现 `Runnable` 接口
- 用 `|` 管道符组合组件
- 自动支持流式、批处理、异步

### 2.2 Runnable 接口

```python
class Runnable:
    def invoke(self, input) -> output          # 单次调用
    def batch(self, inputs) -> outputs         # 批量调用
    def stream(self, input) -> Iterator        # 流式调用
    async def ainvoke(self, input) -> output   # 异步单次
    async def abatch(self, inputs) -> outputs  # 异步批量
    async def astream(self, input) -> AsyncIterator  # 异步流式
```

### 2.3 LCEL 示例

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 组件
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手"),
    ("user", "{input}")
])
llm = ChatOpenAI()
parser = StrOutputParser()

# LCEL 组合
chain = prompt | llm | parser

# 调用
result = chain.invoke({"input": "什么是Python？"})

# 流式
for chunk in chain.stream({"input": "什么是Python？"}):
    print(chunk, end="")

# 批量
results = chain.batch([
    {"input": "什么是Python？"},
    {"input": "什么是JavaScript？"},
])
```

### 2.4 为什么 LCEL 重要？

**面试要点：**

1. **统一接口**: 所有组件都用同一个接口
2. **组合性**: 用 `|` 像搭积木一样组合
3. **流式支持**: 自动支持流式输出
4. **可观察性**: 自动追踪执行过程
5. **可测试性**: 每个组件可以单独测试

**面试话术：**
> "LCEL 是 LangChain 最重要的创新。它用管道符 `|` 组合组件，统一了 Runnable 接口，让所有组件都支持 invoke/batch/stream。这样开发者可以像搭积木一样构建 AI 应用，而且自动支持流式输出和异步执行。"

---

## 三、LangGraph 核心概念

### 3.1 什么是 LangGraph？

LangGraph 用于构建**有状态的、多步骤的** AI 应用。

**与 LangChain 的区别：**

| 特性 | LangChain | LangGraph |
|------|-----------|-----------|
| 核心抽象 | Chain/Agent | Graph/State |
| 执行模型 | 线性/树形 | **图状态机** |
| 循环支持 | 有限 | **原生支持** |
| 状态管理 | Memory 类 | **State 对象** |
| 可视化 | 无 | **支持** |
| 适用场景 | 简单链 | **复杂工作流** |

### 3.2 核心组件

#### State (状态)
```python
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    # 使用 Annotated 和 operator.add 支持状态累加
    messages: Annotated[list, operator.add]
    current_step: str
```

#### Node (节点)
```python
def my_node(state: State) -> dict:
    # 接收状态，返回更新
    return {"messages": ["新消息"]}
```

#### Edge (边)
```python
# 普通边
graph.add_edge("node_a", "node_b")

# 条件边
graph.add_conditional_edges(
    "source_node",           # 源节点
    routing_function,        # 路由函数
    {"path1": "node1", "path2": "node2"}  # 路由映射
)
```

### 3.3 常见模式

#### 1. 线性流程
```
START → Node1 → Node2 → Node3 → END
```

#### 2. 条件分支
```
START → Analyze → (判断) → Path1 → END
                    ↓
                  Path2 → END
```

#### 3. 循环 (ReAct)
```
START → Think → (需要工具?) → UseTool → Think → ...
                                    ↓
                               (不需要) → Respond → END
```

#### 4. 并行执行
```
START → TaskA ─┐
        TaskB ─┼→ Merge → END
        TaskC ─┘
```

### 3.4 面试问答

**Q: 什么时候用 LangGraph？**

A: 
1. 需要循环的 Agent (ReAct 模式)
2. 复杂的多步骤工作流
3. 需要状态管理
4. 需要可视化执行流程
5. 需要人机协作

**Q: LangGraph 如何实现循环？**

A: 通过边指向之前的节点，形成循环。通常配合 `max_iterations` 防止无限循环。

**Q: 什么是条件边？**

A: 根据当前状态决定下一步执行哪个节点。用 `add_conditional_edges()` 定义。

**Q: LangGraph 的优势是什么？**

A:
1. **可视化**: 可以画出执行流程
2. **状态管理**: 显式的状态对象
3. **循环支持**: 原生支持循环
4. **可观测性**: 每一步都可以追踪
5. **持久化**: 支持检查点

---

## 四、实战模式

### 4.1 RAG 工作流

```
用户提问 → 检索文档 → 构建上下文 → LLM 生成 → 返回回答
```

**LangGraph 实现：**
```python
graph.add_node("retrieve", retrieve)
graph.add_node("build_context", build_context)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "build_context")
graph.add_edge("build_context", "generate")
graph.add_edge("generate", END)
```

### 4.2 ReAct Agent

```
Think → (需要工具?) → UseTool → Think → ...
                          ↓
                     (不需要) → Respond → END
```

**关键点：**
- 循环：`UseTool → Think`
- 条件：根据是否需要工具决定
- 终止：达到最大迭代或不需要工具

### 4.3 多 Agent 协作

```
Plan → Agent1 ─┐
       Agent2 ─┼→ Summarize → END
       Agent3 ─┘
```

**关键点：**
- 并行执行多个 Agent
- 汇总所有 Agent 的结果
- 处理 Agent 之间的依赖

### 4.4 人机协作 (Human-in-the-Loop)

```
Plan → WaitApproval → (批准?) → Execute → END
                         ↓
                       (拒绝) → Reject → END
```

**关键点：**
- 使用检查点暂停执行
- 等待人类输入
- 根据人类输入继续或终止

---

## 五、面试高频问答

### Q1: LangChain 和 LangGraph 有什么区别？
**A:** LangChain 用于简单的线性链，LangGraph 用于复杂的有状态工作流。LangGraph 支持循环、分支、并行，适合构建 Agent。

### Q2: LCEL 是什么？
**A:** LangChain Expression Language，用管道符 `|` 组合组件，统一 Runnable 接口，自动支持流式和异步。

### Q3: Runnable 接口有哪些方法？
**A:** invoke (单次)、batch (批量)、stream (流式)，都有异步版本 (ainvoke, abatch, astream)。

### Q4: LangGraph 的核心概念？
**A:** State (状态) + Node (节点) + Edge (边)。状态是数据，节点是处理逻辑，边是连接关系。

### Q5: 什么时候用 LangGraph？
**A:** 需要循环 (ReAct)、复杂状态管理、多 Agent 协作、人机协作、需要可视化执行流程。

### Q6: 什么是条件边？
**A:** 根据当前状态决定下一步执行哪个节点，用 `add_conditional_edges()` 定义。

### Q7: LangGraph 如何实现持久化？
**A:** 使用 Checkpointer (如 MemorySaver)，配合 thread_id 标识会话。

### Q8: LangChain 为什么要做模块化拆分？
**A:** 减少依赖冲突，让用户只安装需要的部分，方便第三方贡献集成。

---

*最后更新: 2026-06-22*
