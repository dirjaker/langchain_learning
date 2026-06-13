# """

::: code-group

```python [02_langgraph_basics.py]
"""
LangGraph 核心概念演示

展示 LangGraph 的图状态机、节点、边、条件分支
"""

# ============================================================
# 1. LangGraph 基础概念
# ============================================================
# 面试要点: LangGraph 是什么？和 LangChain 有什么关系？
#
# LangGraph:
# - 用于构建有状态的、多步骤的 AI 应用
# - 基于图状态机 (Graph State Machine)
# - 支持循环、分支、并行
# - 是 LangChain 的扩展，不是替代
#
# 核心概念:
# - State: 状态 (通常是 TypedDict 或 Pydantic)
# - Node: 节点 (处理函数)
# - Edge: 边 (节点之间的连接)
# - Conditional Edge: 条件边 (根据状态决定下一步)
# ============================================================


def demo_concepts():
    """LangGraph 基础概念"""
    print("=" * 60)
    print("1. LangGraph 基础概念")
    print("=" * 60)

    print("""
    LangGraph vs LangChain:
    ┌─────────────────┬─────────────────┬─────────────────┐
    │ 特性             │ LangChain       │ LangGraph       │
    ├─────────────────┼─────────────────┼─────────────────┤
    │ 核心抽象         │ Chain/Agent     │ Graph/State     │
    │ 执行模型         │ 线性/树形       │ 图状态机        │
    │ 循环支持         │ 有限            │ 原生支持        │
    │ 状态管理         │ Memory 类       │ State 对象      │
    │ 可视化           │ 无              │ 支持            │
    │ 适用场景         │ 简单链          │ 复杂工作流      │
    └─────────────────┴─────────────────┴─────────────────┘

    核心组件:
    1. State (状态): 应用的数据状态，通常是 TypedDict
    2. Node (节点): 处理函数，接收状态，返回更新
    3. Edge (边): 节点之间的连接
    4. Conditional Edge: 条件边，根据状态决定下一步
    5. START/END: 特殊节点，表示开始和结束
    """)
    print()


# ============================================================
# 2. 简单图示例
# ============================================================

def demo_simple_graph():
    """
    简单图示例

    演示 LangGraph 的基本用法
    """
    print("=" * 60)
    print("2. 简单图示例")
    print("=" * 60)

    from typing import TypedDict, Annotated
    from langgraph.graph import StateGraph, START, END
    import operator

    # 1. 定义状态
    class State(TypedDict):
        # 使用 Annotated 和 operator.add 来支持状态累加
        messages: Annotated[list, operator.add]
        current_step: str

    # 2. 定义节点函数
    def node_a(state: State) -> dict:
        """节点 A: 处理输入"""
        print(f"  [Node A] 处理: {state['messages'][-1]}")
        return {
            "messages": ["Node A 处理完成"],
            "current_step": "a_done",
        }

    def node_b(state: State) -> dict:
        """节点 B: 进一步处理"""
        print(f"  [Node B] 处理: {state['messages'][-1]}")
        return {
            "messages": ["Node B 处理完成"],
            "current_step": "b_done",
        }

    def node_c(state: State) -> dict:
        """节点 C: 最终处理"""
        print(f"  [Node C] 处理: {state['messages'][-1]}")
        return {
            "messages": ["Node C 处理完成"],
            "current_step": "done",
        }

    # 3. 创建图
    graph = StateGraph(State)

    # 4. 添加节点
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.add_node("node_c", node_c)

    # 5. 添加边
    graph.add_edge(START, "node_a")  # START → node_a
    graph.add_edge("node_a", "node_b")  # node_a → node_b
    graph.add_edge("node_b", "node_c")  # node_b → node_c
    graph.add_edge("node_c", END)  # node_c → END

    # 6. 编译图
    app = graph.compile()

    # 7. 执行图
    print("\n执行图:")
    result = app.invoke({
        "messages": ["开始"],
        "current_step": "init",
    })

    print(f"\n最终状态:")
    print(f"  messages: {result['messages']}")
    print(f"  current_step: {result['current_step']}")
    print()


# ============================================================
# 3. 条件分支示例
# ============================================================

def demo_conditional_graph():
    """
    条件分支示例

    演示根据状态决定下一步
    """
    print("=" * 60)
    print("3. 条件分支示例")
    print("=" * 60)

    from typing import TypedDict, Annotated, Literal
    from langgraph.graph import StateGraph, START, END
    import operator

    # 1. 定义状态
    class State(TypedDict):
        messages: Annotated[list, operator.add]
        sentiment: str  # 情感: positive/negative/neutral

    # 2. 定义节点
    def analyze_sentiment(state: State) -> dict:
        """分析情感"""
        # 简单模拟: 根据消息内容判断情感
        last_msg = state["messages"][-1].lower()
        if any(w in last_msg for w in ["好", "棒", "喜欢", "开心"]):
            sentiment = "positive"
        elif any(w in last_msg for w in ["差", "糟", "讨厌", "失望"]):
            sentiment = "negative"
        else:
            sentiment = "neutral"

        print(f"  [分析] 情感: {sentiment}")
        return {"sentiment": sentiment}

    def handle_positive(state: State) -> dict:
        """处理正面情感"""
        print(f"  [正面] 感谢您的正面反馈！")
        return {"messages": ["感谢您的正面反馈！"]}

    def handle_negative(state: State) -> dict:
        """处理负面情感"""
        print(f"  [负面] 很抱歉给您带来不好的体验")
        return {"messages": ["很抱歉给您带来不好的体验，我们会改进"]}

    def handle_neutral(state: State) -> dict:
        """处理中性情感"""
        print(f"  [中性] 感谢您的反馈")
        return {"messages": ["感谢您的反馈"]}

    # 3. 条件路由函数
    def route_by_sentiment(state: State) -> Literal["positive", "negative", "neutral"]:
        """根据情感路由"""
        return state["sentiment"]

    # 4. 创建图
    graph = StateGraph(State)

    # 5. 添加节点
    graph.add_node("analyze", analyze_sentiment)
    graph.add_node("positive", handle_positive)
    graph.add_node("negative", handle_negative)
    graph.add_node("neutral", handle_neutral)

    # 6. 添加边
    graph.add_edge(START, "analyze")

    # 条件边: 根据情感选择不同的处理节点
    graph.add_conditional_edges(
        "analyze",  # 源节点
        route_by_sentiment,  # 路由函数
        {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral",
        }
    )

    graph.add_edge("positive", END)
    graph.add_edge("negative", END)
    graph.add_edge("neutral", END)

    # 7. 编译
    app = graph.compile()

    # 8. 测试不同情感
    test_cases = [
        "这个产品太好了，我很喜欢！",
        "服务太差了，非常失望",
        "请问什么时候发货？",
    ]

    for msg in test_cases:
        print(f"\n输入: {msg}")
        result = app.invoke({
            "messages": [msg],
            "sentiment": "",
        })
        print(f"回复: {result['messages'][-1]}")

    print()


# ============================================================
# 4. 循环图示例 (ReAct Agent)
# ============================================================

def demo_react_agent():
    """
    ReAct Agent 示例

    演示 LangGraph 实现 ReAct 模式
    """
    print("=" * 60)
    print("4. ReAct Agent 示例 (循环图)")
    print("=" * 60)

    from typing import TypedDict, Annotated, Literal
    from langgraph.graph import StateGraph, START, END
    import operator

    # 1. 定义状态
    class AgentState(TypedDict):
        messages: Annotated[list, operator.add]
        tool_calls: list
        iterations: int
        max_iterations: int

    # 2. 定义节点
    def think(state: AgentState) -> dict:
        """思考节点: 决定是否需要调用工具"""
        last_msg = state["messages"][-1]
        iterations = state["iterations"] + 1

        print(f"  [Think] 第 {iterations} 轮思考")

        # 简单模拟: 检查是否需要工具
        if "计算" in last_msg or "搜索" in last_msg:
            tool_calls = [{"tool": "search", "query": last_msg}]
            print(f"  [Think] 决定调用工具: {tool_calls[0]['tool']}")
        else:
            tool_calls = []
            print(f"  [Think] 不需要工具，直接回答")

        return {
            "tool_calls": tool_calls,
            "iterations": iterations,
        }

    def use_tool(state: AgentState) -> dict:
        """工具节点: 执行工具调用"""
        tool_call = state["tool_calls"][0]
        print(f"  [Tool] 执行: {tool_call['tool']}")

        # 模拟工具执行
        result = f"工具 {tool_call['tool']} 执行结果: 模拟数据"
        return {"messages": [result], "tool_calls": []}

    def respond(state: AgentState) -> dict:
        """响应节点: 生成最终回答"""
        print(f"  [Respond] 生成回答")
        return {"messages": ["这是我的回答"]}

    # 3. 条件路由
    def should_continue(state: AgentState) -> Literal["use_tool", "respond"]:
        """决定是否继续调用工具"""
        # 如果有工具调用，继续
        if state["tool_calls"]:
            return "use_tool"
        # 如果超过最大迭代次数，停止
        if state["iterations"] >= state["max_iterations"]:
            return "respond"
        # 否则继续思考
        return "use_tool" if state["tool_calls"] else "respond"

    # 4. 创建图
    graph = StateGraph(AgentState)

    # 5. 添加节点
    graph.add_node("think", think)
    graph.add_node("use_tool", use_tool)
    graph.add_node("respond", respond)

    # 6. 添加边
    graph.add_edge(START, "think")
    graph.add_conditional_edges(
        "think",
        should_continue,
        {
            "use_tool": "use_tool",
            "respond": "respond",
        }
    )
    graph.add_edge("use_tool", "think")  # 循环: 工具执行后回到思考
    graph.add_edge("respond", END)

    # 7. 编译
    app = graph.compile()

    # 8. 执行
    print("\n执行 ReAct Agent:")
    result = app.invoke({
        "messages": ["帮我搜索 LangChain 的最新版本"],
        "tool_calls": [],
        "iterations": 0,
        "max_iterations": 3,
    })

    print(f"\n最终结果:")
    print(f"  迭代次数: {result['iterations']}")
    print(f"  最后消息: {result['messages'][-1]}")
    print()


# ============================================================
# 5. LangGraph 面试要点总结
# ============================================================

def demo_interview_points():
    """面试要点总结"""
    print("=" * 60)
    print("5. LangGraph 面试要点总结")
    print("=" * 60)

    print("""
    Q: LangGraph 和 LangChain 有什么区别？
    A: LangChain 用于简单的线性链，LangGraph 用于复杂的有状态工作流。
       LangGraph 支持循环、分支、并行，适合构建 Agent。

    Q: LangGraph 的核心概念是什么？
    A: State (状态) + Node (节点) + Edge (边)
       状态是数据，节点是处理逻辑，边是连接关系。

    Q: 什么是条件边？
    A: 根据当前状态决定下一步执行哪个节点。
       用 add_conditional_edges() 定义。

    Q: LangGraph 如何实现循环？
    A: 通过边指向之前的节点，形成循环。
       通常配合 max_iterations 防止无限循环。

    Q: LangGraph 的优势是什么？
    A: 1. 可视化: 可以画出执行流程
       2. 状态管理: 显式的状态对象
       3. 循环支持: 原生支持循环
       4. 可观测性: 每一步都可以追踪

    Q: 什么时候用 LangGraph？
    A: 1. 需要循环的 Agent (ReAct)
       2. 多步骤工作流
       3. 需要状态管理
       4. 需要可视化执行流程
    """)
    print()


# ============================================================
# 主函数
# ============================================================

def main():
    demo_concepts()
    demo_simple_graph()
    demo_conditional_graph()
    demo_react_agent()
    demo_interview_points()


if __name__ == "__main__":
    main()
```

:::
