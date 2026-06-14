"""
LangGraph 高级模式

展示 LangGraph 的高级用法：子图、持久化、流式、并行等
"""

# ============================================================
# 1. 子图 (Subgraph)
# ============================================================
# 面试要点: 子图用于模块化复杂工作流
#
# 子图优势:
# - 复用性: 同一个子图可以在多处使用
# - 封装性: 隐藏内部实现细节
# - 可测试性: 可以单独测试子图
# ============================================================

def demo_subgraph():
    """子图示例"""
    print("=" * 60)
    print("1. 子图 (Subgraph)")
    print("=" * 60)

    from typing import TypedDict, Annotated
    from langgraph.graph import StateGraph, START, END
    import operator

    # 内部状态
    class InnerState(TypedDict):
        data: str
        processed: str

    # 外部状态
    class OuterState(TypedDict):
        messages: Annotated[list, operator.add]
        result: str

    # 子图: 数据处理
    def process_data(state: InnerState) -> dict:
        print(f"  [子图] 处理数据: {state['data']}")
        return {"processed": f"已处理: {state['data']}"}

    def format_output(state: InnerState) -> dict:
        print(f"  [子图] 格式化: {state['processed']}")
        return {"processed": f"[{state['processed']}]"}

    inner_graph = StateGraph(InnerState)
    inner_graph.add_node("process", process_data)
    inner_graph.add_node("format", format_output)
    inner_graph.add_edge(START, "process")
    inner_graph.add_edge("process", "format")
    inner_graph.add_edge("format", END)
    inner_app = inner_graph.compile()

    # 主图
    def call_subgraph(state: OuterState) -> dict:
        # 调用子图
        result = inner_app.invoke({"data": state["messages"][-1], "processed": ""})
        return {"result": result["processed"]}

    def respond(state: OuterState) -> dict:
        return {"messages": [f"最终结果: {state['result']}"]}

    outer_graph = StateGraph(OuterState)
    outer_graph.add_node("delegate", call_subgraph)
    outer_graph.add_node("respond", respond)
    outer_graph.add_edge(START, "delegate")
    outer_graph.add_edge("delegate", "respond")
    outer_graph.add_edge("respond", END)

    app = outer_graph.compile()

    print("\n执行带子图的工作流:")
    result = app.invoke({"messages": ["测试数据"], "result": ""})
    print(f"\n输出: {result['messages'][-1]}")
    print()


# ============================================================
# 2. 并行执行
# ============================================================
# 面试要点: LangGraph 原生支持并行执行
#
# 并行执行:
# - 多个节点可以同时执行
# - 使用 fan-out/fan-in 模式
# - 提高执行效率
# ============================================================

def demo_parallel():
    """并行执行示例"""
    print("=" * 60)
    print("2. 并行执行 (Fan-out/Fan-in)")
    print("=" * 60)

    from typing import TypedDict, Annotated
    from langgraph.graph import StateGraph, START, END
    import operator
    import time

    class State(TypedDict):
        input: str
        results: Annotated[list, operator.add]
        final: str

    def task_a(state: State) -> dict:
        print(f"  [A] 开始执行...")
        time.sleep(0.1)  # 模拟耗时
        print(f"  [A] 完成")
        return {"results": ["A的结果"]}

    def task_b(state: State) -> dict:
        print(f"  [B] 开始执行...")
        time.sleep(0.1)
        print(f"  [B] 完成")
        return {"results": ["B的结果"]}

    def task_c(state: State) -> dict:
        print(f"  [C] 开始执行...")
        time.sleep(0.1)
        print(f"  [C] 完成")
        return {"results": ["C的结果"]}

    def merge(state: State) -> dict:
        print(f"  [Merge] 合并 {len(state['results'])} 个结果")
        return {"final": f"合并结果: {state['results']}"}

    # 创建图: A, B, C 并行执行
    graph = StateGraph(State)

    graph.add_node("task_a", task_a)
    graph.add_node("task_b", task_b)
    graph.add_node("task_c", task_c)
    graph.add_node("merge", merge)

    # Fan-out: START 同时指向 A, B, C
    graph.add_edge(START, "task_a")
    graph.add_edge(START, "task_b")
    graph.add_edge(START, "task_c")

    # Fan-in: A, B, C 都指向 merge
    graph.add_edge("task_a", "merge")
    graph.add_edge("task_b", "merge")
    graph.add_edge("task_c", "merge")

    graph.add_edge("merge", END)

    app = graph.compile()

    print("\n执行并行任务:")
    start = time.time()
    result = app.invoke({"input": "test", "results": [], "final": ""})
    elapsed = time.time() - start

    print(f"\n结果: {result['final']}")
    print(f"耗时: {elapsed:.2f}s (并行执行)")
    print()


# ============================================================
# 3. 流式输出
# ============================================================
# 面试要点: 流式输出是用户体验的关键
#
# 流式模式:
# - stream: 同步流式
# - astream: 异步流式
# - stream_events: 事件流
# ============================================================

def demo_streaming():
    """流式输出示例"""
    print("=" * 60)
    print("3. 流式输出")
    print("=" * 60)

    from typing import TypedDict, Annotated
    from langgraph.graph import StateGraph, START, END
    import operator

    class State(TypedDict):
        messages: Annotated[list, operator.add]

    def step1(state: State) -> dict:
        return {"messages": ["步骤1完成"]}

    def step2(state: State) -> dict:
        return {"messages": ["步骤2完成"]}

    def step3(state: State) -> dict:
        return {"messages": ["步骤3完成"]}

    graph = StateGraph(State)
    graph.add_node("step1", step1)
    graph.add_node("step2", step2)
    graph.add_node("step3", step3)
    graph.add_edge(START, "step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", END)

    app = graph.compile()

    print("\n流式输出:")
    for event in app.stream({"messages": ["开始"]}):
        # event 是一个字典，key 是节点名，value 是节点输出
        for node_name, output in event.items():
            print(f"  [{node_name}] {output}")

    print()


# ============================================================
# 4. 检查点 (Checkpoint)
# ============================================================
# 面试要点: 检查点用于持久化和恢复
#
# Checkpoint 用途:
# - 持久化状态: 保存到数据库
# - 断点续传: 从上次中断处继续
# - 时间旅行: 回溯到某个状态
# - 人机协作: 暂停等待人类输入
# ============================================================

def demo_checkpoint():
    """检查点示例"""
    print("=" * 60)
    print("4. 检查点 (Checkpoint)")
    print("=" * 60)

    from typing import TypedDict, Annotated
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    import operator

    class State(TypedDict):
        messages: Annotated[list, operator.add]
        step: int

    def step1(state: State) -> dict:
        print(f"  [Step1] 执行")
        return {"messages": ["Step1完成"], "step": 1}

    def step2(state: State) -> dict:
        print(f"  [Step2] 执行")
        return {"messages": ["Step2完成"], "step": 2}

    def step3(state: State) -> dict:
        print(f"  [Step3] 执行")
        return {"messages": ["Step3完成"], "step": 3}

    graph = StateGraph(State)
    graph.add_node("step1", step1)
    graph.add_node("step2", step2)
    graph.add_node("step3", step3)
    graph.add_edge(START, "step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", END)

    # 使用内存检查点
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    # 使用 thread_id 来标识会话
    config = {"configurable": {"thread_id": "session_1"}}

    print("\n执行 (带检查点):")
    result = app.invoke(
        {"messages": ["开始"], "step": 0},
        config=config,
    )

    print(f"\n状态: step={result['step']}, messages={result['messages']}")

    # 获取检查点历史
    print("\n检查点历史:")
    for state in app.get_state_history(config):
        print(f"  step={state.values.get('step', '?')}, "
              f"messages={len(state.values.get('messages', []))}")

    print()


# ============================================================
# 5. Map-Reduce 模式
# ============================================================
# 面试要点: Map-Reduce 用于批量处理
#
# Map-Reduce:
# - Map: 将任务分发到多个节点
# - Reduce: 汇总所有节点的结果
# ============================================================

def demo_map_reduce():
    """Map-Reduce 示例"""
    print("=" * 60)
    print("5. Map-Reduce 模式")
    print("=" * 60)

    from typing import TypedDict, Annotated
    from langgraph.graph import StateGraph, START, END
    import operator

    class State(TypedDict):
        items: list
        results: Annotated[list, operator.add]
        summary: str

    def split(state: State) -> dict:
        """Map: 分割任务"""
        items = state["items"]
        print(f"  [Split] 分割 {len(items)} 个任务")
        # 返回多个结果，触发并行
        return {"results": items}

    def process(state: State) -> dict:
        """处理单个任务"""
        # 这个节点会对每个结果并行执行
        print(f"  [Process] 处理")
        return {}

    def reduce(state: State) -> dict:
        """Reduce: 汇总结果"""
        results = state["results"]
        print(f"  [Reduce] 汇总 {len(results)} 个结果")
        return {"summary": f"汇总: {results}"}

    # 注意: 真正的 Map-Reduce 需要用 Send API
    # 这里简化演示
    graph = StateGraph(State)
    graph.add_node("split", split)
    graph.add_node("reduce", reduce)
    graph.add_edge(START, "split")
    graph.add_edge("split", "reduce")
    graph.add_edge("reduce", END)

    app = graph.compile()

    print("\n执行 Map-Reduce:")
    result = app.invoke({
        "items": ["任务1", "任务2", "任务3"],
        "results": [],
        "summary": "",
    })

    print(f"\n结果: {result['summary']}")
    print()


# ============================================================
# 6. 高级模式总结
# ============================================================

def demo_summary():
    """高级模式总结"""
    print("=" * 60)
    print("6. LangGraph 高级模式总结")
    print("=" * 60)

    print("""
    常用模式:
    ┌─────────────────┬─────────────────┬─────────────────┐
    │ 模式             │ 用途             │ 关键 API        │
    ├─────────────────┼─────────────────┼─────────────────┤
    │ 子图             │ 模块化复用       │ StateGraph      │
    │ 并行执行         │ 提高效率         │ 多边指向同一节点│
    │ 流式输出         │ 用户体验         │ stream()        │
    │ 检查点           │ 持久化/恢复      │ Checkpointer    │
    │ Map-Reduce      │ 批量处理         │ Send API        │
    │ 人机协作         │ 等待人类输入     │ interrupt()     │
    └─────────────────┴─────────────────┴─────────────────┘

    面试关键:
    1. 子图: 复用和封装
    2. 并行: Fan-out/Fan-in 模式
    3. 流式: stream() 方法
    4. 检查点: MemorySaver / SqliteSaver
    5. Map-Reduce: Send API (高级)
    """)
    print()


# ============================================================
# 主函数
# ============================================================

def main():
    demo_subgraph()
    demo_parallel()
    demo_streaming()
    demo_checkpoint()
    demo_map_reduce()
    demo_summary()


if __name__ == "__main__":
    main()
