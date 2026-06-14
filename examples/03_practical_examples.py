"""
LangChain + LangGraph 实战示例

展示如何用 LangGraph 构建完整的 AI 应用
"""

# ============================================================
# 1. RAG 工作流 (使用 LangGraph)
# ============================================================
# 面试要点: RAG 是最常见的 AI 应用模式
#
# RAG (Retrieval-Augmented Generation):
# 1. 用户提问
# 2. 检索相关文档
# 3. 将文档作为上下文
# 4. LLM 生成回答
# ============================================================

def demo_rag_workflow():
    """
    RAG 工作流示例

    使用 LangGraph 实现 RAG
    """
    print("=" * 60)
    print("1. RAG 工作流 (LangGraph)")
    print("=" * 60)

    from typing import TypedDict, Annotated
    from langgraph.graph import StateGraph, START, END
    import operator

    # 1. 定义状态
    class RAGState(TypedDict):
        question: str
        documents: list
        context: str
        answer: str

    # 2. 定义节点
    def retrieve(state: RAGState) -> dict:
        """检索相关文档"""
        question = state["question"]
        print(f"  [Retrieve] 检索: {question}")

        # 模拟检索结果
        documents = [
            "LangChain 是一个用于构建 AI 应用的框架",
            "LangGraph 用于构建有状态的工作流",
            "LCEL 是 LangChain 的表达式语言",
        ]
        return {"documents": documents}

    def build_context(state: RAGState) -> dict:
        """构建上下文"""
        docs = state["documents"]
        context = "\n".join(docs)
        print(f"  [Context] 构建上下文: {len(docs)} 个文档")
        return {"context": context}

    def generate(state: RAGState) -> dict:
        """生成回答"""
        question = state["question"]
        context = state["context"]
        print(f"  [Generate] 生成回答")

        # 模拟 LLM 生成
        answer = f"根据文档，{question} 的答案是: LangChain 是一个 AI 应用框架。"
        return {"answer": answer}

    # 3. 创建图
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("build_context", build_context)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "build_context")
    graph.add_edge("build_context", "generate")
    graph.add_edge("generate", END)

    app = graph.compile()

    # 4. 执行
    print("\n执行 RAG 工作流:")
    result = app.invoke({
        "question": "什么是 LangChain？",
        "documents": [],
        "context": "",
        "answer": "",
    })

    print(f"\n回答: {result['answer']}")
    print()


# ============================================================
# 2. 多 Agent 协作
# ============================================================
# 面试要点: 多 Agent 协作是复杂 AI 应用的关键
#
# 多 Agent 模式:
# - 主 Agent 分配任务
# - 子 Agent 执行任务
# - 汇总结果
# ============================================================

def demo_multi_agent():
    """
    多 Agent 协作示例

    使用 LangGraph 实现 Agent 协作
    """
    print("=" * 60)
    print("2. 多 Agent 协作 (LangGraph)")
    print("=" * 60)

    from typing import TypedDict, Annotated, Literal
    from langgraph.graph import StateGraph, START, END
    import operator

    # 1. 定义状态
    class MultiAgentState(TypedDict):
        task: str
        subtasks: list
        results: Annotated[list, operator.add]
        final_result: str

    # 2. 定义节点
    def plan(state: MultiAgentState) -> dict:
        """规划子任务"""
        task = state["task"]
        print(f"  [Plan] 分解任务: {task}")

        # 模拟任务分解
        subtasks = [
            {"agent": "researcher", "task": "搜索信息"},
            {"agent": "writer", "task": "撰写内容"},
        ]
        return {"subtasks": subtasks}

    def researcher(state: MultiAgentState) -> dict:
        """研究员 Agent"""
        print(f"  [Researcher] 执行搜索")
        return {"results": ["搜索结果: 找到 3 篇相关文章"]}

    def writer(state: MultiAgentState) -> dict:
        """写手 Agent"""
        print(f"  [Writer] 撰写内容")
        return {"results": ["撰写完成: 生成了 500 字的文章"]}

    def summarize(state: MultiAgentState) -> dict:
        """汇总结果"""
        results = state["results"]
        print(f"  [Summarize] 汇总 {len(results)} 个结果")
        return {"final_result": f"任务完成: {'; '.join(results)}"}

    # 3. 创建图
    graph = StateGraph(MultiAgentState)

    graph.add_node("plan", plan)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("summarize", summarize)

    # 规划后并行执行
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "researcher")
    graph.add_edge("plan", "writer")
    graph.add_edge("researcher", "summarize")
    graph.add_edge("writer", "summarize")
    graph.add_edge("summarize", END)

    app = graph.compile()

    # 4. 执行
    print("\n执行多 Agent 协作:")
    result = app.invoke({
        "task": "写一篇关于 LangChain 的文章",
        "subtasks": [],
        "results": [],
        "final_result": "",
    })

    print(f"\n最终结果: {result['final_result']}")
    print()


# ============================================================
# 3. 人机协作 (Human-in-the-Loop)
# ============================================================
# 面试要点: 人机协作是生产环境的重要模式
#
# Human-in-the-Loop:
# - Agent 执行到关键步骤时暂停
# - 等待人类确认
# - 根据人类输入继续执行
# ============================================================

def demo_human_in_loop():
    """
    人机协作示例

    演示如何实现 Human-in-the-Loop
    """
    print("=" * 60)
    print("3. 人机协作 (Human-in-the-Loop)")
    print("=" * 60)

    from typing import TypedDict, Annotated
    from langgraph.graph import StateGraph, START, END
    import operator

    # 1. 定义状态
    class HumanLoopState(TypedDict):
        task: str
        plan: str
        approved: bool
        result: str

    # 2. 定义节点
    def create_plan(state: HumanLoopState) -> dict:
        """创建计划"""
        task = state["task"]
        plan = f"计划: {task} → 执行 → 验证"
        print(f"  [Plan] 创建计划: {plan}")
        return {"plan": plan}

    def wait_approval(state: HumanLoopState) -> dict:
        """等待审批 (模拟)"""
        print(f"  [Wait] 等待人类审批...")
        # 模拟人类审批
        approved = True  # 在实际应用中，这里会暂停等待
        return {"approved": approved}

    def execute(state: HumanLoopState) -> dict:
        """执行计划"""
        print(f"  [Execute] 执行计划")
        return {"result": "执行完成"}

    def reject(state: HumanLoopState) -> dict:
        """拒绝执行"""
        print(f"  [Reject] 计划被拒绝")
        return {"result": "计划被拒绝，需要重新规划"}

    # 3. 条件路由
    def check_approval(state: HumanLoopState):
        if state["approved"]:
            return "execute"
        return "reject"

    # 4. 创建图
    graph = StateGraph(HumanLoopState)

    graph.add_node("create_plan", create_plan)
    graph.add_node("wait_approval", wait_approval)
    graph.add_node("execute", execute)
    graph.add_node("reject", reject)

    graph.add_edge(START, "create_plan")
    graph.add_edge("create_plan", "wait_approval")
    graph.add_conditional_edges(
        "wait_approval",
        check_approval,
        {"execute": "execute", "reject": "reject"}
    )
    graph.add_edge("execute", END)
    graph.add_edge("reject", END)

    app = graph.compile()

    # 5. 执行
    print("\n执行人机协作:")
    result = app.invoke({
        "task": "部署新版本",
        "plan": "",
        "approved": False,
        "result": "",
    })

    print(f"\n结果: {result['result']}")
    print()


# ============================================================
# 4. 实战要点总结
# ============================================================

def demo_practical_points():
    """实战要点总结"""
    print("=" * 60)
    print("4. 实战要点总结")
    print("=" * 60)

    print("""
    场景选择:
    ┌─────────────────┬─────────────────┬─────────────────┐
    │ 场景             │ 推荐方案         │ 原因             │
    ├─────────────────┼─────────────────┼─────────────────┤
    │ 简单问答         │ LangChain Chain │ 线性流程         │
    │ RAG 检索         │ LangGraph       │ 需要状态管理     │
    │ ReAct Agent     │ LangGraph       │ 需要循环         │
    │ 多 Agent        │ LangGraph       │ 需要并行/协作    │
    │ 人机协作         │ LangGraph       │ 需要暂停/恢复    │
    │ 工作流           │ LangGraph       │ 复杂状态管理     │
    └─────────────────┴─────────────────┴─────────────────┘

    最佳实践:
    1. 状态设计: 使用 TypedDict，明确每个字段
    2. 节点职责: 每个节点只做一件事
    3. 错误处理: 在节点中处理异常
    4. 可观测性: 添加日志和追踪
    5. 测试: 为每个节点写单元测试

    常见陷阱:
    1. 状态过大: 只存储必要信息
    2. 无限循环: 设置 max_iterations
    3. 节点过重: 将复杂逻辑拆分
    4. 缺少错误处理: 每个节点都要处理异常
    """)
    print()


# ============================================================
# 主函数
# ============================================================

def main():
    demo_rag_workflow()
    demo_multi_agent()
    demo_human_in_loop()
    demo_practical_points()


if __name__ == "__main__":
    main()
