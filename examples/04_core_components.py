"""
LangChain 核心组件详解

深入讲解 Prompt、Output Parser、Memory、Tool 等核心组件
"""

# ============================================================
# 1. Prompt Template 详解
# ============================================================
# 面试要点: Prompt 是 AI 应用的核心
#
# Prompt Template 类型:
# - ChatPromptTemplate: 聊天模型用
# - PromptTemplate: 文本补全用
# - FewShotPromptTemplate: 少样本提示
# ============================================================

def demo_prompt_templates():
    """Prompt Template 详解"""
    print("=" * 60)
    print("1. Prompt Template 详解")
    print("=" * 60)

    from langchain_core.prompts import (
        ChatPromptTemplate,
        PromptTemplate,
        FewShotPromptTemplate,
        MessagesPlaceholder,
    )
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    # 1. ChatPromptTemplate - 聊天模型
    print("\n✅ ChatPromptTemplate:")
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，擅长{skill}"),
        ("user", "{input}"),
    ])
    # 使用 partial 预填参数
    chat_prompt_partial = chat_prompt.partial(role="Python专家", skill="代码优化")
    messages = chat_prompt_partial.format_messages(input="如何优化代码？")
    for msg in messages:
        print(f"  {msg.__class__.__name__}: {msg.content}")

    # 2. 带历史的聊天模板
    print("\n✅ 带历史的聊天模板:")
    chat_with_history = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手"),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
    ])
    history = [
        HumanMessage(content="你好"),
        AIMessage(content="你好！有什么可以帮你的？"),
    ]
    messages = chat_with_history.format_messages(history=history, input="什么是Python？")
    for msg in messages:
        print(f"  {msg.__class__.__name__}: {msg.content[:30]}...")

    # 3. FewShotPromptTemplate - 少样本提示
    print("\n✅ FewShotPromptTemplate:")
    examples = [
        {"input": "开心", "output": "正面"},
        {"input": "难过", "output": "负面"},
        {"input": "一般", "output": "中性"},
    ]
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="输入: {input}\n输出: {output}"
    )
    few_shot = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="判断以下情感:",
        suffix="输入: {input}\n输出:",
        input_variables=["input"],
    )
    result = few_shot.format(input="愤怒")
    print(f"  {result[:200]}...")

    print()


# ============================================================
# 2. Output Parser 详解
# ============================================================
# 面试要点: 结构化输出是生产环境的关键
#
# Parser 类型:
# - StrOutputParser: 字符串
# - JsonOutputParser: JSON
# - PydanticOutputParser: Pydantic 对象
# - CommaSeparatedListOutputParser: 逗号分隔列表
# ============================================================

def demo_output_parsers():
    """Output Parser 详解"""
    print("=" * 60)
    print("2. Output Parser 详解")
    print("=" * 60)

    from langchain_core.output_parsers import (
        StrOutputParser,
        JsonOutputParser,
        CommaSeparatedListOutputParser,
        PydanticOutputParser,
    )
    from pydantic import BaseModel, Field

    # 1. StrOutputParser
    print("\n✅ StrOutputParser:")
    parser = StrOutputParser()
    result = parser.invoke("Hello World")
    print(f"  输入: 'Hello World' → 输出: '{result}'")

    # 2. JsonOutputParser
    print("\n✅ JsonOutputParser:")
    parser = JsonOutputParser()
    result = parser.invoke('{"name": "Alice", "age": 30}')
    print(f"  输入: JSON字符串 → 输出: {result}")

    # 3. CommaSeparatedListOutputParser
    print("\n✅ CommaSeparatedListOutputParser:")
    parser = CommaSeparatedListOutputParser()
    result = parser.invoke("apple, banana, cherry")
    print(f"  输入: 'apple, banana, cherry' → 输出: {result}")

    # 4. PydanticOutputParser
    print("\n✅ PydanticOutputParser:")
    class Person(BaseModel):
        name: str = Field(description="姓名")
        age: int = Field(description="年龄")
        hobbies: list[str] = Field(description="爱好")

    parser = PydanticOutputParser(pydantic_object=Person)
    print(f"  格式指令: {parser.get_format_instructions()[:100]}...")

    print()


# ============================================================
# 3. Memory 详解
# ============================================================
# 面试要点: Memory 是对话系统的关键
#
# Memory 类型:
# - ConversationBufferMemory: 完整历史
# - ConversationSummaryMemory: 摘要
# - ConversationBufferWindowMemory: 滑动窗口
# ============================================================

def demo_memory():
    """Memory 详解"""
    print("=" * 60)
    print("3. Memory 详解")
    print("=" * 60)

    from langchain_core.messages import HumanMessage, AIMessage

    print("""
    Memory 类型对比:
    ┌─────────────────────────┬─────────────────┬─────────────────┐
    │ 类型                     │ 特点             │ 适用场景         │
    ├─────────────────────────┼─────────────────┼─────────────────┤
    │ ConversationBuffer      │ 保存完整历史     │ 短对话           │
    │ ConversationSummary     │ 保存摘要         │ 长对话           │
    │ ConversationBufferWindow│ 保存最近N轮      │ 中等长度对话     │
    └─────────────────────────┴─────────────────┴─────────────────┘

    现代做法 (LangGraph):
    - 使用 State 对象管理历史
    - 更灵活，更可控
    - 支持持久化存储
    """)

    # 模拟对话历史
    history = [
        HumanMessage(content="你好，我是Alice"),
        AIMessage(content="你好Alice！有什么可以帮你的？"),
        HumanMessage(content="什么是Python？"),
        AIMessage(content="Python是一种编程语言..."),
    ]

    print("✅ 对话历史示例:")
    for msg in history:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content}")

    print()


# ============================================================
# 4. Runnable 组件详解
# ============================================================
# 面试要点: 理解各种 Runnable 的用途
#
# Runnable 类型:
# - RunnablePassthrough: 直接传递
# - RunnableLambda: 函数转换
# - RunnableParallel: 并行执行
# - RunnableBranch: 条件分支
# ============================================================

def demo_runnables():
    """Runnable 组件详解"""
    print("=" * 60)
    print("4. Runnable 组件详解")
    print("=" * 60)

    from langchain_core.runnables import (
        RunnablePassthrough,
        RunnableLambda,
        RunnableParallel,
        RunnableBranch,
    )

    # 1. RunnablePassthrough - 直接传递
    print("\n✅ RunnablePassthrough:")
    passthrough = RunnablePassthrough()
    result = passthrough.invoke({"key": "value"})
    print(f"  输入 = 输出: {result}")

    # 2. RunnableLambda - 函数转换
    print("\n✅ RunnableLambda:")
    def process(x: dict) -> str:
        return f"处理: {x}"

    runnable = RunnableLambda(process)
    result = runnable.invoke("test")
    print(f"  结果: {result}")

    # 3. RunnableParallel - 并行执行
    print("\n✅ RunnableParallel:")
    def add_prefix(x: str) -> str:
        return f"PREFIX: {x}"

    def add_suffix(x: str) -> str:
        return f"{x}: SUFFIX"

    parallel = RunnableParallel(
        with_prefix=RunnableLambda(add_prefix),
        with_suffix=RunnableLambda(add_suffix),
    )
    result = parallel.invoke("hello")
    print(f"  结果: {result}")

    # 4. 链式组合
    print("\n✅ 链式组合:")
    chain = (
        RunnableLambda(lambda x: x.upper())
        | RunnableLambda(lambda x: f"Processed: {x}")
    )
    result = chain.invoke("hello world")
    print(f"  'hello world' → '{result}'")

    print()


# ============================================================
# 5. Tool 定义详解
# ============================================================
# 面试要点: Tool 是 Agent 的核心能力
#
# Tool 定义方式:
# - @tool 装饰器
# - BaseTool 子类
# - StructuredTool
# ============================================================

def demo_tools():
    """Tool 定义详解"""
    print("=" * 60)
    print("5. Tool 定义详解")
    print("=" * 60)

    from langchain_core.tools import tool
    from pydantic import BaseModel, Field

    # 1. @tool 装饰器
    print("\n✅ @tool 装饰器:")
    @tool
    def search(query: str) -> str:
        """搜索信息"""
        return f"搜索结果: {query}"

    print(f"  名称: {search.name}")
    print(f"  描述: {search.description}")

    # 2. 带参数验证的 Tool
    print("\n✅ 带参数验证的 Tool:")
    class CalculatorInput(BaseModel):
        expression: str = Field(description="数学表达式")

    @tool(args_schema=CalculatorInput)
    def calculator(expression: str) -> str:
        """计算数学表达式"""
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except:
            return "计算错误"

    result = calculator.invoke({"expression": "2 + 3 * 4"})
    print(f"  2 + 3 * 4 = {result}")

    # 3. Tool 列表
    print("\n✅ Tool 列表:")
    tools = [search, calculator]
    for t in tools:
        print(f"  - {t.name}: {t.description}")

    print()


# ============================================================
# 6. 核心组件总结
# ============================================================

def demo_summary():
    """核心组件总结"""
    print("=" * 60)
    print("6. 核心组件总结")
    print("=" * 60)

    print("""
    LangChain 核心组件:
    ┌─────────────────┬─────────────────┬─────────────────┐
    │ 组件             │ 用途             │ 关键类           │
    ├─────────────────┼─────────────────┼─────────────────┤
    │ Prompt          │ 构建提示         │ ChatPromptTemplate│
    │ LLM/ChatModel  │ 调用模型         │ ChatOpenAI      │
    │ OutputParser    │ 解析输出         │ JsonOutputParser│
    │ Memory          │ 管理历史         │ State (LangGraph)│
    │ Tool            │ 定义工具         │ @tool           │
    │ Retriever       │ 检索文档         │ VectorStore     │
    │ Chain           │ 组合组件         │ LCEL            │
    └─────────────────┴─────────────────┴─────────────────┘

    面试关键点:
    1. LCEL 是核心创新，用 | 管道符组合
    2. Runnable 接口统一了 invoke/batch/stream
    3. Tool 是 Agent 的核心能力
    4. Memory 现在推荐用 LangGraph State
    """)
    print()


# ============================================================
# 主函数
# ============================================================

def main():
    demo_prompt_templates()
    demo_output_parsers()
    demo_memory()
    demo_runnables()
    demo_tools()
    demo_summary()


if __name__ == "__main__":
    main()
