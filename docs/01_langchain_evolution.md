# """

::: code-group

```python [01_langchain_evolution.py]
"""
LangChain 版本演进演示

展示 LangChain 从 0.0.x 到 1.3.x 的关键变化
"""

# ============================================================
# 1. 旧版 API (0.0.x - 0.1.x) - 已废弃
# ============================================================
# 面试要点: 了解旧版 API，说明为什么被废弃
#
# 旧版特点:
# - LLM 类直接调用
# - Chain 类封装逻辑
# - Memory 类管理上下文
# - 缺乏标准化接口
# ============================================================

def demo_old_api():
    """
    旧版 API 示例 (0.0.x - 0.1.x)

    这些 API 已废弃，但面试中可能会问到
    """
    print("=" * 60)
    print("1. 旧版 API (0.0.x - 0.1.x) - 已废弃")
    print("=" * 60)

    print("""
    # 旧版写法 (已废弃):
    from langchain.llms import OpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    llm = OpenAI(temperature=0.7)
    prompt = PromptTemplate(template="...", input_variables=[...])
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input="...")

    # 问题:
    # 1. 每个组件接口不统一
    # 2. 难以组合和复用
    # 3. 缺乏流式支持
    # 4. 类型提示不完善
    """)
    print()


# ============================================================
# 2. LCEL (0.2.x+) - 核心创新
# ============================================================
# 面试要点: LCEL 是 LangChain 最重要的创新
#
# LCEL (LangChain Expression Language):
# - 使用 | 管道符组合组件
# - 统一的 Runnable 接口
# - 支持流式、批处理、异步
# - 自动类型推断
# ============================================================

def demo_lcel():
    """
    LCEL (LangChain Expression Language) 示例

    这是 LangChain 0.2.x 引入的核心创新
    """
    print("=" * 60)
    print("2. LCEL (0.2.x+) - 核心创新")
    print("=" * 60)

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda

    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手"),
        ("user", "{input}")
    ])

    # 创建输出解析器
    output_parser = StrOutputParser()

    # 使用 LCEL 组合链 (管道符 |)
    # chain = prompt | llm | output_parser

    # 演示 Runnable 组件
    print("\n✅ LCEL 核心概念:")
    print("  1. Runnable 接口 - 所有组件都实现 Runnable")
    print("  2. 管道符 | - 组合组件")
    print("  3. 自动类型推断")
    print("  4. 统一的 invoke/batch/stream 接口")

    # 示例: 使用 RunnableLambda
    def upper(text: str) -> str:
        return text.upper()

    upper_runnable = RunnableLambda(upper)
    result = upper_runnable.invoke("hello world")
    print(f"\n  RunnableLambda 示例: 'hello world' → '{result}'")

    # 示例: 使用 RunnablePassthrough
    passthrough = RunnablePassthrough()
    result = passthrough.invoke({"key": "value"})
    print(f"  RunnablePassthrough 示例: {result}")

    print("\n✅ LCEL 链式调用:")
    print("  chain = prompt | llm | output_parser")
    print("  result = chain.invoke({'input': '...'})")
    print()


# ============================================================
# 3. Runnable 接口 (0.2.x+) - 统一抽象
# ============================================================
# 面试要点: 理解 Runnable 接口的三种方法
#
# Runnable 接口:
# - invoke(input): 单次调用
# - batch(inputs): 批量调用
# - stream(input): 流式调用
# - ainvoke/abatch/astream: 异步版本
# ============================================================

def demo_runnable():
    """
    Runnable 接口演示

    所有 LangChain 组件都实现这个接口
    """
    print("=" * 60)
    print("3. Runnable 接口 (0.2.x+) - 统一抽象")
    print("=" * 60)

    from langchain_core.runnables import RunnableLambda, RunnableParallel

    # 创建多个 Runnable
    def add_one(x: int) -> int:
        return x + 1

    def multiply_two(x: int) -> int:
        return x * 2

    add_runnable = RunnableLambda(add_one)
    multiply_runnable = RunnableLambda(multiply_two)

    # 1. invoke - 单次调用
    result = add_runnable.invoke(5)
    print(f"\n✅ invoke: add_one(5) = {result}")

    # 2. batch - 批量调用
    results = add_runnable.batch([1, 2, 3, 4, 5])
    print(f"✅ batch: add_one([1,2,3,4,5]) = {results}")

    # 3. 并行执行
    parallel = RunnableParallel(
        added=add_runnable,
        multiplied=multiply_runnable,
    )
    result = parallel.invoke(5)
    print(f"✅ parallel: add_one(5)={result['added']}, multiply_two(5)={result['multiplied']}")

    # 4. 链式调用
    chain = add_runnable | multiply_runnable
    result = chain.invoke(5)
    print(f"✅ chain: add_one(5) → multiply_two = {result}")

    print()


# ============================================================
# 4. 结构化输出 (0.3.x+) - Pydantic 集成
# ============================================================
# 面试要点: 结构化输出是生产环境的关键需求
#
# 结构化输出:
# - 使用 Pydantic 定义输出格式
# - 自动验证输出
# - 支持 JSON Schema
# ============================================================

def demo_structured_output():
    """
    结构化输出演示

    0.3.x 引入的 Pydantic 集成
    """
    print("=" * 60)
    print("4. 结构化输出 (0.3.x+) - Pydantic 集成")
    print("=" * 60)

    from pydantic import BaseModel, Field
    from langchain_core.output_parsers import JsonOutputParser

    # 定义输出格式
    class Person(BaseModel):
        """人物信息"""
        name: str = Field(description="姓名")
        age: int = Field(description="年龄")
        occupation: str = Field(description="职业")

    # 创建解析器
    parser = JsonOutputParser(pydantic_object=Person)

    print(f"\n✅ Pydantic 模型定义输出格式:")
    print(f"  {parser.get_format_instructions()}")

    print("\n✅ 使用方式:")
    print("  chain = prompt | llm | parser")
    print("  result = chain.invoke({...})")
    print("  # result 自动验证为 Person 对象")

    print()


# ============================================================
# 5. 版本对比总结
# ============================================================

def demo_version_comparison():
    """版本对比总结"""
    print("=" * 60)
    print("5. LangChain 版本对比总结")
    print("=" * 60)

    print("""
    ┌─────────────────┬─────────────────┬─────────────────┐
    │ 版本             │ 核心变化         │ 面试要点         │
    ├─────────────────┼─────────────────┼─────────────────┤
    │ 0.0.x           │ 初始版本         │ 了解历史         │
    │ 0.1.x           │ 首个稳定版       │ 基础 API        │
    │ 0.2.x           │ LCEL 引入        │ 管道符组合       │
    │ 0.3.x           │ 模块化拆分       │ 包结构变化       │
    │ 1.0.x           │ 正式稳定版       │ API 稳定        │
    │ 1.3.x           │ 性能优化         │ 最新特性         │
    └─────────────────┴─────────────────┴─────────────────┘

    关键变化:
    1. LCEL (0.2.x): 用 | 管道符组合组件，统一 Runnable 接口
    2. 模块化 (0.3.x): 拆分为 langchain-core, langchain-community
    3. 结构化输出 (0.3.x): Pydantic 集成，自动验证
    4. 流式支持 (0.2.x+): 统一的 stream 接口
    5. 异步支持 (0.2.x+): ainvoke, abatch, astream

    包结构 (0.3.x+):
    - langchain-core: 核心抽象 (Runnable, LCEL)
    - langchain-community: 社区集成
    - langchain: 主包 (Chain, Agent, Memory)
    - langchain-openai: OpenAI 集成
    """)
    print()


# ============================================================
# 主函数
# ============================================================

def main():
    demo_old_api()
    demo_lcel()
    demo_runnable()
    demo_structured_output()
    demo_version_comparison()


if __name__ == "__main__":
    main()
```

:::
