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
# 6. 版本升级踩坑指南 — 生产环境真实坑
# ============================================================
# 面试要点: 被问"你遇到过 LangChain 版本升级的坑吗"是高级考点
#
# 这节覆盖真实的跨版本迁移陷阱和解决方案。
# 每个坑都来自实际生产环境，面试时能讲的越具体越好。
# ============================================================

def demo_migration_pitfalls():
    """版本升级踩坑指南 — 生产环境真实经验"""
    print("=" * 60)
    print("6. 版本升级踩坑指南 (Migration Pitfalls)")
    print("=" * 60)

    # ---- 坑 1: Import 路径大迁移 (0.1.x → 0.2.x) ----
    print("\n🕳️ 坑 1: Import 路径变更 (0.1.x → 0.2.x)")
    print("─" * 50)
    # 面试官会问: "从旧版升级到新版本，Import 路径变了怎么办？"
    # 标准答: 这是最常见的坑。LangChain 在 0.2 之后将很多模块重新组织了。
    # 比如 llms 被移到 langchain_community，core 抽象被独立出来。
    print("""
  旧版 (0.1.x)                新版 (0.2.x+)
  ─────────────────────────────────────────────────
  langchain.llms.OpenAI    →  langchain_openai.ChatOpenAI
  langchain.chains.LLMChain →  (废弃，用 LCEL)
  langchain.prompts         →  langchain_core.prompts
  langchain.memory          →  langchain_community.memory
  langchain.agents          →  langgraph.prebuilt.create_react_agent

  核心原则: langchain-core 放抽象，langchain-community 放集成，
           专用包(如 langchain-openai)放特定提供商实现。

  解救方案:
  - 用 ruff 或 pyupgrade 自动修复 import
  - 在 requirements.txt 中明确锁定所有 langchain-* 包的版本
  - CI 中加入 import 检查: python -c "from my_app import *"
""")

    # ---- 坑 2: LLMChain 废弃 (0.1.x → 0.2.x) ----
    print("\n🕳️ 坑 2: LLMChain 废弃 — 必须迁移到 LCEL")
    print("─" * 50)
    # 面试官追问: "LLMChain 被废弃了，怎么迁移？"
    # 最重要的坑：无数教程还在用 LLMChain，但它在 0.2+ 已废弃
    print("""
  旧写法 (已废弃):
    from langchain.chains import LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input="...")

  新写法 (LCEL):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"input": "..."})

  关键差异:
  1. run() → invoke(): run() 已在 0.2+ 废弃
  2. 参数传递: 旧版用关键字参数，新版用字典
  3. 类型安全: LCEL 链有自动类型推断

  面试加分点: "LLMChain 废弃是因为它不支持流式、异步，
  且与 Runnable 接口不兼容。LangChain 团队推荐完全迁移到 LCEL。"
""")

    # ---- 坑 3: ChatPromptTemplate 的 format_strings 陷阱 ----
    print("\n🕳️ 坑 3: ChatPromptTemplate.format_messages() 参数冲突")
    print("─" * 50)
    # 这是最隐蔽的坑！实际调试常常花半天
    print("""
  问题: prompt 中有 {role} 占位符时，format_messages(role="...") 会抛异常

  # ❌ 错误写法
  prompt = ChatPromptTemplate.from_messages([
      ("system", "你是一个{role}。"),
      ("user", "{input}")
  ])
  messages = prompt.format_messages(role="expert", input="问题")
  # KeyError: 'role' — Python str.format() 试图解析 role

  # ✅ 正确写法: 用 partial() 预填充
  prompt = ChatPromptTemplate.from_messages([
      ("system", "你是一个{role}。"),
      ("user", "{input}")
  ])
  prompt_partial = prompt.partial(role="expert")
  messages = prompt_partial.format_messages(input="问题")

  原因: Python 的 str.format() 与 LangChain 的占位符解析冲突。
  format_messages() 底层调用 str.format()，如果参数名恰好
  等于占位符名，会先被 Python 解析掉导致 LangChain 找不到。

  总结: 只要 prompt 中有 {role}/{skill} 等占位符，
  一律用 .partial() 而不是传参给 format_messages()。
""")

    # ---- 坑 4: Agent 框架重构 (0.2.x → 0.3.x) ----
    print("\n🕳️ 坑 4: Agent 框架从 langchain.agents 迁移到 LangGraph")
    print("─" * 50)
    print("""
  旧版 (0.2.x):
    from langchain.agents import initialize_agent, AgentType
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

  新版 (0.3.x+):
    from langgraph.prebuilt import create_react_agent
    agent = create_react_agent(llm, tools)

  关键变化:
  1. initialize_agent 完全废弃 — LangGraph 替代了所有 Agent 创建
  2. AgentType 枚举废弃 — 现在用 create_react_agent() 统一入口
  3. AgentExecutor 废弃 — LangGraph 的 StateGraph 管理执行循环
  4. Agent 状态管理从 Memory 变为 LangGraph 的 State

  面试技巧: "Agent 从 Chain 变为 Graph，核心原因是 Graph 支持
  循环、条件分支、多步骤交互，而 Chain 只能单向执行。"
""")

    # ---- 坑 5: 包拆分导致版本冲突 (0.2.x → 0.3.x) ----
    print("\n🕳️ 坑 5: 包拆分导致版本不一致冲突")
    print("─" * 50)
    print("""
  现象:
    ImportError: cannot import name 'BaseMessage' from 'langchain_core.messages'
    AttributeError: module 'langchain_core' has no attribute '...'

  根因: langchain-core 版本与 langchain 版本不匹配
  例如: langchain 1.3.2 需要 langchain-core >= 1.4.0，
        但如果 pip 锁定了旧版本就会冲突。

  解决方案:
  # 1. 统一更新所有 langchain 相关包
  pip install --upgrade langchain langchain-core langchain-community

  # 2. 锁定版本范围而非精确版本
  # requirements.txt:
  langchain>=1.0,<2.0
  langchain-core>=1.0,<2.0
  langchain-community>=0.4,<1.0

  # 3. 用 pip check 验证依赖一致性
  pip check

  # 4. CI 中加版本一致性检查
  python -c "import langchain; import langchain_core; print('OK')"
""")

    # ---- 坑 6: ChatOpenAI 参数变化 (1.0.x) ----
    print("\n🕳️ 坑 6: ChatOpenAI 参数名变化 (1.0.x)")
    print("─" * 50)
    print("""
  # ❌ 旧版 (0.3.x) — 以下参数已重命名或废弃:
  model_name="gpt-4"    # → model="gpt-4"
  max_tokens=1000       # → max_completion_tokens=1000
  request_timeout=30    # → timeout=30

  # ✅ 新版 (1.0.x+):
  from langchain_openai import ChatOpenAI
  llm = ChatOpenAI(
      model="gpt-4o",
      temperature=0.7,
      max_completion_tokens=1000,
      timeout=30,
  )

  面试加分: "OpenAI 自己的 API 改了参数名，LangChain 跟着适配。
  关注 OpenAI 的 CHANGELOG 可以提前预防这类变更。"
""")

    # ---- 坑 7: Memory 类的 import 变化 ----
    print("\n🕳️ 坑 7: 对话 Memory 的 import 路径变更")
    print("─" * 50)
    print("""
  旧版 (0.1.x):
    from langchain.memory import ConversationBufferMemory

  过渡期 (0.2.x):  # 会警告但还能用
    from langchain.memory import ConversationBufferMemory

  新版 (0.3.x+):
    from langchain_community.memory import ConversationBufferMemory

  注意: 大部分 Memory 类虽然还能 import 但已经不推荐使用。
  LangGraph 用 Checkpoint 系统替代了传统的 Memory。

  替代方案:
  - 简单场景: 把历史消息放进 prompt 里
  - 多轮对话: 用 LangGraph 的 MemorySaver (checkpoint)
  - RAG: 用向量数据库 + 对话历史混合检索
""")

    # ---- 坑 8: Pydantic v1 → v2 兼容性 ----
    print("\n🕳️ 坑 8: Pydantic v1 vs v2 版本兼容")
    print("─" * 50)
    print("""
  LangChain 0.3+ 全面拥抱 Pydantic v2，与 v1 不兼容。

  常见报错:
    pydantic.errors.PydanticImportError:
    'pydantic:BaseSettings has been moved...'

  解决方案:
  # 1. 确保只用 Pydantic v2
  pip install "pydantic>=2.0"

  # 2. 检查依赖中是否有锁死 pydantic v1 的包
  pip list | grep pydantic

  # 3. 代码迁移: 如果还有 v1 代码
  # v1                          # v2
  class Config:                 model_config = ConfigDict(...)
      orm_mode = True           from_attributes = True
  @validator('field')           @field_validator('field')
  regex=r'...'                  pattern=r'...'

  面试要点: "Pydantic v1→v2 是 Python 生态最大的 breaking change 之一。
  理解 LangChain 为什么坚持 v2——类型安全、性能提升、更好的 JSON Schema 生成。"
""")

    # ---- 坑 9: Callback 系统变化 ----
    print("\n🕳️ 坑 9: Callback 系统降级为可选")
    print("─" * 50)
    print("""
  旧版 (0.1.x): Callback 是核心机制，每个组件都支持
  新版 (0.3.x+): Callback 仍然存在但官方推荐 LangSmith 替代

  迁移路径:
  # 旧版: 自定义 Callback
  class MyCallback(BaseCallbackHandler):
      def on_llm_start(self, ...):
          ...

  # 新版: 用 LangSmith + @traceable 替代
  from langsmith import traceable

  @traceable(run_type="chain")
  def my_function(input):
      ...

  注意: Callback 不会废弃，但新项目建议直接用 LangSmith。
  如果要兼容旧代码，Callback 仍然可以正常工作。
""")

    # ---- 升级检查清单 ----
    print("\n📋 版本升级检查清单 (面试速记)")
    print("─" * 50)
    print("""
  □ 1. Import 路径: llms→langchain_openai, chains→废弃
  □ 2. LLMChain → LCEL: run()→invoke(), 字典传参
  □ 3. Prompt: format_messages() 参数冲突 → .partial()
  □ 4. Agent: initialize_agent → create_react_agent (LangGraph)
  □ 5. 包版本: pip check 验证一致性
  □ 6. ChatOpenAI: model_name→model, max_tokens→max_completion_tokens
  □ 7. Memory: langchain.memory → langchain_community.memory → LangGraph Checkpoint
  □ 8. Pydantic: 确保 v2，from_attributes 替代 orm_mode
  □ 9. Callback: 可选 → 建议用 LangSmith
  □ 10. CI 验证: python -c "import langchain; import langchain_core"
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
    demo_migration_pitfalls()


if __name__ == "__main__":
    main()
