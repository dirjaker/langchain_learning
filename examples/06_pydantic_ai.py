"""
Pydantic AI 框架学习

深入讲解 Pydantic AI 的核心概念：Agent、Tool、依赖注入、结构化输出、
流式输出、多 Agent 协作，并与 LangChain 进行对比。
"""

# ============================================================
# 1. Pydantic AI 基础概念
# ============================================================
# 面试要点: Pydantic AI 是什么？与 LangChain 有何不同？
#
# Pydantic AI 核心概念:
# - Agent: 封装了 LLM 调用 + 工具使用 + 结构化输出验证的核心组件
# - Model: 模型抽象层，支持 OpenAI/Anthropic/Google 等多个 Provider
# - Tool: 工具函数，使用 @agent.tool 装饰器注册
# - RunContext: 运行上下文，携带依赖注入的数据
# - output_type: 使用 Pydantic 模型定义结构化输出类型
#
# 与 LangChain 的关键区别:
# - LangChain: 以 Chain/Graph 为中心，Agent 是其中一种节点
# - Pydantic AI: 以 Agent 为中心，一切围绕 Agent 展开
# - LangChain: 需要手动构建工作流图
# - Pydantic AI: Agent 自动处理工具调用循环
# ============================================================

def demo_basic_concepts():
    """
    Pydantic AI 基础概念

    展示 Agent、Model、Tool 三大核心概念
    """
    print("=" * 60)
    print("1. Pydantic AI 基础概念")
    print("=" * 60)

    from pydantic_ai import Agent, RunContext
    from dataclasses import dataclass

    # ---- 核心概念 1: Agent ----
    # Agent 是 Pydantic AI 的核心，封装了 LLM + 工具 + 输出验证
    # 使用 'test' 模型（无需 API key，用于演示）
    agent = Agent(
        'test',  # 模型名（test 是内置测试模型，无需真实 API）
        system_prompt='你是一个乐于助人的 AI 助手，使用中文回答。',
    )

    print("\n✅ 核心概念 1: Agent")
    print("  Agent 封装了模型调用 + 工具 + 输出验证")

    # ---- 核心概念 2: 运行 Agent ----
    result = agent.run_sync('你好，请用一句话介绍你自己')
    print(f"\n✅ 核心概念 2: 运行 Agent")
    print(f"  输入: '你好，请用一句话介绍你自己'")
    print(f"  输出: {result.output}")
    print(f"  消息数: {len(result.all_messages())}")

    # ---- 核心概念 3: Tool(工具) ----
    @agent.tool
    def get_current_time(ctx: RunContext[object]) -> str:
        """
        获取当前时间的工具函数。

        Args:
            ctx: 运行上下文

        Returns:
            格式化的时间字符串
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @agent.tool_plain
    def calculate_bmi(weight_kg: float, height_m: float) -> str:
        """
        计算 BMI 指数。

        Args:
            weight_kg: 体重（千克）
            height_m: 身高（米）

        Returns:
            BMI 计算结果
        """
        if height_m <= 0:
            return "错误: 身高必须大于 0"
        bmi = weight_kg / (height_m ** 2)
        return f"BMI = {bmi:.1f}"

    print(f"\n✅ 核心概念 3: Tool(工具)")
    print(f"  @agent.tool: 需要 ctx 参数的工具")
    print(f"  @agent.tool_plain: 无需 ctx 的纯函数工具")

    # 运行带工具的 Agent
    result = agent.run_sync('现在几点了？')
    print(f"\n  带工具运行结果: {result.output}")

    print()

    # ---- 面试要点总结 ----
    print("📌 面试要点:")
    print("  1. Agent = 模型 + 工具 + 输出类型 + 系统提示")
    print("  2. 'test' 模型是内置测试模型，无需真实 API")
    print("  3. @agent.tool 用于需要运行上下文的工具")
    print("  4. @agent.tool_plain 用于纯函数工具（无需上下文）")
    print()


# ============================================================
# 2. 定义 Agent — system_prompt、result_type、Tool 装饰器
# ============================================================
# 面试要点: Agent 是 Pydantic AI 的一等公民
#
# Agent 初始化参数:
# - model: 模型名或 Model 实例
# - system_prompt: 系统提示词
# - output_type: 输出 Pydantic 模型（结构化输出）
# - deps_type: 依赖类型（用于依赖注入）
# - tools: 工具列表
# - retries: 重试次数
# ============================================================

def demo_agent_definition():
    """
    Agent 定义详解

    展示 system_prompt、output_type、工具注册等
    """
    print("=" * 60)
    print("2. Agent 定义详解")
    print("=" * 60)

    from pydantic_ai import Agent, RunContext
    from pydantic import BaseModel, Field
    from dataclasses import dataclass
    from typing import Optional

    # ---- 方式 1: 基本 Agent（字符串输出） ----
    print("\n✅ 方式 1: 基本 Agent（默认 str 输出）")

    basic_agent = Agent(
        'test',
        system_prompt='你是一个 Python 专家，用中文回答。',
        name='BasicAgent',  # Agent 名称（用于日志/追踪）
    )

    result = basic_agent.run_sync('Python 的列表推导式是什么？')
    print(f"  默认输出类型: str")
    print(f"  结果: {result.output}")

    # ---- 方式 2: 带结构化输出的 Agent ----
    print("\n✅ 方式 2: 带结构化输出的 Agent")

    class WeatherReport(BaseModel):
        """天气报告结构"""
        city: str = Field(description="城市名称")
        temperature: float = Field(description="温度（摄氏度）")
        condition: str = Field(description="天气状况")
        humidity: float = Field(description="湿度百分比")

    weather_agent = Agent(
        'test',
        output_type=WeatherReport,
        system_prompt='你是一个天气预报员。',
    )

    result = weather_agent.run_sync('北京今天天气怎么样？')
    print(f"  输出类型: WeatherReport (Pydantic 模型)")
    print(f"  结果: {result.output}")

    # ---- 方式 3: 带工具和重试的 Agent ----
    print("\n✅ 方式 3: 带工具和重试配置的 Agent")

    search_agent = Agent(
        'test',
        system_prompt='你是一个搜索助手。',
        retries=3,  # 工具/输出验证失败时最多重试 3 次
    )

    @search_agent.tool_plain
    def search_database(query: str) -> str:
        """在数据库中搜索信息。"""
        return f"搜索结果: 找到关于 '{query}' 的 5 条记录"

    result = search_agent.run_sync('搜索关于机器学习的内容')
    print(f"  retries=3 表示工具调用失败或输出验证失败时最多重试 3 次")
    print(f"  结果: {result.output}")

    # ---- 方式 4: 带 instructions 动态指令 ----
    print("\n✅ 方式 4: 动态指令 (instructions)")

    translator = Agent('test')

    # main_instructions 是默认指令
    @translator.instructions
    def main_instructions(ctx: RunContext) -> str:
        return '你是一个翻译助手，默认翻译成中文。'

    # 运行时可以覆盖 instructions
    result = translator.run_sync(
        'Hello, how are you?',
        instructions='请翻译成法语。',  # 运行时覆盖指令
    )
    print(f"  默认指令: 翻译成中文")
    print(f"  运行时指令覆盖: 翻译成法语")
    print(f"  结果: {result.output}")

    print()

    # ---- 面试要点总结 ----
    print("📌 面试要点:")
    print("  1. output_type 决定 Agent 的输出结构（推荐使用 Pydantic BaseModel）")
    print("  2. retries 参数控制工具/输出的重试次数")
    print("  3. system_prompt 可带占位符，通过 deps 注入值")
    print("  4. instructions 装饰器提供动态指令，运行时可通过参数覆盖")
    print("  5. name 用于日志追踪和调试")
    print()


# ============================================================
# 3. 结构化输出 — Pydantic 模型嵌套、验证、自定义验证器
# ============================================================
# 面试要点: Pydantic AI 最大的优势之一是结构化输出
#
# Pydantic AI 使用 Pydantic 模型定义输出结构：
# - 自动 JSON Schema 生成
# - 自动类型验证
# - 支持嵌套模型
# - 支持 Field 约束
# - 支持自定义验证器
# ============================================================

def demo_structured_output():
    """
    结构化输出详解

    展示 Pydantic 模型作为输出类型，包括嵌套、验证等
    """
    print("=" * 60)
    print("3. 结构化输出 (Structured Output)")
    print("=" * 60)

    from pydantic_ai import Agent
    from pydantic import BaseModel, Field, field_validator
    from typing import List, Optional
    from enum import Enum

    # ---- 示例 1: 简单结构化输出 ----
    # 面试技巧: Enum + BaseModel 是 Pydantic AI 最基础的模式，面试几乎必考
    # 面试官追问: "为什么用 Enum 而不是 str?"
    #   答: Enum 提供编译时类型安全，防止 typo（如 'positive'→'postive'），
    #   且 LLM 更容易理解有限枚举值，输出更准确
    print("\n✅ 示例 1: 简单结构化输出")

    class Sentiment(str, Enum):
        POSITIVE = "positive"
        NEGATIVE = "negative"
        NEUTRAL = "neutral"

    class SentimentResult(BaseModel):
        """情感分析结果"""
        sentiment: Sentiment = Field(description="情感分类")
        confidence: float = Field(ge=0.0, le=1.0, description="置信度 0-1")
        reason: str = Field(description="分析原因")

    sentiment_agent = Agent(
        'test',
        output_type=SentimentResult,
        system_prompt='分析文本情感，返回 sentiment、confidence、reason。',
    )

    result = sentiment_agent.run_sync('今天天气真好，心情很棒！')
    print(f"  输出类型: SentimentResult")
    print(f"  结果: {result.output}")

    # ---- 示例 2: 嵌套模型 ----
    # 面试重点: 嵌套模型是区分初级和高级的考点
    # 面试官会问: "如何处理复杂的嵌套 JSON 输出?"
    #   标准答: 用 Pydantic 嵌套模型 + Field(description=) 告诉 LLM 每个字段的含义
    #   这样 LLM 会自动填充所有嵌套层级，无需手动解析 JSON
    print("\n✅ 示例 2: 嵌套 Pydantic 模型")

    class Address(BaseModel):
        """地址信息"""
        street: str = Field(description="街道地址")
        city: str = Field(description="城市")
        zipcode: str = Field(description="邮编")

    class Contact(BaseModel):
        """联系信息"""
        email: str = Field(description="邮箱")
        phone: Optional[str] = Field(default=None, description="电话")

    class Person(BaseModel):
        """人物信息 — 展示嵌套模型"""
        name: str = Field(description="姓名")
        age: int = Field(ge=0, le=150, description="年龄")
        address: Address = Field(description="地址")
        contact: Contact = Field(description="联系方式")
        skills: List[str] = Field(description="技能列表")

    person_agent = Agent(
        'test',
        output_type=Person,
        system_prompt='从文本中提取人物信息，使用中文。',
    )

    result = person_agent.run_sync(
        '张三，28岁，住在北京市海淀区中关村大街1号100080，'
        '邮箱zhangsan@example.com，电话13800138000，'
        '会Python、Java、机器学习'
    )
    print(f"  输出类型: Person（嵌套 Address + Contact）")
    print(f"  结果: {result.output}")

    # ---- 示例 3: 自定义验证器 ----
    print("\n✅ 示例 3: 带自定义验证器的输出")

    class CodeReview(BaseModel):
        """代码审查结果"""
        file_path: str = Field(description="文件路径")
        issues: List[str] = Field(description="发现的问题")
        severity: str = Field(description="严重程度")
        suggestion: str = Field(description="改进建议")

        @field_validator('severity')
        @classmethod
        def validate_severity(cls, v: str) -> str:
            """
            自定义验证器：确保严重程度在允许范围内

            面试要点: field_validator 用于输出验证
            """
            allowed = {'critical', 'high', 'medium', 'low', 'info'}
            v_lower = v.lower()
            if v_lower not in allowed:
                # 返回最接近的匹配（面试中可展示自定义逻辑）
                return 'medium'
            return v_lower

    review_agent = Agent(
        'test',
        output_type=CodeReview,
        system_prompt='审查代码并给出意见。',
    )

    result = review_agent.run_sync('审查 main.py 中的代码')
    print(f"  输出类型: CodeReview（含 field_validator）")
    print(f"  结果: {result.output}")

    # ---- 示例 4: Optional 和 Union 类型 ----
    print("\n✅ 示例 4: Optional 和联合类型")

    class SearchResult(BaseModel):
        """搜索结果 — 展示可选字段"""
        query: str = Field(description="搜索查询")
        answer: str = Field(description="回答（如有）")
        sources: Optional[List[str]] = Field(
            default=None,
            description="信息来源（可能为空）"
        )
        is_complete: bool = Field(default=True, description="是否完整")

    search_agent = Agent(
        'test',
        output_type=SearchResult,
        system_prompt='回答搜索查询，可以提供信息来源。',
    )

    result = search_agent.run_sync('什么是零知识证明？')
    print(f"  输出类型: SearchResult（含 Optional[List[str]]）")
    print(f"  结果: {result.output}")

    print()

    # ---- 面试要点总结 ----
    print("📌 面试要点:")
    print("  1. Pydantic AI 的核心创新：类型即 Schema（Pydantic 模型自动生成 JSON Schema）")
    print("  2. 支持 Enum、嵌套模型、Optional、List、Union 等复杂类型")
    print("  3. field_validator 可用于自定义输出验证（面试常问）")
    print("  4. Field(ge=, le=) 等约束会自动包含在 Schema 中")
    print("  5. 输出类型验证失败时，Agent 会自动重试（通过 retries 配置）")
    print("  6. 对比 LangChain: LangChain 的 OutputParser 需要手动实现，Pydantic AI 自动处理")
    print()


# ============================================================
# 4. 依赖注入 — deps_type、RunContext、运行时传递上下文
# ============================================================
# 面试要点: Pydantic AI 的依赖注入系统是其核心优势
#
# 依赖注入 (Dependency Injection):
# - 通过 deps_type 声明依赖类型
# - 通过 run_sync(deps=...) 传递依赖实例
# - 工具函数通过 RunContext[DepType] 访问依赖
# - 支持复杂对象（数据库连接、配置、客户端等）
# ============================================================

def demo_dependency_injection():
    """
    依赖注入详解

    展示 deps_type、RunContext、运行时依赖传递
    """
    print("=" * 60)
    print("4. 依赖注入 (Dependency Injection)")
    print("=" * 60)

    from pydantic_ai import Agent, RunContext
    from dataclasses import dataclass, field
    from typing import Dict, List

    # ---- 示例 1: 基本依赖注入 ----
    # 面试重点: 依赖注入是 Pydantic AI 区别于 LangChain 的核心特性
    # 面试官会问: "Pydantic AI 的依赖注入和 FastAPI 的 Depends() 有什么异同?"
    #   答: 理念相同——都解耦了"创建"和"使用"。Pydantic AI 通过 deps_type + RunContext
    #   实现，在 Agent 运行时通过 run_sync(deps=...) 注入。
    #   FastAPI 用 Depends() 函数，Pydantic AI 用 dataclass + 类型标注。
    print("\n✅ 示例 1: 基本依赖注入")

    @dataclass
    class AppConfig:
        """应用配置 — 作为依赖注入 Agent"""
        app_name: str
        version: str
        debug: bool = False

    config_agent = Agent(
        'test',
        deps_type=AppConfig,
        system_prompt='你是一个应用助手，了解应用配置。',
    )

    @config_agent.tool
    def get_app_info(ctx: RunContext[AppConfig]) -> str:
        """
        获取应用信息（从依赖中读取配置）。

        Args:
            ctx: 运行上下文，ctx.deps 是 AppConfig 实例
        """
        return (
            f"应用名: {ctx.deps.app_name}, "
            f"版本: {ctx.deps.version}, "
            f"调试模式: {ctx.deps.debug}"
        )

    # 运行时传递依赖
    result = config_agent.run_sync(
        '请告诉我应用信息',
        deps=AppConfig(app_name="MyApp", version="1.0.0", debug=True)
    )
    print(f"  依赖类型: AppConfig")
    print(f"  运行时传递: AppConfig(app_name='MyApp', version='1.0.0', debug=True)")
    print(f"  工具通过 ctx.deps 访问")
    print(f"  结果: {result.output}")

    # ---- 示例 2: 模拟数据库连接依赖 ----
    print("\n✅ 示例 2: 模拟数据库连接依赖（面试高频）")

    @dataclass
    class DatabaseConnection:
        """模拟数据库连接"""
        db_url: str
        _connected: bool = field(default=False, init=False)

        def connect(self):
            """建立连接（模拟）"""
            self._connected = True
            return f"已连接到 {self.db_url}"

        def query(self, sql: str) -> List[Dict]:
            """执行查询（模拟）"""
            if not self._connected:
                return [{"error": "未连接"}]
            # 模拟查询结果
            return [
                {"id": 1, "name": "Alice", "role": "工程师"},
                {"id": 2, "name": "Bob", "role": "设计师"},
            ]

    db_agent = Agent(
        'test',
        deps_type=DatabaseConnection,
        system_prompt='你是一个数据库助手。',
    )

    @db_agent.tool
    def connect_db(ctx: RunContext[DatabaseConnection]) -> str:
        """连接到数据库并返回状态。"""
        return ctx.deps.connect()

    @db_agent.tool
    def query_users(ctx: RunContext[DatabaseConnection], department: str) -> str:
        """查询部门用户。"""
        results = ctx.deps.query(f"SELECT * FROM users WHERE dept='{department}'")
        return str(results)

    # 创建数据库连接并传递
    db = DatabaseConnection(db_url="postgresql://localhost:5432/mydb")
    result = db_agent.run_sync(
        '连接数据库并查询工程师部门的用户',
        deps=db,
    )
    print(f"  依赖: DatabaseConnection（模拟数据库）")
    print(f"  connect_db 工具通过 ctx.deps.connect() 建立连接")
    print(f"  query_users 工具通过 ctx.deps.query() 执行查询")
    print(f"  结果: {result.output}")

    # ---- 示例 3: 模拟 API 客户端依赖 ----
    print("\n✅ 示例 3: 模拟 API 客户端依赖")

    @dataclass
    class WeatherAPIClient:
        """模拟天气 API 客户端"""
        api_key: str
        base_url: str = "https://api.weather.com"

        def get_weather(self, city: str) -> dict:
            """模拟获取天气数据"""
            return {
                "city": city,
                "temperature": 22.5,
                "condition": "晴朗",
                "humidity": 65,
            }

    weather_agent = Agent(
        'test',
        deps_type=WeatherAPIClient,
        system_prompt='你是一个天气查询助手。使用中文回答。',
    )

    @weather_agent.tool
    def fetch_weather(ctx: RunContext[WeatherAPIClient], city: str) -> dict:
        """
        从 API 获取天气数据。

        Args:
            ctx: 运行上下文
            city: 城市名
        """
        return ctx.deps.get_weather(city)

    result = weather_agent.run_sync(
        '查询北京的天气',
        deps=WeatherAPIClient(api_key="sk-test"),
    )
    print(f"  依赖: WeatherAPIClient（模拟 API 客户端）")
    print(f"  fetch_weather 工具通过 ctx.deps 访问 API")
    print(f"  结果: {result.output}")

    # ---- 示例 4: 工具通过 ctx 获取更多上下文 ----
    print("\n✅ 示例 4: RunContext 的其他信息")

    @dataclass
    class SimpleDeps:
        name: str

    ctx_agent = Agent('test', deps_type=SimpleDeps)

    @ctx_agent.tool
    def show_context_info(ctx: RunContext[SimpleDeps]) -> str:
        """展示 RunContext 提供的所有信息。"""
        info = {
            "deps_name": ctx.deps.name,
            "prompt": ctx.prompt[:50] + "..." if len(ctx.prompt) > 50 else ctx.prompt,
            "retry_count": ctx.retry,
            "tool_name": ctx.tool_name,
        }
        return str(info)

    result = ctx_agent.run_sync(
        '显示当前上下文信息',
        deps=SimpleDeps(name="DepsTest"),
    )
    print(f"  RunContext 提供: deps, prompt, retry, tool_name 等")
    print(f"  结果: {result.output}")

    print()

    # ---- 面试要点总结 ----
    print("📌 面试要点:")
    print("  1. deps_type 声明依赖类型（用于静态类型检查）")
    print("  2. 运行时通过 run_sync(deps=...) 传递依赖实例")
    print("  3. 工具函数通过 RunContext[DepType] 类型注解访问依赖")
    print("  4. RunContext 还提供 prompt, retry, messages 等信息")
    print("  5. 对比 FastAPI Depends: Pydantic AI 的依赖注入类似 FastAPI")
    print("  6. 对比 LangChain: LangChain 没有内置依赖注入，需要手动传递")
    print()


# ============================================================
# 5. 工具定义与调用 — 同步/异步工具、错误处理、工具链
# ============================================================
# 面试要点: 工具是 Agent 与外部世界交互的桥梁
#
# 工具类型:
# - @agent.tool: 需要 RunContext 的工具
# - @agent.tool_plain: 纯函数工具（无需上下文）
# - 同步/异步工具均支持
# - 工具可以返回 str, dict, Pydantic 模型
# ============================================================

def demo_tools():
    """
    工具定义与调用详解

    展示各种工具类型、错误处理、工具链
    """
    print("=" * 60)
    print("5. 工具定义与调用")
    print("=" * 60)

    from pydantic_ai import Agent, RunContext
    from pydantic import BaseModel, Field
    from dataclasses import dataclass

    # ---- 示例 1: 同步工具 ----
    print("\n✅ 示例 1: 同步工具（tool + tool_plain）")

    math_agent = Agent('test', system_prompt='你是一个数学助手。')

    @math_agent.tool_plain
    def add(a: float, b: float) -> str:
        """加法运算。"""
        return f"{a} + {b} = {a + b}"

    @math_agent.tool_plain
    def multiply(a: float, b: float) -> str:
        """乘法运算。"""
        return f"{a} × {b} = {a * b}"

    result = math_agent.run_sync('计算 15 + 27 的结果')
    print(f"  工具: add, multiply（tool_plain 纯函数）")
    print(f"  结果: {result.output}")

    # ---- 示例 2: 工具返回 Pydantic 模型 ----
    print("\n✅ 示例 2: 工具返回 Pydantic 模型")

    class StockInfo(BaseModel):
        """股票信息"""
        symbol: str = Field(description="股票代码")
        price: float = Field(description="当前价格")
        change_pct: float = Field(description="涨跌幅百分比")

    stock_agent = Agent('test')

    @stock_agent.tool_plain
    def get_stock_price(symbol: str) -> StockInfo:
        """
        获取股票价格（模拟）。

        Args:
            symbol: 股票代码
        """
        # 模拟数据
        return StockInfo(
            symbol=symbol.upper(),
            price=150.25,
            change_pct=2.35,
        )

    result = stock_agent.run_sync('查询 AAPL 股票价格')
    print(f"  工具返回类型: StockInfo (Pydantic 模型)")
    print(f"  结果: {result.output}")

    # ---- 示例 3: 异步工具 ----
    print("\n✅ 示例 3: 异步工具")

    import asyncio

    async_agent = Agent('test', system_prompt='你是一个信息助手。')

    @async_agent.tool_plain
    async def fetch_news_async(topic: str) -> str:
        """
        异步获取新闻（模拟）。

        Args:
            topic: 新闻主题
        """
        # 模拟异步延迟
        await asyncio.sleep(0.1)
        return f"关于 '{topic}' 的最新新闻: [模拟新闻内容]"

    # 注意：异步工具需要用 run() 而非 run_sync()
    async def run_async_example():
        result = await async_agent.run('获取关于 AI 的新闻')
        return result

    result = asyncio.run(run_async_example())
    print(f"  工具类型: async def（异步工具）")
    print(f"  调用方式: agent.run()（异步）")
    print(f"  结果: {result.output}")

    # ---- 示例 4: 工具链（Tool Chaining） ----
    print("\n✅ 示例 4: 工具链模式")

    @dataclass
    class PipelineState:
        data: str = ""

    pipeline_agent = Agent(
        'test',
        deps_type=PipelineState,
        system_prompt='你是一个数据处理流水线助手。'
    )

    @pipeline_agent.tool
    def step_extract(ctx: RunContext[PipelineState]) -> str:
        """步骤 1: 提取数据"""
        ctx.deps.data = "raw_data|user_id=123|action=login"
        return f"已提取: {ctx.deps.data}"

    @pipeline_agent.tool
    def step_transform(ctx: RunContext[PipelineState]) -> str:
        """步骤 2: 转换数据"""
        parts = ctx.deps.data.split("|")
        parsed = {p.split("=")[0]: p.split("=")[1] if "=" in p else p
                  for p in parts}
        ctx.deps.data = str(parsed)
        return f"已转换: {parsed}"

    @pipeline_agent.tool
    def step_load(ctx: RunContext[PipelineState]) -> str:
        """步骤 3: 加载数据"""
        return f"已加载数据到数据库: {ctx.deps.data}"

    state = PipelineState()
    result = pipeline_agent.run_sync(
        '执行 ETL 流水线: 提取、转换、加载数据',
        deps=state,
    )
    print(f"  模式: 3 个工具组成 ETL 流水线")
    print(f"  共享状态: 通过 deps (PipelineState) 传递")
    print(f"  最终状态: {state.data}")
    print(f"  结果: {result.output}")

    # ---- 示例 5: 工具错误处理 ----
    print("\n✅ 示例 5: 工具错误处理")

    error_agent = Agent(
        'test',
        system_prompt='你是一个安全的数据查询助手。',
        retries=2,  # 工具失败时允许重试
    )

    @error_agent.tool_plain
    def safe_query(query: str) -> str:
        """
        安全查询 — 模拟可能失败的场景。

        面试要点: 工具应处理异常，Agent 支持自动重试
        """
        # 模拟异常情况
        forbidden = ['DROP', 'DELETE', 'TRUNCATE']
        query_upper = query.upper()
        for word in forbidden:
            if word in query_upper:
                return f"拒绝执行: 不允许使用 {word} 操作"
        return f"查询成功: SELECT 结果 (query='{query}')"

    result = error_agent.run_sync('查询用户表: SELECT * FROM users')
    print(f"  正常查询: {result.output}")

    result = error_agent.run_sync('DROP TABLE users')
    print(f"  危险查询: {result.output}")

    print()

    # ---- 面试要点总结 ----
    print("📌 面试要点:")
    print("  1. @agent.tool: 需要 RunContext 才能工作的工具")
    print("  2. @agent.tool_plain: 纯函数工具，无需上下文")
    print("  3. 同步工具用 run_sync()，异步工具用 run()")
    print("  4. 工具链通过 deps 共享状态传递数据")
    print("  5. Agent 支持自动重试（retries 参数）")
    print("  6. 工具应返回有意义的字符串/Pydantic 模型")
    print()


# ============================================================
# 6. 流式输出 — run_stream()、增量事件
# ============================================================
# 面试要点: 流式输出是提升用户体验的关键
#
# Pydantic AI 流式输出:
# - agent.run_stream(): 异步流式运行
# - stream.stream_text(): 流式文本增量
# - stream.stream_output(): 流式结构化输出增量
# - stream.stream_events(): 完整事件流（包含工具调用）
# ============================================================

def demo_streaming():
    """
    流式输出详解

    展示 run_stream()、stream_text、stream_output 等
    """
    print("=" * 60)
    print("6. 流式输出 (Streaming)")
    print("=" * 60)

    from pydantic_ai import Agent
    from pydantic import BaseModel, Field
    import asyncio

    # ---- 示例 1: 流式文本输出 ----
    print("\n✅ 示例 1: stream_text — 逐词流式输出")

    story_agent = Agent(
        'test',
        system_prompt='你是一个故事讲述者。',
    )

    async def stream_text_demo():
        print("  流式文本输出:")
        async with story_agent.run_stream('给我讲一个关于程序员的短故事') as stream:
            async for text_chunk in stream.stream_text():
                print(f"    >> {text_chunk}")

    asyncio.run(stream_text_demo())

    # ---- 示例 2: 流式结构化输出 ----
    print("\n✅ 示例 2: stream_output — 结构化输出增量")

    class Story(BaseModel):
        """故事结构"""
        title: str = Field(description="故事标题")
        content: str = Field(description="故事内容")
        moral: str = Field(description="寓意")

    story_struct_agent = Agent(
        'test',
        output_type=Story,
        system_prompt='创作一个短故事。',
    )

    async def stream_output_demo():
        print("  流式结构化输出 (每个增量是 Story 的部分更新):")
        async with story_struct_agent.run_stream('创作一个关于勇气的短故事') as stream:
            async for output_delta in stream.stream_output():
                print(f"    >> {output_delta}")

    asyncio.run(stream_output_demo())

    # ---- 示例 3: stream_events — 完整事件流 ----
    print("\n✅ 示例 3: 了解事件类型（概念演示）")

    print("""
    Pydantic AI 流式事件类型:
    ┌─────────────────────────┬──────────────────────────────┐
    │ 事件类型                  │ 说明                          │
    ├─────────────────────────┼──────────────────────────────┤
    │ PartStartEvent           │ 某部分开始（文本/工具调用）      │
    │ PartDeltaEvent           │ 增量更新                       │
    │ PartEndEvent             │ 某部分结束                      │
    │ TextPartDelta            │ 文本增量                       │
    │ ToolCallPart             │ 工具调用开始                    │
    │ ToolReturnPart           │ 工具调用返回                    │
    │ FinalResultEvent         │ 最终结果事件                    │
    │ FunctionToolCallEvent    │ 函数工具调用事件                │
    │ FunctionToolResultEvent  │ 函数工具结果事件                │
    └─────────────────────────┴──────────────────────────────┘
    """)

    # ---- 示例 4: 带工具的流式输出 ----
    print("✅ 示例 4: 带工具的流式输出")

    stream_tool_agent = Agent('test', system_prompt='你是一个天气助手。')

    @stream_tool_agent.tool_plain
    def get_temp(city: str) -> str:
        """获取城市温度。"""
        temps = {"北京": 25, "上海": 28, "深圳": 30}
        return f"{city}: {temps.get(city, 22)}°C"

    async def stream_with_tools():
        print("  流式输出（含工具调用）:")
        async with stream_tool_agent.run_stream('北京和上海今天气温多少？') as stream:
            async for text_chunk in stream.stream_text():
                print(f"    >> {text_chunk}")

    asyncio.run(stream_with_tools())

    print()

    # ---- 面试要点总结 ----
    print("📌 面试要点:")
    print("  1. run_stream() 返回异步上下文管理器（async with）")
    print("  2. stream_text() 适合聊天式界面，逐词/逐句显示")
    print("  3. stream_output() 适合结构化输出的增量更新")
    print("  4. stream_events() 提供完整的可观测性（调试用）")
    print("  5. 对比 LangChain: LangGraph 的 streaming 需要 stream_mode 参数")
    print("  6. Pydantic AI 的流式输出自动处理工具调用的中间状态")
    print()


# ============================================================
# 7. 多 Agent 协作 — Agent 间委托、Chain Agent 模式
# ============================================================
# 面试要点: 多 Agent 协作是复杂 AI 应用的核心
#
# Pydantic AI 多 Agent 模式:
# - Agent 作为工具: 一个 Agent 调用另一个 Agent
# - 委托模式: 主 Agent 将子任务委托给专业 Agent
# - 决策路由: 根据输入将请求路由到不同 Agent
# ============================================================

def demo_multi_agent():
    """
    多 Agent 协作

    展示 Agent 间调用、委托、路由模式
    """
    print("=" * 60)
    print("7. 多 Agent 协作")
    print("=" * 60)

    from pydantic_ai import Agent, RunContext
    from pydantic import BaseModel, Field
    from typing import Literal
    from dataclasses import dataclass

    # ---- 示例 1: Agent 作为工具（Agent-as-Tool） ----
    print("\n✅ 示例 1: Agent 作为工具（子 Agent 被父 Agent 调用）")

    # 子 Agent: 专业翻译
    translator_sub = Agent(
        'test',
        system_prompt='你是一个专业翻译，将任意语言翻译成中文。'
    )

    # 子 Agent: 专业摘要
    summarizer_sub = Agent(
        'test',
        system_prompt='你是一个摘要专家，生成简洁的摘要。'
    )

    # 主 Agent: 协调者
    coordinator = Agent(
        'test',
        system_prompt='你是一个内容处理协调者。先翻译，再生成摘要。',
    )

    @coordinator.tool_plain
    def translate_content(text: str) -> str:
        """调用翻译子 Agent 翻译内容。"""
        result = translator_sub.run_sync(text)
        return result.output

    @coordinator.tool_plain
    def summarize_content(text: str) -> str:
        """调用摘要子 Agent 生成摘要。"""
        result = summarizer_sub.run_sync(f"请总结: {text}")
        return result.output

    result = coordinator.run_sync(
        '请将 "Artificial Intelligence is transforming the world" 翻译成中文并总结'
    )
    print(f"  父 Agent: coordinator（协调者）")
    print(f"  子 Agent: translator_sub, summarizer_sub")
    print(f"  工具: translate_content, summarize_content 内部调用子 Agent")
    print(f"  结果: {result.output}")

    # ---- 示例 2: 委托模式（结构化决策） ----
    print("\n✅ 示例 2: 委托模式 — 根据类型路由到专业 Agent")

    class TaskAnalysis(BaseModel):
        """任务分析结果"""
        task_type: Literal["code", "writing", "math"] = Field(
            description="任务类型"
        )
        complexity: int = Field(ge=1, le=5, description="复杂度 1-5")

    task_router = Agent(
        'test',
        output_type=TaskAnalysis,
        system_prompt='分析任务类型和复杂度。',
    )

    # 专业 Agent
    coder = Agent('test', system_prompt='你是一个 Python 编程专家。')
    writer = Agent('test', system_prompt='你是一个专业作家。')
    mathematician = Agent('test', system_prompt='你是一个数学家。')

    @dataclass
    class RouterDeps:
        task: str

    router_agent = Agent(
        'test',
        deps_type=RouterDeps,
        system_prompt='根据任务类型路由到合适的专业 Agent。',
    )

    @router_agent.tool
    def route_task(ctx: RunContext[RouterDeps]) -> str:
        """
        分析任务并将请求路由到合适的专业 Agent。

        面试要点: 展示 Agent 间的委托模式
        """
        # 步骤 1: 分析任务
        analysis = task_router.run_sync(
            f"分析这个任务: {ctx.deps.task}"
        ).output

        print(f"    [路由分析] 任务类型: {analysis.task_type}, 复杂度: {analysis.complexity}")

        # 步骤 2: 委托给专业 Agent
        if analysis.task_type == "code":
            result = coder.run_sync(ctx.deps.task)
        elif analysis.task_type == "writing":
            result = writer.run_sync(ctx.deps.task)
        elif analysis.task_type == "math":
            result = mathematician.run_sync(ctx.deps.task)
        else:
            result = coder.run_sync(ctx.deps.task)

        return f"[{analysis.task_type}专家] {result.output}"

    result = router_agent.run_sync(
        '请写一个 Python 函数计算斐波那契数列',
        deps=RouterDeps(task='写一个 Python 函数计算斐波那契数列'),
    )
    print(f"  结果: {result.output}")

    # ---- 示例 3: 并行多 Agent 对比（Pydantic AI vs LangGraph） ----
    print("\n✅ 示例 3: 多 Agent 对比 (Pydantic AI vs LangGraph)")

    print("""
    ┌────────────────────┬─────────────────────────┬──────────────────────────┐
    │ 维度                 │ Pydantic AI              │ LangGraph                 │
    ├────────────────────┼─────────────────────────┼──────────────────────────┤
    │ Agent 定义           │ Agent(model, ...)        │ StateGraph + Node         │
    │ 工具注册              │ @agent.tool              │ @tool + ToolNode          │
    │ 多 Agent 协作         │ Agent 作为工具调用         │ 子图(Subgraph)             │
    │ 状态管理              │ 依赖注入(deps)            │ TypedDict State           │
    │ 条件路由              │ 工具内部判断              │ add_conditional_edges     │
    │ 并行执行              │ 需要手动 asyncio         │ 原生 Fan-out              │
    │ 流式输出              │ run_stream()             │ stream() + stream_mode    │
    │ 输出验证              │ Pydantic 模型自动验证       │ 需要手动 OutputParser     │
    │ 学习曲线              │ 较低（快速上手）           │ 中等（需要理解图结构）      │
    │ 适用场景              │ 结构化响应、单Agent场景     │ 复杂工作流、多步骤流程      │
    └────────────────────┴─────────────────────────┴──────────────────────────┘
    """)

    print()

    # ---- 面试要点总结 ----
    print("📌 面试要点:")
    print("  1. Agent 作为工具是最简单的多 Agent 协作模式")
    print("  2. 委托模式: 用 Router Agent 分析任务后转发给专业 Agent")
    print("  3. Pydantic AI 更适合快速构建单 Agent 应用")
    print("  4. LangGraph 更适合复杂多步骤工作流")
    print("  5. 两者可以组合使用: LangGraph 编排 + Pydantic AI Agent")
    print()


# ============================================================
# 8. Pydantic AI vs LangChain 全面对比
# ============================================================
# 面试要点: 这是面试高频问题，需要清楚两者的定位差异
#
# 定位差异:
# - LangChain: AI 应用开发框架（构建一切）
# - Pydantic AI: Agent 开发框架（专注 Agent）
# - LangGraph: 流程编排框架（构建工作流）
#
# 选择建议:
# - 简单 Agent → Pydantic AI
# - 复杂工作流 → LangGraph
# - 需要灵活性 → LangChain + LangGraph
# - 追求类型安全 → Pydantic AI
# ============================================================

def demo_comparison():
    """
    Pydantic AI vs LangChain 全面对比

    从架构、使用、场景等维度进行对比
    """
    print("=" * 60)
    print("8. Pydantic AI vs LangChain 全面对比")
    print("=" * 60)

    from pydantic_ai import Agent
    from pydantic import BaseModel, Field

    # ---- 对比 1: 代码量对比 ----
    print("\n✅ 对比 1: 代码量 — 实现同样的功能")

    class WeatherOutput(BaseModel):
        city: str = Field(description="城市")
        temperature: float = Field(description="温度")
        condition: str = Field(description="天气")

    # Pydantic AI: 约 15 行
    print("\n  【Pydantic AI 实现】 (~15 行)")
    print("  ```python")
    print("  agent = Agent(")
    print("      'openai:gpt-4o',")
    print("      output_type=WeatherOutput,")
    print("      system_prompt='你是天气预报员。'")
    print("  )")
    print("  result = agent.run_sync('北京天气？')")
    print("  weather: WeatherOutput = result.output")
    print("  ```")

    # LangChain: 约 40 行
    print("\n  【LangChain 实现】 (~40 行)")
    print("  ```python")
    print("  from langchain_core.prompts import ChatPromptTemplate")
    print("  from langchain_openai import ChatOpenAI")
    print("  from langchain_core.output_parsers import PydanticOutputParser")
    print("")
    print("  parser = PydanticOutputParser(pydantic_object=WeatherOutput)")
    print("  prompt = ChatPromptTemplate.from_messages([")
    print("      ('system', '{format_instructions}'),")
    print("      ('user', '{input}')")
    print("  ])")
    print("  llm = ChatOpenAI(model='gpt-4o')")
    print("  chain = prompt | llm | parser")
    print("  weather = chain.invoke({")
    print("      'input': '北京天气？',")
    print("      'format_instructions': parser.get_format_instructions()")
    print("  })")
    print("  ```")

    # ---- 对比 2: 核心概念映射 ----
    print("\n✅ 对比 2: 核心概念映射")

    print("""
    ┌──────────────────────────┬───────────────────────────────┐
    │ LangChain/LangGraph       │ Pydantic AI                    │
    ├──────────────────────────┼───────────────────────────────┤
    │ Chain (Runnable)          │ Agent (封装自循环)              │
    │ @tool 装饰器               │ @agent.tool / @agent.tool_plain│
    │ PydanticOutputParser      │ output_type (自动处理)          │
    │ StateGraph + State        │ deps_type + RunContext         │
    │ ToolNode + tools          │ Agent 内置工具循环              │
    │ add_conditional_edges     │ 工具内部路由逻辑                 │
    │ LangGraph 子图             │ Agent 作为工具调用               │
    │ PromptTemplate            │ system_prompt (字符串)           │
    │ ChatModel                 │ model 参数 ("openai:gpt-4o")   │
    │ RunnablePassthrough       │ RunContext (携带上下文)          │
    └──────────────────────────┴───────────────────────────────┘
    """)

    # ---- 对比 3: 场景选择建议 ----
    print("✅ 对比 3: 场景选择建议")

    print("""
    📋 选择 Pydantic AI 的场景:
    ─────────────────────────────────
    1. 构建结构化数据提取 Agent（如发票解析、简历分析）
    2. 构建单轮/多轮对话 Agent
    3. 需要强类型安全的应用
    4. 快速原型开发
    5. 简单的工具调用 Agent

    📋 选择 LangChain 的场景:
    ─────────────────────────────────
    1. 需要复杂 Prompt 模板管理
    2. 需要多种 LLM Provider 切换
    3. 需要与向量数据库、文档加载器等生态集成
    4. 需要 LCEL 管道式组合

    📋 选择 LangGraph 的场景:
    ─────────────────────────────────
    1. 多步骤工作流（如 RAG、ETL）
    2. 需要条件分支和循环
    3. 需要并行执行
    4. 需要检查点/持久化
    5. Human-in-the-Loop
    6. 复杂多 Agent 协作

    💡 最佳实践: 组合使用
    ─────────────────────────────────
    - LangGraph 做流程编排（外层）
    - Pydantic AI Agent 作为节点（内层）
    - LangChain 组件提供工具生态（工具层）
    """)

    print()

    # ---- 面试要点总结 ----
    print("📌 面试高频问题 — 标准答案:")
    print()
    print("  Q: Pydantic AI 和 LangChain 的 Agent 有什么区别？")
    print("  A: ")
    print("     1. 设计哲学不同: LangChain 以 Chain/Graph 为中心，Agent 是运行模式；")
    print("        Pydantic AI 以 Agent 为中心，一切围绕 Agent 展开。")
    print("     2. 类型安全: Pydantic AI 利用 Pydantic 模型做输入输出验证，类型安全更强；")
    print("        LangChain 需要手动集成 OutputParser。")
    print("     3. 代码量: Pydantic AI 显著更少（约 1/3），内置工具循环和重试机制。")
    print("     4. 灵活性: LangChain 生态更丰富（向量数据库、文档加载器等）。")
    print()
    print("  Q: 什么时候用 Pydantic AI，什么时候用 LangGraph？")
    print("  A: ")
    print("     简单的单 Agent 任务用 Pydantic AI（快速开发、类型安全）；")
    print("     复杂的多步骤工作流用 LangGraph（条件分支、并行、持久化）；")
    print("     两者可以组合：LangGraph 编排流程，Pydantic AI Agent 作为节点。")
    print()


# ============================================================
# 主函数
# ============================================================

def main():
    """运行所有 Pydantic AI 学习示例"""
    print("\n" + "=" * 60)
    print("  Pydantic AI 框架学习")
    print("  Python 3.12+ | pydantic-ai v2.0")
    print("=" * 60)
    print()

    demo_basic_concepts()
    demo_agent_definition()
    demo_structured_output()
    demo_dependency_injection()
    demo_tools()
    demo_streaming()
    demo_multi_agent()
    demo_comparison()

    print("=" * 60)
    print("  全部示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
