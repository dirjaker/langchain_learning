# """

::: code-group

```python [07_langsmith.py]
"""
LangSmith 观测与评估

LangSmith 是 LangChain 生态的 LLM 应用观测、调试、测试和评估平台。
本章节覆盖 LangSmith 的核心概念：Tracing、Dataset、Evaluation、
Feedback、Hub 以及与 LangChain/LangGraph 的集成。

面试重点：
- LangSmith 在 LangChain 生态中的定位
- Tracing 的工作原理
- 如何构建评估流水线
- Hub 的 Prompt 版本管理
- 生产环境成本控制策略
"""

# ============================================================
# 1. LangSmith 概述与架构
# ============================================================
# 面试要点: LangSmith 是什么？为什么需要它？
#
# LangSmith 定位:
#   - LLM 应用的可观测性平台 (Observability)
#   - 贯穿开发 → 测试 → 生产全生命周期
#   - 提供 Tracing、Dataset、Evaluation、Feedback、Hub 五大核心功能
#
# 为什么需要 LangSmith？
#   1. LLM 应用具有非确定性：同样的输入可能产生不同输出
#   2. 调试困难：链式调用中难以定位问题
#   3. 缺乏评估标准：如何判断输出质量？
#   4. 成本不透明：token 消耗难以追踪
#   5. Prompt 迭代混乱：缺乏版本管理
#
# LangSmith vs 竞品:
#   - Weights & Biases: 侧重模型训练追踪，LangSmith 侧重 LLM 应用
#   - MLflow: 通用 ML 实验管理，LangSmith 深度集成 LangChain
#   - Arize/Phoenix: 更侧重模型监控，LangSmith 覆盖全生命周期
#   - 自建方案: 灵活性高但维护成本大
#
# 核心组件架构:
#   ┌─────────────────────────────────────────────────────┐
#   │                    LangSmith                        │
#   │  ┌──────────┐ ┌──────────┐ ┌────────────────────┐  │
#   │  │ Tracing  │ │ Dataset  │ │   Evaluation       │  │
#   │  │ 调用追踪  │ │ 数据集    │ │   评估             │  │
#   │  └──────────┘ └──────────┘ └────────────────────┘  │
#   │  ┌──────────┐ ┌──────────────────────────────────┐ │
#   │  │ Feedback │ │   Hub (Prompt 注册中心)           │ │
#   │  │ 反馈系统  │ │                                   │ │
#   │  └──────────┘ └──────────────────────────────────┘ │
#   └─────────────────────────────────────────────────────┘
# ============================================================

def demo_langsmith_overview():
    """LangSmith 概述与架构"""
    print("=" * 60)
    print("1. LangSmith 概述与架构")
    print("=" * 60)

    print("""
    LangSmith 五大核心功能:

    ┌─────────────┬──────────────────────────────────────────┐
    │ 功能         │ 说明                                     │
    ├─────────────┼──────────────────────────────────────────┤
    │ Tracing     │ 自动/手动追踪 LLM 调用链，可视化执行流程   │
    │ Dataset     │ 创建和管理测试数据集，支持版本化           │
    │ Evaluation  │ 在线/离线评估，支持自定义和内置评估器      │
    │ Feedback    │ 收集用户和 AI 反馈，持续改进应用           │
    │ Hub         │ Prompt 注册中心，支持版本管理和协作        │
    └─────────────┴──────────────────────────────────────────┘

    工作流程:
    Develop (开发) → Test (测试) → Monitor (监控) → Improve (改进)
         │               │               │               │
         ▼               ▼               ▼               ▼
    Hub管理Prompt   Dataset+Eval   Tracing+Feedback  迭代优化

    LangSmith Python SDK 安装: pip install langsmith
    环境变量: LANGCHAIN_API_KEY=ls__xxx
             LANGCHAIN_PROJECT=my-project
             LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
    """)
    print()


# ============================================================
# 2. Tracing（追踪）
# ============================================================
# 面试要点: Tracing 是 LangSmith 最核心的功能
#
# Tracing 核心概念:
#   - Trace: 一次完整的调用（一个用户请求）
#   - Run: Trace 中的一个步骤（一次 LLM 调用、检索等）
#   - Span: Run 的子步骤
#
# 追踪方式:
#   1. 自动追踪: 设置环境变量后自动捕获 LangChain/LangGraph 调用
#   2. @traceable 装饰器: 手动标记需要追踪的函数
#   3. RunTree API: 编程方式创建自定义追踪
#
# 追踪数据流:
#   应用代码 → RunTree → LangSmith SDK → LangSmith 服务端
#                                   ↓
#                           UI 可视化 / API 查询
#
# 面试常见追问:
#   Q: 追踪如何保证不影响应用性能？
#   A: 异步批量发送，采样策略，本地缓冲
#   Q: 如何追踪自定义函数？
#   A: 使用 @traceable 装饰器或 RunTree API
# ============================================================

def demo_tracing():
    """Tracing 追踪示例"""
    print("=" * 60)
    print("2. Tracing（追踪）")
    print("=" * 60)

    # --- 模拟 RunTree 数据结构 ---
    print("\n✅ RunTree 数据结构模拟:")

    class MockRun:
        """模拟 LangSmith Run 对象"""
        def __init__(self, name, run_type, inputs, outputs=None, parent=None):
            self.name = name
            self.run_type = run_type
            self.inputs = inputs
            self.outputs = outputs or {}
            self.parent = parent
            self.children = []
            self.tags = []
            self.metadata = {}
            if parent:
                parent.children.append(self)

        def add_tag(self, tag):
            self.tags.append(tag)

        def to_dict(self):
            return {
                "name": self.name,
                "run_type": self.run_type,
                "inputs": self.inputs,
                "outputs": self.outputs,
                "tags": self.tags,
                "metadata": self.metadata,
                "children": [c.to_dict() for c in self.children],
            }

    # 模拟一个 RAG 调用的完整 Trace
    root = MockRun(
        name="rag_chain",
        run_type="chain",
        inputs={"question": "什么是 LangChain？"},
    )
    root.add_tag("production")

    # LLM 调用 (Run)
    llm_run = MockRun(
        name="ChatOpenAI",
        run_type="llm",
        inputs={"messages": [{"role": "user", "content": "什么是 LangChain？"}]},
        outputs={"content": "LangChain 是一个用于构建 AI 应用的框架"},
        parent=root,
    )
    llm_run.metadata = {"model": "gpt-4", "temperature": 0.7, "tokens": 150}

    # 检索器调用 (Run)
    retriever_run = MockRun(
        name="vector_store_retriever",
        run_type="retriever",
        inputs={"query": "什么是 LangChain？"},
        outputs={"documents": ["文档1", "文档2", "文档3"]},
        parent=root,
    )

    # 嵌套子步骤 (Span)
    embedding_run = MockRun(
        name="text_embedding",
        run_type="embedding",
        inputs={"texts": ["什么是 LangChain？"]},
        outputs={"embeddings": [[0.1, 0.2, 0.3, 0.4]]},
        parent=retriever_run,
    )

    print(f"  根 Trace: {root.name} ({root.run_type})")
    print(f"  子 Run 数量: {len(root.children)}")
    for child in root.children:
        print(f"    └─ {child.name} ({child.run_type})")
        for sub in child.children:
            print(f"       └─ {sub.name} ({sub.run_type})")

    # --- @traceable 装饰器模拟 ---
    print("\n✅ @traceable 装饰器模拟:")

    # 在实际使用中:
    # from langsmith import traceable
    #
    # @traceable(run_type="tool", name="search_tool")
    # def search(query: str) -> list:
    #     return ["结果1", "结果2"]

    # 这里模拟装饰器的行为
    class MockTraceable:
        """模拟 @traceable 装饰器"""

        def __init__(self, run_type="chain", name=None):
            self.run_type = run_type
            self.name = name

        def __call__(self, func):
            def wrapper(*args, **kwargs):
                print(f"  [Traceable] 开始追踪: {self.name or func.__name__}")
                run = MockRun(
                    name=self.name or func.__name__,
                    run_type=self.run_type,
                    inputs={"args": args, "kwargs": kwargs},
                )
                result = func(*args, **kwargs)
                run.outputs = {"result": result}
                print(f"  [Traceable] 完成追踪: {self.name or func.__name__}")
                print(f"    输入: {run.inputs}")
                print(f"    输出: {run.outputs}")
                return result
            return wrapper

    @MockTraceable(run_type="tool", name="web_search")
    def search_web(query: str):
        """模拟网络搜索工具"""
        return [f"搜索结果 for '{query}': 条目1", f"条目2"]

    result = search_web("LangSmith 教程")
    print(f"  最终结果: {result}")

    # --- 采样策略模拟 ---
    print("\n✅ 采样策略模拟:")

    import random

    class SamplingConfig:
        """采样配置"""
        def __init__(self, rate=1.0):
            self.rate = rate  # 采样率: 0.0 - 1.0

        def should_sample(self):
            """是否应该采样这个 Trace"""
            return random.random() < self.rate

    # 生产环境通常使用 10%-30% 采样率
    configs = [
        SamplingConfig(1.0),   # 开发环境: 100%
        SamplingConfig(0.3),   # 生产环境: 30%
        SamplingConfig(0.1),   # 高流量: 10%
    ]

    for i, config in enumerate(configs):
        sampled = sum(config.should_sample() for _ in range(100))
        print(f"  采样率 {config.rate}: 100 次中采样 {sampled} 次")

    print()


# ============================================================
# 3. Dataset（数据集）
# ============================================================
# 面试要点: Dataset 用于构建评估基准
#
# Dataset 用途:
#   1. 收集真实用户输入作为测试用例
#   2. 手工构建边界情况数据集
#   3. 从 Tracing 中导出示例
#   4. 版本化管理测试数据
#
# 数据集结构:
#   Dataset {
#       name: str           # 数据集名称
#       description: str    # 描述
#       examples: [         # 示例列表
#           {
#               inputs: {},     # 输入（问题、上下文等）
#               outputs: {},    # 期望输出（可选，用于评估）
#               metadata: {}    # 元数据（标签、来源等）
#           }
#       ]
#   }
#
# 面试常见追问:
#   Q: 如何从生产环境构建数据集？
#   A: 通过 Filter 从 Tracing 中筛选高质量示例导出
#   Q: 数据集如何版本化管理？
#   A: 每次修改创建新版本，保留历史记录
# ============================================================

def demo_dataset():
    """Dataset 数据集示例"""
    print("=" * 60)
    print("3. Dataset（数据集）")
    print("=" * 60)

    # --- 模拟 Dataset 数据结构 ---
    print("\n✅ 创建数据集模拟:")

    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class Example:
        """数据集示例"""
        inputs: dict
        outputs: dict = field(default_factory=dict)
        metadata: dict = field(default_factory=dict)
        id: str | None = None

    @dataclass
    class MockDataset:
        """模拟 LangSmith Dataset"""
        name: str
        description: str = ""
        examples: list[Example] = field(default_factory=list)

        def add_example(self, inputs, outputs=None, metadata=None):
            example = Example(
                inputs=inputs,
                outputs=outputs or {},
                metadata=metadata or {},
            )
            self.examples.append(example)
            return example

        def to_dict(self):
            return {
                "name": self.name,
                "description": self.description,
                "example_count": len(self.examples),
                "examples": [
                    {"inputs": e.inputs, "outputs": e.outputs}
                    for e in self.examples
                ],
            }

    # 实际 API:
    # from langsmith import Client
    # client = Client()
    # dataset = client.create_dataset("my-dataset", description="...")
    # client.create_example(
    #     inputs={"question": "什么是 LangChain？"},
    #     outputs={"answer": "LangChain 是一个 AI 框架"},
    #     dataset_id=dataset.id,
    # )

    # --- RAG 评估数据集 ---
    print("── RAG 评估数据集 ──")
    rag_dataset = MockDataset(
        name="rag-evaluation-v1",
        description="用于评估 RAG 系统质量的数据集",
    )

    rag_dataset.add_example(
        inputs={"question": "什么是 LangChain？"},
        outputs={"answer": "LangChain 是一个用于构建 LLM 应用的开源框架"},
        metadata={"difficulty": "easy", "topic": "概述"},
    )
    rag_dataset.add_example(
        inputs={"question": "LangGraph 的 StateGraph 如何工作？"},
        outputs={
            "answer": "StateGraph 通过节点和边定义工作流，每个节点接收和返回状态"
        },
        metadata={"difficulty": "medium", "topic": "LangGraph"},
    )
    rag_dataset.add_example(
        inputs={"question": "如何实现 RAG 的混合检索？"},
        outputs={"answer": "结合向量检索和关键词检索，使用加权融合排序"},
        metadata={"difficulty": "hard", "topic": "RAG"},
    )

    print(f"  数据集: {rag_dataset.name}")
    print(f"  描述: {rag_dataset.description}")
    print(f"  示例数: {len(rag_dataset.examples)}")
    for i, ex in enumerate(rag_dataset.examples):
        print(f"    {i+1}. Q: {ex.inputs['question'][:40]}...")
        print(f"       A: {ex.outputs.get('answer', 'N/A')[:40]}...")
        print(f"       难度: {ex.metadata.get('difficulty', 'N/A')}")

    # --- 从 Tracing 导出数据集 ---
    print("\n✅ 从 Tracing 导出数据集模拟:")

    traces = [
        {"question": "如何使用 LCEL？", "answer": "使用 | 管道符组合 Runnable",
         "latency_ms": 200, "user_rating": 5},
        {"question": "LangChain 支持哪些模型？",
         "answer": "支持 OpenAI、Anthropic、Google 等", "latency_ms": 150,
         "user_rating": 4},
        {"question": "什么是 AgentExecutor？", "answer": "废弃的旧 API",
         "latency_ms": 300, "user_rating": 2},
        {"question": "错误: xxx not found", "answer": "错误", "latency_ms": 50,
         "user_rating": 1},
    ]

    # 实际 API:
    # client.create_dataset_from_run_tree(
    #     dataset_name="from-production",
    #     filter='eq(feedback_score, 5)',  # 只要 5 星好评
    # )

    def filter_traces(traces, min_rating=4, max_latency=500):
        """模拟 Filter 过滤"""
        filtered = [
            t for t in traces
            if t["user_rating"] >= min_rating
            and t["latency_ms"] < max_latency
            and "错误" not in t["answer"]
        ]
        return filtered

    high_quality = filter_traces(traces, min_rating=4)
    print(f"  原始 Trace 数: {len(traces)}")
    print(f"  过滤后 (评分≥4, 无错误): {len(high_quality)}")
    for t in high_quality:
        print(f"    Q: {t['question']} (评分: {t['user_rating']})")

    # --- 数据集版本管理 ---
    print("\n✅ 数据集版本管理:")
    versions = [
        {"version": "v1", "examples": 10, "description": "初始版本"},
        {"version": "v2", "examples": 25, "description": "添加边缘case"},
        {"version": "v3", "examples": 50, "description": "从生产环境导入"},
    ]
    for v in versions:
        print(f"  {v['version']}: {v['examples']} 个示例 - {v['description']}")

    print()


# ============================================================
# 4. Evaluation（评估）
# ============================================================
# 面试要点: 评估是 LangSmith 的质量保障核心
#
# 评估类型:
#   1. 离线评估 (Offline): 对历史数据集批量评估
#   2. 在线评估 (Online): 实时评估生产流量
#   3. 对比评估 (Comparative): 比较两个版本的输出
#
# 内置 Evaluator:
#   - correctness: 正确性评估
#   - helpfulness: 有用性评估
#   - conciseness: 简洁性评估
#   - harmfulness: 有害性检测
#   - custom: 自定义评估函数
#
# 评估流程:
#   数据集 → 运行应用 → 收集输出 → 应用 Evaluator → 生成报告
#
# 面试常见追问:
#   Q: 如何使用 LLM-as-Judge 评估？
#   A: 用另一个 LLM 评估输出质量，定义评分标准
#   Q: 如何处理主观性评估？
#   A: 结合人工反馈和 AI 评估，设置多维度评分
# ============================================================

def demo_evaluation():
    """Evaluation 评估示例"""
    print("=" * 60)
    print("4. Evaluation（评估）")
    print("=" * 60)

    # --- 自定义 Evaluator ---
    print("\n✅ 自定义 Evaluator 模拟:")

    from dataclasses import dataclass

    @dataclass
    class EvaluationResult:
        """评估结果"""
        key: str            # 评估指标名称
        score: float        # 得分 (0-1)
        comment: str = ""   # 评语
        metadata: dict | None = None  # 额外信息

    class MockEvaluator:
        """模拟 LangSmith Evaluator"""

        def __init__(self, key: str, description: str = ""):
            self.key = key
            self.description = description

        def evaluate(
            self, run_output: dict, example: dict
        ) -> EvaluationResult:
            raise NotImplementedError

    # 实际 API:
    # from langsmith.evaluation import evaluate, LangChainStringEvaluator
    #
    # def my_evaluator(run: Run, example: Example) -> dict:
    #     score = compute_score(run.outputs, example.outputs)
    #     return {"key": "my_score", "score": score}

    # --- 正确性评估器 ---
    print("── 正确性评估器 ──")

    class CorrectnessEvaluator(MockEvaluator):
        """评估输出的正确性"""

        def __init__(self):
            super().__init__("correctness", "评估回答是否与参考答案一致")

        def evaluate(self, run_output: dict, example: dict) -> EvaluationResult:
            expected = example.get("outputs", {}).get("answer", "")
            actual = run_output.get("answer", "")

            # 模拟 LLM-as-Judge 评估
            # 实际中会调用 LLM 来判断语义相似度
            if not expected or not actual:
                return EvaluationResult("correctness", 0.0, "缺少答案")

            # 简单模拟: 基于关键词匹配
            keywords = expected.lower().split()
            matched = sum(1 for kw in keywords if kw.lower() in actual.lower())
            score = min(matched / max(len(keywords), 1), 1.0)

            comment = f"关键词匹配: {matched}/{len(keywords)}"
            return EvaluationResult("correctness", round(score, 2), comment)

    # --- 质量评估器 ---
    class QualityEvaluator(MockEvaluator):
        """多维度质量评估"""

        def __init__(self):
            super().__init__("quality", "评估回答的综合质量")

        def evaluate(self, run_output: dict, example: dict) -> EvaluationResult:
            answer = run_output.get("answer", "")
            scores = {}

            # 1. 完整性: 答案长度是否足够
            completeness = min(len(answer) / 200, 1.0)
            scores["completeness"] = round(completeness, 2)

            # 2. 相关性: 是否包含问题关键词
            question = example.get("inputs", {}).get("question", "")
            q_keywords = set(question.lower().split())
            a_keywords = set(answer.lower().split())
            relevance = len(q_keywords & a_keywords) / max(len(q_keywords), 1)
            scores["relevance"] = round(min(relevance * 3, 1.0), 2)

            # 3. 简洁性: 不要太冗长 (模拟)
            conciseness = max(0, 1 - len(answer) / 1000)
            scores["conciseness"] = round(conciseness, 2)

            # 综合得分: 加权平均
            weights = {"completeness": 0.3, "relevance": 0.5, "conciseness": 0.2}
            total = sum(scores[k] * weights[k] for k in weights)

            return EvaluationResult(
                "quality",
                round(total, 2),
                f"综合得分: {total:.2f}",
                metadata={"sub_scores": scores},
            )

    # --- 运行评估 ---
    print("── 批量评估运行 ──")

    test_cases = [
        {
            "inputs": {"question": "什么是 LangChain？"},
            "outputs": {"answer": "LangChain 是一个开源的 LLM 应用开发框架"},
            "expected": {"outputs": {"answer": "LangChain 是用于构建 LLM 应用的开源框架"}},
        },
        {
            "inputs": {"question": "如何安装 LangChain？"},
            "outputs": {"answer": "pip install langchain"},
            "expected": {"outputs": {"answer": "使用 pip install langchain 安装"}},
        },
        {
            "inputs": {"question": "LangGraph 的核心概念？"},
            "outputs": {"answer": "核心概念包括状态图、节点、边"},  # 较简短
            "expected": {"outputs": {"answer": "StateGraph、Node、Edge 和条件分支是 LangGraph 的核心概念"}},
        },
    ]

    correctness_eval = CorrectnessEvaluator()
    quality_eval = QualityEvaluator()

    print(f"\n{'用例':<4} {'问题':<30} {'正确性':>6} {'质量':>6}")
    print("-" * 50)

    for i, case in enumerate(test_cases):
        question = case["inputs"]["question"]
        output = case["outputs"]

        c_result = correctness_eval.evaluate(output, case["expected"])
        q_result = quality_eval.evaluate(output, case["expected"])

        print(f"{i+1:<4} {question[:28]:<30} {c_result.score:>6.2f} {q_result.score:>6.2f}")
        if q_result.metadata:
            subs = q_result.metadata["sub_scores"]
            print(f"     {'':>4} 子分数: 完整={subs['completeness']:.2f} "
                  f"相关={subs['relevance']:.2f} 简洁={subs['conciseness']:.2f}")

    # --- 对比评估 ---
    print("\n✅ 对比评估 (A/B Test):")

    # 模拟两个版本的 Prompt 输出
    results_a = {
        "prompt_version": "v1 - 简单提示",
        "answers": [
            "LangChain 是一个框架。",  # 太简短
            "使用 pip install。",
        ],
        "avg_score": 0.65,
    }
    results_b = {
        "prompt_version": "v2 - 优化提示",
        "answers": [
            "LangChain 是一个用于构建 LLM 应用的开源框架，支持链式调用和 Agent。",
            "通过 pip install langchain 安装，支持 Python 3.9+。",
        ],
        "avg_score": 0.88,
    }

    print(f"  {results_a['prompt_version']}: 平均分 {results_a['avg_score']}")
    print(f"  {results_b['prompt_version']}: 平均分 {results_b['avg_score']}")
    print(f"  改进: +{results_b['avg_score'] - results_a['avg_score']:.2f}")
    print(f"  结论: v2 优于 v1，建议上线 v2")

    print()


# ============================================================
# 5. Feedback（反馈）
# ============================================================
# 面试要点: Feedback 是持续改进的关键
#
# Feedback 类型:
#   1. 用户反馈 (Human Feedback):
#      - 点赞/点踩 (thumbs up/down)
#      - 评分 (1-5 星)
#      - 修正回复
#   2. AI 反馈 (AI Feedback):
#      - LLM-as-Judge 自动评分
#      - 自动检测有害内容
#      - 自动评估质量指标
#
# Feedback 应用场景:
#   - 构建 RLHF 数据集
#   - 识别低质量输出
#   - A/B 测试效果对比
#   - 生成评估基线
#
# 面试常见追问:
#   Q: 用户反馈和 AI 反馈如何结合？
#   A: 用户反馈更权威但稀疏，AI 反馈全覆盖但可能有偏差
#       两者结合: AI 反馈用于快速筛选，用户反馈用于校准
# ============================================================

def demo_feedback():
    """Feedback 反馈示例"""
    print("=" * 60)
    print("5. Feedback（反馈）")
    print("=" * 60)

    # --- 反馈数据模型 ---
    print("\n✅ 反馈系统模拟:")

    from dataclasses import dataclass, field
    from datetime import datetime

    @dataclass
    class Feedback:
        """模拟 LangSmith Feedback"""
        key: str            # 反馈类型: "user_score", "correctness", "helpfulness"
        score: float        # 得分 (通常 0-1 或 0-5)
        comment: str = ""   # 评语
        source: str = "user"  # "user" 或 "ai"
        created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    class MockFeedbackCollector:
        """模拟反馈收集器"""

        def __init__(self):
            self.feedbacks: list[Feedback] = []

        def add_user_feedback(self, run_output, score, comment=""):
            """添加用户反馈"""
            fb = Feedback(
                key="user_score",
                score=score,
                comment=comment,
                source="user",
            )
            self.feedbacks.append(fb)
            return fb

        def add_ai_feedback(self, run_output, score, evaluator_type):
            """添加 AI 反馈"""
            fb = Feedback(
                key=evaluator_type,
                score=score,
                source="ai",
            )
            self.feedbacks.append(fb)
            return fb

        def get_stats(self):
            """获取反馈统计"""
            users = [f for f in self.feedbacks if f.source == "user"]
            ais = [f for f in self.feedbacks if f.source == "ai"]
            return {
                "total": len(self.feedbacks),
                "user_avg": sum(f.score for f in users) / len(users) if users else 0,
                "ai_avg": sum(f.score for f in ais) / len(ais) if ais else 0,
                "rating_distribution": self._rating_distribution(),
            }

        def _rating_distribution(self):
            """评分分布"""
            dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for fb in self.feedbacks:
                bucket = min(5, max(1, int(fb.score)))
                dist[bucket] += 1
            return dist

    # --- 收集反馈 ---
    collector = MockFeedbackCollector()

    # 模拟用户反馈
    conversations = [
        {"question": "什么是 LangChain？", "answer": "LangChain 是一个 AI 框架",
         "user_score": 5, "comment": "很清晰"},
        {"question": "如何安装？", "answer": "pip install langchain",
         "user_score": 4, "comment": "简洁"},
        {"question": "LangGraph 和 LangChain 的区别？",
         "answer": "LangGraph 是有状态的工作流引擎...",
         "user_score": 5, "comment": "详细"},
        {"question": "Agent 是什么？", "answer": "相关的概念...",
         "user_score": 3, "comment": "不够清晰"},
        {"question": "如何调试？", "answer": "用 print", "user_score": 1,
         "comment": "太敷衍"},
    ]

    for conv in conversations:
        collector.add_user_feedback(
            run_output={"answer": conv["answer"]},
            score=conv["user_score"],
            comment=conv["comment"],
        )

    # 模拟 AI 反馈
    ai_scores = [
        ("correctness", 0.9),
        ("helpfulness", 0.85),
        ("conciseness", 0.7),
        ("correctness", 0.6),
        ("helpfulness", 0.4),
    ]
    for eval_type, score in ai_scores:
        collector.add_ai_feedback(
            run_output={},
            score=score * 5,  # 转换为 0-5 分
            evaluator_type=eval_type,
        )

    # --- 反馈分析 ---
    print("── 反馈统计 ──")
    stats = collector.get_stats()
    print(f"  总反馈数: {stats['total']}")
    print(f"  用户平均评分: {stats['user_avg']:.2f}/5")
    print(f"  AI 平均评分: {stats['ai_avg']:.2f}/5")

    print("\n── 评分分布 ──")
    dist = stats["rating_distribution"]
    max_count = max(dist.values()) if dist.values() else 1
    for rating, count in sorted(dist.items()):
        bar = "█" * (count * 20 // max_count)
        print(f"  {rating} 星: {bar} ({count})")

    # --- 反馈驱动的改进 ---
    print("\n✅ 反馈驱动改进流程:")

    improvement_cycle = """
    1. 收集反馈 → 识别低分回复
    2. 分析原因 → 归类问题类型
    3. 改进方案 → 更新 Prompt / 增加文档 / 调优模型
    4. 验证效果 → 对比改进前后评分
    5. 上线 → 持续监控
    """

    low_score_items = [
        fb for fb in collector.feedbacks
        if fb.source == "user" and fb.score <= 3
    ]
    print(f"\n  识别到 {len(low_score_items)} 个低分反馈:")
    for fb in low_score_items:
        print(f"    评分: {fb.score} - {fb.comment}")

    # 实际 API:
    # client.create_feedback(
    #     run_id="xxx",
    #     key="user_score",
    #     score=5,
    #     comment="很好的回答",
    # )

    print()


# ============================================================
# 6. Hub（Prompt 注册中心）
# ============================================================
# 面试要点: Hub 是 Prompt 的 GitHub
#
# Hub 功能:
#   1. Prompt 版本管理: 类似 Git 的 commit 历史
#   2. 团队协作: 共享和复用 Prompt
#   3. 一键拉取: langsmith pull 命令
#   4. 社区共享: 公开 Prompt 供他人使用
#
# Hub 工作流:
#   编写 Prompt → 测试 → push 到 Hub → 团队 review → 应用拉取 → 生产使用
#
# 面试常见追问:
#   Q: Hub 和 Git 管理 Prompt 有什么区别？
#   A: Hub 专为 Prompt 设计，支持结构化 Prompt、自动版本、一键部署
#       Git 管理的是文本文件，缺少专门的 Diff 和测试集成
# ============================================================

def demo_hub():
    """Hub Prompt 注册中心示例"""
    print("=" * 60)
    print("6. Hub（Prompt 注册中心）")
    print("=" * 60)

    # --- Prompt 版本管理 ---
    print("\n✅ Prompt 版本管理模拟:")

    from dataclasses import dataclass, field

    @dataclass
    class PromptVersion:
        """Prompt 版本"""
        version: str
        commit_hash: str
        prompt_text: str
        description: str
        metadata: dict = field(default_factory=dict)

    @dataclass
    class MockHubPrompt:
        """模拟 Hub 上的 Prompt"""
        owner: str
        name: str
        description: str = ""
        versions: list[PromptVersion] = field(default_factory=list)
        tags: list[str] = field(default_factory=list)

        def add_version(self, prompt_text: str, description: str,
                        metadata: dict | None = None):
            import hashlib
            commit = hashlib.sha256(prompt_text.encode()).hexdigest()[:7]
            version_num = f"v{len(self.versions) + 1}"
            pv = PromptVersion(
                version=version_num,
                commit_hash=commit,
                prompt_text=prompt_text,
                description=description,
                metadata=metadata or {},
            )
            self.versions.append(pv)
            return pv

    # --- RAG Prompt 版本演进 ---
    rag_prompt = MockHubPrompt(
        owner="my-org",
        name="rag-qa-prompt",
        description="RAG 问答系统的 System Prompt",
        tags=["rag", "qa", "production"],
    )

    # v1: 初始版本
    rag_prompt.add_version(
        prompt_text="你是一个问答助手。根据以下上下文回答问题：\n上下文: {context}\n问题: {question}",
        description="初始版本，基础 RAG Prompt",
        metadata={"author": "Alice", "tested_on": "gpt-4"},
    )

    # v2: 优化版本 - 添加角色设定
    rag_prompt.add_version(
        prompt_text="""你是一个专业的知识问答助手。请严格基于提供的上下文信息回答问题。

规则:
1. 只使用上下文中的信息
2. 如果上下文不包含答案，诚实地说"我不知道"
3. 回答要简洁准确

上下文:
{context}

问题: {question}

回答:""",
        description="v2: 增加角色设定和回答规则，减少幻觉",
        metadata={"author": "Bob", "tested_on": "gpt-4", "hallucination_rate": "reduced_40%"},
    )

    # v3: 添加引用要求
    rag_prompt.add_version(
        prompt_text="""你是一个专业的知识问答助手。请严格基于提供的上下文信息回答问题。

【核心规则】
1. 只使用上下文中的信息，禁止编造
2. 如果上下文不包含答案，如实回答"根据已有信息无法回答"
3. 回答要简洁准确，包含引用编号，如 [1]、[2]
4. 多个信息源时，综合归纳

【上下文】
{context}

【问题】
{question}

【回答格式】
答案: [你的回答]
参考: [引用的文档编号]""",
        description="v3: 添加引用要求和结构化输出格式",
        metadata={"author": "Carol", "tested_on": "gpt-4-turbo", "format": "structured"},
    )

    # --- 展示版本历史 ---
    print(f"  Hub Prompt: {rag_prompt.owner}/{rag_prompt.name}")
    print(f"  描述: {rag_prompt.description}")
    print(f"  标签: {', '.join(rag_prompt.tags)}")
    print(f"\n  版本历史:")
    print(f"  {'版本':<6} {'Commit':<10} {'描述':<50} {'作者':<10}")
    print(f"  {'-'*6} {'-'*10} {'-'*50} {'-'*10}")
    for v in rag_prompt.versions:
        author = v.metadata.get("author", "Unknown")
        print(f"  {v.version:<6} {v.commit_hash:<10} {v.description[:48]:<50} {author:<10}")

    # --- 拉取 Prompt ---
    print("\n✅ 拉取和使用 Prompt:")

    # 实际 API:
    # from langchain import hub
    # prompt = hub.pull("my-org/rag-qa-prompt:v3")
    # chain = prompt | llm | output_parser

    latest = rag_prompt.versions[-1]
    print(f"  hub.pull('{rag_prompt.owner}/{rag_prompt.name}:{latest.version}')")
    print(f"\n  拉取的 Prompt:")
    print(f"  {latest.prompt_text[:200]}...")

    # --- 推送 Prompt ---
    print("\n✅ 推送 Prompt:")

    # 实际 API:
    # hub.push("my-org/rag-qa-prompt", prompt_object, new_repo=False)
    print(f"  hub.push('{rag_prompt.owner}/{rag_prompt.name}', prompt)")

    # --- Hub 社区探索 ---
    print("\n✅ Hub 社区 Prompt 示例:")
    community_prompts = [
        ("hwchase17/react", "ReAct Agent Prompt"),
        ("langchain-ai/rag-prompt", "RAG 最佳实践 Prompt"),
        ("rlm/rag-prompt", "RAG 提示模板"),
        ("efriis/bangladesh-rag", "孟加拉语 RAG Prompt"),
    ]
    for path, desc in community_prompts:
        print(f"  hub.pull('{path}')  ← {desc}")

    print()


# ============================================================
# 7. 与 LangChain/LangGraph 集成
# ============================================================
# 面试要点: LangSmith 深度集成 LangChain 生态
#
# 集成方式:
#   1. 环境变量自动集成: 设置 LANGCHAIN_TRACING_V2=true 即可
#   2. Callback 回调: 使用 LangChainTracer 回调
#   3. @traceable 装饰器: 手动标记函数
#   4. Context Manager: trace() 上下文管理器
#
# 自动捕获的内容:
#   - LLM 调用 (模型名、token 数、延迟)
#   - Chain 执行 (每步输入输出)
#   - Retriever 调用 (查询和返回文档)
#   - Tool 调用 (工具名和参数)
#   - Agent 决策 (思考过程)
#
# 面试常见追问:
#   Q: 如何在不入侵代码的情况下集成？
#   A: 设置环境变量即可，LangSmith 自动捕获所有 LangChain 调用
#   Q: LangGraph 的 StateGraph 如何追踪？
#   A: LangSmith 自动追踪每个节点和状态转换
# ============================================================

def demo_integration():
    """与 LangChain/LangGraph 集成示例"""
    print("=" * 60)
    print("7. 与 LangChain/LangGraph 集成")
    print("=" * 60)

    # --- 环境变量自动集成 ---
    print("\n✅ 环境变量自动集成:")

    config_example = {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_API_KEY": "ls__your_api_key_here",
        "LANGCHAIN_PROJECT": "my-production-app",
    }
    for key, value in config_example.items():
        masked = value if "api" not in key.lower() else "ls__***"
        print(f"  export {key}={masked}")

    # --- Callback 手动集成 ---
    print("\n✅ Callback 手动集成:")
    print("""
    from langchain.callbacks.tracers import LangChainTracer
    from langsmith import Client

    client = Client()
    tracer = LangChainTracer(
        project_name="my-project",
        client=client,
    )

    # 在 invoke 时传入
    chain.invoke(
        {"question": "什么是 LangChain？"},
        config={"callbacks": [tracer]}
    )
    """)

    # --- @traceable 装饰器自定义追踪 ---
    print("✅ @traceable 自定义追踪:")
    print("""
    from langsmith import traceable

    @traceable(run_type="tool", name="custom_retriever")
    def my_retriever(query: str) -> list[str]:
        '''自定义检索器，LangSmith 自动追踪'''
        # 检索逻辑
        return ["doc1", "doc2"]

    @traceable(run_type="chain", name="document_processor")
    def process_documents(docs: list[str]) -> str:
        '''文档处理链'''
        return "\\n".join(docs)
    """)

    # --- LangGraph 自动追踪 ---
    print("\n✅ LangGraph 自动追踪:")
    print("""
    from langgraph.graph import StateGraph, START, END

    # LangGraph 的每个节点和状态转换都会被自动追踪
    graph = StateGraph(MyState)
    graph.add_node("retrieve", retrieve)    # 自动追踪
    graph.add_node("generate", generate)    # 自动追踪
    graph.add_node("evaluate", evaluate)    # 自动追踪

    # 执行时自动创建 Trace
    app = graph.compile()
    result = app.invoke({"question": "..."})
    # → LangSmith 自动记录: 节点调用顺序、状态变化、耗时等
    """)

    # --- 模拟集成追踪 ---
    print("✅ 集成追踪数据模拟:")

    class TraceCollector:
        """模拟 LangChain 集成时的追踪收集"""

        def __init__(self):
            self.records = []

        def record(self, component, action, details):
            self.records.append({
                "component": component,
                "action": action,
                "details": details,
            })

    collector = TraceCollector()

    # 模拟 LangChain + LangSmith 自动收集的追踪数据
    collector.record("ChatOpenAI", "llm_call", {
        "model": "gpt-4",
        "input_tokens": 150,
        "output_tokens": 80,
        "latency_ms": 1200,
    })
    collector.record("Retriever", "search", {
        "query": "什么是 LangChain？",
        "k": 3,
        "documents_found": 3,
        "latency_ms": 45,
    })
    collector.record("PromptTemplate", "format", {
        "template": "rag-qa-prompt:v3",
        "variables": ["context", "question"],
    })
    collector.record("StateGraph", "node_execution", {
        "node": "retrieve",
        "state_before": {"question": "...", "documents": []},
        "state_after": {"question": "...", "documents": ["doc1", "doc2"]},
    })
    collector.record("StateGraph", "node_execution", {
        "node": "generate",
        "state_before": {"question": "...", "documents": ["doc1", "doc2"]},
        "state_after": {"answer": "LangChain 是一个 AI 应用框架"},
    })

    print(f"\n  自动捕获 {len(collector.records)} 个事件:")
    for r in collector.records:
        print(f"    [{r['component']}] {r['action']}: {r['details']}")

    # --- 客户端 API ---
    print("\n✅ LangSmith Client 主要 API:")

    apis = [
        ("client.list_runs()", "查询所有 Run"),
        ("client.read_run(run_id)", "读取单个 Run 详情"),
        ("client.create_dataset(name)", "创建数据集"),
        ("client.list_datasets()", "列出所有数据集"),
        ("client.create_example(inputs, outputs, dataset_id)", "添加示例到数据集"),
        ("client.create_feedback(run_id, key, score)", "创建反馈"),
        ("client.list_feedbacks(run_ids)", "查询反馈"),
        ("client.share_run(run_id)", "分享 Run"),
        ("client.read_project(project_name)", "读取项目信息"),
    ]
    for api, desc in apis:
        print(f"  {api:<45} ← {desc}")

    print()


# ============================================================
# 8. 生产环境最佳实践
# ============================================================
# 面试要点: 在生产环境中高效使用 LangSmith
#
# 核心原则:
#   1. 成本控制: 采样、过滤、数据保留策略
#   2. 性能优化: 异步发送、本地缓冲、批量提交
#   3. 安全合规: 数据脱敏、访问控制、数据留存
#   4. 监控告警: 设置质量阈值、延迟告警
#
# 生产 Checklist:
#   ☐ 设置采样率 (10-30%)
#   ☐ 过滤敏感数据 (PII 脱敏)
#   ☐ 异步发送追踪数据
#   ☐ 设置数据保留期限
#   ☐ 配置告警规则
#   ☐ 定期检查成本报表
#   ☐ 建立评估流水线
#
# 常见陷阱:
#   1. 100% 采样导致成本过高
#   2. 忘记过滤敏感数据（API key、用户数据）
#   3. 同步发送影响应用延迟
#   4. 缺乏数据清理导致存储成本膨胀
# ============================================================

def demo_production():
    """生产环境最佳实践示例"""
    print("=" * 60)
    print("8. 生产环境最佳实践")
    print("=" * 60)

    # --- 成本控制 ---
    print("\n✅ 成本控制策略:")

    from dataclasses import dataclass

    @dataclass
    class CostEstimate:
        """成本估算"""
        daily_traces: int
        avg_spans_per_trace: int
        cost_per_1k_spans: float = 0.005  # 假设每 1000 span $0.005

        @property
        def monthly_spans(self):
            return self.daily_traces * self.avg_spans_per_trace * 30

        @property
        def monthly_cost(self):
            return self.monthly_spans * self.cost_per_1k_spans / 1000

    # 不同规模的估算
    scenarios = [
        ("开发环境", CostEstimate(100, 3)),
        ("小型生产", CostEstimate(1000, 5)),
        ("中型生产", CostEstimate(10000, 5)),
        ("大型生产", CostEstimate(100000, 8)),
    ]

    print(f"  {'场景':<12} {'日请求':>8} {'月 Span':>12} {'月成本(全量)':>12} {'30%采样':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*12} {'-'*10}")
    for name, est in scenarios:
        print(f"  {name:<12} {est.daily_traces:>8} {est.monthly_spans:>12,} "
              f"${est.monthly_cost:>10.2f} ${est.monthly_cost*0.3:>9.2f}")

    # --- 采样策略 ---
    print("\n✅ 采样策略实现:")

    import random
    import hashlib

    class SamplingStrategy:
        """采样策略"""

        def __init__(self, sample_rate: float = 0.3):
            self.sample_rate = sample_rate

        def should_sample_random(self) -> bool:
            """随机采样"""
            return random.random() < self.sample_rate

        def should_sample_deterministic(self, user_id: str) -> bool:
            """基于用户 ID 的确定性采样"""
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            return (hash_val % 1000) / 1000.0 < self.sample_rate

        def should_sample_error(self, has_error: bool) -> bool:
            """错误必须全量采样"""
            if has_error:
                return True
            return self.should_sample_random()

    strategy = SamplingStrategy(0.3)

    print("  采样决策示例:")
    test_cases = [
        ("random", {}),
        ("user_alice", {"user_id": "alice@example.com"}),
        ("user_bob", {"user_id": "bob@example.com"}),
        ("has_error", {"has_error": True}),
        ("user_carol", {"user_id": "carol@example.com"}),
    ]

    for label, kwargs in test_cases:
        if "user_id" in kwargs:
            decision = strategy.should_sample_deterministic(kwargs["user_id"])
        elif "has_error" in kwargs:
            decision = strategy.should_sample_error(kwargs["has_error"])
        else:
            decision = strategy.should_sample_random()
        print(f"    {label:<15} → {'采样 ✓' if decision else '跳过 ✗'}")

    # --- 数据脱敏 ---
    print("\n✅ 数据脱敏 (PII Masking):")

    import re

    class PIIMasker:
        """PII 数据脱敏器"""

        PATTERNS = {
            "email": (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]'),
            "phone": (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),
            "api_key": (r'(?:sk-|ls__|api_key=)[a-zA-Z0-9_-]+', '[API_KEY]'),
            "credit_card": (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[CC]'),
        }

        @classmethod
        def mask(cls, text: str) -> str:
            """对文本进行脱敏处理"""
            masked = text
            for name, (pattern, replacement) in cls.PATTERNS.items():
                masked = re.sub(pattern, replacement, masked)
            return masked

    # 测试脱敏
    sensitive_texts = [
        "请联系 alice@example.com 获取帮助",
        "API key: sk-proj-abc123def456",
        "电话: 138-1234-5678",
        "信用卡: 1234 5678 9012 3456",
        "用户问: 我的邮箱是 bob@test.com",
    ]

    print("  脱敏测试:")
    for text in sensitive_texts:
        masked = PIIMasker.mask(text)
        if masked != text:
            print(f"    原始: {text}")
            print(f"    脱敏: {masked}")
            print()

    # --- 异步发送 ---
    print("✅ 异步发送配置:")

    print("""
    import os

    # 异步发送追踪数据，不阻塞主流程
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    # 使用 BackgroundTracer（默认异步）
    # 数据在后台线程中批量发送

    # 或手动控制发送
    from langsmith import Client

    client = Client(
        auto_batch_tracing=True,      # 自动批量
        batch_size=100,               # 每批 100 条
        batch_interval=5.0,           # 每 5 秒发送一次
    )
    """)

    # --- 监控告警 ---
    print("✅ 监控告警设置:")

    alerts = [
        ("延迟告警", "P95 延迟 > 3s", "通知 on-call"),
        ("质量告警", "平均评分 < 3.5", "通知 PM"),
        ("成本告警", "月度成本 > $500", "通知工程负责人"),
        ("错误率告警", "错误率 > 5%", "立即通知"),
        ("Token 用量告警", "日均 Token > 1M", "通知团队"),
    ]

    print(f"  {'告警类型':<16} {'触发条件':<20} {'处理方式':<15}")
    print(f"  {'-'*16} {'-'*20} {'-'*15}")
    for name, condition, action in alerts:
        print(f"  {name:<16} {condition:<20} {action:<15}")

    # --- 数据保留 ---
    print("\n✅ 数据保留策略:")
    retention = [
        ("开发环境", "7 天", "仅用于调试"),
        ("测试环境", "30 天", "保留评估结果"),
        ("生产环境", "90 天", "满足审计要求"),
        ("归档", "1 年+", "导出到数据仓库"),
    ]
    for env, duration, purpose in retention:
        print(f"  {env}: {duration} - {purpose}")

    print()


# ============================================================
# 9. 总结与面试要点
# ============================================================

def demo_summary():
    """LangSmith 面试要点总结"""
    print("=" * 60)
    print("9. LangSmith 面试要点总结")
    print("=" * 60)

    print("""
    LangSmith 面试速查卡:

    ┌─────────────────┬──────────────────────────────────────────────┐
    │ 概念             │ 关键点                                       │
    ├─────────────────┼──────────────────────────────────────────────┤
    │ 定位             │ LLM 应用全生命周期可观测性平台                │
    │ Tracing         │ 自动追踪 LLM 调用链，可视化调试               │
    │ 自动追踪         │ 设置 LANGCHAIN_TRACING_V2=true 即可          │
    │ 手动追踪         │ @traceable 装饰器 / RunTree API              │
    │ Dataset         │ 测试用例管理，支持版本化和导出                │
    │ Evaluation      │ 内置 + 自定义评估器，支持 LLM-as-Judge        │
    │ Feedback        │ 用户反馈 + AI 反馈，驱动持续改进              │
    │ Hub             │ Prompt 注册中心，版本管理 + 团队协作          │
    │ 集成             │ 环境变量 / Callback / @traceable / Context   │
    │ 采样             │ 生产环境建议 10-30% 采样 + 错误全量          │
    │ 成本控制         │ 采样 + 过滤 + 数据保留 + 异步发送            │

    高频面试题 & 标准答案:

    Q1: LangSmith 是什么？在 LangChain 生态中扮演什么角色？
    A1: LangSmith 是 LLM 应用的可观测性和评估平台，贯穿开发→测试→生产
        全生命周期。核心功能包括 Tracing（调试）、Dataset（测试）、
        Evaluation（评估）、Feedback（反馈）和 Hub（Prompt 管理）。
        它是 LangChain 生态的"质量保障系统"。

    Q2: LangSmith 的 Tracing 机制如何工作？
    A2: 基于 RunTree 数据结构，每个 Trace 包含多个 Run（步骤）。
        自动追踪：设置环境变量后，LangChain/LangGraph 的每次调用
        自动创建 Run。手动追踪：使用 @traceable 装饰器或 RunTree API。
        数据通过异步批量发送到 LangSmith 服务端，不影响应用性能。

    Q3: 如何用 LangSmith 评估 RAG 系统质量？
    A3: 1) 创建评估数据集（包含问题-答案对）
        2) 运行 RAG 系统生成回答
        3) 使用 Evaluator 评估：正确性、相关性、完整性、有用性
        4) 支持 LLM-as-Judge 评估主观指标
        5) 进行 A/B 对比实验，选择最优方案

    Q4: LangSmith Hub 的作用是什么？
    A4: Hub 是 Prompt 的 GitHub。提供：
        - 版本管理：每次修改自动生成版本
        - 团队协作：共享和复用 Prompt
        - 一键部署：hub.push/hub.pull 命令
        - 社区生态：公开 Prompt 库

    Q5: 如何在生产环境中高效使用 LangSmith？
    A5: 1) 采样：10-30% 随机采样 + 错误全量
        2) 脱敏：自动过滤 PII、API key
        3) 异步：后台批量发送，不阻塞主流程
        4) 成本：设置数据保留期限
        5) 监控：配置质量/延迟/成本告警

    常见陷阱:
    1. ❌ 全量采样导致成本失控 → ✅ 使用 10-30% 采样率
    2. ❌ 忘记脱敏敏感数据 → ✅ 配置 PII 过滤器
    3. ❌ 同步发送影响延迟 → ✅ 使用异步批量发送
    4. ❌ 缺少评估管线 → ✅ 建立 CI/CD 评估流水线
    5. ❌ 忽略反馈循环 → ✅ 定期分析用户反馈改进应用

    与竞品对比:
    ┌──────────────┬────────────┬──────────────┬──────────────┐
    │ 特性          │ LangSmith  │ W&B          │ MLflow       │
    ├──────────────┼────────────┼──────────────┼──────────────┤
    │ LLM 调用追踪  │ ✅ 原生     │ ✅ 插件      │ ❌           │
    │ LangChain集成 │ ✅ 深度     │ ⚠️ 需配置    │ ❌           │
    │ Prompt 管理   │ ✅ Hub     │ ❌           │ ❌           │
    │ 用户反馈      │ ✅ 原生     │ ❌           │ ❌           │
    │ 模型训练追踪  │ ❌          │ ✅ 核心      │ ✅ 核心      │
    │ 自托管        │ ✅ 企业版   │ ✅           │ ✅           │
    │ 开源          │ ✅ SDK     │ ✅           │ ✅           │
    └──────────────┴────────────┴──────────────┴──────────────┘

    快速上手步骤:
    1. pip install langsmith
    2. 设置环境变量 LANGCHAIN_API_KEY
    3. 设置 LANGCHAIN_TRACING_V2=true
    4. 运行你的 LangChain 应用
    5. 在 https://smith.langchain.com 查看追踪
    """)
    print()


# ============================================================
# 主函数
# ============================================================

def main():
    """运行所有 LangSmith 示例"""
    print("""
╔══════════════════════════════════════════════════════════╗
║           LangSmith 观测与评估 — 完整示例                 ║
║                                                          ║
║  全部代码使用模拟数据，无需真实 API key 即可运行           ║
╚══════════════════════════════════════════════════════════╝
""")
    demo_langsmith_overview()
    demo_tracing()
    demo_dataset()
    demo_evaluation()
    demo_feedback()
    demo_hub()
    demo_integration()
    demo_production()
    demo_summary()


if __name__ == "__main__":
    main()
```

:::

---

## 📖 章节概述

LangSmith 是 LangChain 生态系统中的 **LLM 应用可观测性与评估平台**，由 LangChain 团队开发维护。它贯穿 LLM 应用的完整生命周期：从开发阶段的调试追踪，到测试阶段的评估验证，再到生产环境的监控反馈。

### 学习目标

完成本章学习后，你将能够：

1. **理解 LangSmith 的定位**：在 LLM 应用开发中的角色和价值
2. **掌握 Tracing 机制**：如何自动和手动追踪 LLM 调用链
3. **构建评估流水线**：使用 Dataset 和 Evaluator 建立质量保障体系
4. **运用反馈系统**：收集用户和 AI 反馈持续改进应用
5. **管理 Prompt 版本**：通过 Hub 实现团队协作和版本控制
6. **生产环境部署**：掌握采样、脱敏、成本控制等最佳实践

---

## 🎯 面试高频问答

### Q1: LangSmith 是什么？它在 LangChain 生态中扮演什么角色？

**标准答案：**

LangSmith 是 LangChain 官方推出的 LLM 应用全生命周期管理平台。它在 LangChain 生态中扮演 **"质量保障系统"** 的角色。

LangSmith 定位为四个层次：

| 层次 | 阶段 | LangSmith 功能 |
|------|------|----------------|
| 开发层 | Develop | Hub 管理 Prompt，Tracing 调试调用链 |
| 测试层 | Test | Dataset 管理测试用例，Evaluation 自动评估 |
| 生产层 | Monitor | Tracing 监控生产流量，Feedback 收集反馈 |
| 优化层 | Improve | 基于反馈数据迭代 Prompt 和应用逻辑 |

**与 LangChain/LangGraph 的关系：**
- LangChain 负责**构建** → LangSmith 负责**观测**
- LangGraph 负责**编排** → LangSmith 负责**追踪**
- 三者配合形成完整的 **构建→观测→评估→迭代** 闭环

**追问：为什么不用传统的日志系统？**
传统日志系统（如 ELK）设计用于确定性系统，而 LLM 应用具有以下特点：
- **非确定性**：同一输入可能产生不同输出，需要追踪完整的上下文
- **链式调用**：一个请求可能触发多层嵌套调用，需要树状结构
- **多模态**：涉及文本、嵌入向量、工具调用等多种数据类型
- **评估需求**：需要专门的质量评估工具，而非简单的错误日志

---

### Q2: LangSmith 的 Tracing 机制是如何工作的？

**标准答案：**

LangSmith 的 Tracing 基于 **RunTree** 数据结构，核心概念包括：

```
Trace (一次用户请求)
  ├── Run (LLM 调用)
  │   ├── Span (Token 计数、延迟)
  │   └── Span (模型参数)
  ├── Run (检索器调用)
  │   └── Span (查询嵌入)
  └── Run (输出解析)
```

**追踪方式对比：**

| 方式 | 入侵性 | 粒度 | 适用场景 |
|------|--------|------|----------|
| 环境变量自动追踪 | 零入侵 | 粗粒度（组件级） | 快速接入，标准 LangChain 应用 |
| @traceable 装饰器 | 低入侵 | 中粒度（函数级） | 自定义函数、工具 |
| RunTree API | 高入侵 | 细粒度（任意级别） | 完全自定义追踪 |
| Callback 回调 | 中入侵 | 中粒度 | 需要精确控制追踪时机 |

**数据流架构：**
```
应用代码 → RunTree 构建 → 本地缓冲队列 → 异步批量发送 → LangSmith 服务端
                                                              ↓
                                              Web UI 可视化 / API 查询 / 导出
```

**面试加分点：**
- 支持分布式追踪：通过 `trace_id` 和 `parent_run_id` 跨服务关联
- 异步非阻塞：追踪数据在后台线程批量发送，不影响主流程延迟（<1ms 开销）
- 元数据丰富：自动记录 token 消耗、模型参数、延迟百分位等
- 支持采样：随机采样、基于用户采样、错误全量采样等策略

---

### Q3: 如何用 LangSmith 评估 RAG 系统的质量？

**标准答案：**

LangSmith 评估 RAG 系统遵循五步法：

**第一步：构建评估数据集**
```python
# 创建包含典型问答对的数据集
dataset = client.create_dataset("rag-eval-v1")
client.create_example(
    inputs={"question": "什么是 LangChain？"},
    outputs={"answer": "LangChain 是用于构建 LLM 应用的开源框架"},
    dataset_id=dataset.id,
)
```

**第二步：定义评估指标**
LangSmith 支持多维度评估：

| 指标类型 | 评估内容 | 评估方法 |
|----------|----------|----------|
| 正确性 (Correctness) | 回答是否与参考答案一致 | LLM-as-Judge + 参考答案 |
| 忠实度 (Faithfulness) | 回答是否基于提供的上下文 | LLM-as-Judge 检查幻觉 |
| 相关性 (Relevance) | 检索文档是否与问题相关 | 向量相似度 / LLM 判断 |
| 完整性 (Completeness) | 是否覆盖所有关键信息点 | 关键信息覆盖率 |
| 有害性 (Harmfulness) | 回答是否包含有害内容 | 内容安全检测 |

**第三步：运行评估**
```python
from langsmith.evaluation import evaluate

results = evaluate(
    lambda inputs: my_rag_chain(inputs),  # 待评估的应用
    data="rag-eval-v1",                    # 数据集
    evaluators=[
        correctness_evaluator,
        faithfulness_evaluator,
        relevance_evaluator,
    ],
    experiment_prefix="rag-v2-",
)
```

**第四步：对比实验**
```python
# A/B 对比不同配置
results_a = evaluate(rag_v1, data="rag-eval-v1", experiment_prefix="v1-")
results_b = evaluate(rag_v2, data="rag-eval-v1", experiment_prefix="v2-")
# 在 LangSmith UI 中并排对比两个实验
```

**第五步：持续监控**
- 将生产流量采样到评估数据集
- 定期（每日/每周）运行评估
- 设置质量阈值告警

**面试加分点：**
- **LLM-as-Judge** 是评估主观指标的主要方式，但需要校准（与人工评估对比）
- **Pairwise 对比**：同时展示两个版本的输出让人工判断哪个更好
- **在线评估**：除了离线评估，LangSmith 也支持在生产环境实时评估每条请求

---

### Q4: LangSmith Hub 的作用是什么？

**标准答案：**

LangSmith Hub 可以理解为 **Prompt 的 GitHub**，它解决了 LLM 应用开发中的 Prompt 管理痛点。

**核心功能：**

| 功能 | 说明 | 类比 Git |
|------|------|----------|
| 版本管理 | 每次修改自动生成 commit，可回溯历史 | `git commit` |
| 分支协作 | 团队成员可 fork/push Prompt | `git branch/push` |
| 一键部署 | `hub.pull()` 直接拉取到代码中使用 | `git pull` |
| 社区共享 | 浏览和使用社区公开 Prompt | GitHub 公开仓库 |

**使用示例：**
```python
# 拉取社区 Prompt
from langchain import hub
prompt = hub.pull("hwchase17/react")  # ReAct Agent Prompt

# 推送团队 Prompt
hub.push("my-org/rag-prompt", my_prompt)

# 指定版本拉取
prompt = hub.pull("my-org/rag-prompt:v3")
```

**为什么不用 Git 管理 Prompt？**
1. **结构化理解**：Hub 理解 Prompt 模板的变量结构，Git 只看到纯文本
2. **专用 Diff**：Hub 展示 Prompt 变量的增删改，而非文本行变化
3. **直接集成**：可以在 LangSmith UI 中测试 Prompt 后直接部署
4. **权限管理**：细粒度的组织和仓库权限控制
5. **评估联动**：Hub 版本与评估实验关联，追踪每次 Prompt 变更的效果

---

### Q5: 如何在生产环境中高效使用 LangSmith？

**标准答案：**

生产环境使用 LangSmith 需要平衡**观测覆盖度**和**成本性能**。以下是核心策略：

**1. 采样策略（最重要）**

```
开发环境: 100% 采样（全量调试）
灰度环境: 50% 采样（验证稳定性）
生产环境: 10-30% 采样（成本控制）
错误请求: 100% 采样（必须全量，用于问题排查）
```

**2. 成本控制矩阵**

| 日请求量 | 建议采样率 | 预估月成本 | 策略 |
|----------|-----------|-----------|------|
| <1,000 | 100% | <$5 | 全量采样 |
| 1,000-10,000 | 30% | $5-$15 | 随机采样 |
| 10,000-100,000 | 10% | $15-$50 | 随机 + 错误全量 |
| >100,000 | 5-10% | $50+ | 分层采样 + 聚合 |

**3. 数据安全（必须执行）**

```python
# 必须配置 PII 脱敏
# 包括但不限于：
- 邮箱地址 → 替换为 [EMAIL]
- 手机号码 → 替换为 [PHONE]
- API Key → 替换为 [API_KEY]
- 信用卡号 → 替换为 [CC]
- 身份证号 → 替换为 [ID_CARD]
```

**4. 异步发送配置**

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# 默认后台异步发送，不阻塞请求
# 额外优化：
# - 设置合理的 batch_size（100-500）
# - 设置 batch_interval（1-5秒）
# - 设置最大重试次数（3次）
```

**5. 监控告警体系**

```
告警层级:
  P0（立即）：错误率 > 5%，P99 延迟 > 10s
  P1（5分钟）：用户平均评分 < 3.0，Token 消耗异常
  P2（每日）：月度成本超预算 80%
```

---

## 🔄 LangSmith vs 竞品对比

### 与主流 LLM 观测平台的详细对比

| 维度 | LangSmith | Weights & Biases | MLflow | Arize/Phoenix |
|------|-----------|------------------|--------|---------------|
| **核心定位** | LLM 应用全生命周期 | 模型训练实验管理 | 通用 ML 生命周期 | 模型监控与分析 |
| **LLM 调用追踪** | ✅ 原生深度支持 | ⚠️ 通过插件 | ❌ 不支持 | ⚠️ 部分支持 |
| **LangChain 集成** | ✅ 零配置自动集成 | ⚠️ 需手动配置 | ❌ 不适用 | ⚠️ 需适配 |
| **Prompt 管理** | ✅ Hub 注册中心 | ❌ 无 | ❌ 无 | ❌ 无 |
| **评估框架** | ✅ 内置评估器 | ⚠️ 需自定义 | ✅ 通用评估 | ❌ 不适用 |
| **用户反馈收集** | ✅ 原生 API | ❌ 无 | ❌ 无 | ❌ 无 |
| **模型训练追踪** | ❌ 非核心 | ✅ 核心能力 | ✅ 核心能力 | ❌ 非核心 |
| **自托管部署** | ✅ 企业版支持 | ✅ 支持 | ✅ 支持 | ✅ 支持 |
| **开源程度** | SDK 开源，平台闭源 | ✅ 完全开源 | ✅ 完全开源 | ✅ 完全开源 |
| **定价模式** | 免费额度 + 按量付费 | 免费额度 + 按量 | 开源免费 | 开源免费 |
| **学习曲线** | 低（与 LangChain 天然集成） | 中 | 高（概念较多） | 中 |

**选型建议：**

| 场景 | 推荐方案 |
|------|----------|
| 深度使用 LangChain/LangGraph | **LangSmith**（首选） |
| 需要模型训练实验管理 | Weights & Biases |
| 通用 ML 项目管理 | MLflow |
| 纯模型性能监控 | Arize/Phoenix |
| 预算有限 + 需要自托管 | MLflow + 自建追踪 |
| 全面解决方案 | LangSmith + W&B 组合 |

---

## ⚠️ 常见陷阱与解决

### 1. 全量追踪导致成本爆炸

**问题**：在生产环境开启 100% 追踪，日请求 10 万，月成本超 $500。

**解决**：
- 使用 10-30% 随机采样
- 错误请求全量采样（通过 `should_sample_error()` 判断）
- 设置月度预算告警
- 定期审查和清理过期数据

### 2. 敏感数据泄露

**问题**：用户对话中的邮箱、电话、API Key 被记录到 LangSmith。

**解决**：
```python
# 必须配置 PII Masking
from langsmith import Client
client = Client(
    hide_inputs=["email", "phone", "credit_card"],
    hide_outputs=["api_key", "token"],
)
```

### 3. 同步发送影响延迟

**问题**：同步发送追踪数据导致 API 响应延迟增加 50-100ms。

**解决**：LangSmith 默认使用异步发送，确保：
- 不手动调用 `client.flush()` 在请求路径中
- 检查是否错误配置了同步模式
- 设置合理的 batch 参数

### 4. 评估标准过于主观

**问题**：仅用 LLM-as-Judge 评估，缺乏客观标准。

**解决**：
- 建立参考答案（Golden Dataset）
- 结合人工审核校准 AI 评估结果
- 使用多维度评分（正确性 + 完整性 + 简洁性）
- 定期抽查 AI 评估的准确性

### 5. 忽略反馈闭环

**问题**：收集了反馈但从未分析改进。

**解决**：
- 建立每周反馈分析机制
- 低分回复自动归类（幻觉/不完整/跑题/格式错误）
- 反馈数据驱动 Prompt 迭代
- 设置反馈驱动的自动告警

---

## 📊 快速参考

### 核心 API 速查

| API | 用途 |
|-----|------|
| `client.create_dataset(name)` | 创建评估数据集 |
| `client.create_example(inputs, outputs, dataset_id)` | 添加示例 |
| `client.list_runs(project_name=...)` | 查询追踪记录 |
| `client.create_feedback(run_id, key, score)` | 创建反馈 |
| `hub.pull("org/name")` | 拉取 Prompt |
| `hub.push("org/name", prompt)` | 推送 Prompt |
| `evaluate(fn, data, evaluators)` | 运行评估 |
| `@traceable(run_type="tool")` | 装饰器手动追踪 |

### 环境变量速查

| 变量 | 说明 |
|------|------|
| `LANGCHAIN_API_KEY` | API 密钥（必需） |
| `LANGCHAIN_TRACING_V2` | 开启追踪（设为 true） |
| `LANGCHAIN_PROJECT` | 项目名称 |
| `LANGCHAIN_ENDPOINT` | API 端点（自托管时修改） |
| `LANGCHAIN_HUB_API_URL` | Hub API 地址 |

---

## 🔗 扩展阅读

- [LangSmith 官方文档](https://docs.smith.langchain.com/)
- [LangSmith Hub](https://smith.langchain.com/hub)
- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangSmith Cookbook](https://github.com/langchain-ai/langsmith-cookbook)

---

## 🏗️ 架构深入：RunTree 详解

### RunTree 是 LangSmith 追踪的核心数据结构

理解 RunTree 对于面试和实际开发都至关重要。RunTree 本质上是一棵树，其中每个节点代表一个执行步骤。

**节点的生命周期：**

```
创建 (POST) → 运行中 (PATCH 更新) → 结束 (PATCH 最终状态) → 可选：添加反馈 (PATCH)
```

**节点类型与自动捕获的内容：**

| 节点类型 | 自动捕获的元数据 | 典型来源 |
|----------|-----------------|----------|
| `llm` | model, temperature, tokens (input/output), latency, generation | ChatOpenAI, ChatAnthropic |
| `chain` | chain_type, steps, input/output | LCEL 链, Chain 实例 |
| `retriever` | query, k, documents_returned, latency | VectorStoreRetriever |
| `tool` | tool_name, tool_args, tool_output | @tool 装饰的函数 |
| `embedding` | model, batch_size, embedding_dim | OpenAIEmbeddings |
| `agent` | agent_type, iterations, tools_used | AgentExecutor, LangGraph Agent |
| `parser` | parser_type, format_instructions | OutputParser |

**RunTree 的分布式追踪能力：**

在一个微服务架构中，一次用户请求可能跨越多个服务。LangSmith 通过以下机制实现全链路追踪：

1. **trace_id**：全局唯一的追踪标识符，跨服务传递
2. **parent_run_id**：父子关系建立，还原调用树
3. **dotted_order**：全局排序字段，保证时序正确

示例场景：用户通过 Web 服务提问，Web 服务调用 RAG 服务，RAG 服务调用向量检索服务和 LLM 服务。

```
Web Service (Trace: abc123)
  └── RAG Service (parent: abc123, run: def456)
        ├── Vector Service (parent: def456, run: ghi789)
        └── LLM Service (parent: def456, run: jkl012)
```

所有服务的追踪数据最终汇聚到 LangSmith，形成完整的调用视图。

---

## 🧩 集成深入：与 LangGraph 的高级配合

### LangGraph 节点级别的自动追踪

LangGraph 与 LangSmith 的集成非常深入。当你在 LangGraph 中编译并运行一个 StateGraph 时，LangSmith 会自动追踪每个节点和边：

**自动捕获的信息包括：**

- 每个节点的执行时间
- 节点输入/输出的状态变化
- 条件边的决策结果
- 检查点（checkpoint）的创建和恢复
- 子图的嵌套执行

**代码示例：**

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class MyState(TypedDict):
    query: str
    result: str

graph = StateGraph(MyState)
graph.add_node("step_a", node_a)
graph.add_node("step_b", node_b)
graph.add_edge(START, "step_a")
graph.add_conditional_edges("step_a", router, {"go": "step_b", "stop": END})
graph.add_edge("step_b", END)

# 编译并执行 — LangSmith 自动追踪所有节点
app = graph.compile()
result = app.invoke({"query": "hello"})

# 在 LangSmith UI 中你会看到:
# Trace: my_state_graph
#   ├── step_a (2.3s) — 输入状态, 输出状态
#   ├── [条件判断] router → "go"
#   └── step_b (1.8s) — 输入状态, 输出状态
```

### Human-in-the-Loop 的追踪

当使用 LangGraph 的 `interrupt()` 功能暂停等待人类输入时，LangSmith 同样会追踪这个等待过程：

- 暂停点会被记录为特殊的 Run 状态
- 人类输入被视为该 Run 的更新
- 完整的等待时间被记录下来

这对于分析系统性能和用户体验非常有用。

---

## 📐 评估深入：自定义评估器开发

### 编写生产级自定义评估器

虽然 LangSmith 提供了内置评估器，但实际项目中经常需要编写自定义评估器来满足特定业务需求。

**评估器设计原则：**

1. **明确评分标准**：定义清晰的评分区间（0-1 或 0-5）
2. **提供详细评语**：不只是分数，要说明为什么
3. **处理边界情况**：空输入、超长输出、格式错误
4. **避免评分偏差**：使用校准样本验证评估器

**自定义评估器模板：**

```python
from langsmith.evaluation import EvaluationResult

def my_custom_evaluator(run, example):
    """
    自定义评估器模板

    Args:
        run: 应用的运行结果
        example: 数据集中的示例

    Returns:
        EvaluationResult: 评估结果
    """
    # 1. 提取输入输出
    actual_output = run.outputs.get("answer", "")
    expected_output = example.outputs.get("answer", "")

    # 2. 计算评分
    # ... 你的评估逻辑 ...

    # 3. 返回结果
    return EvaluationResult(
        key="my_metric",
        score=score,  # 0.0 - 1.0
        comment=f"评估详情: ...",
    )
```

### 评估流水线的 CI/CD 集成

将 LangSmith 评估集成到 CI/CD 流水线中是确保质量不退化的重要实践：

```yaml
# GitHub Actions 示例
name: LangSmith Evaluation
on:
  pull_request:
    paths:
      - 'src/**'
      - 'prompts/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run LangSmith Eval
        env:
          LANGCHAIN_API_KEY: ${{ secrets.LANGCHAIN_API_KEY }}
        run: |
          python -m pytest tests/evaluation/test_quality.py
          # 如果评分低于阈值，CI 失败
```

**CI 评估的最佳实践：**

- 每次 PR 自动运行评估
- 设置最低质量门槛（如平均评分 ≥ 0.8）
- 对比当前版本与生产版本的差异
- 评估结果作为 PR 评论自动发布
- 关键指标退化时自动阻止合并

---

## 🔧 故障排查指南

### 常见问题与解决

**问题 1：追踪数据未出现在 LangSmith UI**

排查步骤：
1. 检查 `LANGCHAIN_API_KEY` 是否正确设置
2. 确认 `LANGCHAIN_TRACING_V2=true`（注意 V2）
3. 检查网络连接：`curl https://api.smith.langchain.com`
4. 查看 SDK 日志：设置 `LANGCHAIN_TRACING_LOG_LEVEL=DEBUG`
5. 确认项目名称：在 UI 中切换到正确的 Project

**问题 2：某些调用没有被追踪**

可能原因：
- 使用了未集成的第三方库（需要添加 @traceable 装饰器）
- 异步调用未被正确捕获（检查 event loop）
- 采样策略过滤掉了（检查采样配置）
- 数据被 PII 过滤器误过滤（检查脱敏规则）

**问题 3：追踪延迟过高**

优化措施：
- 确认使用异步模式（默认行为）
- 减小 payload 大小（过滤不必要的元数据）
- 调整 batch 参数（更大的 batch，更长的间隔）
- 检查网络延迟（考虑使用同区域部署）

**问题 4：评估结果不稳定**

原因与解决：
- LLM-as-Judge 的随机性 → 使用较低 temperature（0 或 0.1）
- 评估 Prompt 不够精确 → 细化评分标准
- 缺少参考答案 → 构建 Golden Dataset
- 单一维度评估 → 多维度综合评估

---

## 🎓 面试模拟：进阶追问

### 追问 1：LangSmith 如何处理大规模分布式追踪？

在大规模分布式系统中，一次用户请求可能触发数十甚至上百个微服务调用。LangSmith 的处理策略：

- **全局 trace_id 传递**：通过 HTTP header 或消息队列元数据传递
- **异步批量上报**：每个服务独立异步上报，不阻塞业务
- **服务端聚合**：LangSmith 后端根据 trace_id 和 dotted_order 重建调用树
- **采样与过滤**：大规模场景下配合采样策略控制数据量

### 追问 2：如果不用 LangSmith，有哪些替代方案？

| 方案 | 优点 | 缺点 |
|------|------|------|
| 自建追踪系统 | 完全可控、无外部依赖 | 开发维护成本高 |
| OpenTelemetry + Jaeger | 通用标准、生态丰富 | 缺少 LLM 专用语义 |
| 自定义日志 + ELK | 技术栈熟悉 | 缺少树状结构和评估能力 |
| MLflow Tracking | ML 项目兼容 | LLM 调用追踪能力弱 |

**结论**：如果深度使用 LangChain，LangSmith 是最佳选择。如果是通用 LLM 应用，可以考虑 OpenTelemetry 方案。如果是非 LLM 的 ML 项目，MLflow 更合适。

### 追问 3：LangSmith 的数据安全性如何？

- **传输加密**：所有数据通过 HTTPS 加密传输
- **数据隔离**：每个组织有独立的数据空间
- **PII 过滤**：支持自动脱敏敏感数据
- **访问控制**：基于角色的细粒度权限管理
- **自托管选项**：企业版支持完全私有化部署
- **合规认证**：SOC 2 Type II 认证
- **数据留存**：可配置数据保留策略

---

> 💡 **面试技巧**：当被问到 LangSmith 时，不要只停留在功能介绍层面。展现你对 LLM 应用开发痛点的理解 —— 为什么需要可观测性？为什么传统监控不够？然后用 LangSmith 的功能来回答这些痛点。这会让你从"会用工具"的候选人升级为"理解原理"的工程师。

---

## 📝 本章总结

本章全面覆盖了 LangSmith 的核心功能和使用场景，从基础的追踪概念到高级的生产环境部署策略。关键收获如下：

- **LangSmith 不是可选品，而是 LLM 应用开发的必需品**：没有可观测性，你无法知道你的应用在做什么，更无法系统性地改进它。
- **Tracing 是调试的起点**：在复杂的链式调用中，没有追踪如同盲人摸象。
- **评估是质量的保障**：用数据说话，而不是凭感觉判断应用好坏。
- **反馈驱动迭代**：建立从用户反馈到产品改进的完整闭环。
- **生产环境需要策略**：采样、脱敏、成本控制缺一不可。

建议在学习完本章后，实际注册一个 LangSmith 账号（有免费额度），将前面章节中的 LangChain/LangGraph 示例接入 LangSmith 追踪，亲身体验从"黑盒"到"透明"的变化过程。只有真正用过，面试时才能自信地讲述 LangSmith 为你的项目带来的价值。
