"""Prometheus 指标定义与便捷记录函数。

覆盖设计文档 6.1 节：
- 应用层：请求耗时、Token 消耗
- 检索层：检索命中率、重排耗时
- LLM 层：首 Token 延迟、总延迟、输入/输出 Token 数
- 用户反馈评分
"""
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

# 使用独立 registry 便于测试隔离
REGISTRY = CollectorRegistry()

_INFO = Info("ops_rag", "运维 RAG 助手元信息", registry=REGISTRY)
_INFO.info({"version": "0.1.0", "component": "advanced-rag"})

# 应用层
REQUEST_LATENCY = Histogram(
    "ops_rag_request_latency_seconds",
    "端到端请求延迟（秒）",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30),
    registry=REGISTRY,
)
REQUEST_COUNT = Counter(
    "ops_rag_requests_total",
    "请求总数",
    ["status"],
    registry=REGISTRY,
)

# 检索层
RETRIEVAL_HITS = Gauge(
    "ops_rag_retrieval_hits",
    "最近一次检索命中的文档数",
    registry=REGISTRY,
)
RETRIEVAL_HIT_RATIO = Gauge(
    "ops_rag_retrieval_hit_ratio",
    "检索命中率（命中数/召回数）",
    registry=REGISTRY,
)
RERANK_LATENCY = Histogram(
    "ops_rag_rerank_latency_seconds",
    "重排耗时（秒）",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2),
    registry=REGISTRY,
)

# LLM 层
LLM_TOTAL_LATENCY = Histogram(
    "ops_rag_llm_latency_seconds",
    "LLM 总延迟（秒）",
    buckets=(0.5, 1, 2, 5, 10, 30),
    registry=REGISTRY,
)
LLM_FIRST_TOKEN_LATENCY = Histogram(
    "ops_rag_llm_first_token_seconds",
    "首 Token 延迟（秒）",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5),
    registry=REGISTRY,
)
TOKEN_USAGE = Counter(
    "ops_rag_token_usage_total",
    "Token 消耗总数",
    ["direction"],  # input / output
    registry=REGISTRY,
)
TOKEN_COST_USD = Gauge(
    "ops_rag_token_cost_usd",
    "估算 Token 成本（美元）",
    registry=REGISTRY,
)

# 用户反馈
FEEDBACK_SCORE = Histogram(
    "ops_rag_feedback_score",
    "用户反馈评分 1-5",
    buckets=(1, 2, 3, 4, 5),
    registry=REGISTRY,
)

# 在线评估（LLM 自评）
FAITHFULNESS_SCORE = Gauge(
    "ops_rag_faithfulness_score",
    "LLM 自评回答忠实度 1-5（在线评估）",
    registry=REGISTRY,
)
ANSWER_RELEVANCY_SCORE = Gauge(
    "ops_rag_answer_relevancy_score",
    "LLM 自评回答相关度 1-5（在线评估）",
    registry=REGISTRY,
)
HALLUCINATION_SCORE = Gauge(
    "ops_rag_hallucination_score",
    "LLM 自评幻觉程度 0-1（在线评估）",
    registry=REGISTRY,
)


# ===== 便捷记录函数 =====
def RECORD_REQUEST(latency_ms: float, status: str = "ok") -> None:
    REQUEST_LATENCY.observe(latency_ms / 1000.0)
    REQUEST_COUNT.labels(status=status).inc()


def RECORD_TOKEN_USAGE(input_tokens: int, output_tokens: int) -> None:
    TOKEN_USAGE.labels(direction="input").inc(input_tokens)
    TOKEN_USAGE.labels(direction="output").inc(output_tokens)
    # 粗略成本估算（gpt-4o 输入 $2.5/M，输出 $10/M）
    cost = (input_tokens * 2.5 + output_tokens * 10.0) / 1_000_000
    TOKEN_COST_USD.set(cost)


def RECORD_RETRIEVAL(hits: int, recalled: int) -> None:
    RETRIEVAL_HITS.set(hits)
    if recalled > 0:
        RETRIEVAL_HIT_RATIO.set(hits / recalled)


def RECORD_RERANK(latency_ms: float) -> None:
    RERANK_LATENCY.observe(latency_ms / 1000.0)


def RECORD_LLM(latency_ms: float, first_token_ms: float = 0.0) -> None:
    LLM_TOTAL_LATENCY.observe(latency_ms / 1000.0)
    if first_token_ms:
        LLM_FIRST_TOKEN_LATENCY.observe(first_token_ms / 1000.0)


def RECORD_FEEDBACK(rating: int) -> None:
    REQUEST_COUNT.labels(status="feedback").inc()


def RECORD_FAITHFULNESS(faithfulness: float, relevancy: float = 0.0, hallucination: float = 0.0) -> None:
    """记录在线评估分数（由 evaluate 节点调用）。"""
    FAITHFULNESS_SCORE.set(round(faithfulness, 2))
    ANSWER_RELEVANCY_SCORE.set(round(relevancy, 2))
    HALLUCINATION_SCORE.set(round(hallucination, 3))


def export() -> bytes:
    """供 /metrics 端点导出。"""
    return generate_latest(REGISTRY)
