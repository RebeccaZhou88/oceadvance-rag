"""Prometheus metric definitions and convenience recording functions.

Covers design document section 6.1:
- Application layer: request latency, Token consumption
- Retrieval layer: retrieval hit rate, rerank latency
- LLM layer: first Token latency, total latency, input/output Token count
- User feedback score
"""
from collections import deque

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

# Use a separate registry for test isolation
REGISTRY = CollectorRegistry()

_INFO = Info("ops_rag", "Ops RAG assistant metadata", registry=REGISTRY)
_INFO.info({"version": "0.1.0", "component": "advanced-rag"})

# Application layer
REQUEST_LATENCY = Histogram(
    "ops_rag_request_latency_seconds",
    "End-to-end request latency (seconds)",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30),
    registry=REGISTRY,
)
REQUEST_COUNT = Counter(
    "ops_rag_requests_total",
    "Total number of requests",
    ["status"],
    registry=REGISTRY,
)
# Precise quantile: in-memory sliding window (latest 200 requests), more accurate than histogram bucket estimation
REQUEST_P50_MS = Gauge(
    "ops_rag_request_latency_p50_ms",
    "End-to-end latency P50 (milliseconds, sliding window of latest 200 requests)",
    registry=REGISTRY,
)
REQUEST_P95_MS = Gauge(
    "ops_rag_request_latency_p95_ms",
    "End-to-end latency P95 (milliseconds, sliding window of latest 200 requests)",
    registry=REGISTRY,
)
REQUEST_P99_MS = Gauge(
    "ops_rag_request_latency_p99_ms",
    "End-to-end latency P99 (milliseconds, sliding window of latest 200 requests)",
    registry=REGISTRY,
)

# Business latency: end-to-end excluding evaluate_answer (online evaluation) duration
BUSINESS_LATENCY = Histogram(
    "ops_rag_business_latency_seconds",
    "Business latency (end-to-end excluding online evaluation evaluate_answer, seconds)",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30),
    registry=REGISTRY,
)
BUSINESS_P50_MS = Gauge(
    "ops_rag_business_latency_p50_ms",
    "Business latency P50 (milliseconds, excluding evaluate, sliding window of latest 200 requests)",
    registry=REGISTRY,
)
BUSINESS_P95_MS = Gauge(
    "ops_rag_business_latency_p95_ms",
    "Business latency P95 (milliseconds, excluding evaluate, sliding window of latest 200 requests)",
    registry=REGISTRY,
)
BUSINESS_P99_MS = Gauge(
    "ops_rag_business_latency_p99_ms",
    "Business latency P99 (milliseconds, excluding evaluate, sliding window of latest 200 requests)",
    registry=REGISTRY,
)

# Latency per LangGraph node (distinguished by node name)
NODE_LATENCY = Histogram(
    "ops_rag_node_latency_seconds",
    "Latency per LangGraph node (seconds)",
    ["node"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30),
    registry=REGISTRY,
)

# Retrieval layer
RETRIEVAL_HITS = Gauge(
    "ops_rag_retrieval_hits",
    "Documents returned by Azure on the latest retrieval call",
    registry=REGISTRY,
)
# Cumulative counters across ALL RAG calls (cache-hit requests do NOT increment)
# Note: Prometheus Counter auto-strips a trailing "_total" from its collect name,
# so we must NOT name these anything that collides with the Gauge above.
_RETRIEVAL_EFFECTIVE_CUM = Counter(
    "ops_rag_retrieval_effective_cum",
    "Cumulative sum of effective document counts across all retrieval calls",
    registry=REGISTRY,
)
_RETRIEVAL_HITS_CUM = Counter(
    "ops_rag_retrieval_hits_cum",
    "Cumulative sum of Azure-returned document counts across all retrieval calls",
    registry=REGISTRY,
)
# Aggregated effective ratio = effective_total / hits_total (smoother than last-call gauge)
RETRIEVAL_EFFECTIVE_RATIO = Gauge(
    "ops_rag_retrieval_effective_ratio",
    "Retrieval effective ratio — cumulative average across all RAG queries (cache hits excluded)",
    registry=REGISTRY,
)
RERANK_LATENCY = Histogram(
    "ops_rag_rerank_latency_seconds",
    "Rerank latency (seconds)",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2),
    registry=REGISTRY,
)

# LLM layer
LLM_TOTAL_LATENCY = Histogram(
    "ops_rag_llm_latency_seconds",
    "LLM total latency (seconds)",
    buckets=(0.5, 1, 2, 5, 10, 30),
    registry=REGISTRY,
)
LLM_FIRST_TOKEN_LATENCY = Histogram(
    "ops_rag_llm_first_token_seconds",
    "First Token latency (seconds)",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5),
    registry=REGISTRY,
)
TOKEN_USAGE = Counter(
    "ops_rag_token_usage_total",
    "Total Token consumption",
    ["direction"],  # input / output
    registry=REGISTRY,
)
TOKEN_COST_USD = Gauge(
    "ops_rag_token_cost_usd",
    "Estimated Token cost (US dollars)",
    registry=REGISTRY,
)

# Exact cache
CACHE_HITS = Counter(
    "ops_rag_cache_hits_total",
    "Exact cache hits (RAG pipeline skipped)",
    registry=REGISTRY,
)
CACHE_MISSES = Counter(
    "ops_rag_cache_misses_total",
    "Exact cache misses (RAG pipeline executed)",
    registry=REGISTRY,
)
CACHE_SIZE = Gauge(
    "ops_rag_cache_size",
    "Number of entries currently in the exact cache",
    registry=REGISTRY,
)


# User feedback
FEEDBACK_SCORE = Histogram(
    "ops_rag_feedback_score",
    "User feedback score 1-5",
    buckets=(1, 2, 3, 4, 5),
    registry=REGISTRY,
)

# Online evaluation (LLM self-evaluation)
FAITHFULNESS_SCORE = Gauge(
    "ops_rag_faithfulness_score",
    "LLM self-evaluated answer faithfulness 1-5 (online evaluation)",
    registry=REGISTRY,
)
ANSWER_RELEVANCY_SCORE = Gauge(
    "ops_rag_answer_relevancy_score",
    "LLM self-evaluated answer relevancy 1-5 (online evaluation)",
    registry=REGISTRY,
)
HALLUCINATION_SCORE = Gauge(
    "ops_rag_hallucination_score",
    "LLM self-evaluated hallucination degree 0-1 (online evaluation)",
    registry=REGISTRY,
)


# ===== Convenience recording functions =====
# End-to-end latency (milliseconds) sliding window of the latest N requests, used for precise quantiles
_latency_window: deque[float] = deque(maxlen=200)
_business_window: deque[float] = deque(maxlen=200)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolation percentile (smoother than nearest-rank on small samples)."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    # rank: 1-indexed position of the percentile in the sorted array
    rank = pct / 100.0 * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    # Linear interpolation between sorted_vals[lo] and sorted_vals[hi]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def RECORD_REQUEST(latency_ms: float, status: str = "ok") -> None:
    REQUEST_LATENCY.observe(latency_ms / 1000.0)
    REQUEST_COUNT.labels(status=status).inc()
    # Update precise quantile (sliding window)
    _latency_window.append(float(latency_ms))
    ordered = sorted(_latency_window)
    REQUEST_P50_MS.set(round(_percentile(ordered, 50), 1))
    REQUEST_P95_MS.set(round(_percentile(ordered, 95), 1))
    REQUEST_P99_MS.set(round(_percentile(ordered, 99), 1))


def RECORD_BUSINESS(latency_ms: float) -> None:
    """Record business latency (end-to-end excluding evaluate_answer online evaluation)."""
    BUSINESS_LATENCY.observe(latency_ms / 1000.0)
    _business_window.append(float(latency_ms))
    ordered = sorted(_business_window)
    BUSINESS_P50_MS.set(round(_percentile(ordered, 50), 1))
    BUSINESS_P95_MS.set(round(_percentile(ordered, 95), 1))
    BUSINESS_P99_MS.set(round(_percentile(ordered, 99), 1))


def RECORD_NODE(node: str, latency_ms: float) -> None:
    """Record latency for a single LangGraph node."""
    NODE_LATENCY.labels(node=node).observe(latency_ms / 1000.0)


def RECORD_TOKEN_USAGE(input_tokens: int, output_tokens: int) -> None:
    TOKEN_USAGE.labels(direction="input").inc(input_tokens)
    TOKEN_USAGE.labels(direction="output").inc(output_tokens)
    # Rough cost estimate (gpt-4o input $2.5/M, output $10/M)
    cost = (input_tokens * 2.5 + output_tokens * 10.0) / 1_000_000
    TOKEN_COST_USD.set(cost)


# Cumulative accumulators for retrieval effective ratio (updated only on actual RAG calls, cache hits excluded)
_retrieval_effective_sum = 0.0
_retrieval_hits_sum = 0.0


def RECORD_RETRIEVAL(hits: int, recalled: int, scores: list[float] | None = None,
                     threshold: float = 0.5) -> None:
    """Record retrieval metrics.

    Args:
        hits: Number of documents actually returned by Azure (after filter + similarity threshold)
        recalled: Requested top_k
        scores: List of @search.score for documents returned by Azure, used to compute effective quality metrics
        threshold: Similarity threshold for determining "effective documents" (default 0.5)
    """
    RETRIEVAL_HITS.set(hits)

    if scores:
        effective = sum(1 for s in scores if s > threshold)
        # Prometheus Counters (for external monitoring)
        _RETRIEVAL_EFFECTIVE_CUM.inc(effective)
        _RETRIEVAL_HITS_CUM.inc(hits)
        # Local cumulative sums → update aggregated ratio Gauge
        global _retrieval_effective_sum, _retrieval_hits_sum
        _retrieval_effective_sum += effective
        _retrieval_hits_sum += hits
        if _retrieval_hits_sum > 0:
            RETRIEVAL_EFFECTIVE_RATIO.set(round(_retrieval_effective_sum / _retrieval_hits_sum, 4))


def RECORD_RERANK(latency_ms: float) -> None:
    RERANK_LATENCY.observe(latency_ms / 1000.0)


def RECORD_LLM(latency_ms: float, first_token_ms: float = 0.0) -> None:
    LLM_TOTAL_LATENCY.observe(latency_ms / 1000.0)
    if first_token_ms:
        LLM_FIRST_TOKEN_LATENCY.observe(first_token_ms / 1000.0)


def RECORD_FEEDBACK(rating: int) -> None:
    REQUEST_COUNT.labels(status="feedback").inc()


def RECORD_FAITHFULNESS(faithfulness: float, relevancy: float = 0.0, hallucination: float = 0.0) -> None:
    """Record online evaluation scores (called by the evaluate node)."""
    FAITHFULNESS_SCORE.set(round(faithfulness, 2))
    ANSWER_RELEVANCY_SCORE.set(round(relevancy, 2))
    HALLUCINATION_SCORE.set(round(hallucination, 3))


def RECORD_CACHE(hit: bool, size: int = -1) -> None:
    """Record a cache hit or miss."""
    if hit:
        CACHE_HITS.inc()
    else:
        CACHE_MISSES.inc()
    if size >= 0:
        CACHE_SIZE.set(size)


def export() -> bytes:
    """Export for the /metrics endpoint."""
    return generate_latest(REGISTRY)
