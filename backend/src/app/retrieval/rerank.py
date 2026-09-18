"""重排模块：支持三种策略。

- none:            不重排，直接取前 final_top_k（仍记指标，latency=0）
- cross-encoder:    本地 Cross-Encoder（BAAI/bge-reranker-base）
- llm:              用 LLM 对 Top-20 逐条打分
- semantic:         语义已在检索层做，这里只截断（等同 noop）

所有路径都会调 RECORD_RERANK，确保「重排平均延迟」指标不为 0。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from app.config import Settings
from app.observability.metrics import RECORD_RERANK

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """重排协议：返回**按重排分数降序排好**的完整 docs 列表（不截断）。"""

    async def rerank(
        self, query: str, docs: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """返回已排好序的 docs 全量，由调用方按 top_k 截断。"""


class NoopReranker:
    """不做真正重排，按现有 score 排序（返回全量，不截断）。"""

    async def rerank(
        self, query: str, docs: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        RECORD_RERANK(0.0)
        return sorted(docs, key=lambda d: -d.get("score", 0.0))


class CrossEncoderReranker:
    """本地 Cross-Encoder 重排。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self._model_name = model_name
        self._model = None   # None=未加载, False=加载失败, 对象=成功
        self._logged_fail = False

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            logger.info("正在加载 CrossEncoder 重排模型 %s ...", self._model_name)
            self._model = CrossEncoder(self._model_name)
            logger.info("CrossEncoder 加载完成。")
        except Exception as exc:  # noqa: BLE001
            if not self._logged_fail:
                logger.warning("CrossEncoder 加载失败 (%s)，退化为无重排。首次加载需下载 ~400MB 模型。", exc)
                self._logged_fail = True
            self._model = False

    async def rerank(
        self, query: str, docs: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        self._load()
        if self._model is False or not docs:
            RECORD_RERANK(0.0)
            return sorted(docs, key=lambda d: -d.get("score", 0.0))
        start = time.perf_counter()
        pairs = [(query, d.get("content", "")) for d in docs]
        scores = self._model.predict(pairs) if pairs else []
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        ordered = sorted(docs, key=lambda d: -d.get("rerank_score", 0.0))
        RECORD_RERANK((time.perf_counter() - start) * 1000)
        return ordered


class LLMReranker:
    """LLM 批量重排。让 LLM 直接输出 JSON 数组，更鲁棒。"""

    def __init__(self, llm_client) -> None:
        self._llm = llm_client

    async def rerank(
        self, query: str, docs: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        if not docs:
            RECORD_RERANK(0.0)
            return []
        start = time.perf_counter()
        joined = "\n".join(
            f"[{i}] {d.get('content', '')[:200]}" for i, d in enumerate(docs)
        )
        # 要 LLM 输出严格 JSON，避免 idx:score 解析脆弱
        prompt = (
            "You are a relevance evaluator. Rate each doc snippet 0-10 against the query.\n"
            "Return ONLY a JSON array like [8,3,6,0,9] — no other text.\n"
            f"Query: {query}\nDocs:\n{joined}"
        )
        try:
            resp = await self._llm.chat([{"role": "user", "content": prompt}])
            text = (resp.get("answer") or "").strip()
            scores = self._parse_scores(text, len(docs))
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLMReranker 调用/解析失败: %s，退化为原序", exc)
            scores = [float(d.get("score", 0.0)) for d in docs]

        for i, d in enumerate(docs):
            d["rerank_score"] = scores[i] if i < len(scores) else 0.0

        # 记录日志帮助排查
        logger.info(
            "LLMRerank 结果: %s",
            [(i, scores[i], docs[i].get("source", "")) for i in range(min(len(docs), 5))],
        )

        ordered = sorted(docs, key=lambda d: -d.get("rerank_score", 0.0))
        RECORD_RERANK((time.perf_counter() - start) * 1000)
        return ordered

    @staticmethod
    def _parse_scores(text: str, expected: int) -> list[float]:
        """尽力从 LLM 输出解析出 float 列表。支持 JSON / 逗号分隔 / 纯数字。"""
        import re

        # 尝试直接 json.loads
        try:
            import json
            obj = json.loads(text)
            if isinstance(obj, list):
                return [float(x) for x in obj[:expected]]
        except Exception:
            pass

        # 兜底：正则捞所有数字
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        out = []
        for n in nums[:expected]:
            try:
                out.append(float(n))
            except ValueError:
                continue
        # 长度不够补 0
        while len(out) < expected:
            out.append(0.0)
        return out


def build_reranker(settings: Settings, llm_client=None) -> Reranker:
    strategy = settings.rerank_strategy
    if strategy == "none":
        logger.info("Reranker: noop (strategy=none)")
        return NoopReranker()
    if strategy == "cross-encoder":
        logger.info("Reranker: cross-encoder (strategy=cross-encoder)")
        return CrossEncoderReranker()
    if strategy == "llm":
        logger.info("Reranker: llm (strategy=llm)")
        if llm_client is None:
            from app.llm.azure_openai import build_llm_client
            llm_client = build_llm_client(settings)
        return LLMReranker(llm_client)
    if strategy == "semantic":
        logger.info("Reranker: noop (strategy=semantic, 语义已在检索层做)")
        return NoopReranker()
    logger.warning("Reranker: 未知策略 '%s'，退回 noop", strategy)
    return NoopReranker()
