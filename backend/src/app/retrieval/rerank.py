# @Author: RebeccaZhou
# @Description: Pluggable rerankers: Cross-Encoder ONNX / LLM / Semantic / none
#              可插拔重排器：Cross-Encoder ONNX / LLM / Semantic / none

"""Reranking module: supports three strategies.

- none:            No reranking, directly take top final_top_k (still records metrics, latency=0)
- cross-encoder:   Local Cross-Encoder (BAAI/bge-reranker-base)
- llm:             Use LLM to score Top-20 one by one
- semantic:        Semantic search already done at retrieval layer, only truncate here (same as noop)

All paths call RECORD_RERANK to ensure the "rerank average latency" metric is not 0.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from app.config import Settings
from app.observability.metrics import RECORD_RERANK

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """Reranker protocol: returns the full docs list **sorted descending by rerank score** (no truncation)."""

    async def rerank(
        self, query: str, docs: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Return the fully sorted docs list; caller truncates by top_k."""


class NoopReranker:
    """No actual reranking, sort by existing score (return full list, no truncation)."""

    async def rerank(
        self, query: str, docs: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        RECORD_RERANK(0.0)
        return sorted(docs, key=lambda d: -d.get("score", 0.0))


class CrossEncoderReranker:
    """Local Cross-Encoder reranking."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self._model_name = model_name
        self._model = None   # None=not loaded, False=load failed, object=success
        self._logged_fail = False

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading CrossEncoder rerank model %s ...", self._model_name)
            self._model = CrossEncoder(self._model_name)
            logger.info("CrossEncoder loaded.")
        except Exception as exc:  # noqa: BLE001
            if not self._logged_fail:
                logger.warning("CrossEncoder load failed (%s), falling back to no reranking. First load needs to download ~400MB model.", exc)
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
    """LLM batch reranking. Let LLM directly output a JSON array, more robust."""

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
        # Ask LLM to output strict JSON to avoid fragile idx:score parsing
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
            logger.warning("LLMReranker call/parse failed: %s, falling back to original order", exc)
            scores = [float(d.get("score", 0.0)) for d in docs]

        for i, d in enumerate(docs):
            d["rerank_score"] = scores[i] if i < len(scores) else 0.0

        # Log to help troubleshooting
        logger.info(
            "LLMRerank result: %s",
            [(i, scores[i], docs[i].get("source", "")) for i in range(min(len(docs), 5))],
        )

        ordered = sorted(docs, key=lambda d: -d.get("rerank_score", 0.0))
        RECORD_RERANK((time.perf_counter() - start) * 1000)
        return ordered

    @staticmethod
    def _parse_scores(text: str, expected: int) -> list[float]:
        """Best-effort parse a float list from LLM output. Supports JSON / comma-separated / plain numbers."""
        import re

        # Try direct json.loads
        try:
            import json
            obj = json.loads(text)
            if isinstance(obj, list):
                return [float(x) for x in obj[:expected]]
        except Exception:
            pass

        # Fallback: regex extract all numbers
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        out = []
        for n in nums[:expected]:
            try:
                out.append(float(n))
            except ValueError:
                continue
        # Pad with 0 if length insufficient
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
            from app.llm.azure_openai import build_chat_client
            llm_client = build_chat_client(settings)
        return LLMReranker(llm_client)
    if strategy == "semantic":
        logger.info("Reranker: noop (strategy=semantic, semantic search done at retrieval layer)")
        return NoopReranker()
    logger.warning("Reranker: unknown strategy '%s', falling back to noop", strategy)
    return NoopReranker()
