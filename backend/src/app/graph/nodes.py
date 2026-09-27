# @Author: RebeccaZhou
# @Description: LangGraph nodes: cache lookup, intent, retrieve, rerank, answer, evaluate
#              LangGraph 节点：缓存查找、意图识别、检索、重排、回答、评估

"""LangGraph node implementations.

Each node is a callable class (__call__ is async), configured via __init__.
LangGraph invokes nodes via the callable protocol with a unified signature:
    async def __call__(self, state: GraphState) -> dict

Nodes:
1. IntentCheckNode    - intent classification (query/followup/chitchat)
2. RetrieveDocsNode   - hybrid retrieval + permission filtering
3. RerankSequenceNode - rerank + context assembly (records before/after comparison)
4. GenerateAnswerNode - generate cited answer
5. EvaluateAnswerNode - online lightweight evaluation (Faithfulness / Relevance / Hallucination)

Each node appends a structured step log to state.trace for the frontend to display the orchestration process,
and also logs concise key info to the terminal via logger.info for real-time observability.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from app.observability.metrics import (
    RECORD_LLM,
    RECORD_RETRIEVAL,
    RECORD_FAITHFULNESS,
    RECORD_NODE,
    RECORD_CACHE,
)
from app.retrieval.rerank import Reranker
from app.retrieval.search import Retriever
from app.security.permissions import assert_no_leak, build_search_filter
from app.llm.azure_openai import BaseLLMClient
from app.cache import ExactCache
from app.graph.state import GraphState
from app.prompts import (
    CHITCHAT_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    EVALUATE_SYSTEM_PROMPT,
    build_context,
)

logger = logging.getLogger(__name__)


def _trace(step: str, status: str, duration_ms: float, **detail) -> dict[str, Any]:
    """Build a trace record."""
    return {
        "step": step,
        "status": status,
        "duration_ms": round(duration_ms, 3),
        "ts": time.time(),
        "detail": detail,
    }


class CacheLookupNode:
    """Exact cache lookup node — runs before everything else.

    On a hit, returns the cached answer + citations and sets from_cache=True
    so the workflow can route directly to END, skipping the entire RAG pipeline.
    On a miss, returns from_cache=False and the workflow continues to intent_check.
    """

    def __init__(self, cache: ExactCache | None = None) -> None:
        self.cache = cache

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        start = time.perf_counter()
        question = state.get("question", "")
        user_group = state.get("user_group", "ops")

        # Cache disabled or not configured → miss
        if self.cache is None:
            duration_ms = (time.perf_counter() - start) * 1000
            RECORD_NODE("cache_lookup", duration_ms)
            RECORD_CACHE(hit=False)
            logger.info("s [cache_lookup] disabled | %.2fms", duration_ms)
            return {
                "from_cache": False,
                "trace": [_trace(
                    "cache_lookup", "skipped", duration_ms,
                    reason="cache disabled",
                )],
            }

        logger.info(">> [cache_lookup] question=%.60r | group=%s", question, user_group)
        cached = self.cache.get(question, user_group)
        duration_ms = (time.perf_counter() - start) * 1000
        RECORD_NODE("cache_lookup", duration_ms)

        if cached is not None:
            RECORD_CACHE(hit=True, size=len(self.cache._store))
            logger.info(
                "v [cache_lookup] HIT | %.2fms | answer_len=%d | citations=%d",
                duration_ms, len(cached.get("answer", "")), len(cached.get("citations", [])),
            )
            return {
                "answer": cached["answer"],
                "citations": cached.get("citations", []),
                "from_cache": True,
                "intent": "query",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": duration_ms,
                "trace": [_trace(
                    "cache_lookup", "hit", duration_ms,
                    question=question[:60],
                    user_group=user_group,
                    citations=len(cached.get("citations", [])),
                    kb_version=cached.get("kb_version", ""),
                )],
            }

        RECORD_CACHE(hit=False, size=len(self.cache._store))
        logger.info("x [cache_lookup] MISS | %.2fms", duration_ms)
        return {
            "from_cache": False,
            "trace": [_trace(
                "cache_lookup", "miss", duration_ms,
                question=question[:60],
                user_group=user_group,
            )],
        }


class IntentCheckNode:
    """Intent classification: query / followup / chitchat. No external dependencies, pure heuristic."""

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        start = time.perf_counter()
        question = state.get("question", "")
        history = state.get("messages", [])
        logger.info(">> [intent_check] question=%.60r | history_turns=%d", question, len(history))
        # Use word-boundary matching so cues do not fire inside other words
        # (e.g. "hi" inside "high"/"this", "it" inside "with"/"site")
        followup_cues = ("it", "this", "that", "above", "also", "then", "continue", "furthermore", "another")
        is_followup = (
            any(re.search(rf"\b{re.escape(c)}\b", question.lower()) for c in followup_cues)
            and bool(history)
        )
        chitchat_cues = ("hello", "hi", "thanks", "thank you", "bye", "goodbye", "who are you")
        is_chitchat = any(re.search(rf"\b{re.escape(c)}\b", question.lower()) for c in chitchat_cues)
        if is_chitchat:
            intent = "chitchat"
        elif is_followup:
            intent = "followup"
        else:
            intent = "query"
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("v [intent_check] intent=%s | duration %.2fms", intent, duration_ms)
        RECORD_NODE("intent_check", duration_ms)
        return {
            "intent": intent,
            "trace": [_trace(
                "intent_check", "ok", duration_ms,
                intent=intent,
                question=question[:60],
                history_turns=len(history),
            )],
        }


class RetrieveDocsNode:
    """Hybrid retrieval + permission filtering."""

    def __init__(self, retriever: Retriever, top_k: int = 20):
        self.retriever = retriever
        self.top_k = top_k

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        start = time.perf_counter()
        question = state["question"]
        user_group = state.get("user_group", "ops")
        search_filter = build_search_filter(user_group)
        logger.info(
            ">> [retrieve_docs] query=%.60r | user_group=%s | filter=%s | top_k=%d",
            question, user_group, search_filter, self.top_k,
        )
        try:
            docs = await self.retriever.hybrid_search(question, user_group, top_k=self.top_k)
            assert_no_leak(docs, user_group)
            status = "ok"
        except PermissionError as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error("x [retrieve_docs] permission filter failed: %s | duration %.2fms", exc, duration_ms)
            RECORD_NODE("retrieve_docs", duration_ms)
            return {"retrieved_docs": [], "trace": [_trace(
                "retrieve_docs", "error", duration_ms,
                error=f"permission filter failed: {exc}",
            )]}
        recalled = self.top_k
        hits = len(docs)
        from app.config import get_settings
        threshold = get_settings().effective_threshold
        scores = [float(d.get("score", 0.0)) for d in docs]
        top1_score = max(scores) if scores else 0.0
        effective_count = sum(1 for s in scores if s > threshold)
        RECORD_RETRIEVAL(hits=hits, recalled=recalled, scores=scores, threshold=threshold)
        duration_ms = (time.perf_counter() - start) * 1000
        RECORD_NODE("retrieve_docs", duration_ms)
        # Retrieval summary: top 3 category + source + score
        top_hits = [
            f"#{i+1} {d.get('category','?')}/{os.path.basename(d.get('source',''))} "
            f"(score={float(d.get('score',0.0)):.4f})"
            for i, d in enumerate(docs[:3])
        ]
        logger.info(
            "v [retrieve_docs] retrieved %d docs | duration %.2fms | Top3: %s",
            hits, duration_ms, "; ".join(top_hits) if top_hits else "(empty)",
        )
        before = [
            {
                "doc_id": d.get("id", ""),
                "category": d.get("category", ""),
                "source": d.get("source", ""),
                "score": round(float(d.get("score", 0.0)), 6),
            }
            for d in docs
        ]
        return {
            "retrieved_docs": docs,
            "trace": [_trace(
                "retrieve_docs", status, duration_ms,
                question=question[:60],
                user_group=user_group,
                search_filter=search_filter,
                recalled=recalled,
                hits=hits,
                top1_score=round(top1_score, 4),
                effective_threshold=threshold,
                effective_count=effective_count,
                effective_ratio=round(effective_count / max(hits, 1), 4),
                before=before,
            )],
        }


class RerankSequenceNode:
    """Rerank + context assembly (records before/after comparison)."""

    def __init__(self, reranker: Reranker, final_top_k: int = 5, strategy: str = "none"):
        self.reranker = reranker
        self.final_top_k = final_top_k
        self.strategy = strategy

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        start = time.perf_counter()
        docs = state.get("retrieved_docs", [])
        question = state["question"]
        logger.info(
            ">> [rerank_sequence] strategy=%s | final_top_k=%d | input %d docs",
            self.strategy, self.final_top_k, len(docs),
        )

        if not docs:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info("s [rerank_sequence] skipped (no retrieved docs) | duration %.2fms", duration_ms)
            RECORD_NODE("rerank_sequence", duration_ms)
            return {
                "citations": [],
                "trace": [_trace(
                    "rerank_sequence", "skipped", duration_ms,
                    reason="no retrieved docs, rerank skipped",
                    strategy=self.strategy,
                )],
            }

        before_snapshot = [
            {
                "doc_id": d.get("id", ""),
                "category": d.get("category", ""),
                "score": round(float(d.get("score", 0.0)), 6),
            }
            for d in sorted(docs, key=lambda x: -float(x.get("score", 0.0)))
        ]

        all_ranked = await self.reranker.rerank(question, docs, top_k=self.final_top_k)
        duration_ms = (time.perf_counter() - start) * 1000
        RECORD_NODE("rerank_sequence", duration_ms)

        after_snapshot = [
            {
                "doc_id": d.get("id", ""),
                "category": d.get("category", ""),
                "score": round(float(d.get("rerank_score", d.get("score", 0.0))), 6),
                "original_score": round(float(d.get("score", 0.0)), 6),
            }
            for d in all_ranked
        ]

        ranked = all_ranked[:self.final_top_k]
        # Rerank result summary
        rank_summary = []
        for i, d in enumerate(ranked):
            old_rank = next(
                (j + 1 for j, b in enumerate(before_snapshot) if b["doc_id"] == d.get("id")),
                "-",
            )
            rank_summary.append(
                f"#{i+1}(was#{old_rank}) {d.get('category','?')} "
                f"score={float(d.get('rerank_score',0.0)):.3f}"
            )
        eliminated_ids = [d["doc_id"] for d in after_snapshot[self.final_top_k:]]
        logger.info(
            "v [rerank_sequence] output %d docs | duration %.2fms | Top: %s | eliminated %d",
            len(ranked), duration_ms, "; ".join(rank_summary), len(eliminated_ids),
        )

        before_ids = [d["doc_id"] for d in before_snapshot]
        after_ids = [d["doc_id"] for d in after_snapshot]
        rank_changes = []
        for new_rank, did in enumerate(after_ids, start=1):
            if did in before_ids:
                old_rank = before_ids.index(did) + 1
                change = old_rank - new_rank
                rank_changes.append({
                    "doc_id": did,
                    "old_rank": old_rank,
                    "new_rank": new_rank,
                    "change": change,
                })
            else:
                rank_changes.append({
                    "doc_id": did,
                    "old_rank": None,
                    "new_rank": new_rank,
                    "change": "new",
                })
        eliminated = [d["doc_id"] for d in after_snapshot[self.final_top_k:]]

        citations = [
            {
                "doc_id": d.get("id", ""),
                "source": d.get("source", ""),
                "category": d.get("category", ""),
                "snippet": d.get("content", "")[:300],
                "score": float(d.get("rerank_score", d.get("score", 0.0))),
            }
            for d in ranked
        ]
        return {
            "retrieved_docs": ranked,
            "citations": citations,
            "trace": [_trace(
                "rerank_sequence", "ok", duration_ms,
                strategy=self.strategy,
                final_top_k=self.final_top_k,
                input_count=len(docs),
                output_count=len(ranked),
                before=before_snapshot,
                after=after_snapshot,
                rank_changes=rank_changes,
                eliminated=eliminated,
            )],
        }


class GenerateAnswerNode:
    """Generate a cited answer."""

    def __init__(self, llm: BaseLLMClient, cache: ExactCache | None = None):
        self.llm = llm
        self.cache = cache

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        start = time.perf_counter()
        docs = state.get("retrieved_docs", [])
        question = state["question"]
        history = state.get("messages", [])
        citations = state.get("citations", [])
        intent = state.get("intent", "query")
        logger.info(
            ">> [generate_answer] intent=%s | context %d chunks | citations %d | history %d turns",
            intent, len(docs), len(citations), len(history),
        )

        # ====== chitchat: let the LLM answer directly even without context ======
        if intent == "chitchat":
            messages = [{"role": "system", "content": CHITCHAT_SYSTEM_PROMPT}]
            for m in history[-4:]:
                if isinstance(m, dict):
                    role = m.get("role", "user")
                    content = m.get("content", "")
                else:
                    role = getattr(m, "type", getattr(m, "role", "user"))
                    content = getattr(m, "content", "")
                if role in ("user", "assistant"):
                    messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": question})
            resp = await self.llm.chat(messages, citations_hint=[])
            latency_ms = (time.perf_counter() - start) * 1000
            RECORD_LLM(latency_ms=latency_ms)
            RECORD_NODE("generate_answer", latency_ms)
            logger.info(
                "v [generate_answer] chitchat | tokens=%d/%d | LLM latency %.0fms | total %.2fms",
                resp["input_tokens"], resp["output_tokens"],
                resp.get("latency_ms", 0), latency_ms,
            )
            return {
                "answer": resp["answer"],
                "input_tokens": resp["input_tokens"],
                "output_tokens": resp["output_tokens"],
                "latency_ms": latency_ms,
                "messages": [{"role": "assistant", "content": resp["answer"]}],
                "trace": [_trace(
                    "generate_answer", "ok", latency_ms,
                    intent=intent,
                    context_chunks=0,
                    citations=0,
                    input_tokens=resp["input_tokens"],
                    output_tokens=resp["output_tokens"],
                    llm_latency_ms=round(resp.get("latency_ms", 0.0), 3),
                )],
            }

        # ====== query/followup but retrieval is empty ======
        if not docs:
            duration_ms = (time.perf_counter() - start) * 1000
            RECORD_NODE("generate_answer", duration_ms)
            logger.info(
                "s [generate_answer] LLM skipped (no context, saves Token) | duration %.2fms",
                duration_ms,
            )
            return {
                "answer": "No relevant knowledge found, unable to answer this question.",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0.0,
                "messages": [{"role": "assistant", "content": "No relevant knowledge found, unable to answer this question."}],
                "trace": [_trace(
                    "generate_answer", "skipped", duration_ms,
                    reason="no context, LLM call skipped (saves Token)",
                    intent=intent,
                )],
            }

        context = build_context(docs)
        messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
        for m in history[-4:]:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
            else:
                role = getattr(m, "type", getattr(m, "role", "user"))
                content = getattr(m, "content", "")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})
        messages.append({
            "role": "system",
            "content": f"Retrieved context:\n{context}",
        })
        messages.append({"role": "user", "content": question})

        resp = await self.llm.chat(messages, citations_hint=citations)
        latency_ms = (time.perf_counter() - start) * 1000
        RECORD_LLM(latency_ms=latency_ms)
        RECORD_NODE("generate_answer", latency_ms)
        answer_preview = (resp["answer"] or "").replace("\n", " ")[:80]
        logger.info(
            "v [generate_answer] RAG | tokens=%d/%d | LLM latency %.0fms | total %.2fms | answer: %.80s",
            resp["input_tokens"], resp["output_tokens"],
            resp.get("latency_ms", 0), latency_ms, answer_preview,
        )

        # Write to exact cache (only for successful query-intent RAG answers with docs)
        if self.cache is not None and intent == "query" and resp.get("answer"):
            self.cache.set(
                question, state.get("user_group", "ops"),
                resp["answer"], citations, content_type="general",
            )

        return {
            "answer": resp["answer"],
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp["output_tokens"],
            "latency_ms": latency_ms,
            "messages": [{"role": "assistant", "content": resp["answer"]}],
            "trace": [_trace(
                "generate_answer", "ok", latency_ms,
                intent=intent,
                context_chunks=len(docs),
                citations=len(citations),
                input_tokens=resp["input_tokens"],
                output_tokens=resp["output_tokens"],
                llm_latency_ms=round(resp.get("latency_ms", 0.0), 3),
                cached=bool(self.cache is not None and intent == "query"),
            )],
        }


class EvaluateAnswerNode:
    """Online lightweight evaluation: LLM self-eval Faithfulness / Relevance / Hallucination.

    Three scenarios:
    1. chitchat -> skip (no context, self-eval is meaningless)
    2. empty retrieval -> skip (no context to evaluate faithfulness against)
    3. normal RAG -> call LLM self-eval and record scores
    """

    def __init__(self, llm: BaseLLMClient):
        self.llm = llm

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        from app.config import get_settings
        start = time.perf_counter()
        intent = state.get("intent", "query")
        docs = state.get("retrieved_docs", [])
        answer = state.get("answer", "")
        question = state.get("question", "")
        logger.info(">> [evaluate_answer] intent=%s | context %d | has_answer=%s",
                    intent, len(docs), bool(answer))

        if intent == "chitchat":
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info("s [evaluate_answer] skipped (chitchat) | %.2fms", duration_ms)
            RECORD_NODE("evaluate_answer", duration_ms)
            return {"trace": [_trace(
                "evaluate_answer", "skipped", duration_ms,
                reason="chitchat intent, no faithfulness evaluation needed",
                intent=intent,
            )]}

        if not get_settings().enable_online_eval:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info("s [evaluate_answer] skipped (ENABLE_ONLINE_EVAL=false) | %.2fms", duration_ms)
            RECORD_NODE("evaluate_answer", duration_ms)
            return {"trace": [_trace(
                "evaluate_answer", "skipped", duration_ms,
                reason="ENABLE_ONLINE_EVAL=false, off by default in production",
                intent=intent,
            )]}

        if not docs or not answer or answer.startswith("No relevant knowledge"):
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info("s [evaluate_answer] skipped (no context or no answer) | %.2fms", duration_ms)
            RECORD_NODE("evaluate_answer", duration_ms)
            return {"trace": [_trace(
                "evaluate_answer", "skipped", duration_ms,
                reason="no context or no answer, cannot evaluate",
                intent=intent,
            )]}

        try:
            context = build_context(docs)
            eval_messages = [
                {"role": "system", "content": EVALUATE_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"User question: {question}\n\n"
                    f"Retrieved context:\n{context}\n\n"
                    f"Assistant's answer:\n{answer}"
                )},
            ]
            resp = await self.llm.chat(eval_messages, citations_hint=[])
            duration_ms = (time.perf_counter() - start) * 1000
            RECORD_NODE("evaluate_answer", duration_ms)

            raw = resp.get("answer", "")
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {}
                for key in ("faithfulness", "answer_relevancy"):
                    m = re.search(rf'"{key}"\s*[:=]\s*(\d+(?:\.\d+)?)', raw)
                    if m:
                        data[key] = float(m.group(1))
                m = re.search(r'"hallucination_score"\s*[:=]\s*(\d+(?:\.\d+)?)', raw)
                if m:
                    data["hallucination_score"] = float(m.group(1))
                m = re.search(r'"notes"\s*[:=]\s*"([^"]*)"', raw)
                if m:
                    data["notes"] = m.group(1)

            faithfulness = float(data.get("faithfulness", 0.0))
            relevancy = float(data.get("answer_relevancy", 0.0))
            hallucination = float(data.get("hallucination_score", 0.0))
            notes = data.get("notes", "")

            RECORD_FAITHFULNESS(
                faithfulness=faithfulness,
                relevancy=relevancy,
                hallucination=hallucination,
            )
            logger.info(
                "v [evaluate_answer] faithfulness=%.2f | relevancy=%.2f | halluc=%.2f | duration %.2fms",
                faithfulness, relevancy, hallucination, duration_ms,
            )
            return {"trace": [_trace(
                "evaluate_answer", "ok", duration_ms,
                faithfulness=faithfulness,
                answer_relevancy=relevancy,
                hallucination_score=hallucination,
                notes=notes,
                llm_eval_latency_ms=round(resp.get("latency_ms", 0.0), 3),
            )]}

        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error("x [evaluate_answer] evaluation LLM call failed: %s | duration %.2fms", exc, duration_ms)
            RECORD_NODE("evaluate_answer", duration_ms)
            return {"trace": [_trace(
                "evaluate_answer", "error", duration_ms,
                error=f"evaluation LLM call failed: {exc}",
            )]}
