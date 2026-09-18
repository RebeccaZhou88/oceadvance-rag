"""LangGraph 节点实现。

每个节点是一个可调用类（__call__ 为 async），配置走 __init__，
LangGraph 调节点时走可调用对象协议，签名统一为
    async def __call__(self, state: GraphState) -> dict

节点：
1. IntentCheckNode    - 意图识别（query/followup/chitchat）
2. RetrieveDocsNode   - 混合检索 + 权限过滤
3. RerankSequenceNode - 重排 + 上下文组装（记录 before/after 对比）
4. GenerateAnswerNode - 生成带引用回答
5. EvaluateAnswerNode - 在线轻量评估（Faithfulness / Relevance / Hallucination）

每个节点都会往 state.trace 追加一条结构化步骤日志，供前端展示编排过程。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from app.observability.metrics import (
    RECORD_LLM,
    RECORD_RETRIEVAL,
    RECORD_FAITHFULNESS,
)
from app.retrieval.rerank import Reranker
from app.retrieval.search import Retriever
from app.security.permissions import assert_no_leak, build_search_filter
from app.llm.azure_openai import BaseLLMClient
from app.graph.state import GraphState
from app.prompts import (
    CHITCHAT_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    EVALUATE_SYSTEM_PROMPT,
    build_context,
)


def _trace(step: str, status: str, duration_ms: float, **detail) -> dict[str, Any]:
    """构造一条 trace 记录。"""
    return {
        "step": step,
        "status": status,
        "duration_ms": round(duration_ms, 3),
        "ts": time.time(),
        "detail": detail,
    }


class IntentCheckNode:
    """意图识别：query / followup / chitchat。无外部依赖，纯启发式。"""

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        start = time.perf_counter()
        question = state.get("question", "")
        history = state.get("messages", [])
        followup_cues = ("它", "这个", "上面", "那", "另外", "接着", "继续")
        is_followup = any(c in question for c in followup_cues) and bool(history)
        chitchat_cues = ("你好", "谢谢", "再见", "你是谁")
        is_chitchat = any(c in question for c in chitchat_cues)
        if is_chitchat:
            intent = "chitchat"
        elif is_followup:
            intent = "followup"
        else:
            intent = "query"
        duration_ms = (time.perf_counter() - start) * 1000
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
    """混合检索 + 权限过滤。"""

    def __init__(self, retriever: Retriever, top_k: int = 20):
        self.retriever = retriever
        self.top_k = top_k

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        start = time.perf_counter()
        question = state["question"]
        user_group = state.get("user_group", "ops")
        search_filter = build_search_filter(user_group)
        try:
            docs = await self.retriever.hybrid_search(question, user_group, top_k=self.top_k)
            assert_no_leak(docs, user_group)
            status = "ok"
        except PermissionError as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            return {"retrieved_docs": [], "trace": [_trace(
                "retrieve_docs", "error", duration_ms,
                error=f"权限过滤失效: {exc}",
            )]}
        recalled = self.top_k
        hits = len(docs)
        RECORD_RETRIEVAL(hits=hits, recalled=recalled)
        duration_ms = (time.perf_counter() - start) * 1000
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
                before=before,
            )],
        }


class RerankSequenceNode:
    """重排 + 上下文组装（记录 before/after 对比）。"""

    def __init__(self, reranker: Reranker, final_top_k: int = 5, strategy: str = "none"):
        self.reranker = reranker
        self.final_top_k = final_top_k
        self.strategy = strategy

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        start = time.perf_counter()
        docs = state.get("retrieved_docs", [])
        question = state["question"]

        if not docs:
            duration_ms = (time.perf_counter() - start) * 1000
            return {
                "citations": [],
                "trace": [_trace(
                    "rerank_sequence", "skipped", duration_ms,
                    reason="无召回文档，跳过重排",
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
    """生成带引用回答。"""

    def __init__(self, llm: BaseLLMClient):
        self.llm = llm

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        start = time.perf_counter()
        docs = state.get("retrieved_docs", [])
        question = state["question"]
        history = state.get("messages", [])
        citations = state.get("citations", [])
        intent = state.get("intent", "query")

        # ====== chitchat: 无上下文也让 LLM 直接回答 ======
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

        # ====== query/followup 但检索为空 ======
        if not docs:
            duration_ms = (time.perf_counter() - start) * 1000
            return {
                "answer": "未找到相关知识，无法回答该问题。",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0.0,
                "messages": [{"role": "assistant", "content": "未找到相关知识，无法回答该问题。"}],
                "trace": [_trace(
                    "generate_answer", "skipped", duration_ms,
                    reason="无上下文，跳过 LLM 调用（节省 Token）",
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
            "content": f"检索到的上下文：\n{context}",
        })
        messages.append({"role": "user", "content": question})

        resp = await self.llm.chat(messages, citations_hint=citations)
        latency_ms = (time.perf_counter() - start) * 1000
        RECORD_LLM(latency_ms=latency_ms)
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
            )],
        }


class EvaluateAnswerNode:
    """在线轻量评估：LLM 自评 Faithfulness / Relevance / Hallucination。

    三种场景：
    1. chitchat → 跳过（无上下文，自评无意义）
    2. 检索为空 → 跳过（无上下文可评估忠实度）
    3. 正常 RAG → 调用 LLM 自评并记录分数
    """

    def __init__(self, llm: BaseLLMClient):
        self.llm = llm

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        start = time.perf_counter()
        intent = state.get("intent", "query")
        docs = state.get("retrieved_docs", [])
        answer = state.get("answer", "")
        question = state.get("question", "")

        if intent == "chitchat":
            duration_ms = (time.perf_counter() - start) * 1000
            return {"trace": [_trace(
                "evaluate_answer", "skipped", duration_ms,
                reason="闲聊意图，无需评估忠实度",
                intent=intent,
            )]}

        if not docs or not answer or answer.startswith("未找到相关知识"):
            duration_ms = (time.perf_counter() - start) * 1000
            return {"trace": [_trace(
                "evaluate_answer", "skipped", duration_ms,
                reason="无上下文或无答案，无法评估",
                intent=intent,
            )]}

        try:
            context = build_context(docs)
            eval_messages = [
                {"role": "system", "content": EVALUATE_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"用户问题：{question}\n\n"
                    f"检索到的上下文：\n{context}\n\n"
                    f"助手的回答：\n{answer}"
                )},
            ]
            resp = await self.llm.chat(eval_messages, citations_hint=[])
            duration_ms = (time.perf_counter() - start) * 1000

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
            return {"trace": [_trace(
                "evaluate_answer", "error", duration_ms,
                error=f"评估 LLM 调用失败: {exc}",
            )]}
