# @Author: RebeccaZhou
# @Description: LangGraph workflow: entry node, conditional edges, fallback assembly
#              LangGraph 工作流：入口节点、条件边与降级组装

"""LangGraph workflow assembly and external call entry point.

Flow:
  START -> intent_check_node -> {chitchat: answer directly} / {query|followup: retrieve_docs_node}
  retrieve_docs_node -> rerank_sequence_node -> generate_answer_node -> evaluate_answer_node -> END
  If retrieval is empty, skip generate_answer_node and reply directly (saves Token)
"""
from __future__ import annotations

import logging
import time
from typing import Any

from app.config import Settings
from app.llm.azure_openai import build_chat_client
from app.retrieval.rerank import build_reranker
from app.retrieval.search import build_retriever
from app.cache import ExactCache
from app.graph.nodes import (
    CacheLookupNode,
    IntentCheckNode,
    RetrieveDocsNode,
    RerankSequenceNode,
    GenerateAnswerNode,
    EvaluateAnswerNode,
)
from app.graph.state import GraphState

# Visualize the workflow graph in a Jupyter environment (silently skips without IPython)
try:
    from IPython.display import Image, display  # type: ignore

    def _in_jupyter() -> bool:
        try:
            from IPython import get_ipython  # type: ignore
            return get_ipython() is not None
        except Exception:  # pragma: no cover
            return False
except ImportError:  # pragma: no cover
    Image = None  # type: ignore
    display = None  # type: ignore

    def _in_jupyter() -> bool:
        return False

logger = logging.getLogger(__name__)

# Session-level history storage (in-memory for prototype; production should use Redis/DB)
_SESSIONS: dict[str, list[dict[str, Any]]] = {}


class RAGWorkflow:
    """Wraps LangGraph orchestration."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = build_chat_client(settings)  # chat path (DashScope qwen)
        self.retriever = build_retriever(settings)
        self.reranker = build_reranker(settings, llm_client=self.llm)
        # Exact cache (single-turn): enabled by default, keyed by normalized question hash
        self.cache = ExactCache(kb_version=settings.cache_version) if settings.cache_enabled else None
        self._graph = None

    async def a_init(self) -> None:
        """Async initialization (load model/index)."""
        try:
            self._build_graph()
        except Exception as exc:  # noqa: BLE001
            logger.warning("LangGraph unavailable, falling back to sequential execution: %s", exc)
            self._graph = None

    def _build_graph(self) -> None:
        from langgraph.graph import END, StateGraph

        g = StateGraph(GraphState)
        # Node names must NOT collide with GraphState TypedDict fields (LangGraph strict check)
        g.add_node("cache_lookup_node", CacheLookupNode(self.cache))
        g.add_node("intent_check_node", IntentCheckNode())
        g.add_node("retrieve_docs_node", RetrieveDocsNode(self.retriever, self.settings.hybrid_top_k))
        g.add_node("rerank_sequence_node", RerankSequenceNode(self.reranker, self.settings.final_top_k, self.settings.rerank_strategy))
        g.add_node("generate_answer_node", GenerateAnswerNode(self.llm, self.cache))
        g.add_node("evaluate_answer_node", EvaluateAnswerNode(self.llm))

        # Entry point: cache lookup first
        g.set_entry_point("cache_lookup_node")

        # On cache hit → END (skip entire RAG pipeline)
        # On cache miss → intent_check
        def after_cache(state: GraphState) -> str:
            if state.get("from_cache"):
                return "__end__"
            return "intent_check_node"

        g.add_conditional_edges("cache_lookup_node", after_cache, {
            "intent_check_node": "intent_check_node",
            "__end__": END,
        })

        def after_intent_check(state: GraphState) -> str:
            if state.get("intent") == "chitchat":
                return "generate_answer_node"
            return "retrieve_docs_node"

        g.add_conditional_edges("intent_check_node", after_intent_check, {
            "retrieve_docs_node": "retrieve_docs_node",
            "generate_answer_node": "generate_answer_node",
        })
        g.add_edge("retrieve_docs_node", "rerank_sequence_node")
        g.add_edge("rerank_sequence_node", "generate_answer_node")
        g.add_edge("generate_answer_node", "evaluate_answer_node")
        g.add_edge("evaluate_answer_node", END)
        self._graph = g.compile()

        # Draw the workflow DAG only in a Jupyter environment (display(Image) warns in a plain terminal)
        if display is not None and Image is not None and _in_jupyter():
            try:
                display(Image(
                    self._graph.get_graph().draw_mermaid_png(
                        output_file_path="./OCEAdvanceRAG.png"
                    )
                ))
            except Exception:
                pass

    async def arun(
        self, question: str, user_group: str, session_id: str
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        history = _SESSIONS.setdefault(session_id, [])
        logger.info(
            "== [request start] session=%s | group=%s | history_turns=%d | question=%.80r",
            session_id, user_group, len(history), question,
        )
        state: GraphState = {
            "question": question,
            "user_group": user_group,
            "messages": list(history),
        }
        if self._graph is not None:
            result = await self._graph.ainvoke(state)
        else:
            result = await self._fallback_run(state)
        # Concise log: only key fields, no full dump
        answer_len = len(result.get("answer", "") or "")
        trace_count = len(result.get("trace", []))
        retrieved_count = len(result.get("retrieved_docs", []))
        intent = result.get("intent", "-")
        latency = result.get("latency_ms", 0.0)
        logger.info(
            "== [request done] session=%s | intent=%s | retrieved=%d | answer_len=%d | trace=%d steps | "
            "workflow_latency=%.0fms | total %.2fs",
            session_id, intent, retrieved_count, answer_len, trace_count,
            latency, time.monotonic() - t0,
        )
        # Update session history (always use dict format, avoid LangGraph Message object pollution)
        history.append({"role": "user", "content": question})
        if "answer" in result:
            history.append({"role": "assistant", "content": result["answer"]})
        _SESSIONS[session_id] = history[-10:]  # keep last 10
        # Clean up result: convert LangChain Message objects to dict and remove messages (frontend doesn't need them)
        result = dict(result)
        messages = result.get("messages")
        if messages:
            result["messages"] = [
                {"role": getattr(m, "type", getattr(m, "role", "unknown")), "content": getattr(m, "content", "")}
                if not isinstance(m, dict) else m
                for m in messages
            ]
        return result

    async def _fallback_run(self, state: GraphState) -> dict[str, Any]:
        """Sequential fallback implementation when LangGraph is missing."""
        cache_lookup_node = CacheLookupNode(self.cache)
        intent_check_node = IntentCheckNode()
        retrieve_docs_node = RetrieveDocsNode(self.retriever, self.settings.hybrid_top_k)
        rerank_sequence_node = RerankSequenceNode(self.reranker, self.settings.final_top_k, self.settings.rerank_strategy)
        generate_answer_node = GenerateAnswerNode(self.llm, self.cache)
        evaluate_answer_node = EvaluateAnswerNode(self.llm)

        accumulated_trace: list[dict] = []

        def _merge(node_result: dict) -> None:
            """Merge node result into state and manually accumulate trace."""
            trace_chunk = node_result.pop("trace", [])
            if trace_chunk:
                accumulated_trace.extend(trace_chunk)
            state.update(node_result)

        # Cache lookup first
        _merge(await cache_lookup_node(state))
        if state.get("from_cache"):
            state["trace"] = accumulated_trace
            return state

        _merge(await intent_check_node(state))
        if state.get("intent") != "chitchat":
            _merge(await retrieve_docs_node(state))
            _merge(await rerank_sequence_node(state))
        _merge(await generate_answer_node(state))
        _merge(await evaluate_answer_node(state))
        state["trace"] = accumulated_trace
        return state

    async def a_close(self) -> None:
        # Release resources (Azure client etc.)
        close = getattr(self.llm, "aclose", None)
        if callable(close):
            await close()
            logging.info("LLM resources released")
