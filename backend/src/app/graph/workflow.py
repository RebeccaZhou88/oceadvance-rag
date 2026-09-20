"""LangGraph 工作流装配与对外调用入口。

流程：
  START -> intent_check_node -> {chitchat: 直接回答} / {query|followup: retrieve_docs_node}
  retrieve_docs_node -> rerank_sequence_node -> generate_answer_node -> evaluate_answer_node -> END
  若检索为空则跳过 generate_answer_node 直接回复（节省 Token）
"""
from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.llm.azure_openai import build_llm_client
from app.retrieval.rerank import build_reranker
from app.retrieval.search import build_retriever
from app.graph.nodes import (
    IntentCheckNode,
    RetrieveDocsNode,
    RerankSequenceNode,
    GenerateAnswerNode,
    EvaluateAnswerNode,
)
from app.graph.state import GraphState

# Jupyter 环境下可视化工作流图（无 IPython 时静默跳过）
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

# 会话级历史存储（原型用内存，生产应换 Redis/数据库）
_SESSIONS: dict[str, list[dict[str, Any]]] = {}


class RAGWorkflow:
    """封装 LangGraph 编排。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = build_llm_client(settings)
        self.retriever = build_retriever(settings)
        self.reranker = build_reranker(settings, llm_client=self.llm)
        self._graph = None

    async def a_init(self) -> None:
        """异步初始化（加载模型/索引）。"""
        try:
            self._build_graph()
        except Exception as exc:  # noqa: BLE001
            logger.warning("LangGraph 不可用，退化为顺序调用：%s", exc)
            self._graph = None

    def _build_graph(self) -> None:
        from langgraph.graph import END, StateGraph

        g = StateGraph(GraphState)
        # 节点名不能与 GraphState TypedDict 字段重名（LangGraph 强校验）
        g.add_node("intent_check_node", IntentCheckNode())
        g.add_node("retrieve_docs_node", RetrieveDocsNode(self.retriever, self.settings.hybrid_top_k))
        g.add_node("rerank_sequence_node", RerankSequenceNode(self.reranker, self.settings.final_top_k, self.settings.rerank_strategy))
        g.add_node("generate_answer_node", GenerateAnswerNode(self.llm))
        g.add_node("evaluate_answer_node", EvaluateAnswerNode(self.llm))

        g.set_entry_point("intent_check_node")

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

        # 仅 Jupyter 环境画出工作流 DAG（纯终端下 display(Image) 会打警告）
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
        history = _SESSIONS.setdefault(session_id, [])
        logging.info("history: %s", history)
        state: GraphState = {
            "question": question,
            "user_group": user_group,
            "messages": list(history),
        }
        if self._graph is not None:
            result = await self._graph.ainvoke(state)
        else:
            result = await self._fallback_run(state)
        logging.info("result: %s", result)
        # 更新会话历史(始终用 dict 格式,避免 LangGraph 的 Message 对象污染)
        history.append({"role": "user", "content": question})
        if "answer" in result:
            history.append({"role": "assistant", "content": result["answer"]})
        _SESSIONS[session_id] = history[-10:]  # 保留最近 10 条
        # 清理返回结果:把 LangChain Message 对象转成 dict,并移除 messages(前端不需要)
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
        """LangGraph 缺失时的顺序兜底实现。"""
        intent_check_node = IntentCheckNode()
        retrieve_docs_node = RetrieveDocsNode(self.retriever, self.settings.hybrid_top_k)
        rerank_sequence_node = RerankSequenceNode(self.reranker, self.settings.final_top_k, self.settings.rerank_strategy)
        generate_answer_node = GenerateAnswerNode(self.llm)
        evaluate_answer_node = EvaluateAnswerNode(self.llm)

        accumulated_trace: list[dict] = []

        def _merge(node_result: dict) -> None:
            """合并节点结果到 state,并手动累积 trace。"""
            trace_chunk = node_result.pop("trace", [])
            if trace_chunk:
                accumulated_trace.extend(trace_chunk)
            state.update(node_result)

        _merge(await intent_check_node(state))
        if state.get("intent") != "chitchat":
            _merge(await retrieve_docs_node(state))
            _merge(await rerank_sequence_node(state))
        _merge(await generate_answer_node(state))
        _merge(await evaluate_answer_node(state))
        state["trace"] = accumulated_trace
        return state

    async def a_close(self) -> None:
        # 释放资源（Azure 客户端等）
        close = getattr(self.llm, "aclose", None)
        if callable(close):
            await close()
            logging.info("LLM 资源已释放")
