# @Author: RebeccaZhou
# @Description: GraphState TypedDict shared across nodes
#              GraphState TypedDict：节点间共享状态定义

"""Conversation state definition (design doc 4.2)."""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

try:
    from langgraph.graph.message import add_messages  # type: ignore
except Exception:  # fallback when langgraph is not installed
    def add_messages(left: list, right: list) -> list:
        return left + right


def add_trace(left: list, right: list) -> list:
    """Reducer that appends to the trace field: trace fragments returned by nodes are appended to the existing list."""
    if right is None:
        return left
    if isinstance(right, list):
        return left + right
    return left + [right]


class GraphState(TypedDict, total=False):
    """LangGraph shared state.

    - messages: conversation history (HumanMessage / AIMessage dicts)
    - user_group: current user group, used for permission filtering
    - question: current-turn user question
    - intent: intent classification query / followup / chitchat
    - retrieved_docs: document snippets after retrieval + rerank
    - citations: citation source list
    - answer: final answer
    - input_tokens / output_tokens / latency_ms: observability metrics
    - from_cache: whether the answer was served from the exact cache (skips RAG pipeline)
    - trace: structured step log for each node, shown in the frontend for the orchestration process
    """
    messages: Annotated[list[dict[str, Any]], add_messages]
    user_group: str
    question: str
    intent: str
    retrieved_docs: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    from_cache: bool
    trace: Annotated[list[dict[str, Any]], add_trace]
