"""对话状态定义（设计文档 4.2）。"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

try:
    from langgraph.graph.message import add_messages  # type: ignore
except Exception:  # langgraph 未安装时的兜底
    def add_messages(left: list, right: list) -> list:
        return left + right


def add_trace(left: list, right: list) -> list:
    """trace 字段的追加 reducer：节点返回 trace 片段会被拼接到已有列表。"""
    if right is None:
        return left
    if isinstance(right, list):
        return left + right
    return left + [right]


class GraphState(TypedDict, total=False):
    """LangGraph 共享状态。

    - messages: 对话历史（HumanMessage / AIMessage dict）
    - user_group: 当前用户组，用于权限过滤
    - question: 当前轮用户问题
    - intent: 意图分类 query / followup / chitchat
    - retrieved_docs: 检索+重排后的文档片段
    - citations: 引用来源列表
    - answer: 最终回答
    - input_tokens / output_tokens / latency_ms: 观测指标
    - trace: 每个节点的结构化步骤日志，供前端展示编排过程
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
    trace: Annotated[list[dict[str, Any]], add_trace]
