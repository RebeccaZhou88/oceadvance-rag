"""请求/响应 Pydantic 模型。"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """引用来源。"""
    doc_id: str
    source: str = Field(description="原始文档路径或 URL")
    category: str = Field(description="runbook/postmortem/kusto/icm")
    snippet: str = Field(description="被引用的片段文本")
    score: float = Field(ge=0.0, description="检索/重排得分（不同策略量纲不同：检索 0-1，LLM 重排 0-10）")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="用户问题")
    user_group: str = Field(description="用户组，用于权限过滤，如 sre/dev/ops")
    session_id: str = Field(description="多轮对话会话 ID")


class TraceStep(BaseModel):
    """单个编排步骤日志。"""
    step: str
    status: str
    duration_ms: float
    ts: float
    detail: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    session_id: str
    retrieved_docs_count: int = 0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    trace: list[TraceStep] = Field(default_factory=list, description="编排步骤日志")


class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    answer: str
    rating: int = Field(ge=1, le=5, description="1-5 分，5 为最满意")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str = "ok"
    received_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    backend: str = Field(description="azure | mock")
