"""Request/response Pydantic models."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Citation source."""
    doc_id: str
    source: str = Field(description="Original document path or URL")
    category: str = Field(description="runbook/postmortem/kusto/icm")
    snippet: str = Field(description="The referenced text snippet")
    score: float = Field(ge=0.0, description="Retrieval/rerank score (different strategies have different scales: retrieval 0-1, LLM rerank 0-10)")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="User question")
    user_group: str = Field(description="User group, used for permission filtering, e.g. sre/dev/ops")
    session_id: str = Field(description="Multi-turn conversation session ID")


class TraceStep(BaseModel):
    """Single orchestration step log."""
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
    from_cache: bool = False
    intent: str = "query"
    trace: list[TraceStep] = Field(default_factory=list, description="Orchestration step logs")
    governance_task_id: Optional[str] = Field(default=None, description="Background task id returned when quality governance rules are matched")


class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    answer: str
    rating: int = Field(ge=1, le=5, description="1-5 score, 5 being the most satisfactory")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str = "ok"
    received_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    backend: str = Field(description="azure | mock")
