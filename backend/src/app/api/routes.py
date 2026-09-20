"""FastAPI 路由：/chat /feedback /health /metrics。"""
import time
from typing import TYPE_CHECKING
import logging

from fastapi import APIRouter, HTTPException
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
)
from app.config import get_settings
from app.observability.metrics import (
    FEEDBACK_SCORE,
    RECORD_FEEDBACK,
    RECORD_REQUEST,
    RECORD_TOKEN_USAGE,
    export as export_metrics,
)

if TYPE_CHECKING:
    from app.graph.workflow import RAGWorkflow

router = APIRouter()

# 运行时注入的工作流实例（由 main.py lifespan 注入）
_workflow: "RAGWorkflow | None" = None


def set_workflow(workflow: "RAGWorkflow") -> None:
    global _workflow
    _workflow = workflow


def _get_workflow() -> "RAGWorkflow":
    if _workflow is None:
        raise HTTPException(status_code=503, detail="RAG workflow not initialized")
    return _workflow


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    backend = _detect_backend(settings)
    from app import __version__
    return HealthResponse(status="ok", version=__version__, backend=backend)


def _detect_backend(settings) -> str:
    if settings.has_llm:
        return "llm"
    if settings.has_azure_openai and settings.has_azure_search:
        return "azure"
    return "mock"


@router.get("/config")
async def config() -> dict:
    """暴露当前运行配置，供前端展示。"""
    settings = get_settings()
    return {
        "backend": _detect_backend(settings),
        "rerank_strategy": settings.rerank_strategy,
        "hybrid_top_k": settings.hybrid_top_k,
        "final_top_k": settings.final_top_k,
        "rrf_k": settings.rrf_k,
        "chat_deployment": settings.model_name,
        "embedding_deployment": settings.azure_openai_embedding_deployment or "-",
        "index_name": settings.azure_search_index_name,
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    workflow = _get_workflow()
    try:
        result = await workflow.arun(
            question=req.question,
            user_group=req.user_group,
            session_id=req.session_id,
        )
    except Exception as exc:  # noqa: BLE001
        import traceback as tb
        logging.error("Chat pipeline failed:\n%s", tb.format_exc())
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}") from exc

    latency_ms = (time.perf_counter() - start) * 1000
    RECORD_REQUEST(latency_ms=latency_ms, status="ok")
    RECORD_TOKEN_USAGE(
        input_tokens=result.get("input_tokens", 0),
        output_tokens=result.get("output_tokens", 0),
    )
    return ChatResponse(
        answer=result["answer"],
        citations=result.get("citations", []),
        session_id=req.session_id,
        retrieved_docs_count=result.get("retrieved_docs_count", 0),
        latency_ms=latency_ms,
        input_tokens=result.get("input_tokens", 0),
        output_tokens=result.get("output_tokens", 0),
        trace=result.get("trace", []),
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest) -> FeedbackResponse:
    RECORD_FEEDBACK(rating=req.rating)
    FEEDBACK_SCORE.observe(req.rating)
    # 真实场景可落库用于后续离线评估
    return FeedbackResponse()


@router.get("/metrics")
async def metrics():
    from fastapi import Response
    return Response(content=export_metrics(), media_type="text/plain; version=0.0.4; charset=utf-8")
