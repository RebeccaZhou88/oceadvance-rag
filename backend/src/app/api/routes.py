"""FastAPI routes: /chat /feedback /health /metrics."""
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
    RECORD_BUSINESS,
    RECORD_FEEDBACK,
    RECORD_REQUEST,
    RECORD_TOKEN_USAGE,
    export as export_metrics,
)

if TYPE_CHECKING:
    from app.graph.workflow import RAGWorkflow

router = APIRouter()

# Runtime-injected workflow instance (injected by main.py lifespan)
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
    """Determine the retrieval backend based on has_azure_search and use_mock_backend.

    Note: the chat pipeline (has_llm) is unrelated to the retrieval backend, do not include it in this check.
    """
    if settings.use_mock_backend or not settings.has_azure_search:
        return "mock"
    # At this point = Azure AI Search is actually in use (pure vector / hybrid both count as azure)
    return "azure"


@router.get("/config")
async def config() -> dict:
    """Expose the current runtime configuration for the frontend to display."""
    settings = get_settings()
    return {
        "backend": _detect_backend(settings),
        "search_mode": settings.azure_search_mode,
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

    # Business latency = end-to-end - evaluate_answer (online evaluation) duration
    trace = result.get("trace", [])
    eval_ms = sum(
        t.get("duration_ms", 0.0)
        for t in trace
        if t.get("step") == "evaluate_answer" and t.get("status") != "skipped"
    )
    business_ms = max(0.0, latency_ms - eval_ms)
    RECORD_BUSINESS(business_ms)

    RECORD_TOKEN_USAGE(
        input_tokens=result.get("input_tokens", 0),
        output_tokens=result.get("output_tokens", 0),
    )

    # Quality governance conditional routing: empty recall / low evaluation score -> async trigger CrewAI, does not block answer
    governance_task_id = None
    from app.api.governance_routes import _manager as _gov_manager
    if _gov_manager is not None:
        try:
            governance_task_id = await _gov_manager.maybe_trigger_after_chat(
                result=result,
                question=req.question,
                user_group=req.user_group,
                session_id=req.session_id,
            )
        except Exception:  # noqa: BLE001
            logging.getLogger("app").exception("Governance trigger failed (does not affect main flow)")

    return ChatResponse(
        answer=result["answer"],
        citations=result.get("citations", []),
        session_id=req.session_id,
        retrieved_docs_count=result.get("retrieved_docs_count", 0),
        latency_ms=latency_ms,
        input_tokens=result.get("input_tokens", 0),
        output_tokens=result.get("output_tokens", 0),
        from_cache=result.get("from_cache", False),
        intent=result.get("intent", "query"),
        trace=result.get("trace", []),
        governance_task_id=governance_task_id,
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest) -> FeedbackResponse:
    RECORD_FEEDBACK(rating=req.rating)
    FEEDBACK_SCORE.observe(req.rating)
    # In real scenarios this can be persisted for offline evaluation later
    return FeedbackResponse()


@router.get("/metrics")
async def metrics():
    from fastapi import Response
    return Response(content=export_metrics(), media_type="text/plain; version=0.0.4; charset=utf-8")
