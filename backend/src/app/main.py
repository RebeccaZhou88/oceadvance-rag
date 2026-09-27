# @Author: RebeccaZhou
# @Description: FastAPI application: logging bootstrap, lifespan, routers
#              FastAPI 应用：日志初始化、lifespan 与路由挂载

"""FastAPI entry point: wires up routes, workflow, observability middleware, and static frontend."""
import logging
import os
import sys
from contextlib import asynccontextmanager

# -- Configure logging at the very top (must happen before uvicorn/other third-party imports) --
# uvicorn takes over the root logger once it starts, so basicConfig becomes a no-op,
# so we must configure it at module load time to ensure all logger.info calls under app.xxx stream to the terminal in real time.
_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _level, logging.INFO),
    format="%(asctime)s [%(levelname)5s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# Suppress third-party HTTP request/response detail logs to keep terminal output clean
for _name in ("azure", "azure.core", "httpx", "openai", "httpcore", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app.api.governance_routes import router as governance_router  # noqa: E402
from app.api.governance_routes import set_governance  # noqa: E402
from app.api.routes import router, set_workflow  # noqa: E402
from app.config import get_settings  # noqa: E402


def _log_settings_summary(s) -> None:
    """Only log key config to avoid leaking api_key in plaintext."""
    logging.info(
        "= RAG assistant config = mock=%s | rerank=%s | embed=%s/%ddim | "
        "llm_model=%s | az_endpoint=%s | idx=%s | gov=%s",
        s.use_mock_backend,
        s.rerank_strategy,
        s.embedding_model_name,
        s.embedding_dimensions,
        s.model_name or "-",
        s.azure_search_endpoint,
        s.azure_search_index_name,
        s.governance_enabled,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _log_settings_summary(settings)
    logging.info("Starting RAG assistant | mock_backend=%s | rerank=%s",
                 settings.use_mock_backend, settings.rerank_strategy)
    # Lazy import to avoid circular dependency
    from app.graph.workflow import RAGWorkflow
    workflow = RAGWorkflow(settings=settings)
    await workflow.a_init()
    set_workflow(workflow)
    app.state.workflow = workflow
    logging.info("Workflow initialized")

    # Knowledge-base quality governance (CrewAI multi-agent + SQLite task store)
    gov_store = None
    gov_manager = None
    if settings.governance_enabled:
        from app.governance.crew import is_crewai_available
        from app.governance.manager import GovernanceManager
        from app.governance.store import GovernanceTaskStore
        gov_store = GovernanceTaskStore(settings.governance_db_file)
        gov_manager = GovernanceManager(settings, gov_store)
        set_governance(gov_manager, gov_store)
        logging.getLogger("app").info(
            "Quality governance enabled | crewai=%s | db=%s",
            is_crewai_available(), settings.governance_db_file,
        )

    yield

    if gov_manager is not None:
        await gov_manager.aclose()
    if gov_store is not None:
        gov_store.close()
    await workflow.a_close()
    logging.info("Workflow shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="OCE Operations Knowledge-Base Advanced RAG Assistant",
        version="0.1.0",
        description="Enterprise-grade RAG prototype with hybrid retrieval + rerank + citation traceability + permission filtering",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(governance_router)

    # Mount static frontend (prefer the FRONTEND_DIR env var)
    # src layout: src/app/main.py -> ../../../frontend (frontend/ at project root)
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    static_dir = os.environ.get(
        "FRONTEND_DIR",
        os.path.join(_root, "frontend"),
    )
    static_dir = os.path.abspath(static_dir)
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/", include_in_schema=False)
        async def index():
            return FileResponse(os.path.join(static_dir, "index.html"))

        logging.getLogger("app").info("Static frontend dir: %s", static_dir)

    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "dev",
    )
