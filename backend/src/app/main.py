"""FastAPI 入口：装配路由、工作流、可观测性中间件、静态前端页面。"""
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router, set_workflow
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.error("settings: %s", settings)
    logging.basicConfig(level=settings.log_level.upper())
    logging.getLogger("app").info(
        "启动 RAG 助手 | mock_backend=%s | rerank=%s",
        settings.use_mock_backend, settings.rerank_strategy,
    )
    # 延迟导入避免循环依赖
    from app.graph.workflow import RAGWorkflow
    workflow = RAGWorkflow(settings=settings)
    await workflow.a_init()
    set_workflow(workflow)
    app.state.workflow = workflow
    yield
    await workflow.a_close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="运维知识库 Advanced RAG 助手",
        version="0.1.0",
        description="混合检索 + 重排 + 引用溯源 + 权限过滤的企业级 RAG 原型",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    # 挂载静态前端页面（优先环境变量 FRONTEND_DIR）
    # src layout: src/app/main.py → ../../../frontend (项目根的 frontend/)
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

        logging.getLogger("app").info("静态前端目录: %s", static_dir)

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
