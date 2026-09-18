"""核心单测：权限过滤、Chunking、Mock 检索、API 端到端。"""
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# 让项目根可导入
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("USE_MOCK_BACKEND", "true")
os.environ.setdefault("RERANK_STRATEGY", "none")


def test_filter_docs_by_group_allows_public_and_group():
    from app.security.permissions import filter_docs_by_group

    docs = [
        {"id": "1", "content": "公开", "allowed_groups": []},
        {"id": "2", "content": "仅 sre", "allowed_groups": ["sre"]},
        {"id": "3", "content": "仅 dev", "allowed_groups": ["dev"]},
    ]
    result = filter_docs_by_group(docs, "sre")
    assert {d["id"] for d in result} == {"1", "2"}


def test_build_search_filter_escapes_quote():
    from app.security.permissions import build_search_filter

    flt = build_search_filter("a'b")
    assert "a''b" in flt
    assert flt.startswith("allowed_groups/any(g: g eq '")


def test_chunk_markdown_splits_by_heading():
    from ingestion.chunking import chunk_document

    md = "## H2 标题一\n段落内容一\n\n## H2 标题二\n段落内容二\n"
    chunks = chunk_document(md, "sample.md", "runbooks", ["sre"])
    assert len(chunks) >= 2
    assert any("标题一" in c.heading for c in chunks)
    assert all("sre" in c.allowed_groups for c in chunks)


def test_chunk_kusto_keeps_statements():
    from ingestion.chunking import chunk_document

    kql = "Table1 | where x>1;\nTable2 | where y<2;\n"
    chunks = chunk_document(kql, "q.kql", "kusto", ["sre", "dev"])
    assert len(chunks) >= 1
    assert all(c.category == "kusto" for c in chunks)


def test_assert_no_leak_raises():
    from app.security.permissions import assert_no_leak

    with pytest.raises(PermissionError):
        assert_no_leak([{"allowed_groups": ["dev"]}], "sre")


@pytest.mark.asyncio
async def test_mock_workflow_chat_returns_answer():
    from app.config import get_settings
    from app.graph.workflow import RAGWorkflow

    settings = get_settings()
    wf = RAGWorkflow(settings=settings)
    await wf.a_init()
    result = await wf.arun("SharePoint 高延迟如何排查", "sre", "test-sess")
    assert "answer" in result
    assert isinstance(result.get("citations"), list)
    await wf.a_close()


def test_health_endpoint():
    from app.main import create_app

    # TestClient 不触发 lifespan 的异步初始化，直接测 health
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["backend"] in ("azure", "mock")
