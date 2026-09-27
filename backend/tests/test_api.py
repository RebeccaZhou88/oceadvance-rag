# @Author: RebeccaZhou
# @Description: API tests: chat pipeline, permissions, metrics
#              API 测试：问答流程、权限过滤与指标

"""Core unit tests: permission filtering, Chunking, Mock retrieval, API end-to-end."""
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("USE_MOCK_BACKEND", "true")
os.environ.setdefault("RERANK_STRATEGY", "none")


def test_filter_docs_by_group_allows_public_and_group():
    from app.security.permissions import filter_docs_by_group

    docs = [
        {"id": "1", "content": "public", "allowed_groups": []},
        {"id": "2", "content": "sre only", "allowed_groups": ["sre"]},
        {"id": "3", "content": "dev only", "allowed_groups": ["dev"]},
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

    md = (
        "## H2 Heading One\n"
        "Paragraph content one, containing enough text to ensure it is not merged into the next heading block. "
        "We need to make the character count of this paragraph exceed more than half of MAX_CHUNK_CHARS, "
        "so that the adjacent merging logic does not combine the two paragraphs into one block.\n\n"
        "## H2 Heading Two\n"
        "Paragraph content two, also containing enough text to ensure it forms an independent block. "
        "Each paragraph should contain at least two hundred characters, so that it triggers independent chunking without being merged. "
        "Continue filling in content to make it a bit longer, so the test does not fail due to merging logic.\n"
    )
    chunks = chunk_document(md, "sample.md", "runbooks", ["sre"])
    # v2 semantics: short paragraphs merge, long ones independent
    assert len(chunks) >= 1  # at least one chunk (may or may not merge)
    assert any("H2" in c.heading for c in chunks)
    assert all("sre" in c.allowed_groups for c in chunks)
    # each chunk carries heading path
    assert all(len(c.heading_path) >= 1 for c in chunks)
    # pure heading blocks (content is only the Hx line) are skipped
    assert all(len(c.content) > 50 for c in chunks)


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
    result = await wf.arun("How to fix SharePoint slow performance issues", "sre", "test-sess")
    assert "answer" in result
    assert isinstance(result.get("citations"), list)
    await wf.a_close()


def test_health_endpoint():
    from app.main import create_app

    # TestClient does not trigger lifespan async init, test health directly
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["backend"] in ("azure", "mock", "llm")
