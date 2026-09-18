"""索引写入器：Azure AI Search 上传合并文档 / 本地 JSONL 落盘（Mock）。

用法：
  python -m ingestion.indexer --data data/raw
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone

from app.config import Settings, get_settings
from app.security.permissions import build_search_filter
from ingestion.chunking import chunk_document
from ingestion.embedding import embed_texts


async def index_directory(data_dir: str = "data/raw", settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    files = []
    for root, _, names in os.walk(data_dir):
        cat = os.path.basename(root) or "misc"
        for n in names:
            if n.endswith((".md", ".kql", ".txt")):
                files.append((os.path.join(root, n), cat))
    if not files:
        print(f"[indexer] 未发现可索引文件于 {data_dir}")
        return 0

    all_chunks = []
    for path, cat in files:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        groups = ["sre"] if cat in ("postmortems", "icm_summaries") else ["sre", "dev", "ops"]
        all_chunks.extend(chunk_document(content, path, cat, groups))

    print(f"[indexer] 共生成 {len(all_chunks)} 个 chunk")
    vectors = await embed_texts([c.content for c in all_chunks], settings)

    if settings.use_mock_backend or not settings.has_azure_search:
        _write_local(all_chunks, vectors)
    else:
        await _upload_azure(all_chunks, vectors, settings)
    return len(all_chunks)


def _write_local(chunks, vectors) -> None:
    os.makedirs("data/index", exist_ok=True)
    out_path = os.path.join("data", "index", "mock_index.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for c, v in zip(chunks, vectors):
            f.write(json.dumps({
                "id": c.id,
                "content": c.content,
                "content_vector": v,
                "source": c.source,
                "category": c.category,
                "allowed_groups": c.allowed_groups,
                "heading": c.heading,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False) + "\n")
    print(f"[indexer] 本地索引已写入 {out_path}")


async def _upload_azure(chunks, vectors, settings: Settings) -> None:
    from azure.search.documents.aio import SearchClient
    from azure.core.credentials import AzureKeyCredential

    client = SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index_name,
        credential=AzureKeyCredential(settings.azure_search_api_key),
    )
    docs = []
    for c, v in zip(chunks, vectors):
        docs.append({
            "id": c.id,
            "content": c.content,
            "content_vector": v,
            "source": c.source,
            "category": c.category,
            "allowed_groups": c.allowed_groups,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "@search.action": "mergeOrUpload",
        })
    # 分批上传
    batch = 100
    for i in range(0, len(docs), batch):
        await client.merge_or_upload_documents(documents=docs[i:i + batch])
    print(f"[indexer] 已上传 {len(docs)} 条到 Azure AI Search 索引 {settings.azure_search_index_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="索引运维知识库文档")
    parser.add_argument("--data", default="data/raw", help="原始数据目录")
    args = parser.parse_args()
    n = asyncio.run(index_directory(args.data))
    print(f"[indexer] 完成，共 {n} 个 chunk")


if __name__ == "__main__":
    main()
