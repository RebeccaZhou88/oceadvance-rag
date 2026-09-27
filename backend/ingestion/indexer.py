# @Author: RebeccaZhou
# @Description: Document ingestion pipeline: chunk, embed, merge-or-upload to index
#              文档摄入流水线：分块、嵌入并 merge-or-upload 上传索引

"""Index writer: Azure AI Search upload merge docs / local JSONL dump (Mock).

Usage:
  python -m ingestion.indexer --data data/raw
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone

from app.config import Settings, get_settings
from app.security.permissions import build_search_filter
from ingestion.chunking import chunk_document
from ingestion.embedding import embed_texts

logger = logging.getLogger(__name__)

# Restricted category: sre only
_RESTRICTED = ("postmortems", "icm_summaries")


async def index_directory(data_dir: str = "data/raw", settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    t_start = time.monotonic()

    # ── 1. Scan files ──
    logger.info("══ [1/4] Scanning data dir: %s", data_dir)
    files = []
    for root, _, names in os.walk(data_dir):
        cat = os.path.basename(root) or "misc"
        for n in sorted(names):
            if n.endswith((".md", ".kql", ".txt")):
                files.append((os.path.join(root, n), cat))

    if not files:
        logger.warning("No indexable files found in %s", data_dir)
        return 0

    for path, cat in files:
        groups = ["sre"] if cat in _RESTRICTED else ["sre", "dev", "ops"]
        logger.info("  ▸ %-40s category=%-16s groups=%s", os.path.relpath(path, data_dir), cat, groups)
    logger.info("  Scan done: %d files total", len(files))

    # ── 2. Split chunks ──
    logger.info("══ [2/4] Document chunking")
    all_chunks = []
    for path, cat in files:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        groups = ["sre"] if cat in _RESTRICTED else ["sre", "dev", "ops"]
        chunks = chunk_document(content, path, cat, groups)
        heading_preview = chunks[0].heading[:50] if chunks else ""
        logger.info(
            "  ▸ %-40s %d chars → %d chunks, first heading=%s",
            os.path.relpath(path, data_dir),
            len(content),
            len(chunks),
            heading_preview,
        )
        all_chunks.extend(chunks)
    logger.info("  Chunking done: %d chunks total", len(all_chunks))

    # ── 3. Generate vectors ──
    logger.info("══ [3/4] Generate vectors (embedding)")
    vectors = await embed_texts([c.content for c in all_chunks], settings)

    # ── 4. Write to storage ──
    logger.info("══ [4/4] Write to storage")
    if settings.use_mock_backend or not settings.has_azure_search:
        logger.info("  Mode: local Mock (USE_MOCK_BACKEND=%s)", settings.use_mock_backend)
        _write_local(all_chunks, vectors)
    else:
        logger.info(
            "  Mode: Azure AI Search (endpoint=%s, index=%s)",
            settings.azure_search_endpoint,
            settings.azure_search_index_name,
        )
        await _upload_azure(all_chunks, vectors, settings)

    logger.info(
        "══ All done: %d chunks total, elapsed %.2fs",
        len(all_chunks),
        time.monotonic() - t_start,
    )
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
    logger.info("  ✔ Local index written to %s (%d records)", out_path, len(chunks))


async def _upload_azure(chunks, vectors, settings: Settings) -> None:
    from azure.search.documents.aio import SearchClient
    from azure.core.credentials import AzureKeyCredential

    # Dimension check: when embedding API fails & falls back, it's 256-dim hash vectors; uploading to 1536-dim index will fail, intercept early
    actual_dim = len(vectors[0]) if vectors else 0
    if actual_dim != settings.embedding_dimensions:
        logger.error(
            "✘ Dimension check failed: vector dim=%d, index dim=%d, upload aborted",
            actual_dim,
            settings.embedding_dimensions,
        )
        raise RuntimeError(
            f"Vector dimension {actual_dim} does not match Azure index dimension {settings.embedding_dimensions}, "
            f"upload aborted. Please check LLM_EMBEDDING_MODEL_NAME (currently {settings.embedding_model_name}) "
            "for availability and quota (embedding failure silently falls back to hash vectors)."
        )
    logger.info("  Dimension check passed: %d dims ✓", actual_dim)

    async with SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index_name,
        credential=AzureKeyCredential(settings.azure_search_api_key),
    ) as client:
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
        # Upload in batches
        batch = 100
        total_batches = (len(docs) + batch - 1) // batch
        for i in range(0, len(docs), batch):
            t0 = time.monotonic()
            result = await client.merge_or_upload_documents(documents=docs[i:i + batch])
            batch_no = i // batch + 1
            logger.info(
                "  ▸ Upload batch %d/%d: %d records, elapsed %.2fs",
                batch_no,
                total_batches,
                len(docs[i:i + batch]),
                time.monotonic() - t0,
            )
    logger.info("  ✔ Uploaded %d records to Azure AI Search index %s", len(docs), settings.azure_search_index_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index ops knowledge base docs")
    parser.add_argument("--data", default="data/raw", help="Raw data directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show DEBUG level logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)5s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress third-party lib HTTP request/response detail logs, keep terminal output clean
    for name in ("azure", "azure.core", "httpx", "openai", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    n = asyncio.run(index_directory(args.data))
    logger.info("indexer exiting, %d chunks processed", n)


if __name__ == "__main__":
    main()
