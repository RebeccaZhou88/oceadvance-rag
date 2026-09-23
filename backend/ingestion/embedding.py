"""Embedding generation: batch call Azure OpenAI / Mock."""
from __future__ import annotations

import logging
import time
from typing import Sequence

from app.config import Settings
from app.llm.azure_openai import build_embed_client

logger = logging.getLogger(__name__)

BATCH_SIZE = 32


async def embed_texts(texts: Sequence[str], settings: Settings) -> list[list[float]]:
    client = build_embed_client(settings)
    # Try to get embedding model id: Azure client uses deployment name, others use model_name
    embed_label = getattr(client, "_embed_model", None) or settings.embedding_model_name or "mock"
    logger.info(
        "══ embedding phase start: %d texts total, batch=%d, client=%s",
        len(texts), BATCH_SIZE, embed_label,
    )
    vectors: list[list[float]] = []
    total = len(texts)
    for i in range(0, total, BATCH_SIZE):
        batch = list(texts[i:i + BATCH_SIZE])
        t0 = time.monotonic()
        result = await client.embed(batch)
        vectors.extend(result)
        logger.info(
            "  ▸ batch %d/%d done: %d in batch, cumulative %d/%d, dim=%d, elapsed %.2fs",
            i // BATCH_SIZE + 1,
            (total + BATCH_SIZE - 1) // BATCH_SIZE,
            len(batch),
            len(vectors),
            total,
            len(result[0]) if result else 0,
            time.monotonic() - t0,
        )
    dim = len(vectors[0]) if vectors else 0
    logger.info("══ embedding phase done: %d vectors total, dim=%d", len(vectors), dim)
    return vectors
