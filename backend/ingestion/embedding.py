"""Embedding 生成：批量调用 Azure OpenAI / Mock。"""
from __future__ import annotations

import logging
from typing import Sequence

from app.config import Settings
from app.llm.azure_openai import build_llm_client

logger = logging.getLogger(__name__)

BATCH_SIZE = 32


async def embed_texts(texts: Sequence[str], settings: Settings) -> list[list[float]]:
    client = build_llm_client(settings)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = list(texts[i:i + BATCH_SIZE])
        vectors.extend(await client.embed(batch))
    logger.info("已生成 %d 条向量", len(vectors))
    return vectors
