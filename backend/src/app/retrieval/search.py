# @Author: RebeccaZhou
# @Description: Azure AI Search client: vector / hybrid (BM25+Vector+RRF) modes
#              Azure AI Search 检索客户端：vector / hybrid（BM25+Vector+RRF）双模式

"""Hybrid retrieval module: Azure AI Search (BM25 + Vector + RRF) or local in-memory mock backend.

Returns a unified structure: list[dict], each chunk contains
  id / content / source / category / allowed_groups / score / last_updated
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Protocol

from app.config import Settings
from app.security.permissions import build_search_filter, filter_docs_by_group

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    async def hybrid_search(
        self, query: str, user_group: str, top_k: int = 20
    ) -> list[dict[str, Any]]:
        ...


class AzureAISearchRetriever:
    """Real Azure AI Search hybrid retrieval.

    - BM25 keyword search
    - Vector search (same embedding model as ingestion, default text-embedding-v2 / 1536 dim)
    - RRF fusion
    - OData permission filtering
    """

    def __init__(self, settings: Settings) -> None:
        from azure.search.documents.aio import SearchClient
        from azure.core.credentials import AzureKeyCredential

        self.settings = settings
        self._client = SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index_name,
            credential=AzureKeyCredential(settings.azure_search_api_key),
        )

    async def hybrid_search(
        self, query: str, user_group: str, top_k: int = 20
    ) -> list[dict[str, Any]]:
        from azure.search.documents.models import VectorizedQuery

        # Inject permission filtering OData
        search_filter = build_search_filter(user_group)
        vector = await self._embed(query)
        logger.info(
            "  🔎 [Azure Search] filter=%s | vector_dim=%d | top_k=%d | query_type=%s",
            search_filter, len(vector), top_k,
            "semantic" if self.settings.rerank_strategy == "semantic" else "simple",
        )

        vector_query = VectorizedQuery(
            vector=vector, k_nearest_neighbors=top_k, fields="content_vector"
        )
        # search_mode 决定是否启用 BM25 关键词检索 + RRF 融合
        #   vector: 仅向量检索，中文 BM25 简单解析器效果差，纯向量通常更好
        #   hybrid: search_text=query + vector_queries，Azure 自动 RRF 融合两路结果
        search_kwargs: dict[str, Any] = {
            "vector_queries": [vector_query],
            "filter": search_filter,
            "top": top_k,
        }
        if self.settings.azure_search_mode == "hybrid":
            search_kwargs["search_text"] = query
        mode_label = "hybrid" if self.settings.azure_search_mode == "hybrid" else "vector-only"
        logger.info(
            "  🔎 [Azure Search] mode=%s | filter=%s | vector_dim=%d | top_k=%d",
            mode_label, search_filter, len(vector), top_k,
        )
        results = await self._client.search(**search_kwargs)
        docs: list[dict[str, Any]] = []
        async for r in results:
            docs.append(self._normalize(r))
        logger.info("  🔎 [Azure Search] returned %d docs", len(docs))
        return docs

    async def _embed(self, text: str) -> list[float]:
        # Query vectorization must use the same embedding model as ingestion
        # (same dimensions), uniformly go through the factory
        from app.llm.azure_openai import build_embed_client

        client = build_embed_client(self.settings)
        return (await client.embed([text]))[0]

    @staticmethod
    def _normalize(r: Any) -> dict[str, Any]:
        return {
            "id": r.get("id"),
            "content": r.get("content"),
            "source": r.get("source"),
            "category": r.get("category"),
            "allowed_groups": r.get("allowed_groups") or [],
            "score": r.get("@search.score", 0.0),
            "last_updated": r.get("last_updated"),
        }


class MockRetriever:
    """In-memory mock backend: BM25 (term-frequency based) + Vector (cosine) + RRF fusion.

    Read synthetic documents from data/raw/* to build a local index, convenient
    for development and evaluation without Azure credentials.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._docs: list[dict[str, Any]] = []
        self._vectors: list[list[float]] = []
        self._embedder = None  # lazy initialization

    def load_local(self, data_dir: str = "data/raw") -> None:
        """Recursively load .md/.kql/.txt documents from data/raw."""
        if not os.path.isdir(data_dir):
            logger.warning("Local data directory does not exist: %s", data_dir)
            return
        for root, _, files in os.walk(data_dir):
            category = os.path.basename(root) or "misc"
            for fname in files:
                if not fname.endswith((".md", ".kql", ".txt")):
                    continue
                path = os.path.join(root, fname)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Permissions: runbooks/kusto default public, postmortems/icm sre-only
                if category in ("postmortems", "icm_summaries"):
                    groups = ["sre"]
                else:
                    groups = ["sre", "dev", "ops"]
                self._docs.append({
                    "id": f"{category}:{fname}",
                    "content": content,
                    "source": path,
                    "category": category,
                    "allowed_groups": groups,
                    "score": 0.0,
                    "last_updated": None,
                })

    async def _ensure_embedded(self) -> None:
        if self._vectors or not self._docs:
            return
        if self._embedder is None:
            from app.llm.azure_openai import build_embed_client

            self._embedder = build_embed_client(self.settings)
        texts = [d["content"] for d in self._docs]
        # Batch to avoid overly long requests
        batch = 32
        for i in range(0, len(texts), batch):
            self._vectors.extend(await self._embedder.embed(texts[i:i + batch]))

    async def hybrid_search(
        self, query: str, user_group: str, top_k: int = 20
    ) -> list[dict[str, Any]]:
        await self._ensure_embedded()
        # First do permission filtering
        candidates = filter_docs_by_group(self._docs, user_group)
        if not candidates:
            return []
        candidate_idx = [self._docs.index(c) for c in candidates]

        # BM25 term-frequency scoring
        bm25_scores = self._bm25_scores(query, candidates)
        # Vector cosine scoring
        vec_scores = await self._vector_scores(query, candidate_idx)

        # RRF fusion
        k = self.settings.rrf_k
        bm25_rank = sorted(range(len(candidates)), key=lambda i: -bm25_scores[i])
        vec_rank = sorted(range(len(candidates)), key=lambda i: -vec_scores[i])
        rrf = [0.0] * len(candidates)
        for rank, i in enumerate(bm25_rank):
            rrf[i] += 1.0 / (k + rank + 1)
        for rank, i in enumerate(vec_rank):
            rrf[i] += 1.0 / (k + rank + 1)

        order = sorted(range(len(candidates)), key=lambda i: -rrf[i])[:top_k]
        results: list[dict[str, Any]] = []
        for i in order:
            doc = dict(candidates[i])
            doc["score"] = rrf[i]
            results.append(doc)
        return results

    def _bm25_scores(self, query: str, docs: list[dict[str, Any]]) -> list[float]:
        q_terms = set(query.lower().split())
        scores: list[float] = []
        avg_len = max(sum(len(d["content"].split()) for d in docs) / max(len(docs), 1), 1)
        for d in docs:
            tokens = d["content"].lower().split()
            tf = sum(1 for t in tokens if t in q_terms)
            idf = math.log(1 + (len(docs) - tf + 0.5) / (tf + 0.5)) if tf else 0.0
            score = idf * (tf * (1.2 + 1) / (tf + 1.2 * (0.25 + 0.75 * len(tokens) / avg_len)))
            scores.append(score)
        return scores

    async def _vector_scores(self, query: str, idxs: list[int]) -> list[float]:
        if self._embedder is None:
            from app.llm.azure_openai import build_embed_client

            self._embedder = build_embed_client(self.settings)
        qv = (await self._embedder.embed([query]))[0]
        scores: list[float] = []
        for i in idxs:
            dv = self._vectors[i] if i < len(self._vectors) else [0.0] * len(qv)
            denom = (math.sqrt(sum(a * a for a in qv)) * math.sqrt(sum(b * b for b in dv))) or 1.0
            scores.append(sum(a * b for a, b in zip(qv, dv)) / denom)
        return scores


def build_retriever(settings: Settings) -> Retriever:
    if settings.use_mock_backend or not settings.has_azure_search:
        logger.warning("Using in-memory Mock retrieval (Azure AI Search not configured)")
        retriever = MockRetriever(settings)
        retriever.load_local()
        return retriever
    return AzureAISearchRetriever(settings)
