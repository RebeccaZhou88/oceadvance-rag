"""混合检索模块：Azure AI Search (BM25 + Vector + RRF) 或本地内存模拟后端。

返回统一结构：list[dict]，每个片段含
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
    """真实 Azure AI Search 混合检索。

    - BM25 关键词检索
    - 向量检索（text-embedding-3-small）
    - RRF 融合
    - OData 权限过滤
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

        # 注入权限过滤 OData
        search_filter = build_search_filter(user_group)
        # 注：向量查询需先 embedding，此处通过依赖的 LLM 客户端
        # 为简洁，这里假定调用方已传入 query 文本，向量化在外层处理
        # 真实实现中应注入 embedding client
        vector = await self._embed(query)

        vector_query = VectorizedQuery(
            vector=vector, k_nearest_neighbors=top_k, fields="content_vector"
        )
        results = await self._client.search(
            search_text=query,
            vector_queries=[vector_query],
            filter=search_filter,
            top=top_k,
            query_type="semantic" if self.settings.rerank_strategy == "semantic" else "simple",
        )
        docs: list[dict[str, Any]] = []
        async for r in results:
            docs.append(self._normalize(r))
        return docs

    async def _embed(self, text: str) -> list[float]:
        # 由 workflow 注入 embedding；此处兜底走 Azure
        from app.llm.azure_openai import AzureOpenAIClient

        client = AzureOpenAIClient(self.settings)
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
    """内存模拟后端：BM25（基于词频）+ 向量（余弦）+ RRF 融合。

    从 data/raw/* 读取合成文档建立本地索引，便于无 Azure 凭证时开发与评估。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._docs: list[dict[str, Any]] = []
        self._vectors: list[list[float]] = []
        self._embedder = None  # 延迟初始化

    def load_local(self, data_dir: str = "data/raw") -> None:
        """从 data/raw 递归加载 .md/.kql/.txt 文档。"""
        if not os.path.isdir(data_dir):
            logger.warning("本地数据目录不存在：%s", data_dir)
            return
        for root, _, files in os.walk(data_dir):
            category = os.path.basename(root) or "misc"
            for fname in files:
                if not fname.endswith((".md", ".kql", ".txt")):
                    continue
                path = os.path.join(root, fname)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                # 权限：runbooks/kusto 默认公开，postmortems/icm 仅 sre 可见
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
            from app.llm.azure_openai import build_llm_client

            self._embedder = build_llm_client(self.settings)
        texts = [d["content"] for d in self._docs]
        # 分批避免过长
        batch = 32
        for i in range(0, len(texts), batch):
            self._vectors.extend(await self._embedder.embed(texts[i:i + batch]))

    async def hybrid_search(
        self, query: str, user_group: str, top_k: int = 20
    ) -> list[dict[str, Any]]:
        await self._ensure_embedded()
        # 先做权限过滤
        candidates = filter_docs_by_group(self._docs, user_group)
        if not candidates:
            return []
        candidate_idx = [self._docs.index(c) for c in candidates]

        # BM25 词频打分
        bm25_scores = self._bm25_scores(query, candidates)
        # 向量余弦打分
        vec_scores = await self._vector_scores(query, candidate_idx)

        # RRF 融合
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
            from app.llm.azure_openai import build_llm_client

            self._embedder = build_llm_client(self.settings)
        qv = (await self._embedder.embed([query]))[0]
        scores: list[float] = []
        for i in idxs:
            dv = self._vectors[i] if i < len(self._vectors) else [0.0] * len(qv)
            denom = (math.sqrt(sum(a * a for a in qv)) * math.sqrt(sum(b * b for b in dv))) or 1.0
            scores.append(sum(a * b for a, b in zip(qv, dv)) / denom)
        return scores


def build_retriever(settings: Settings) -> Retriever:
    if settings.use_mock_backend or not settings.has_azure_search:
        logger.warning("使用内存 Mock 检索（未配置 Azure AI Search）")
        retriever = MockRetriever(settings)
        retriever.load_local()
        return retriever
    return AzureAISearchRetriever(settings)
