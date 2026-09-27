# @Author: RebeccaZhou
# @Description: One-off Azure AI Search index creation with vector config
#              Azure AI Search 索引创建（一次性）：含向量字段配置

"""Create/rebuild Azure AI Search index schema.

Fields:
  id / content / content_vector / source / category
  / allowed_groups (permission filter) / last_updated

Usage:
  python -m ingestion.create_index               # error if already exists
  python -m ingestion.create_index --recreate     # force delete old index then rebuild
"""
from __future__ import annotations

import argparse
import asyncio
import logging

import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)5s] %(message)s",
    datefmt="%H:%M:%S",
)

from app.config import get_settings

logger = logging.getLogger(__name__)


async def create_index(recreate: bool = False) -> None:
    settings = get_settings()
    if not settings.has_azure_search:
        logger.warning("Azure Search credentials not configured, skipping.")
        return
    logger.info("══ [create_index] endpoint=%s | index=%s | vector_dim=%d",
                settings.azure_search_endpoint,
                settings.azure_search_index_name,
                settings.embedding_dimensions)

    from azure.search.documents.indexes.aio import SearchIndexClient
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        VectorSearch,
        VectorSearchProfile,
    )
    from azure.core.credentials import AzureKeyCredential

    async with SearchIndexClient(
        endpoint=settings.azure_search_endpoint,
        credential=AzureKeyCredential(settings.azure_search_api_key),
    ) as client:
        # 1. Check existing index
        existing = None
        try:
            existing = await client.get_index(settings.azure_search_index_name)
        except Exception:
            existing = None

        if existing and recreate:
            logger.warning("⚠️  Old index already exists (%d fields), will delete then rebuild...", len(existing.fields))
            await client.delete_index(settings.azure_search_index_name)
            logger.info("  ✔ Old index deleted")
        elif existing:
            logger.error(
                "✘ Index %s already exists, add --recreate to force rebuild, or manually delete in Azure Portal",
                settings.azure_search_index_name,
            )
            return

        # 2. Build schema
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
            profiles=[VectorSearchProfile(name="default-profile", algorithm_configuration_name="hnsw-config")],
        )
        fields = [
            SearchField(name="id", type="Edm.String", key=True),
            SearchField(name="content", type="Edm.String", searchable=True, filterable=True),
            SearchField(
                name="content_vector",
                type="Collection(Edm.Single)",
                searchable=True,
                vector_search_dimensions=settings.embedding_dimensions,
                vector_search_profile_name="default-profile",
            ),
            SearchField(name="source", type="Edm.String", filterable=True, retrievable=True),
            SearchField(name="category", type="Edm.String", filterable=True, facetable=True),
            SearchField(
                name="allowed_groups",
                type="Collection(Edm.String)",
                filterable=True,
            ),
            SearchField(name="last_updated", type="Edm.DateTimeOffset", filterable=True, sortable=True),
        ]
        index = SearchIndex(
            name=settings.azure_search_index_name,
            fields=fields,
            vector_search=vector_search,
        )

        # 3. Create
        logger.info("▶ Creating index schema (%d fields, HNSW profile, vector dim=%d)",
                     len(fields), settings.embedding_dimensions)
        await client.create_index(index)
        logger.info("✔ Index %s created", settings.azure_search_index_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Azure AI Search index")
    parser.add_argument("--recreate", "-r", action="store_true",
                        help="Force delete old index then rebuild (will clear existing docs)")
    args = parser.parse_args()
    asyncio.run(create_index(recreate=args.recreate))


if __name__ == "__main__":
    main()
