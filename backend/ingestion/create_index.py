"""创建 Azure AI Search 索引 schema（设计文档 3.2）。

字段：
  id / content / content_vector / source / category
  / allowed_groups（权限过滤）/ last_updated

用法：
  python -m ingestion.create_index
"""
from __future__ import annotations

import asyncio

from app.config import get_settings


async def create_index() -> None:
    settings = get_settings()
    if not settings.has_azure_search:
        print("[create_index] 未配置 Azure Search 凭证，跳过。")
        return

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

    client = SearchIndexClient(
        endpoint=settings.azure_search_endpoint,
        credential=AzureKeyCredential(settings.azure_search_api_key),
    )
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
            vector_search_dimensions=1536,
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
    await client.create_index(index)
    print(f"[create_index] 索引 {settings.azure_search_index_name} 已创建")
    await client.close()


def main() -> None:
    asyncio.run(create_index())


if __name__ == "__main__":
    main()
