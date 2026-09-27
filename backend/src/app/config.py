# @Author: RebeccaZhou
# @Description: Pydantic-settings configuration: all environment variables typed
#              Pydantic-settings 配置：全部环境变量类型化定义

"""Centralized configuration: loaded from environment variables, automatically falls back to an in-memory mock backend when Azure credentials are not configured."""
import os
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env sits under backend/, src layout: src/app/config.py -> ../../.env
_ENV_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,        # strict case, prevent accidental matches with system env vars
        populate_by_name=True,      # allow using the field name (api_key) or its alias (API_KEY)
        protected_namespaces=(),    # allow model_base_url / model_name field names (pydantic reserves model_ prefix by default)
    )

    # Azure OpenAI
    azure_openai_endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(default="", alias="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = Field(default="", alias="AZURE_OPENAI_API_VERSION")
    azure_openai_chat_deployment: str = Field(default="", alias="AZURE_OPENAI_CHAT_DEPLOYMENT")
    azure_openai_embedding_deployment: str = Field(default="", alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    # QWen / DeepSeek / any OpenAI-compatible endpoint
    # alias only accepts uppercase env var names, so DEEPSEEK_API_KEY won't accidentally match api_key
    api_key: str = Field(default="", alias="LLM_API_KEY")
    model_base_url: str = Field(default="", alias="LLM_MODEL_BASE_URL")
    model_name: str = Field(default="", alias="LLM_MODEL_NAME")
    # ================================================================
    # Vector dimension iron rule: writes and queries MUST use the same embedding model
    # Currently Azure OpenAI ada-002 (1536 dim) is the active main path
    #   -> AZURE_OPENAI_EMBEDDING_DEPLOYMENT (.env) is what's actually used
    #   -> the two fields below only serve as a fallback when Azure OpenAI is not configured
    # ================================================================
    # Fallback embedding model (only effective when Azure OpenAI is unavailable, via OpenAI-compatible endpoint)
    embedding_model_name: str = Field(default="text-embedding-v2", alias="LLM_EMBEDDING_MODEL_NAME")
    # Azure AI Search index vector dimension (used by create_index.py; consistency check before upload/query)
    # Changing this value REQUIRES rebuilding the index + re-running the indexer, otherwise dimension mismatch will raise errors
    embedding_dimensions: int = Field(default=1536, alias="LLM_EMBEDDING_DIMENSIONS")

    # Azure AI Search
    azure_search_endpoint: str = Field(default="", alias="AZURE_SEARCH_ENDPOINT")
    azure_search_api_key: str = Field(default="", alias="AZURE_SEARCH_API_KEY")
    azure_search_index_name: str = Field(default="ops-kb", alias="AZURE_SEARCH_INDEX_NAME")
    # vector: 纯向量检索（当前默认，中文下 BM25 简单解析器效果差）
    # hybrid: BM25 关键词 + Vector 向量 + RRF 融合（英文/中英文混合知识库可试）
    azure_search_mode: Literal["vector", "hybrid"] = Field(
        default="hybrid", alias="AZURE_SEARCH_MODE"
    )

    # Rerank strategy
    rerank_strategy: Literal["semantic", "cross-encoder", "llm", "none"] = Field(
        default="llm", alias="RERANK_STRATEGY"
    )

    # App
    app_env: str = Field(default="dev", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    use_mock_backend: bool = Field(default=False, alias="USE_MOCK_BACKEND")

    # Online evaluation (EvaluateAnswerNode adds one extra LLM self-eval call, ~15-20s)
    enable_online_eval: bool = Field(default=True, alias="ENABLE_ONLINE_EVAL")

    # Retrieval "effective document" threshold: Azure @search.score > this value counts as effective
    # Note: Azure vector search @search.score = (1 + cosine)/2, 0.5 means completely irrelevant,
    # and ada-002 is anisotropic (irrelevant text often lands at 0.75-0.80).
    # Empirically 0.82 (~cosine 0.64) gives the best separation: chunks from the matching document pass, irrelevant chunks are blocked.
    retrieval_effective_threshold: float = Field(default=0.82, alias="RETRIEVAL_EFFECTIVE_THRESHOLD")
    # Hybrid (BM25 + Vector + RRF) 模式下 Azure 返回 @search.score 是 RRF 融合分数，量级 0.01-0.05
    # 0.02 通常可区分"确实切题的融合结果"和"关键词/向量单边弱召回"
    hybrid_effective_threshold: float = Field(default=0.02, alias="RETRIEVAL_EFFECTIVE_THRESHOLD_HYBRID")

    # Alert thresholds
    alert_p95_latency_seconds: float = Field(default=5.0, alias="ALERT_P95_LATENCY_SECONDS")
    alert_daily_token_cost_usd: float = Field(default=10.0, alias="ALERT_DAILY_TOKEN_COST_USD")
    alert_hallucination_rate: float = Field(default=0.05, alias="ALERT_HALLUCINATION_RATE")

    # Retrieval parameters
    hybrid_top_k: int = Field(default=20, alias="HYBRID_TOP_K")    # Azure recall pool size (candidates before rerank)
    final_top_k: int = Field(default=4, alias="FINAL_TOP_K")       # final snippets kept for the LLM after rerank
    rrf_k: int = Field(default=60, alias="RRF_K")                   # RRF fusion parameter (larger k = closer to original rank)

    # Exact cache (single-turn): skip full RAG on repeated questions
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_version: str = Field(default="v2", alias="CACHE_VERSION")

    # Knowledge-base quality governance (CrewAI multi-agent, runs async in background)
    governance_enabled: bool = Field(default=True, alias="GOVERNANCE_ENABLED")
    governance_db_path: str = Field(default="", alias="GOVERNANCE_DB_PATH")
    # Evaluation score threshold (1-5 scale; below threshold triggers governance)
    governance_min_faithfulness: float = Field(default=3.5, alias="GOVERNANCE_MIN_FAITHFULNESS")
    governance_min_relevancy: float = Field(default=3.5, alias="GOVERNANCE_MIN_RELEVANCY")
    governance_max_hallucination: float = Field(default=0.3, alias="GOVERNANCE_MAX_HALLUCINATION")

    @property
    def has_azure_openai(self) -> bool:
        return bool(self.azure_openai_endpoint and self.azure_openai_api_key)

    @property
    def has_llm(self) -> bool:
        return bool(self.api_key and self.model_base_url)

    @property
    def has_azure_search(self) -> bool:
        return bool(self.azure_search_endpoint and self.azure_search_api_key)

    @property
    def effective_threshold(self) -> float:
        """按当前检索模式返回正确的 effective_threshold。

        vector 模式下用 retrieval_effective_threshold（默认 0.82，对应向量分数），
        hybrid 模式下用 hybrid_effective_threshold（默认 0.02，对应 RRF 融合分数）。
        """
        if self.azure_search_mode == "hybrid":
            return self.hybrid_effective_threshold
        return self.retrieval_effective_threshold

    @property
    def governance_db_file(self) -> str:
        """Absolute path to the governance task SQLite DB, default backend/data/governance/tasks.db."""
        if self.governance_db_path:
            return self.governance_db_path
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "governance", "tasks.db")
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
