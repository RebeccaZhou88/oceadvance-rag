"""集中化配置：通过环境变量加载，未配置 Azure 凭证时自动降级为内存模拟后端。"""
import os
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env 位于 backend/ 目录下，src layout: src/app/config.py → ../../.env
_ENV_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,        # 严格大小写，防止系统环境变量误匹配
        populate_by_name=True,      # 允许用字段名本身（api_key）或 alias（API_KEY）
        protected_namespaces=(),    # 允许 model_base_url / model_name 字段名（pydantic 默认保留 model_ 前缀）
    )

    # Azure OpenAI
    azure_openai_endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(default="", alias="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = Field(default="", alias="AZURE_OPENAI_API_VERSION")
    azure_openai_chat_deployment: str = Field(default="", alias="AZURE_OPENAI_CHAT_DEPLOYMENT")
    azure_openai_embedding_deployment: str = Field(default="", alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    # QWen / DeepSeek / 任何 OpenAI 兼容接口
    # alias 只接受大写环境变量名，DEEPSEEK_API_KEY 不会误匹配到 api_key
    api_key: str = Field(default="", alias="LLM_API_KEY")
    model_base_url: str = Field(default="", alias="LLM_MODEL_BASE_URL")
    model_name: str = Field(default="", alias="LLM_MODEL_NAME")

    # Azure AI Search
    azure_search_endpoint: str = Field(default="", alias="AZURE_SEARCH_ENDPOINT")
    azure_search_api_key: str = Field(default="", alias="AZURE_SEARCH_API_KEY")
    azure_search_index_name: str = Field(default="ops-kb", alias="AZURE_SEARCH_INDEX_NAME")

    # 重排策略
    rerank_strategy: Literal["semantic", "cross-encoder", "llm", "none"] = Field(
        default="llm", alias="RERANK_STRATEGY"
    )

    # 应用
    app_env: str = Field(default="dev", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    use_mock_backend: bool = Field(default=True, alias="USE_MOCK_BACKEND")

    # 告警阈值
    alert_p95_latency_seconds: float = Field(default=5.0, alias="ALERT_P95_LATENCY_SECONDS")
    alert_daily_token_cost_usd: float = Field(default=10.0, alias="ALERT_DAILY_TOKEN_COST_USD")
    alert_hallucination_rate: float = Field(default=0.05, alias="ALERT_HALLUCINATION_RATE")

    # 检索参数
    hybrid_top_k: int = Field(default=3, alias="HYBRID_TOP_K")
    final_top_k: int = Field(default=2, alias="FINAL_TOP_K")
    rrf_k: int = Field(default=60, alias="RRF_K")

    @property
    def has_azure_openai(self) -> bool:
        return bool(self.azure_openai_endpoint and self.azure_openai_api_key)

    @property
    def has_llm(self) -> bool:
        return bool(self.api_key and self.model_base_url)

    @property
    def has_azure_search(self) -> bool:
        return bool(self.azure_search_endpoint and self.azure_search_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
