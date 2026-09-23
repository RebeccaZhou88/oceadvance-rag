"""Azure OpenAI client wrapper, falls back to local Mock LLM when credentials not configured.

Provides unified interface:
- embed(texts) -> list[list[float]]
- chat(messages) -> dict(answer, citations hint, tokens, latency)
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)

# Local fallback vector dimension (only used when embedding API unavailable,
# cannot be used for 1536-dim Azure index)
MOCK_EMBED_DIM = 256


def _mock_embed_one(text: str, dim: int = MOCK_EMBED_DIM) -> list[float]:
    """Deterministic hash vector: fallback when offline/embedding API fails."""
    import math

    vec = [0.0] * dim
    for i, ch in enumerate(text):
        vec[i % dim] += ord(ch) % 97
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class BaseLLMClient:
    """Unified LLM interface."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def chat(
        self, messages: list[dict[str, str]], citations_hint: list[dict] | None = None
    ) -> dict[str, Any]:
        raise NotImplementedError


class AzureOpenAIClient(BaseLLMClient):
    """Real Azure OpenAI client."""

    def __init__(self, settings: Settings) -> None:
        from openai import AsyncAzureOpenAI

        self.settings = settings
        self._client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self._embed_model = settings.azure_openai_embedding_deployment
        self._chat_model = settings.azure_openai_chat_deployment

    async def embed(self, texts: list[str]) -> list[list[float]]:
        logger.info(
            "▶ [Azure OpenAI] embedding call started: deployment=%s, endpoint=%s, %d inputs",
            self._embed_model, self.settings.azure_openai_endpoint, len(texts),
        )
        t0 = time.monotonic()
        try:
            resp = await self._client.embeddings.create(
                model=self._embed_model, input=texts
            )
            vectors = [item.embedding for item in resp.data]
            logger.info(
                "✔ [Azure OpenAI] embedding done: returned %d vectors, dim=%d, elapsed %.2fs",
                len(vectors), len(vectors[0]) if vectors else 0, time.monotonic() - t0,
            )
            return vectors
        except Exception:  # noqa: BLE001
            logger.error(
                "✘ [Azure OpenAI] embedding call failed (deployment=%s, elapsed %.2fs)",
                self._embed_model, time.monotonic() - t0, exc_info=True,
            )
            raise

    async def chat(
        self, messages: list[dict[str, str]], citations_hint: list[dict] | None = None
    ) -> dict[str, Any]:
        start = time.perf_counter()
        resp = await self._client.chat.completions.create(
            model=self._chat_model,
            messages=messages,
            temperature=0.0,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        choice = resp.choices[0].message
        usage = resp.usage
        return {
            "answer": choice.content or "",
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "latency_ms": latency_ms,
        }


class MockLLMClient(BaseLLMClient):
    """Deterministic Mock used locally without Azure credentials, convenient for development and testing."""

    # Simple deterministic embedding: normalize to fixed dimension based on char hash
    EMBED_DIM = MOCK_EMBED_DIM

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_mock_embed_one(t, self.EMBED_DIM) for t in texts]

    async def chat(
        self, messages: list[dict[str, str]], citations_hint: list[dict] | None = None
    ) -> dict[str, Any]:
        # Extract context from system messages (chunks marked with [1] ...)
        context_parts = []
        question = ""
        for m in messages:
            if m["role"] == "system" and "retrieved context" in m.get("content", ""):
                context_parts.append(m["content"])
            elif m["role"] == "user":
                question = m.get("content", "")

        if context_parts:
            context_text = "\n".join(context_parts)
            answer = self._generate_answer(question, context_text, citations_hint or [])
        else:
            # Small talk or no context: respond with heuristics
            if any(c in question for c in ("hello", "hi", "thanks", "thank you", "bye")):
                answer = "Hello! I'm the ops knowledge base assistant. I can help you with questions about operations, Runbooks, Kusto queries, and more."
            else:
                answer = "No relevant knowledge found, unable to answer this question."

        # Rough token estimation
        input_tokens = sum(len(m["content"]) // 4 for m in messages)
        output_tokens = len(answer) // 4
        return {
            "answer": answer,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": 8.0,
        }

    def _generate_answer(self, question: str, context: str, citations: list[dict]) -> str:
        """Real 'read chunks -> synthesize analysis -> compose answer', simulating LLM RAG response."""
        import re
        import json

        # Extract the body of each [N] chunk from context
        chunks: list[tuple[int, str]] = []
        for m in re.finditer(r"\[(\d+)\]\s*\((\w+)\)\s*(.+?)(?=\[\d+\]|\Z)", context, re.DOTALL):
            idx = int(m.group(1))
            text = m.group(3).strip()
            chunks.append((idx, text))

        if not chunks:
            return "Based on the retrieved chunk information, not enough content found to answer this question."

        # Match chunks by keywords, extract relevant sentences
        question_lower = question.lower()
        keywords = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]+", question_lower)
        keywords = [k for k in keywords if k not in ("how", "what", "why", "when", "where", "which", "troubleshoot", "issue", "problem", "solve", "fix", "check", "view", "write")]

        # Find sentences containing keywords from each chunk
        extracted: list[tuple[int, str]] = []
        for idx, text in chunks:
            sentences = re.split(r"(?<=[。！？.!?\n])", text)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if any(k.lower() in sent.lower() for k in keywords):
                    # Take first 120 chars
                    extracted.append((idx, sent[:120].replace("\n", " ")))

        # Compose answer
        lines: list[str] = []

        # 1. Summary paragraph
        if extracted:
            lines.append(f"Based on {len(chunks)} relevant documents retrieved from the ops knowledge base, here is a summary for the question '{question[:40]}':")
            lines.append("")
            # After deduplication, take 3-5 most relevant sentences
            seen = set()
            summary_sents = []
            for idx, sent in extracted:
                key = sent[:30]
                if key not in seen:
                    seen.add(key)
                    summary_sents.append((idx, sent))
            for idx, sent in summary_sents[:4]:
                cat = citations[idx - 1]["category"] if idx - 1 < len(citations) else ""
                lines.append(f"• {sent}[{idx}]  ({cat})")
        else:
            # When no keywords matched, use first 200 chars of the first chunk as overview
            idx, text = chunks[0]
            lines.append(f"For the question '{question[:40]}', synthesizing ops knowledge base info:")
            lines.append("")
            lines.append(f"According to the Runbook documentation, troubleshooting this issue requires step-by-step localization following the standard process[{idx}].")
            # Extract key steps
            steps = re.findall(r"[-*]\s*(.+?)(?:\n|$)", text)
            steps += re.findall(r"\d+\.\s*(.+?)(?:\n|$)", text)
            if steps:
                lines.append("")
                lines.append("## Recommended Troubleshooting Steps")
                for step in steps[:5]:
                    lines.append(f"  {step.strip()}[{idx}]")

        lines.append("")

        # 2. Referenced Kusto code blocks (if any kusto chunks)
        kusto_chunks = [(idx, t) for idx, t in chunks if citations[idx - 1]["category"] == "kusto_templates" if idx - 1 < len(citations)]
        if kusto_chunks:
            idx, text = kusto_chunks[0]
            lines.append("## Related Kusto Query Templates")
            # Extract Kusto statements
            kql_lines = re.findall(r"AppGwLogs.*?(?=\n\s*\n|$)", text, re.DOTALL)
            if not kql_lines:
                kql_lines = [text[:200]]
            for kql in kql_lines[:2]:
                lines.append("```kusto")
                lines.append(kql.strip())
                lines.append("```")
            lines.append(f"[Source: {citations[idx-1]['source']}]")

        lines.append("")
        lines.append("## References")
        for idx, text in chunks:
            if idx - 1 < len(citations):
                src = citations[idx - 1]["source"]
                cat = citations[idx - 1]["category"]
                lines.append(f"[{idx}] {src} ({cat})")

        return "\n".join(lines)


class LLMClient(BaseLLMClient):
    """Call QWen or DeepSeek via OpenAI-compatible API."""

    def __init__(self, settings: Settings) -> None:
        from langchain_openai import ChatOpenAI
        from openai import AsyncOpenAI

        self.settings = settings
        self._llm = ChatOpenAI(
            model=settings.model_name,
            api_key=settings.api_key,
            base_url=settings.model_base_url,
            temperature=0.0,
        )
        # embeddings API on the same OpenAI-compatible endpoint (DashScope text-embedding-v2, etc.)
        self._embed_client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.model_base_url,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI-compatible embeddings API to generate real vectors.

        On API failure (no such model / no quota / offline), fall back to
        deterministic hash vector and warn, ensuring local mock retrieval still
        works; Azure upload path will be intercepted by dimension validation in
        the indexer.
        """
        logger.info(
            "▶ embedding call started: model=%s, endpoint=%s, %d input texts",
            self.settings.embedding_model_name,
            self.settings.model_base_url,
            len(texts),
        )
        t0 = time.monotonic()
        try:
            resp = await self._embed_client.embeddings.create(
                model=self.settings.embedding_model_name,
                input=list(texts),
            )
            vectors = [item.embedding for item in resp.data]
            logger.info(
                "✔ embedding done: returned %d vectors, dim=%d, elapsed %.2fs",
                len(vectors),
                len(vectors[0]) if vectors else 0,
                time.monotonic() - t0,
            )
            return vectors
        except Exception:  # noqa: BLE001
            logger.error(
                "✘ embedding API call failed (model=%s, elapsed %.2fs), falling back to %d-dim hash vector; "
                "to upload to Azure, check LLM_EMBEDDING_MODEL_NAME/quota",
                self.settings.embedding_model_name,
                time.monotonic() - t0,
                MOCK_EMBED_DIM,
                exc_info=True,
            )
            return [_mock_embed_one(t) for t in texts]

    async def chat(
        self, messages: list[dict[str, str]], citations_hint: list[dict] | None = None
    ) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        start = time.perf_counter()
        lc_msgs = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                lc_msgs.append(SystemMessage(content=content))
            elif role == "user":
                lc_msgs.append(HumanMessage(content=content))
            else:
                lc_msgs.append(HumanMessage(content=content))

        resp = await self._llm.ainvoke(lc_msgs)
        latency_ms = (time.perf_counter() - start) * 1000

        answer = resp.content if hasattr(resp, "content") else str(resp)
        # Try to get tokens from usage_metadata
        usage = getattr(resp, "usage_metadata", None) or {}
        return {
            "answer": answer or "",
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "latency_ms": latency_ms,
        }


def build_embed_client(settings: Settings) -> BaseLLMClient:
    """Factory: embedding-specific client.

    Priority (vector write and query must share the same source):
    1. Azure OpenAI has embedding deployment -> use Azure (your configured ada-002 = 1536 dim)
    2. OpenAI-compatible endpoint (DashScope text-embedding-v2 = 1536 dim)
    3. Mock (256-dim hash, for local debugging, cannot upload to Azure)
    """
    if settings.has_azure_openai and settings.azure_openai_embedding_deployment:
        logger.info(
            "[Embed] Azure OpenAI | endpoint=%s | deployment=%s",
            settings.azure_openai_endpoint, settings.azure_openai_embedding_deployment,
        )
        return AzureOpenAIClient(settings)
    if settings.has_llm:
        logger.info(
            "[Embed] OpenAI-compatible endpoint | base=%s | model=%s",
            settings.model_base_url, settings.embedding_model_name,
        )
        return LLMClient(settings)
    logger.warning("[Embed] falling back to Mock LLM (256-dim hash, cannot upload to Azure)")
    return MockLLMClient()


def build_chat_client(settings: Settings) -> BaseLLMClient:
    """Factory: chat-specific client.

    Priority (independent of embedding pipeline):
    1. OpenAI-compatible endpoint (DashScope qwen -- has free quota)
    2. Azure OpenAI (must configure chat deployment)
    3. Mock (pure heuristic response)
    """
    if settings.has_llm:
        logger.info(
            "[Chat] OpenAI-compatible endpoint | model=%s | base=%s",
            settings.model_name, settings.model_base_url,
        )
        return LLMClient(settings)
    if settings.has_azure_openai and settings.azure_openai_chat_deployment:
        logger.info(
            "[Chat] Azure OpenAI | deployment=%s", settings.azure_openai_chat_deployment,
        )
        return AzureOpenAIClient(settings)
    logger.warning("[Chat] falling back to Mock LLM (heuristic response)")
    return MockLLMClient()


def build_llm_client(settings: Settings) -> BaseLLMClient:
    """Default factory for backward compatibility (embed takes priority, chat pipeline automatically uses the same client).

    New code should explicitly choose build_embed_client / build_chat_client.
    """
    return build_embed_client(settings)


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]
