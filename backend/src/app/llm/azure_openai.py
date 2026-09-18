"""Azure OpenAI 客户端封装，未配置凭证时降级为本地 Mock LLM。

提供统一接口：
- embed(texts) -> list[list[float]]
- chat(messages) -> dict(answer, citations 提示, tokens, latency)
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)


class BaseLLMClient:
    """统一 LLM 接口。"""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def chat(
        self, messages: list[dict[str, str]], citations_hint: list[dict] | None = None
    ) -> dict[str, Any]:
        raise NotImplementedError


class AzureOpenAIClient(BaseLLMClient):
    """真实 Azure OpenAI 客户端。"""

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
        resp = await self._client.embeddings.create(
            model=self._embed_model, input=texts
        )
        return [item.embedding for item in resp.data]

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
    """本地无 Azure 凭证时使用的确定性 Mock，便于开发与测试。"""

    # 简单确定性嵌入：基于字符 hash 归一化到固定维度
    EMBED_DIM = 256

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import math

        results: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.EMBED_DIM
            for i, ch in enumerate(text):
                vec[i % self.EMBED_DIM] += ord(ch) % 97
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            results.append([v / norm for v in vec])
        return results

    async def chat(
        self, messages: list[dict[str, str]], citations_hint: list[dict] | None = None
    ) -> dict[str, Any]:
        # 从 system messages 中提取上下文(含 [1] ... 标记的片段)
        context_parts = []
        question = ""
        for m in messages:
            if m["role"] == "system" and "检索到的上下文" in m.get("content", ""):
                context_parts.append(m["content"])
            elif m["role"] == "user":
                question = m.get("content", "")

        if context_parts:
            context_text = "\n".join(context_parts)
            answer = self._generate_answer(question, context_text, citations_hint or [])
        else:
            # 闲聊或无上下文:用启发式回应
            if any(c in question for c in ("你好", "谢谢", "再见")):
                answer = "你好！我是运维知识库助手，可以帮你解答运维、Runbook、Kusto 查询等相关问题。"
            else:
                answer = "未找到相关知识，无法回答该问题。"

        # 粗略 token 估算
        input_tokens = sum(len(m["content"]) // 4 for m in messages)
        output_tokens = len(answer) // 4
        return {
            "answer": answer,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": 8.0,
        }

    def _generate_answer(self, question: str, context: str, citations: list[dict]) -> str:
        """真正的"读片段→综合分析→组织答案",模拟 LLM 的 RAG 回答。"""
        import re
        import json

        # 从 context 中提取每个 [N] 片段的正文
        chunks: list[tuple[int, str]] = []
        for m in re.finditer(r"\[(\d+)\]\s*\((\w+)\)\s*(.+?)(?=\[\d+\]|\Z)", context, re.DOTALL):
            idx = int(m.group(1))
            text = m.group(3).strip()
            chunks.append((idx, text))

        if not chunks:
            return "综合检索到的片段信息,未找到足够内容回答该问题。"

        # 按关键词匹配片段,提取相关句
        question_lower = question.lower()
        keywords = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]+", question_lower)
        keywords = [k for k in keywords if k not in ("如何", "怎么", "什么", "排查", "问题", "解决", "查看", "写")]

        # 从每个片段中找含关键词的句子
        extracted: list[tuple[int, str]] = []
        for idx, text in chunks:
            sentences = re.split(r"(?<=[。！？.!?\n])", text)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if any(k.lower() in sent.lower() for k in keywords):
                    # 取前 120 字
                    extracted.append((idx, sent[:120].replace("\n", " ")))

        # 组织答案
        lines: list[str] = []

        # 1. 总结段
        if extracted:
            lines.append(f"根据运维知识库检索到的 {len(chunks)} 条相关文档,针对「{question[:40]}」问题,总结如下:")
            lines.append("")
            # 去重后取 3-5 条最相关句子
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
            # 没匹配到关键词时,用第一个片段的前 200 字做概述
            idx, text = chunks[0]
            lines.append(f"针对「{question[:40]}」问题,综合运维知识库信息:")
            lines.append("")
            lines.append(f"从 Runbook 文档中可知,该问题的排查需要按标准流程逐步定位[{idx}]。")
            # 提取关键步骤
            steps = re.findall(r"[-*]\s*(.+?)(?:\n|$)", text)
            steps += re.findall(r"\d+\.\s*(.+?)(?:\n|$)", text)
            if steps:
                lines.append("")
                lines.append("## 推荐排查步骤")
                for step in steps[:5]:
                    lines.append(f"  {step.strip()}[{idx}]")

        lines.append("")

        # 2. 引用的 Kusto 代码块(如果有 kusto 片段)
        kusto_chunks = [(idx, t) for idx, t in chunks if citations[idx - 1]["category"] == "kusto_templates" if idx - 1 < len(citations)]
        if kusto_chunks:
            idx, text = kusto_chunks[0]
            lines.append("## 相关 Kusto 查询模板")
            # 提取 Kusto 语句
            kql_lines = re.findall(r"AppGwLogs.*?(?=\n\s*\n|$)", text, re.DOTALL)
            if not kql_lines:
                kql_lines = [text[:200]]
            for kql in kql_lines[:2]:
                lines.append("```kusto")
                lines.append(kql.strip())
                lines.append("```")
            lines.append(f"[来源 {citations[idx-1]['source']}]")

        lines.append("")
        lines.append("## 参考来源")
        for idx, text in chunks:
            if idx - 1 < len(citations):
                src = citations[idx - 1]["source"]
                cat = citations[idx - 1]["category"]
                lines.append(f"[{idx}] {src} ({cat})")

        return "\n".join(lines)


class LLMClient(BaseLLMClient):
    """通过 OpenAI 兼容接口调用 QWen 或 DeepSeek。"""

    def __init__(self, settings: Settings) -> None:
        from langchain_openai import ChatOpenAI

        self.settings = settings
        self._llm = ChatOpenAI(
            model=settings.model_name,
            api_key=settings.api_key,
            base_url=settings.model_base_url,
            temperature=0.0,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """DeepSeek 无 embedding 接口,回退 Mock 嵌入。"""
        import math
        dim = 256
        results: list[list[float]] = []
        for text in texts:
            vec = [0.0] * dim
            for i, ch in enumerate(text):
                vec[i % dim] += ord(ch) % 97
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            results.append([v / norm for v in vec])
        return results

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
        # 尝试从 usage_metadata 取 token
        usage = getattr(resp, "usage_metadata", None) or {}
        return {
            "answer": answer or "",
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "latency_ms": latency_ms,
        }


def build_llm_client(settings: Settings) -> BaseLLMClient:
    """工厂方法:DeepSeek/QWen > Azure > Mock。"""
    logger.error("settings: %s", settings)
    if settings.has_llm:
        logger.info("使用 QWen or DeepSeek LLM (model=%s)", settings.model_name)
        return LLMClient(settings)
    if settings.has_azure_openai:
        logger.info("使用 Azure OpenAI")
        return AzureOpenAIClient(settings)
    logger.warning("使用 Mock LLM(未配置任何 LLM 凭证)")
    return MockLLMClient()


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]
