"""Exact cache (single-turn version).

Key:   SHA-1 hash of the normalized user question, prefixed with cache version
       and user_group for permission isolation.
Value: JSON containing answer, citations, kb_version, created_at.

On a cache hit the full RAG pipeline (retrieval → rerank → generate) is skipped,
reducing latency and token cost to near zero.

TTL is tiered by content type (real-time / general / static / reference) with a
sensible default. The in-memory store can be swapped for Redis without changing
the public interface.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# TTL by content type (seconds)
TTL_MAP: dict[str, int] = {
    "realtime": 300,       # 5 min  – dashboards / status pages
    "general": 3600,       # 1 hour – general Q&A (default)
    "static": 86400,       # 1 day  – runbooks / postmortems
    "reference": 604800,   # 7 days – API docs / architecture
}
DEFAULT_TTL = 3600


def normalize(q: str) -> str:
    """Normalize a question: lowercase, strip punctuation, collapse whitespace."""
    q = q.strip().lower()
    q = re.sub(r"[^\w\s]", "", q)   # remove punctuation
    q = re.sub(r"\s+", " ", q)      # collapse whitespace
    return q


def cache_key(question: str, user_group: str, kb_version: str) -> str:
    """Build the cache key: qa:{version}:{user_group}:exact:{sha1}."""
    h = hashlib.sha1(normalize(question).encode()).hexdigest()
    return f"qa:{kb_version}:{user_group}:exact:{h}"


class ExactCache:
    """In-memory exact cache with TTL expiry.

    For production, replace the internal dict with a Redis client — the public
    API (get / set / stats) stays the same.
    """

    def __init__(self, kb_version: str = "v1") -> None:
        self._kb_version = kb_version
        # key -> (json_value_str, expiry_timestamp)
        self._store: dict[str, tuple[str, float]] = {}
        # stats
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, question: str, user_group: str) -> dict[str, Any] | None:
        """Return the cached value dict, or None on miss / expiry."""
        key = cache_key(question, user_group, self._kb_version)
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        value_str, expires_at = entry
        if time.time() > expires_at:
            # Lazy eviction
            del self._store[key]
            self._misses += 1
            return None
        self._hits += 1
        return json.loads(value_str)

    def set(
        self,
        question: str,
        user_group: str,
        answer: str,
        citations: list[dict[str, Any]],
        content_type: str = "general",
    ) -> None:
        """Write a cache entry. TTL is chosen by content_type."""
        key = cache_key(question, user_group, self._kb_version)
        value = json.dumps({
            "answer": answer,
            "citations": citations,
            "kb_version": self._kb_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        ttl = TTL_MAP.get(content_type, DEFAULT_TTL)
        self._store[key] = (value, time.time() + ttl)
        logger.info(
            "[exact_cache] WRITE key=%s | ttl=%ds | content_type=%s",
            key[-12:], ttl, content_type,
        )

    def stats(self) -> dict[str, int]:
        """Return hit / miss / size counters."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._store),
        }

    def clear(self) -> None:
        """Drop all cached entries (used when kb_version changes)."""
        self._store.clear()
        self._hits = 0
        self._misses = 0
        logger.info("[exact_cache] cache cleared")
