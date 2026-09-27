# @Author: RebeccaZhou
# @Description: Permission filter: OData allowed_groups injection + leak assertion
#              权限过滤：OData allowed_groups 注入与泄露断言

"""Document-level permission filtering based on user group.

Design doc 3.3:
- Index field allowed_groups stores ["sre","dev","ops"]
- Inject user group via OData filter during retrieval
- Filter at retrieval stage to prevent unauthorized content from entering LLM context
"""
from __future__ import annotations

from typing import Any


def build_search_filter(user_group: str) -> str:
    """Construct Azure AI Search OData filter, injecting user group permission filtering.

    Equivalent to: allowed_groups/any(g: g eq '<user_group>')
    """
    safe_group = user_group.replace("'", "''")
    return f"allowed_groups/any(g: g eq '{safe_group}')"


def filter_docs_by_group(docs: list[dict[str, Any]], user_group: str) -> list[dict[str, Any]]:
    """In-memory fallback permission filtering (used by Mock backend or local retrieval)."""
    allowed: list[dict[str, Any]] = []
    for doc in docs:
        groups = doc.get("allowed_groups") or []
        if not groups or user_group in groups:
            # No allowed_groups means public
            allowed.append(doc)
    return allowed


def assert_no_leak(docs: list[dict[str, Any]], user_group: str) -> None:
    """Assert returned results contain no unauthorized documents, for self-check and testing."""
    leaked = [
        d for d in docs
        if d.get("allowed_groups") and user_group not in d["allowed_groups"]
    ]
    if leaked:
        raise PermissionError(
            f"Permission filtering failed: found {len(leaked)} unauthorized documents about to enter LLM context"
        )
