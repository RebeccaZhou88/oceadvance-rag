"""基于用户组的文档级权限过滤。

设计文档 3.3：
- 索引字段 allowed_groups 存储 ["sre","dev","ops"]
- 检索时通过 OData filter 注入用户组
- 在检索阶段即过滤，避免无权限内容进入 LLM 上下文
"""
from __future__ import annotations

from typing import Any


def build_search_filter(user_group: str) -> str:
    """构造 Azure AI Search OData filter，注入用户组权限过滤。

    等价于：allowed_groups/any(g: g eq '<user_group>')
    """
    safe_group = user_group.replace("'", "''")
    return f"allowed_groups/any(g: g eq '{safe_group}')"


def filter_docs_by_group(docs: list[dict[str, Any]], user_group: str) -> list[dict[str, Any]]:
    """内存侧兜底权限过滤（Mock 后端或本地检索使用）。"""
    allowed: list[dict[str, Any]] = []
    for doc in docs:
        groups = doc.get("allowed_groups") or []
        if not groups or user_group in groups:
            # 无 allowed_groups 视为公开
            allowed.append(doc)
    return allowed


def assert_no_leak(docs: list[dict[str, Any]], user_group: str) -> None:
    """断言返回结果不含越权文档，用于自检与测试。"""
    leaked = [
        d for d in docs
        if d.get("allowed_groups") and user_group not in d["allowed_groups"]
    ]
    if leaked:
        raise PermissionError(
            f"权限过滤失效：发现 {len(leaked)} 篇越权文档将进入 LLM 上下文"
        )
