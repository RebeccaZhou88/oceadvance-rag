"""Chunking 策略（设计文档 3.2）。

- Markdown 按标题层级（H2/H3）切分
- 代码块（Kusto）单独切分，保留上下文注释
- chunk size 512 tokens，overlap 50 tokens
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


@dataclass
class Chunk:
    id: str
    content: str
    source: str
    category: str
    heading: str = ""
    allowed_groups: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _token_len(text: str) -> int:
    # 粗略 token 估算：英文按词，中文按字
    return len(re.findall(r"\w+|[^\s\w]", text))


def _sliding_window(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    tokens = text.split()
    if len(tokens) <= size:
        return [text] if text.strip() else []
    out: list[str] = []
    step = max(size - overlap, 1)
    for i in range(0, len(tokens), step):
        piece = " ".join(tokens[i:i + size])
        if piece.strip():
            out.append(piece)
        if i + size >= len(tokens):
            break
    return out


def _chunk_markdown(md: str, source: str, category: str, allowed_groups: list[str]) -> list[Chunk]:
    """按 H2/H3 切分，再对过长段落做滑动窗口。"""
    chunks: list[Chunk] = []
    # 先保护代码块
    code_blocks: list[str] = []
    def _stash(m: re.Match) -> str:
        code_blocks.append(m.group(0))
        return f"@@CODEBLOCK{len(code_blocks) - 1}@@"

    protected = re.sub(r"```[\s\S]*?```", _stash, md)

    current_heading = ""
    for part in re.split(r"(?m)^(#{2,3}\s.*)$", protected):
        if not part.strip():
            continue
        if re.match(r"^#{2,3}\s", part):
            current_heading = part.strip()
            continue
        # 还原代码块
        def _restore(m: re.Match) -> str:
            return code_blocks[int(m.group(1))]

        restored = re.sub(r"@@CODEBLOCK(\d+)@@", _restore, part)
        for piece in _sliding_window(restored):
            chunks.append(Chunk(
                id=_make_id(source, current_heading, piece),
                content=f"{current_heading}\n{piece}" if current_heading else piece,
                source=source,
                category=category,
                heading=current_heading,
                allowed_groups=list(allowed_groups),
                metadata={"token_len": _token_len(piece)},
            ))
    return chunks


def _chunk_kusto(kql: str, source: str, allowed_groups: list[str]) -> list[Chunk]:
    """Kusto 单独切分，保留注释上下文。"""
    chunks: list[Chunk] = []
    statements = re.split(r"(?<=;)\s*\n", kql)
    buf = ""
    for stmt in statements:
        if _token_len(buf + stmt) > CHUNK_SIZE and buf:
            chunks.append(Chunk(
                id=_make_id(source, "kusto", buf),
                content=buf.strip(),
                source=source,
                category="kusto",
                heading="kusto",
                allowed_groups=list(allowed_groups),
                metadata={"token_len": _token_len(buf)},
            ))
            buf = stmt
        else:
            buf += "\n" + stmt
    if buf.strip():
        chunks.append(Chunk(
            id=_make_id(source, "kusto", buf),
            content=buf.strip(),
            source=source,
            category="kusto",
            heading="kusto",
            allowed_groups=list(allowed_groups),
        ))
    return chunks


def _make_id(source: str, heading: str, content: str) -> str:
    raw = f"{source}:{heading}:{content[:64]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def chunk_document(content: str, source: str, category: str, allowed_groups: list[str] | None = None) -> list[Chunk]:
    groups = allowed_groups or ["sre", "dev", "ops"]
    if category == "kusto" or source.endswith(".kql"):
        return _chunk_kusto(content, source, groups)
    return _chunk_markdown(content, source, category, groups)
