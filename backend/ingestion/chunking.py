# @Author: RebeccaZhou
# @Description: Markdown/KQL document chunking with title-path context
#              Markdown/KQL 文档分块：继承标题路径上下文

"""Chunking strategy v2 — semantic preservation first.

Core design:
  1. Full heading path inheritance (H1 → H2 → H3): each chunk content prefixed with hierarchical path
     so ada-002 embedding and LLM can understand the fragment's position in the document
  2. Skip pure heading blocks (< 50 chars and only heading with no content)
  3. Merge adjacent short chunks: same document + total content < MAX_CHUNK_CHARS
     (Chinese scenario controlled by char count not token count, since Chinese has no space tokenization)
  4. True sliding window (by char count) only triggered on super long paragraphs
  5. Kusto keeps independent strategy (split by statement)

Why not use LangChain's MarkdownHeaderTextSplitter:
  - Its token estimation is rough for Chinese (splits by word)
  - It treats each heading as independent chunk, no adjacent merging
  - Our needs are more customized: Ops Runbook (step-based) vs Postmortem (timeline-based) vs Kusto (code-based)
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

# ═══════════════════════════════════════════════════════════════════
# ║ Chunking params (Chinese estimated by char count, closer to     ║
# ║ actual chunk length than token)                                  ║
# ║                                                                  ║
# ║ Target range: 300-600 chars/chunk → ada-002 embedding semantic  ║
# ║          stable, LLM rerank & answer gen get complete fragments  ║
# ╚══════════════════════════════════════════════════════════════════

# Target max chunk char count (sliding window trigger threshold; also adjacent merge upper bound)
MAX_CHUNK_CHARS = 500
# Sliding window overlap (char count)
OVERLAP_CHARS = 80
# Pure heading block threshold: below this deemed pure heading, skipped
MIN_CHUNK_CHARS = 50


@dataclass
class Chunk:
    id: str
    content: str
    source: str
    category: str
    heading: str = ""            # current chunk's main heading (full path)
    heading_path: list[str] = field(default_factory=list)  # [H1, H2, H3]
    allowed_groups: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _make_id(source: str, heading_path: list[str], content: str) -> str:
    raw = f"{source}:{'/'.join(heading_path) if heading_path else '-'}:{content[:64]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _sliding_window_chars(text: str, max_chars: int = MAX_CHUNK_CHARS,
                           overlap: int = OVERLAP_CHARS) -> list[str]:
    """Sliding window by char count (compatible with Chinese & English)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    step = max(max_chars - overlap, 50)
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = end - overlap
    return out


# ═══════════════════════════════════════════════════════════════════
# Markdown chunking
# ═══════════════════════════════════════════════════════════════════

def _chunk_markdown(md: str, source: str, category: str,
                    allowed_groups: list[str]) -> list[Chunk]:
    """Split by heading level → inherit full path → skip pure headings → merge adjacent short blocks."""

    # ── Step 1: Protect code blocks ──
    code_blocks: list[str] = []

    def _stash(m: re.Match) -> str:
        code_blocks.append(m.group(0))
        return f"@@CODEBLOCK{len(code_blocks) - 1}@@"

    protected = re.sub(r"```[\s\S]*?```", _stash, md)

    # ── Step 2: Split by all headings (H1/H2/H3/... all captured, build path) ──
    # Capture group 1 = level symbols (## etc), group 2 = heading text
    heading_re = re.compile(r"(?m)^(#{1,6})\s+(.+)$")

    segments: list[tuple[list[str], str]] = []  # (heading_path, body_text)
    current_path: list[str] = []
    current_body: list[str] = []

    def _flush() -> None:
        body = "\n".join(current_body).strip()
        if body:
            segments.append((list(current_path), body))

    for line in protected.split("\n"):
        m = heading_re.match(line.strip())
        if m:
            # Heading hit: flush previously accumulated body first
            _flush()
            current_body = []
            hashes = m.group(1)
            level = len(hashes)      # 1=H1, 2=H2, ...
            title_text = m.group(2).strip()
            # Update path: cut off same-level or deeper, then append
            current_path = current_path[:level - 1] + [title_text]
        else:
            current_body.append(line)
    _flush()

    # ── Step 3: Restore code blocks + skip pure heading blocks + build initial chunks ──
    def _restore(text: str) -> str:
        return re.sub(r"@@CODEBLOCK(\d+)@@",
                      lambda mm: code_blocks[int(mm.group(1))], text)

    raw_chunks: list[Chunk] = []
    for path, body in segments:
        content = _restore(body).strip()
        if not content:
            continue
        # Pure heading block: content < MIN_CHUNK_CHARS and only one heading in path → skip
        if len(content) < MIN_CHUNK_CHARS and len(path) <= 1:
            continue
        # Super long paragraph: sliding window
        for piece in _sliding_window_chars(content):
            raw_chunks.append(Chunk(
                id="",  # generated uniformly later
                content=_build_content_with_path(path, piece),
                source=source,
                category=category,
                heading=path[-1] if path else "",
                heading_path=list(path),
                allowed_groups=list(allowed_groups),
                metadata={"char_len": len(piece)},
            ))

    # ── Step 4: Merge adjacent short chunks (same doc + same category + total chars <= MAX_CHUNK_CHARS) ──
    merged = _merge_adjacent_short(raw_chunks)

    # ── Step 5: Generate stable id ──
    for c in merged:
        c.id = _make_id(source, c.heading_path, c.content)

    return merged


def _build_content_with_path(path: list[str], body: str) -> str:
    """Inject heading hierarchy path at start of content so embedding senses context.

    E.g. path=['SharePoint high latency troubleshooting Runbook', 'Step 2: troubleshoot backend DB']
    → content:
        [SharePoint high latency troubleshooting Runbook › Step 2: troubleshoot backend DB]
        If app gateway is normal, enter SharePoint backend SQL Server. Check sys.dm_tran_locks ...
    """
    if not path:
        return body
    # Strip the last level's number/Hx prefix (already stored in heading field)
    hier = " › ".join(path)
    return f"[{hier}]\n{body}"


def _strip_path_tag(content: str) -> str:
    """Strip the [hierarchy path] tag line at the start of content (used to rebuild unified prefix when merging)."""
    return re.sub(r"^\[[^\]]*\]\n?", "", content, count=1)


def _merge_adjacent_short(chunks: list[Chunk]) -> list[Chunk]:
    """Merge adjacent short chunks within the same source (total chars <= MAX_CHUNK_CHARS).

    Rules:
      - Same source + same category + adjacent
      - Merged total content <= MAX_CHUNK_CHARS
      - Merged heading_path takes the longest one (preserve full hierarchy)
      - content uniformly uses one most complete hierarchy path tag, stripping both original ones
    """
    if not chunks:
        return chunks

    out: list[Chunk] = [chunks[0]]
    for curr in chunks[1:]:
        prev = out[-1]
        same_source = prev.source == curr.source and prev.category == curr.category
        # When merging, strip each path tag first then count length, to avoid double counting
        prev_body = _strip_path_tag(prev.content)
        curr_body = _strip_path_tag(curr.content)
        combined_len = len(prev_body) + len(curr_body)

        if same_source and combined_len <= MAX_CHUNK_CHARS:
            # Unify path: take the deepest-level one
            merged_path = (
                curr.heading_path if len(curr.heading_path) > len(prev.heading_path)
                else prev.heading_path
            )
            merged_body = prev_body + "\n" + curr_body
            out[-1] = Chunk(
                id="",
                content=_build_content_with_path(merged_path, merged_body),
                source=prev.source,
                category=prev.category,
                heading=merged_path[-1] if merged_path else "",
                heading_path=merged_path,
                allowed_groups=prev.allowed_groups,
                metadata={"char_len": len(merged_body), "merged_from": 2},
            )
        else:
            out.append(curr)
    return out


# ═══════════════════════════════════════════════════════════════════
# Kusto chunking: split by statement, preserve comment context
# ═══════════════════════════════════════════════════════════════════

def _chunk_kusto(kql: str, source: str, allowed_groups: list[str]) -> list[Chunk]:
    statements = re.split(r"(?<=;)\s*\n", kql)
    buf = ""
    out: list[Chunk] = []
    for stmt in statements:
        if len(buf) + len(stmt) > MAX_CHUNK_CHARS and buf:
            out.append(Chunk(
                id=_make_id(source, ["kusto"], buf.strip()),
                content=buf.strip(),
                source=source,
                category="kusto",
                heading="kusto",
                heading_path=["kusto"],
                allowed_groups=list(allowed_groups),
                metadata={"char_len": len(buf)},
            ))
            buf = stmt
        else:
            buf += "\n" + stmt
    if buf.strip():
        out.append(Chunk(
            id=_make_id(source, ["kusto"], buf.strip()),
            content=buf.strip(),
            source=source,
            category="kusto",
            heading="kusto",
            heading_path=["kusto"],
            allowed_groups=list(allowed_groups),
        ))
    return out


# ═══════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════

def chunk_document(content: str, source: str, category: str,
                   allowed_groups: list[str] | None = None) -> list[Chunk]:
    groups = allowed_groups or ["sre", "dev", "ops"]
    if category == "kusto" or source.endswith(".kql"):
        return _chunk_kusto(content, source, groups)
    return _chunk_markdown(content, source, category, groups)
