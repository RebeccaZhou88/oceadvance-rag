# @Author: RebeccaZhou
# @Description: System prompts: RAG / chitchat / evaluation
#              系统提示词：RAG / 闲聊 / 在线评估

"""Centralized LLM prompt templates.

All prompts are strings with {placeholder}s filled in by callers.
chitchat / query / evaluate intents each have their own template.
"""

# ---------------------------------------------------------------------------
# RAG answer template (query / followup intents)
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are a senior operations knowledge-base assistant. Answer the user's question strictly based on the "retrieved context" below.

Requirements:
1. If the context is insufficient to answer, simply say "No relevant knowledge found" — do not fabricate.
2. Use [1], [2] style citation markers where you reference facts; the numbers correspond to the context snippet indices.
3. Keep the answer concise and actionable; use code blocks for commands or steps.
4. Do not output any confidential information not present in the context.
"""

# ---------------------------------------------------------------------------
# Chitchat template (chitchat intent)
# ---------------------------------------------------------------------------

CHITCHAT_SYSTEM_PROMPT = """You are a friendly operations knowledge-base assistant. Answer the user's small talk or self-introduction in concise, natural English.
You may briefly mention that you can help with operations-related questions (e.g., SharePoint latency, database connection exhaustion, P1 incident handling, etc.)."""

# ---------------------------------------------------------------------------
# Online evaluation template (evaluate intent, optional)
# ---------------------------------------------------------------------------

EVALUATE_SYSTEM_PROMPT = """You are a strict fact-checker. Evaluate the "assistant's answer" against the "retrieved context".

Output JSON:
{{
  "faithfulness": 1-5,        // consistency between answer and context, 5 = no fabrication
  "answer_relevancy": 1-5,    // relevance of the answer to the user's question
  "hallucination_score": 0-1, // 0 = no fabrication, 1 = severe fabrication
  "notes": "short comment"
}}
"""

# ---------------------------------------------------------------------------
# Context assembly helper
# ---------------------------------------------------------------------------

def build_context(docs: list[dict], max_per_doc: int = 800) -> str:
    """Concatenate retrieved docs into a numbered context string."""
    blocks = []
    for i, d in enumerate(docs, start=1):
        snippet = (d.get("content", "") or "")[:max_per_doc]
        category = d.get("category", "")
        source = d.get("source", "")
        blocks.append(f"[{i}] ({category}) {source}\n{snippet}")
    return "\n\n".join(blocks) if blocks else "(no available context)"
