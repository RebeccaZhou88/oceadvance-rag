# @Author: RebeccaZhou
# @Description: CrewAI sequential crew: Blind Spot / Doc Quality / Remediation agents
#              CrewAI 顺序协作：盲区分析 / 文档质量 / 改进顾问三代理

"""CrewAI document quality governance crew: gap analysis → document quality assessment → improvement suggestions.

Three agents with single responsibilities collaborating sequentially:
1. GapAnalyst: determines the type of knowledge gap (missing/outdated/fragmented/inaccurate) and supporting evidence
2. DocQualityAssessor: evaluates the completeness/accuracy/findability of matched documents
3. RemediationAdvisor: produces actionable document create/update actions with acceptance criteria

CrewAI is an optional dependency: when not installed is_crewai_available()=False and the main flow is unaffected.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Step event callback: (stage, message, agent, detail) -> None
# stage values: agent_step (single agent step executed) / agent_done (stage task completed)
EventSink = Callable[..., None]


def is_crewai_available() -> bool:
    try:
        import crewai  # noqa: F401
        return True
    except ImportError:
        return False


# Disable CrewAI/OpenTelemetry telemetry reporting (local/intranet environment)
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_VALID_GAP_TYPES = {"missing", "outdated", "fragmented", "inaccurate"}
_VALID_PRIORITIES = {"high", "medium", "low"}


def _parse_json_loose(text: str) -> dict[str, Any]:
    """Tolerantly extract JSON from LLM output (handles ```json code block wrapping)."""
    if not text:
        raise ValueError("Empty output")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(cleaned)
        if not m:
            raise
        return json.loads(m.group(0))


def _build_suggestion_md(result: dict[str, Any], question: str) -> str:
    """Render the crew's structured output into a Markdown summary for kanban cards."""
    lines = [f"**Question**: {question}", ""]
    gap = result.get("gap_analysis") or {}
    if gap.get("evidence"):
        lines += ["**Gap Evidence**", gap["evidence"], ""]
    dq = result.get("doc_quality") or []
    if dq:
        lines.append("**Matched Document Issues**")
        for item in dq:
            issues = item.get("issues") or []
            lines.append(f"- `{item.get('doc_id', '?')}`: {'; '.join(issues) if issues else 'No obvious issues'}")
        lines.append("")
    actions = result.get("actions") or []
    if actions:
        lines.append("**Improvement Actions**")
        for i, act in enumerate(actions, 1):
            lines.append(
                f"{i}. [{act.get('type', 'update')}] **{act.get('target_doc', 'Document to be created')}** — {act.get('reason', '')}"
            )
            for point in act.get("outline") or []:
                lines.append(f"   - {point}")
            if act.get("acceptance"):
                lines.append(f"   - Acceptance: {act['acceptance']}")
    return "\n".join(lines)


def _summarize_gap(raw: str) -> str:
    try:
        d = _parse_json_loose(raw)
        evidence = str(d.get("evidence") or d.get("affected_topic") or "")[:100]
        return f"Gap analysis complete: {d.get('gap_type', '?')} — {evidence}".rstrip(" —")
    except Exception:  # noqa: BLE001
        return "Gap analysis complete (output is not valid JSON; final result will be handled tolerantly)"


def _summarize_quality(raw: str) -> str:
    try:
        d = _parse_json_loose(raw)
        docs = d.get("doc_quality") or []
        n_issues = sum(len(x.get("issues") or []) for x in docs if isinstance(x, dict))
        return f"Document quality assessment complete: evaluated {len(docs)} documents, found {n_issues} issues"
    except Exception:  # noqa: BLE001
        return "Document quality assessment complete (output is not valid JSON; final result will be handled tolerantly)"


def _summarize_remediation(raw: str) -> str:
    try:
        d = _parse_json_loose(raw)
        actions = d.get("actions") or []
        return (
            f"Remediation suggestion generation complete: {len(actions)} actions, priority {d.get('priority', '?')}"
            f" — {str(d.get('summary', ''))[:80]}"
        ).rstrip(" —")
    except Exception:  # noqa: BLE001
        return "Remediation suggestion generation complete (output is not valid JSON; final result will be handled tolerantly)"


class GovernanceCrew:
    """Wrapper for the CrewAI governance multi-agents; when crewai is missing the constructor does not error, and run raises RuntimeError."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings

    # ---------- Step event callback factory ----------

    @staticmethod
    def _make_step_cb(agent_name: str, sink: EventSink) -> Callable[..., None]:
        """Triggered once per agent step (one LLM thought/tool call), giving the frontend a 'live' feel."""
        counter = {"n": 0}

        def cb(answer: Any) -> None:  # noqa: ANN401
            counter["n"] += 1
            text = (getattr(answer, "text", "") or str(answer)).strip().replace("\n", " ")
            try:
                sink(
                    "agent_step",
                    f"{agent_name} step {counter['n']}: {text[:120]}",
                    agent_name,
                    {"step": counter["n"], "raw": text[:500]},
                )
            except Exception:  # noqa: BLE001  event reporting failure must never affect crew execution
                logger.debug("Step event reporting failed", exc_info=True)

        return cb

    @staticmethod
    def _make_task_cb(
        agent_name: str, summarize: Callable[[str], str], sink: EventSink
    ) -> Callable[..., None]:
        """Triggered when a single task (one agent stage) completes."""

        def cb(output: Any) -> None:  # noqa: ANN401
            raw = str(getattr(output, "raw", None) or output)
            try:
                sink("agent_done", summarize(raw), agent_name, {"raw": raw[:800]})
            except Exception:  # noqa: BLE001
                logger.debug("Stage completion event reporting failed", exc_info=True)

        return cb

    def _build(self, sink: EventSink):  # type: ignore[no-untyped-def]
        from crewai import Agent, Crew, LLM, Process, Task  # type: ignore

        # Reuse the main project's OpenAI-compatible interface (DeepSeek / Qwen)
        # litellm requires the openai/ prefix to recognize custom endpoints (not needed if main project connects directly via openai SDK)
        model = self.settings.model_name or "gpt-4o-mini"
        if "/" not in model:
            model = f"openai/{model}"
        llm = LLM(
            model=model,
            base_url=self.settings.model_base_url,
            api_key=self.settings.api_key,
            temperature=0.2,
        )

        gap_analyst = Agent(
            role="Knowledge Base Gap Analyst",
            goal="Based on the user's question, retrieval results, and RAG evaluation scores, accurately identify the type of knowledge gap and supporting evidence",
            backstory=(
                "You are a senior operations knowledge base operator, familiar with Runbook/Postmortem/Kusto templates/IcM incident management. "
                "You judge only based on the given evidence and do not speculate about content that does not exist in the knowledge base."
            ),
            llm=llm,
            allow_delegation=False,
            max_iter=3,
            verbose=False,
            step_callback=self._make_step_cb("Gap Analyst", sink),
        )
        doc_assessor = Agent(
            role="Document Quality Assessor",
            goal="Evaluate the completeness, accuracy, and findability of matched documents and point out specific defects",
            backstory=(
                "You are an expert in technical document review, focusing on: whether steps are complete, whether commands/thresholds are accurate, "
                "whether titles and keywords are easy to retrieve, and whether content is fragmented across multiple documents."
            ),
            llm=llm,
            allow_delegation=False,
            max_iter=3,
            verbose=False,
            step_callback=self._make_step_cb("Document Quality Assessor", sink),
        )
        remediation_advisor = Agent(
            role="Document Governance Improvement Advisor",
            goal="Combine gap analysis and quality assessment to provide engineers with directly actionable document create/update actions",
            backstory=(
                "You are responsible for driving the 'discover gap → improve document → improve RAG' closed loop; suggestions must be specific and verifiable, "
                "and you must provide priority and target document path recommendations."
            ),
            llm=llm,
            allow_delegation=False,
            max_iter=3,
            verbose=False,
            step_callback=self._make_step_cb("Document Governance Improvement Advisor", sink),
        )

        gap_task = Task(
            description=(
                "Analyze the following RAG Q&A context and determine the knowledge gap.\n"
                "User question: {question}\nUser group: {user_group}\nTrigger reason: {trigger_type}\n"
                "Evaluation scores: {eval_scores}\nMatched documents (may be empty, empty means no retrieval): {doc_refs}\n\n"
                "Output JSON: {{\"gap_type\": one of missing/outdated/fragmented/inaccurate, "
                "\"affected_topic\": topic, \"evidence\": one sentence of judgment evidence}}"
            ),
            expected_output="Output only a JSON object, no additional explanation",
            agent=gap_analyst,
            callback=self._make_task_cb("Gap Analyst", _summarize_gap, sink),
        )
        quality_task = Task(
            description=(
                "Based on the gap analysis results and the list of matched documents, evaluate document quality one by one.\n"
                "Output JSON: {{\"doc_quality\": [{{\"doc_id\": document id, "
                "\"issues\": [specific defects...], \"scores\": {{\"completeness\": 1-5, "
                "\"accuracy\": 1-5, \"findability\": 1-5}}}}]}}; output an empty array when there are no matched documents."
            ),
            expected_output="Output only a JSON object, no additional explanation",
            agent=doc_assessor,
            context=[gap_task],
            callback=self._make_task_cb("Document Quality Assessor", _summarize_quality, sink),
        )
        remediation_task = Task(
            description=(
                "Combine the outputs of the previous two steps and provide governance actions.\n"
                "Output JSON: {{\"gap_type\": carry over the gap type, \"priority\": high/medium/low, "
                "\"summary\": one sentence summary, "
                "\"actions\": [{{\"type\": create/update/retire, "
                "\"target_doc\": suggested document path or filename, \"reason\": reason, "
                "\"outline\": [sections/key points to include...], "
                "\"acceptance\": verifiable acceptance criteria}}]}}"
            ),
            expected_output="Output only a JSON object, no additional explanation",
            agent=remediation_advisor,
            context=[gap_task, quality_task],
            callback=self._make_task_cb("Document Governance Improvement Advisor", _summarize_remediation, sink),
        )

        return Crew(
            agents=[gap_analyst, doc_assessor, remediation_advisor],
            tasks=[gap_task, quality_task, remediation_task],
            process=Process.sequential,
            verbose=False,
        )

    def run(
        self,
        question: str,
        trigger_type: str,
        user_group: str,
        eval_scores: dict[str, Any],
        doc_refs: list[dict[str, Any]],
        on_event: EventSink | None = None,
    ) -> dict[str, Any]:
        """Synchronous execution (caller wraps with asyncio.to_thread). Returns structured result.

        on_event(stage, message, agent, detail): returns agent execution steps in real time for frontend kanban display.
        """
        if not is_crewai_available():
            raise RuntimeError("crewai is not installed, cannot run governance crew")

        sink: EventSink = on_event or (lambda *a, **k: None)

        # Control context size: keep only document id/title/fragment first 200 chars
        slim_refs = [
            {
                "doc_id": d.get("doc_id", ""),
                "title": d.get("title", ""),
                "category": d.get("category", ""),
                "snippet": (d.get("content") or d.get("text") or "")[:200],
            }
            for d in (doc_refs or [])[:5]
        ]
        crew = self._build(sink)
        raw = crew.kickoff(
            inputs={
                "question": question,
                "user_group": user_group or "unknown",
                "trigger_type": trigger_type,
                "eval_scores": json.dumps(eval_scores, ensure_ascii=False),
                "doc_refs": json.dumps(slim_refs, ensure_ascii=False),
            }
        )
        text = getattr(raw, "raw", None) or str(raw)
        result = _parse_json_loose(text)

        # Normalize enums
        if result.get("gap_type") not in _VALID_GAP_TYPES:
            result["gap_type"] = "missing" if not slim_refs else "fragmented"
        if result.get("priority") not in _VALID_PRIORITIES:
            result["priority"] = "high" if trigger_type == "empty_retrieval" else "medium"
        result["suggestion_md"] = _build_suggestion_md(result, question)
        return result
