# @Author: RebeccaZhou
# @Description: Governance trigger manager: async spawn with dedupe
#              治理触发管理器：异步任务派生与去重

"""Governance orchestration: after the LangGraph main flow ends, perform conditional judgment and asynchronously trigger CrewAI governance.

Responsibility boundaries:
- LangGraph: RAG main flow state transitions (this module does not intervene inside the graph)
- This manager: decides whether to trigger based on main flow outputs (retrieval count/evaluation score), manages background task lifecycle
- GovernanceCrew: multi-agent execution (see crew.py)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.governance.crew import GovernanceCrew, is_crewai_available
from app.governance.store import GovernanceTaskStore

logger = logging.getLogger(__name__)

TRIGGER_EMPTY = "empty_retrieval"   # Empty retrieval / knowledge not found
TRIGGER_LOW_SCORE = "low_score"     # Low evaluation score
TRIGGER_MANUAL = "manual"           # Manually initiated from frontend

TRIGGER_LABELS = {
    TRIGGER_EMPTY: "Empty retrieval",
    TRIGGER_LOW_SCORE: "Low online evaluation score",
    TRIGGER_MANUAL: "Manual initiation",
}

_NO_KNOWLEDGE_PREFIX = "No relevant knowledge found"


def extract_eval_scores(result: dict[str, Any]) -> dict[str, float]:
    """Extract online evaluation scores from the evaluate_answer trace step."""
    for step in result.get("trace", []) or []:
        if step.get("step") == "evaluate_answer" and step.get("status") == "ok":
            detail = step.get("detail") or {}
            return {
                "faithfulness": float(detail.get("faithfulness", 0.0) or 0.0),
                "answer_relevancy": float(detail.get("answer_relevancy", 0.0) or 0.0),
                "hallucination_score": float(detail.get("hallucination_score", 0.0) or 0.0),
            }
    return {}


def detect_trigger(result: dict[str, Any], settings: Any) -> str | None:
    """Conditional routing: returns the trigger type; returns None if not satisfied. Chitchat never triggers."""
    if result.get("intent") == "chitchat":
        return None

    docs = result.get("retrieved_docs") or []
    answer = result.get("answer", "") or ""
    scores = extract_eval_scores(result)

    # Rule 1: empty retrieval / knowledge not found
    if not docs or answer.startswith(_NO_KNOWLEDGE_PREFIX):
        return TRIGGER_EMPTY

    # Rule 2: online evaluation below threshold
    if scores:
        if scores["faithfulness"] and scores["faithfulness"] < settings.governance_min_faithfulness:
            return TRIGGER_LOW_SCORE
        if scores["answer_relevancy"] and scores["answer_relevancy"] < settings.governance_min_relevancy:
            return TRIGGER_LOW_SCORE
        if scores["hallucination_score"] > settings.governance_max_hallucination:
            return TRIGGER_LOW_SCORE
    return None


class GovernanceManager:
    """Singleton (lifespan managed): holds the task store + crew factory + background task set."""

    def __init__(self, settings: Any, store: GovernanceTaskStore) -> None:
        self.settings = settings
        self.store = store
        self.crew = GovernanceCrew(settings)
        self._bg_tasks: set[asyncio.Task] = set()

    @property
    def available(self) -> bool:
        return self.settings.governance_enabled and is_crewai_available()

    # ---------- Trigger entry points ----------

    async def maybe_trigger_after_chat(
        self,
        result: dict[str, Any],
        question: str,
        user_group: str,
        session_id: str,
    ) -> str | None:
        """Called after the chat main flow ends; if rules match, asynchronously initiates governance and returns task id (or None)."""
        if not self.settings.governance_enabled:
            return None
        trigger = detect_trigger(result, self.settings)
        if trigger is None:
            return None
        doc_refs = result.get("citations") or result.get("retrieved_docs") or []
        return await self._spawn(
            question=question,
            trigger_type=trigger,
            user_group=user_group,
            session_id=session_id,
            eval_scores=extract_eval_scores(result),
            doc_refs=doc_refs,
        )

    async def trigger_manual(
        self,
        question: str,
        user_group: str = "",
        session_id: str = "",
        eval_scores: dict | None = None,
        doc_refs: list | None = None,
    ) -> str:
        """Manual entry point for the frontend 'initiate governance' button."""
        return await self._spawn(
            question=question,
            trigger_type=TRIGGER_MANUAL,
            user_group=user_group,
            session_id=session_id,
            eval_scores=eval_scores or {},
            doc_refs=doc_refs or [],
        )

    # ---------- Internal ----------

    async def _spawn(
        self,
        question: str,
        trigger_type: str,
        user_group: str,
        session_id: str,
        eval_scores: dict,
        doc_refs: list,
    ) -> str | None:
        # Dedup: same question already has an open/running task
        if await asyncio.to_thread(self.store.has_open_for_question, question):
            logger.info("Governance task already exists, skipping duplicate trigger: %s", question[:60])
            return None

        if not is_crewai_available():
            logger.warning("crewai is not installed, skipping governance trigger (does not affect RAG main flow)")
            return None

        priority = "high" if trigger_type == TRIGGER_EMPTY else "medium"
        task_row = await asyncio.to_thread(
            self.store.create_task,
            question, trigger_type, user_group, session_id, eval_scores, doc_refs, priority,
        )
        task_id = task_row["id"]

        # Step log 1: task created (including trigger reason and priority)
        await asyncio.to_thread(
            self.store.add_event,
            task_id, "created",
            f"Governance task created (trigger reason: {TRIGGER_LABELS.get(trigger_type, trigger_type)}, "
            f"priority: {priority})",
        )

        bg = asyncio.create_task(
            self._run_crew(task_id, question, trigger_type, user_group, eval_scores, doc_refs),
            name=f"governance-{task_id}",
        )
        self._bg_tasks.add(bg)
        bg.add_done_callback(self._bg_tasks.discard)
        return task_id

    async def _run_crew(
        self,
        task_id: str,
        question: str,
        trigger_type: str,
        user_group: str,
        eval_scores: dict,
        doc_refs: list,
    ) -> None:
        # Step logs are written synchronously by sink in the crew worker thread (store has internal Lock, thread-safe)
        def sink(stage: str, message: str, agent: str = "", detail: dict | None = None) -> None:
            self.store.add_event(task_id, stage, message, agent, detail)

        try:
            await asyncio.to_thread(
                self.store.add_event,
                task_id, "crew_start",
                "CrewAI multi-agents starting sequential analysis: Gap Analyst → Document Quality Assessor → Document Governance Improvement Advisor",
            )
            result = await asyncio.to_thread(
                self.crew.run,
                question, trigger_type, user_group, eval_scores, doc_refs, sink,
            )
            suggestion = result.pop("suggestion_md", "")
            gap_type = result.get("gap_type", "missing")
            await asyncio.to_thread(
                self.store.save_result,
                task_id,
                gap_type,
                result.get("priority", "medium"),
                result,
                suggestion,
            )
            # Step log: completed
            n_actions = len(result.get("actions") or [])
            await asyncio.to_thread(
                self.store.add_event,
                task_id, "done",
                f"Analysis complete: gap type {gap_type}, produced {n_actions} document improvement actions, suggestions saved, awaiting engineer follow-up",
            )
            logger.info("Governance task completed: id=%s gap=%s", task_id, gap_type)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Governance task failed: id=%s", task_id)
            await asyncio.to_thread(self.store.mark_failed, task_id, str(exc))
            await asyncio.to_thread(
                self.store.add_event,
                task_id, "error",
                f"Governance execution failed: {str(exc)[:300]}",
            )

    async def aclose(self) -> None:
        """Graceful shutdown: wait for in-progress governance tasks (max 15s)."""
        if not self._bg_tasks:
            return
        _, pending = await asyncio.wait(self._bg_tasks, timeout=15)
        for t in pending:
            t.cancel()
