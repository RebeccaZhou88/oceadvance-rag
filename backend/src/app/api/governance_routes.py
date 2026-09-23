"""Knowledge base quality governance REST API: task list / status transitions / manual trigger."""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.governance.crew import is_crewai_available
from app.governance.store import VALID_STATUSES, GovernanceTaskStore

router = APIRouter(prefix="/governance", tags=["governance"])

# Injected by main.py lifespan
_manager: Any = None
_store: GovernanceTaskStore | None = None


def set_governance(manager: Any, store: GovernanceTaskStore | None) -> None:
    global _manager, _store
    _manager = manager
    _store = store


def _require_store() -> GovernanceTaskStore:
    if _store is None:
        raise HTTPException(status_code=503, detail="Governance module not initialized")
    return _store


class StatusUpdate(BaseModel):
    status: str = Field(description="open | in_progress | resolved | ignored")


class ManualTrigger(BaseModel):
    question: str = Field(min_length=1)
    user_group: str = "ops"
    session_id: str = ""
    eval_scores: dict = Field(default_factory=dict)
    doc_refs: list[dict] = Field(default_factory=list)


@router.get("/status")
async def governance_status() -> dict:
    """Governance module availability (for the frontend to decide whether to show the dashboard/button)."""
    settings_enabled = bool(_manager and _manager.settings.governance_enabled)
    counts = {}
    if _store is not None:
        tasks = await asyncio.to_thread(_store.list_tasks, None, 500)
        for t in tasks:
            counts[t["status"]] = counts.get(t["status"], 0) + 1
    return {
        "enabled": settings_enabled,
        "crewai_available": is_crewai_available(),
        "task_counts": counts,
    }


@router.get("/tasks")
async def list_tasks(status: Optional[str] = None, limit: int = 100) -> list[dict]:
    store = _require_store()
    if status is not None and status not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail=f"Invalid status, options: {sorted(VALID_STATUSES)}")
    return await asyncio.to_thread(store.list_tasks, status, min(limit, 500))


@router.get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict:
    store = _require_store()
    task = await asyncio.to_thread(store.get_task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task does not exist")
    return task


@router.get("/tasks/{task_id}/events")
async def list_task_events(task_id: str) -> list[dict]:
    """Multi-agent execution step logs for a task (in chronological order, frontend polls to render timeline)."""
    store = _require_store()
    task = await asyncio.to_thread(store.get_task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task does not exist")
    return await asyncio.to_thread(store.list_events, task_id)


@router.patch("/tasks/{task_id}")
async def update_task(task_id: str, body: StatusUpdate) -> dict:
    store = _require_store()
    if body.status not in VALID_STATUSES or body.status == "running":
        raise HTTPException(status_code=400, detail="Dashboard only allows transition to open/in_progress/resolved/ignored")
    task = await asyncio.to_thread(store.update_status, task_id, body.status)
    if task is None:
        raise HTTPException(status_code=404, detail="Task does not exist")
    return task


@router.post("/trigger")
async def manual_trigger(req: ManualTrigger) -> dict:
    if _manager is None:
        raise HTTPException(status_code=503, detail="Governance module not initialized")
    if not is_crewai_available():
        raise HTTPException(status_code=503, detail="crewai is not installed, governance is unavailable")
    task_id = await _manager.trigger_manual(
        question=req.question,
        user_group=req.user_group,
        session_id=req.session_id,
        eval_scores=req.eval_scores,
        doc_refs=req.doc_refs,
    )
    if task_id is None:
        return {"status": "skipped", "reason": "An in-progress governance task already exists for the same question"}
    return {"status": "created", "task_id": task_id}
