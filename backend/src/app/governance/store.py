"""Governance task store: SQLite persistence (task carrier for the question → discover gap → improve document closed loop).

Thread-safe: single connection + Lock, all methods synchronous; caller wraps with asyncio.to_thread.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Task state machine
STATUS_OPEN = "open"            # Pending
STATUS_RUNNING = "running"      # Crew analyzing
STATUS_IN_PROGRESS = "in_progress"  # Engineer working
STATUS_RESOLVED = "resolved"    # Improved/closed
STATUS_IGNORED = "ignored"      # Ignored
VALID_STATUSES = {STATUS_OPEN, STATUS_RUNNING, STATUS_IN_PROGRESS, STATUS_RESOLVED, STATUS_IGNORED}

# Kanban column statuses (running is occupied by the system, others can be manually transitioned)
BOARD_STATUSES = [STATUS_RUNNING, STATUS_OPEN, STATUS_IN_PROGRESS, STATUS_RESOLVED, STATUS_IGNORED]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS governance_tasks (
    id           TEXT PRIMARY KEY,
    trigger_type TEXT NOT NULL,          -- empty_retrieval | low_score | manual
    question     TEXT NOT NULL,
    user_group   TEXT DEFAULT '',
    session_id   TEXT DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'open',
    priority     TEXT NOT NULL DEFAULT 'medium',   -- high | medium | low
    gap_type     TEXT DEFAULT '',                  -- missing | outdated | fragmented | inaccurate
    eval_scores  TEXT DEFAULT '{}',               -- JSON: faithfulness/relevancy/hallucination
    doc_refs     TEXT DEFAULT '[]',               -- JSON: matched document reference snapshot
    crew_result  TEXT DEFAULT '{}',               -- JSON: CrewAI complete structured output
    suggestion   TEXT DEFAULT '',                  -- Markdown improvement suggestion summary
    error        TEXT DEFAULT '',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gt_status ON governance_tasks(status);
CREATE INDEX IF NOT EXISTS idx_gt_created ON governance_tasks(created_at);

-- CrewAI multi-agent execution step log (frontend kanban displays agent progress in real time)
CREATE TABLE IF NOT EXISTS governance_events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    ts      TEXT NOT NULL,                   -- ISO8601 UTC
    stage   TEXT NOT NULL,                   -- created|crew_start|agent_step|agent_done|done|error
    agent   TEXT DEFAULT '',                 -- Gap Analyst / Document Quality Assessor / Improvement Advisor
    message TEXT NOT NULL DEFAULT '',
    detail  TEXT DEFAULT '{}'                -- JSON additional info (truncated raw output, etc.)
);
CREATE INDEX IF NOT EXISTS idx_ge_task ON governance_events(task_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for key in ("eval_scores", "doc_refs", "crew_result"):
        try:
            d[key] = json.loads(d.get(key) or "{}" if key != "doc_refs" else d.get(key) or "[]")
        except (json.JSONDecodeError, TypeError):
            d[key] = {} if key != "doc_refs" else []
    return d


class GovernanceTaskStore:
    """SQLite task store (thread-safe singleton style, initialized/closed by lifespan)."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        logger.info("Governance task store initialized: %s", db_path)

    # ---------- Write ----------

    def create_task(
        self,
        question: str,
        trigger_type: str,
        user_group: str = "",
        session_id: str = "",
        eval_scores: dict | None = None,
        doc_refs: list | None = None,
        priority: str = "medium",
    ) -> dict[str, Any]:
        task_id = uuid.uuid4().hex[:12]
        now = _now()
        question = (question or "").strip()  # Normalize to ensure dedup matching (query side also strips)
        with self._lock:
            self._conn.execute(
                """INSERT INTO governance_tasks
                   (id, trigger_type, question, user_group, session_id, status, priority,
                    eval_scores, doc_refs, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?)""",
                (
                    task_id, trigger_type, question, user_group, session_id, priority,
                    json.dumps(eval_scores or {}, ensure_ascii=False),
                    json.dumps(doc_refs or [], ensure_ascii=False),
                    now, now,
                ),
            )
            self._conn.commit()
        logger.info("Governance task created: id=%s trigger=%s", task_id, trigger_type)
        return self.get_task(task_id)  # type: ignore[return-value]

    def save_result(
        self,
        task_id: str,
        gap_type: str,
        priority: str,
        crew_result: dict[str, Any],
        suggestion: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE governance_tasks
                   SET status='open', gap_type=?, priority=?, crew_result=?, suggestion=?,
                       error='', updated_at=?
                   WHERE id=?""",
                (
                    gap_type, priority,
                    json.dumps(crew_result, ensure_ascii=False),
                    suggestion, _now(), task_id,
                ),
            )
            self._conn.commit()

    def mark_failed(self, task_id: str, error: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE governance_tasks SET status='open', error=?, updated_at=? WHERE id=?",
                (error[:2000], _now(), task_id),
            )
            self._conn.commit()

    def update_status(self, task_id: str, status: str) -> dict[str, Any] | None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        with self._lock:
            self._conn.execute(
                "UPDATE governance_tasks SET status=?, updated_at=? WHERE id=?",
                (status, _now(), task_id),
            )
            self._conn.commit()
        return self.get_task(task_id)

    # ---------- Read ----------

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM governance_tasks WHERE id=?", (task_id,)
            ).fetchone()
        return _row_to_dict(row) if row else None

    def list_tasks(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT * FROM governance_tasks"
        params: tuple = ()
        if status:
            sql += " WHERE status=?"
            params = (status,)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params = params + (limit,)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]

    def has_open_for_question(self, question: str) -> bool:
        """Dedup: do not trigger repeatedly when the same question has running/open tasks."""
        q = question.strip()
        with self._lock:
            row = self._conn.execute(
                """SELECT 1 FROM governance_tasks
                   WHERE question=? AND status IN ('open','running') LIMIT 1""",
                (q,),
            ).fetchone()
        return row is not None

    # ---------- Step event log ----------

    def add_event(
        self,
        task_id: str,
        stage: str,
        message: str,
        agent: str = "",
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Append an agent execution step (called in the crew worker thread, relies on Lock for safety)."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO governance_events (task_id, ts, stage, agent, message, detail)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    task_id, _now(), stage, agent, message[:1000],
                    json.dumps(detail or {}, ensure_ascii=False)[:4000],
                ),
            )
            self._conn.commit()

    def list_events(self, task_id: str) -> list[dict[str, Any]]:
        """Return task step logs in chronological order (id auto-increment equals insertion order)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, stage, agent, message, detail FROM governance_events"
                " WHERE task_id=? ORDER BY id ASC",
                (task_id,),
            ).fetchall()
        events = []
        for row in rows:
            d = dict(row)
            try:
                d["detail"] = json.loads(d.get("detail") or "{}")
            except (json.JSONDecodeError, TypeError):
                d["detail"] = {}
            events.append(d)
        return events

    def close(self) -> None:
        with self._lock:
            self._conn.close()
