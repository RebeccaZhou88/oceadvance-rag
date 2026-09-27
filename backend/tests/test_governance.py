# @Author: RebeccaZhou
# @Description: Governance tests: store, state machine, trigger dedupe
#              治理测试：任务存储、状态机与触发去重

"""Knowledge base document quality governance module tests:

Covers four layers:
1. GovernanceTaskStore: SQLite task store CRUD / state machine / dedup / step events
2. GovernanceCrew: JSON fault-tolerant parsing, Markdown rendering, enum normalization, run() event callback (no real LLM)
3. GovernanceManager: conditional routing matrix, async execution success/failure/dedup/disable
4. REST API: status/tasks/events/trigger all branches
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("USE_MOCK_BACKEND", "true")
os.environ.setdefault("RERANK_STRATEGY", "none")

from app.governance import crew as crew_mod  # noqa: E402
from app.governance.crew import (  # noqa: E402
    GovernanceCrew,
    _build_suggestion_md,
    _parse_json_loose,
    _summarize_gap,
    _summarize_quality,
    _summarize_remediation,
)
from app.governance.manager import (  # noqa: E402
    TRIGGER_EMPTY,
    TRIGGER_LOW_SCORE,
    GovernanceManager,
    detect_trigger,
    extract_eval_scores,
)
from app.governance.store import (  # noqa: E402
    STATUS_IN_PROGRESS,
    STATUS_OPEN,
    STATUS_RESOLVED,
    STATUS_RUNNING,
    GovernanceTaskStore,
)


# ============================================================================
# Test doubles
# ============================================================================

class FakeSettings:
    """manager only depends on switches & thresholds; crew only reads model connection fields."""

    def __init__(self, enabled: bool = True):
        self.governance_enabled = enabled
        self.governance_min_faithfulness = 3.5
        self.governance_min_relevancy = 3.5
        self.governance_max_hallucination = 0.3
        self.api_key = "test-key"
        self.model_name = "test-model"
        self.model_base_url = "https://example.invalid/v1"


class _RawOutput:
    def __init__(self, raw: str):
        self.raw = raw


class _FakeCrewObj:
    """Replace GovernanceCrew._build: kickoff returns preset JSON and simulates stage callbacks."""

    def __init__(self, raw: str, sink):
        self.raw = raw
        self.sink = sink
        self.inputs = None

    def kickoff(self, inputs):
        self.inputs = inputs
        # Simulate completion callbacks for three Tasks (real Crew runs sequentially three times)
        for agent, msg in [
            ("Blind Spot Analyst", "Blind spot analysis done: missing"),
            ("Doc Quality Assessor", "Doc quality evaluation done: evaluated 0 docs"),
            ("Doc Governance Improvement Advisor", "Improvement suggestion generation done: 1 action"),
        ]:
            self.sink("agent_done", msg, agent, {"raw": self.raw[:100]})
        return _RawOutput(self.raw)


class FakeGovernanceCrew:
    """manager layer double: records inputs, injectable failure, accepts on_event per real signature."""

    def __init__(self, result: dict | None = None, fail: bool = False):
        self.fail = fail
        self.result = result or {
            "gap_type": "missing",
            "priority": "high",
            "summary": "Missing Kafka runbook",
            "actions": [
                {
                    "type": "create",
                    "target_doc": "runbooks/kafka.md",
                    "reason": "empty retrieval",
                    "outline": ["troubleshooting steps"],
                    "acceptance": "retrievable hit",
                }
            ],
        }
        self.calls: list[dict] = []

    def run(self, question, trigger_type, user_group, eval_scores, doc_refs, on_event=None):
        self.calls.append(
            {
                "question": question,
                "trigger_type": trigger_type,
                "user_group": user_group,
                "eval_scores": eval_scores,
                "doc_refs": doc_refs,
                "on_event": on_event,
            }
        )
        if on_event:
            on_event("agent_step", "Blind Spot Analyst step 1: analyzing", "Blind Spot Analyst")
            # Simulate three Task completion callbacks
            for agent, msg in [
                ("Blind Spot Analyst", "Blind spot analysis done: missing"),
                ("Doc Quality Assessor", "Doc quality evaluation done: evaluated 0 docs"),
                ("Doc Governance Improvement Advisor", "Improvement suggestion generation done: 1 action"),
            ]:
                on_event("agent_done", msg, agent)
        if self.fail:
            raise RuntimeError("Simulated Crew execution failure")
        out = dict(self.result)
        out["suggestion_md"] = "**Suggestion**: add docs"
        return out


VALID_RESULT_JSON = json.dumps(
    {
        "gap_type": "missing",
        "priority": "high",
        "summary": "Missing Kafka runbook",
        "actions": [
            {"type": "create", "target_doc": "runbooks/kafka.md", "reason": "empty retrieval",
             "outline": ["a", "b"], "acceptance": "top1 hit"}
        ],
    },
    ensure_ascii=False,
)


def _wait_status(store: GovernanceTaskStore, task_id: str, target: set[str], timeout: float = 8.0):
    """Poll-wait for background task to be persisted (event loop in another thread, e.g. TestClient portal, use blocking sleep)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = store.get_task(task_id)
        if row and row["status"] in target:
            return row
        time.sleep(0.05)
    raise AssertionError(f"Task {task_id} status did not enter {target}, current: {store.get_task(task_id)}")


async def _await_status(
    store: GovernanceTaskStore, task_id: str, target: set[str], timeout: float = 8.0
):
    """Wait within same event loop: must yield control to let asyncio background Crew task run."""
    import asyncio

    deadline = time.time() + timeout
    while time.time() < deadline:
        row = store.get_task(task_id)
        if row and row["status"] in target:
            return row
        await asyncio.sleep(0.05)
    raise AssertionError(f"Task {task_id} status did not enter {target}, current: {store.get_task(task_id)}")


# ============================================================================
# 1. Store: task store
# ============================================================================

@pytest.fixture
def store(tmp_path):
    s = GovernanceTaskStore(str(tmp_path / "gov_test.db"))
    yield s
    s.close()


class TestGovernanceTaskStore:
    def test_create_task_defaults_running(self, store):
        row = store.create_task("question?", TRIGGER_EMPTY, "sre", "s1")
        assert row["status"] == STATUS_RUNNING
        assert row["trigger_type"] == TRIGGER_EMPTY
        assert row["priority"] == "medium"
        assert row["eval_scores"] == {}
        assert row["doc_refs"] == []
        assert row["crew_result"] == {}
        assert len(row["id"]) == 12

    def test_save_result_opens_task_and_persists_json(self, store):
        row = store.create_task("question?", TRIGGER_EMPTY)
        result = {"gap_type": "missing", "actions": [{"target_doc": "a.md"}]}
        store.save_result(row["id"], "missing", "high", result, "**Suggestion**")
        updated = store.get_task(row["id"])
        assert updated["status"] == STATUS_OPEN
        assert updated["gap_type"] == "missing"
        assert updated["priority"] == "high"
        assert updated["crew_result"]["actions"][0]["target_doc"] == "a.md"
        assert updated["suggestion"] == "**Suggestion**"
        assert updated["error"] == ""

    def test_mark_failed_reopens_with_error(self, store):
        row = store.create_task("question?", TRIGGER_EMPTY)
        store.mark_failed(row["id"], "boom")
        updated = store.get_task(row["id"])
        assert updated["status"] == STATUS_OPEN
        assert updated["error"] == "boom"

    def test_update_status_state_machine(self, store):
        row = store.create_task("question?", TRIGGER_EMPTY)
        assert store.update_status(row["id"], STATUS_IN_PROGRESS)["status"] == STATUS_IN_PROGRESS
        assert store.update_status(row["id"], STATUS_RESOLVED)["status"] == STATUS_RESOLVED
        assert store.update_status(row["id"], STATUS_OPEN)["status"] == STATUS_OPEN

    def test_update_status_rejects_unknown(self, store):
        row = store.create_task("question?", TRIGGER_EMPTY)
        with pytest.raises(ValueError):
            store.update_status(row["id"], "deleted")
        assert store.update_status("not-exist", STATUS_OPEN) is None

    def test_list_filter_and_ordering(self, store):
        r1 = store.create_task("question one", TRIGGER_EMPTY)
        r2 = store.create_task("question two", "manual")
        store.save_result(r2["id"], "missing", "medium", {}, "md")
        assert {t["id"] for t in store.list_tasks(STATUS_RUNNING)} == {r1["id"]}
        assert {t["id"] for t in store.list_tasks(STATUS_OPEN)} == {r2["id"]}
        all_tasks = store.list_tasks()
        assert len(all_tasks) == 2
        # most recently created sorts first
        assert all_tasks[0]["id"] == r2["id"]
        assert store.get_task("missing-id") is None

    def test_dedup_has_open_for_question(self, store):
        row = store.create_task("  duplicate question?  ", TRIGGER_EMPTY)
        # whitespace stripped: dedup when same question is running/open
        assert store.has_open_for_question("duplicate question?")
        store.save_result(row["id"], "missing", "medium", {}, "")
        assert store.has_open_for_question("duplicate question?")  # open still occupies
        store.update_status(row["id"], STATUS_RESOLVED)
        assert not store.has_open_for_question("duplicate question?")
        assert not store.has_open_for_question("brand new question?")

    def test_events_append_in_order_with_detail(self, store):
        row = store.create_task("question?", TRIGGER_EMPTY)
        store.add_event(row["id"], "created", "created")
        store.add_event(row["id"], "agent_step", "step 1", "Blind Spot Analyst", {"step": 1})
        store.add_event(row["id"], "done", "done")
        events = store.list_events(row["id"])
        assert [e["stage"] for e in events] == ["created", "agent_step", "done"]
        assert events[1]["agent"] == "Blind Spot Analyst"
        assert events[1]["detail"] == {"step": 1}
        assert store.list_events("not-exist") == []

    def test_event_message_truncated(self, store):
        row = store.create_task("question?", TRIGGER_EMPTY)
        store.add_event(row["id"], "agent_step", "x" * 5000)
        message = store.list_events(row["id"])[0]["message"]
        assert len(message) == 1000

    def test_reopen_same_db_persists(self, tmp_path):
        """Simulate service restart: schema idempotent when table exists, historical data retained."""
        db = str(tmp_path / "persist.db")
        s1 = GovernanceTaskStore(db)
        row = s1.create_task("still here after restart?", TRIGGER_EMPTY)
        s1.add_event(row["id"], "created", "x")
        s1.close()
        s2 = GovernanceTaskStore(db)
        try:
            assert s2.get_task(row["id"])["question"] == "still here after restart?"
            assert len(s2.list_events(row["id"])) == 1
        finally:
            s2.close()


# ============================================================================
# 2. Crew: pure functions & run() (no real LLM)
# ============================================================================

class TestCrewParsing:
    def test_parse_plain_json(self):
        assert _parse_json_loose('{"a": 1}') == {"a": 1}

    def test_parse_fenced_json(self):
        text = '```json\n{"a": 2}\n```'
        assert _parse_json_loose(text) == {"a": 2}

    def test_parse_embedded_in_prose(self):
        text = 'OK, the analysis result is as follows: {"gap_type": "missing"} please check'
        assert _parse_json_loose(text)["gap_type"] == "missing"

    def test_parse_empty_and_garbage_raise(self):
        with pytest.raises(ValueError):
            _parse_json_loose("")
        with pytest.raises(json.JSONDecodeError):
            _parse_json_loose("content that is not JSON at all")

    def test_build_suggestion_md_contains_sections(self):
        result = {
            "gap_analysis": {"evidence": "empty retrieval"},
            "doc_quality": [{"doc_id": "d1", "issues": ["missing steps", "outdated threshold"]}],
            "actions": [
                {"type": "update", "target_doc": "rb.md", "reason": "outdated",
                 "outline": ["new section"], "acceptance": "acceptance A"}
            ],
        }
        md = _build_suggestion_md(result, "some question?")
        assert "some question?" in md
        assert "empty retrieval" in md
        assert "`d1`" in md and "missing steps" in md
        assert "[update] **rb.md**" in md
        assert "new section" in md and "acceptance A" in md

    def test_summarizers_valid_and_fallback(self):
        gap = _summarize_gap('{"gap_type": "outdated", "evidence": "threshold is from 2023"}')
        assert "outdated" in gap and "2023" in gap
        quality = _summarize_quality(
            '{"doc_quality": [{"doc_id": "d", "issues": ["a", "b"]}]}'
        )
        assert "evaluated 1" in quality and "2 issues" in quality
        remediation = _summarize_remediation(
            '{"priority": "high", "summary": "missing runbook", "actions": [{}, {}]}'
        )
        assert "2 actions" in remediation and "high" in remediation
        # Invalid JSON falls back to tolerant text, no exception raised
        assert "tolerant" in _summarize_gap("not json")
        assert "tolerant" in _summarize_quality("not json")
        assert "tolerant" in _summarize_remediation("not json")


class TestCrewRunWithFakeCrew:
    def test_run_normalizes_enums_and_attaches_md(self, monkeypatch):
        settings = FakeSettings()
        crew = GovernanceCrew(settings)
        built = {}

        def fake_build(sink):
            obj = _FakeCrewObj('{"gap_type": "weird", "priority": "weird"}', sink)
            built["obj"] = obj
            return obj

        monkeypatch.setattr(crew, "_build", fake_build)
        events = []
        result = crew.run(
            "question?", TRIGGER_EMPTY, "sre", {}, [],
            on_event=lambda *a, **k: events.append((a, k)),
        )
        # Invalid enum normalized: no matched docs → missing; empty_retrieval → high
        assert result["gap_type"] == "missing"
        assert result["priority"] == "high"
        assert "suggestion_md" in result
        # all three stage completion events are passed through to sink
        done_events = [a[0][0] for a in events]
        assert done_events.count("agent_done") == 3

    def test_run_slims_doc_refs(self, monkeypatch):
        settings = FakeSettings()
        crew = GovernanceCrew(settings)
        built = {}

        def fake_build(sink):
            obj = _FakeCrewObj(VALID_RESULT_JSON, sink)
            built["obj"] = obj
            return obj

        monkeypatch.setattr(crew, "_build", fake_build)
        docs = [{"doc_id": f"d{i}", "content": "x" * 500} for i in range(8)]
        crew.run("question?", "low_score", "sre", {"faithfulness": 2.0}, docs)
        slim = json.loads(built["obj"].inputs["doc_refs"])
        assert len(slim) == 5  # max 5 docs
        assert len(slim[0]["snippet"]) == 200  # snippet truncated to 200
        assert built["obj"].inputs["user_group"] == "sre"
        assert json.loads(built["obj"].inputs["eval_scores"])["faithfulness"] == 2.0

    def test_run_without_sink_does_not_raise(self, monkeypatch):
        settings = FakeSettings()
        crew = GovernanceCrew(settings)
        monkeypatch.setattr(
            crew, "_build", lambda sink: _FakeCrewObj(VALID_RESULT_JSON, sink)
        )
        result = crew.run("question?", "manual", "", {}, [])
        assert result["gap_type"] == "missing"

    def test_run_raises_when_crewai_missing(self, monkeypatch):
        settings = FakeSettings()
        crew = GovernanceCrew(settings)
        monkeypatch.setattr(crew_mod, "is_crewai_available", lambda: False)
        with pytest.raises(RuntimeError):
            crew.run("question?", TRIGGER_EMPTY, "sre", {}, [])

    def test_invalid_enum_with_refs_defaults_fragmented(self, monkeypatch):
        settings = FakeSettings()
        crew = GovernanceCrew(settings)
        monkeypatch.setattr(
            crew, "_build",
            lambda sink: _FakeCrewObj('{"gap_type": "??", "priority": "??"}', sink),
        )
        # Has matched docs but type invalid → fragmented; non-empty retrieval defaults to medium priority
        result = crew.run("question?", "low_score", "sre", {}, [{"doc_id": "d1"}])
        assert result["gap_type"] == "fragmented"
        assert result["priority"] == "medium"


# ============================================================================
# 3. Manager: conditional routing + async execution
# ============================================================================

def _trace_result(docs, answer="answer based on knowledge base", **scores):
    trace = []
    if scores:
        trace.append({"step": "evaluate_answer", "status": "ok", "detail": dict(scores)})
    return {
        "intent": "query",
        "retrieved_docs": docs,
        "citations": docs,
        "answer": answer,
        "trace": trace,
    }


class TestDetectTrigger:
    def setup_method(self):
        self.settings = FakeSettings()

    def test_chitchat_never_triggers(self):
        result = _trace_result([], answer="hello")
        result["intent"] = "chitchat"
        assert detect_trigger(result, self.settings) is None

    def test_empty_docs_triggers(self):
        assert detect_trigger(_trace_result([]), self.settings) == TRIGGER_EMPTY

    def test_no_knowledge_prefix_triggers_even_with_docs(self):
        result = _trace_result([{"id": 1}], answer="No relevant knowledge found, please rephrase")
        assert detect_trigger(result, self.settings) == TRIGGER_EMPTY

    def test_healthy_docs_no_trigger(self):
        result = _trace_result(
            [{"id": 1}], faithfulness=4.5, answer_relevancy=4.5, hallucination_score=0.1
        )
        assert detect_trigger(result, self.settings) is None

    def test_docs_without_scores_no_trigger(self):
        assert detect_trigger(_trace_result([{"id": 1}]), self.settings) is None

    def test_low_faithfulness_triggers(self):
        result = _trace_result(
            [{"id": 1}], faithfulness=3.4, answer_relevancy=4.5, hallucination_score=0.1
        )
        assert detect_trigger(result, self.settings) == TRIGGER_LOW_SCORE

    def test_low_relevancy_triggers(self):
        result = _trace_result(
            [{"id": 1}], faithfulness=4.5, answer_relevancy=3.4, hallucination_score=0.1
        )
        assert detect_trigger(result, self.settings) == TRIGGER_LOW_SCORE

    def test_high_hallucination_triggers(self):
        result = _trace_result(
            [{"id": 1}], faithfulness=4.5, answer_relevancy=4.5, hallucination_score=0.31
        )
        assert detect_trigger(result, self.settings) == TRIGGER_LOW_SCORE

    def test_threshold_boundary_not_trigger(self):
        # Exactly equal to threshold: faithfulness/relevancy use < comparison, hallucination uses > comparison
        result = _trace_result(
            [{"id": 1}], faithfulness=3.5, answer_relevancy=3.5, hallucination_score=0.3
        )
        assert detect_trigger(result, self.settings) is None

    def test_zero_scores_ignored(self):
        # When eval step lacks fields it is 0.0, must not be misjudged as low score
        result = _trace_result(
            [{"id": 1}], faithfulness=0.0, answer_relevancy=0.0, hallucination_score=0.0
        )
        assert detect_trigger(result, self.settings) is None


class TestExtractEvalScores:
    def test_extract_from_ok_step(self):
        result = {"trace": [{"step": "evaluate_answer", "status": "ok", "detail": {
            "faithfulness": 4, "answer_relevancy": 3, "hallucination_score": 0.2}}]}
        scores = extract_eval_scores(result)
        assert scores == {"faithfulness": 4.0, "answer_relevancy": 3.0, "hallucination_score": 0.2}

    def test_ignore_failed_step_and_empty(self):
        assert extract_eval_scores({"trace": [
            {"step": "evaluate_answer", "status": "error", "detail": {}}]}) == {}
        assert extract_eval_scores({}) == {}


@pytest.mark.asyncio
class TestManagerFlow:
    async def _make_manager(self, tmp_path, fake_crew=None, enabled=True):
        store = GovernanceTaskStore(str(tmp_path / "mgr.db"))
        manager = GovernanceManager(FakeSettings(enabled), store)
        manager.crew = fake_crew or FakeGovernanceCrew()
        return manager, store

    async def test_happy_path_runs_crew_and_records_events(self, tmp_path):
        manager, store = await self._make_manager(tmp_path)
        result = _trace_result([], answer="No relevant knowledge found")
        task_id = await manager.maybe_trigger_after_chat(result, "Kafka backlog?", "sre", "s1")
        assert task_id and len(task_id) == 12
        # task status is running at creation moment
        assert store.get_task(task_id)["status"] == STATUS_RUNNING

        row = await _await_status(store, task_id, {STATUS_OPEN})
        assert row["gap_type"] == "missing"
        assert "add docs" in row["suggestion"]

        events = store.list_events(task_id)
        stages = [e["stage"] for e in events]
        assert stages[0] == "created"
        assert "Empty retrieval" in events[0]["message"]
        assert "crew_start" in stages
        assert stages.count("agent_done") == 3
        assert stages[-1] == "done"
        assert "1 document" in events[-1]["message"]
        await manager.aclose()
        store.close()

    async def test_failure_marks_error_and_event(self, tmp_path):
        manager, store = await self._make_manager(
            tmp_path, FakeGovernanceCrew(fail=True)
        )
        task_id = await manager.trigger_manual("failed question?", "sre")
        row = await _await_status(store, task_id, {STATUS_OPEN})
        assert "Simulated Crew execution failure" in row["error"]
        stages = [e["stage"] for e in store.list_events(task_id)]
        assert stages[-1] == "error"
        await manager.aclose()
        store.close()

    async def test_dedup_skips_second_trigger(self, tmp_path):
        manager, store = await self._make_manager(tmp_path)
        result = _trace_result([])
        first_id = await manager.maybe_trigger_after_chat(result, "dedup question?", "sre", "s")
        assert first_id is not None
        # First one still running or open after completion, both should be deduped
        assert await manager.maybe_trigger_after_chat(result, "dedup question?", "sre", "s") is None
        await _await_status(store, first_id, {STATUS_OPEN})
        assert await manager.trigger_manual("dedup question?", "sre") is None
        await manager.aclose()
        store.close()

    async def test_disabled_returns_none(self, tmp_path):
        manager, store = await self._make_manager(tmp_path, enabled=False)
        assert await manager.maybe_trigger_after_chat(_trace_result([]), "q", "sre", "s") is None
        await manager.aclose()
        store.close()

    async def test_no_trigger_when_healthy(self, tmp_path):
        manager, store = await self._make_manager(tmp_path)
        result = _trace_result(
            [{"id": 1}], faithfulness=4.8, answer_relevancy=4.8, hallucination_score=0.0
        )
        assert await manager.maybe_trigger_after_chat(result, "normal question", "sre", "s") is None
        assert store.list_tasks() == []
        await manager.aclose()
        store.close()

    async def test_eval_scores_passed_to_crew(self, tmp_path):
        fake = FakeGovernanceCrew()
        manager, store = await self._make_manager(tmp_path, fake)
        result = _trace_result(
            [{"id": 1}], faithfulness=2.0, answer_relevancy=2.5, hallucination_score=0.6
        )
        task_id = await manager.maybe_trigger_after_chat(result, "low score question", "sre", "s")
        await _await_status(store, task_id, {STATUS_OPEN})
        assert fake.calls[0]["trigger_type"] == TRIGGER_LOW_SCORE
        assert fake.calls[0]["eval_scores"]["faithfulness"] == 2.0
        assert callable(fake.calls[0]["on_event"])
        await manager.aclose()
        store.close()


# ============================================================================
# 4. REST API
# ============================================================================

@pytest.fixture
def api_client(tmp_path):
    """Inject temp store + FakeCrew into route globals; TestClient overrides lifespan default instance after entering."""
    from app.api import governance_routes as gr
    from app.main import create_app

    store = GovernanceTaskStore(str(tmp_path / "api.db"))
    manager = GovernanceManager(FakeSettings(), store)
    manager.crew = FakeGovernanceCrew()
    with TestClient(create_app()) as client:
        gr.set_governance(manager, store)
        yield client, store, manager, gr
        gr.set_governance(None, None)
    store.close()


class TestGovernanceAPI:
    def test_status_with_counts(self, api_client):
        client, store, _, _ = api_client
        store.create_task("q1", TRIGGER_EMPTY)
        data = client.get("/governance/status").json()
        assert data["enabled"] is True
        assert data["crewai_available"] is True
        assert data["task_counts"]["running"] == 1

    def test_list_and_filter_tasks(self, api_client):
        client, store, _, _ = api_client
        r1 = store.create_task("q1", TRIGGER_EMPTY)
        store.save_result(r1["id"], "missing", "high", {}, "md")
        assert client.get("/governance/tasks?status=open").json()[0]["id"] == r1["id"]
        assert client.get("/governance/tasks?status=running").json() == []
        bad = client.get("/governance/tasks?status=bogus")
        assert bad.status_code == 400

    def test_get_task_404(self, api_client):
        client, _, _, _ = api_client
        assert client.get("/governance/tasks/nope").status_code == 404

    def test_patch_status_transitions(self, api_client):
        client, store, _, _ = api_client
        tid = store.create_task("q", "manual")["id"]
        store.save_result(tid, "missing", "medium", {}, "")
        assert client.patch(f"/governance/tasks/{tid}", json={"status": "in_progress"}).json()[
            "status"
        ] == "in_progress"
        assert client.patch(f"/governance/tasks/{tid}", json={"status": "resolved"}).json()[
            "status"
        ] == "resolved"

    def test_patch_running_rejected(self, api_client):
        client, store, _, _ = api_client
        tid = store.create_task("q", "manual")["id"]  # still running
        resp = client.patch(f"/governance/tasks/{tid}", json={"status": "resolved"})
        # running is a system-occupied status: routes forbid manually changing a task to 'running';
        # but normal transition of running tasks (→resolved) should be allowed
        assert resp.status_code == 200
        bad = client.patch(f"/governance/tasks/{tid}", json={"status": "running"})
        assert bad.status_code == 400
        invalid = client.patch(f"/governance/tasks/{tid}", json={"status": "deleted"})
        assert invalid.status_code == 400
        assert client.patch("/governance/tasks/missing", json={"status": "open"}).status_code == 404

    def test_events_endpoint(self, api_client):
        client, store, _, _ = api_client
        tid = store.create_task("q", "manual")["id"]
        store.add_event(tid, "created", "Task created")
        store.add_event(tid, "agent_step", "analyzing", "Blind Spot Analyst")
        events = client.get(f"/governance/tasks/{tid}/events").json()
        assert [e["stage"] for e in events] == ["created", "agent_step"]
        assert events[1]["agent"] == "Blind Spot Analyst"
        assert client.get("/governance/tasks/missing/events").status_code == 404

    def test_manual_trigger_creates_and_dedups(self, api_client):
        client, store, _, _ = api_client
        resp = client.post(
            "/governance/trigger",
            json={"question": "manual governance question?", "user_group": "sre", "session_id": "api-1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "created"
        tid = data["task_id"]
        # Events complete after Crew finishes
        row = _wait_status(store, tid, {STATUS_OPEN})
        assert row["gap_type"] == "missing"
        stages = [e["stage"] for e in store.list_events(tid)]
        assert "agent_done" in stages and stages[-1] == "done"
        # Repeated same question → skipped
        again = client.post(
            "/governance/trigger", json={"question": "manual governance question?", "user_group": "sre"}
        ).json()
        assert again["status"] == "skipped"

    def test_manual_trigger_validation(self, api_client):
        client, _, _, _ = api_client
        assert client.post("/governance/trigger", json={"question": ""}).status_code == 422

    def test_crewai_unavailable_returns_503(self, api_client, monkeypatch):
        client, _, _, gr = api_client
        monkeypatch.setattr(gr, "is_crewai_available", lambda: False)
        resp = client.post("/governance/trigger", json={"question": "q?"})
        assert resp.status_code == 503

    def test_store_not_initialized_503(self):
        """When routes not injected (governance disabled), return 503 not 500."""
        from app.api import governance_routes as gr
        from app.main import create_app

        with TestClient(create_app()) as client:
            gr.set_governance(None, None)
            assert client.get("/governance/tasks").status_code == 503
            assert client.get("/governance/status").json()["enabled"] is False
