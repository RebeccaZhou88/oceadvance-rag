"""Offline retrieval evaluation — zero Chat LLM cost, deterministic, repeatable.

Usage (under backend dir):
  $env:PYTHONPATH = "src"
  python -m eval.run_retrieval_eval --dataset data/eval/golden.jsonl
  python -m eval.run_retrieval_eval --rerank          # also evaluate reranked order

Metrics:
  Precision@K / Recall@K (binary, relevance>=1 deemed relevant)
  MRR (reciprocal rank of first relevant snippet)
  nDCG@K (graded relevance 0/1/2)
  Permission violation count (forbidden_sources appearing in results)
  Intent consistency (chitchat cases, pure heuristic no LLM)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.retrieval.search import build_retriever

CUTOFFS = [1, 3, 5]


# ───────────────────────── Annotation parsing ─────────────────────────

def _doc_grade(doc: dict[str, Any], labels: list[dict[str, Any]]) -> int:
    """Return doc's annotation grade 0/1/2 for a query; take highest match in label order."""
    content = doc.get("content") or ""
    source = doc.get("source") or ""
    grade = 0
    for lab in labels:
        if not source.endswith(lab["source"]):
            continue
        if "heading" in lab and lab["heading"] not in content:
            continue
        if "contains" in lab and not all(t in content for t in lab["contains"]):
            continue
        grade = max(grade, int(lab.get("relevance", 1)))
    return grade


# ───────────────────────── Metric computation ─────────────────────────

def _dcg(grades: list[int], k: int) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def _per_query(grades: list[int], gold_total: int) -> dict[str, float]:
    """grades: each grade sorted by system. gold_total: gold standard relevant count."""
    out: dict[str, float] = {}
    for k in CUTOFFS:
        topk = grades[:k]
        hits = sum(1 for g in topk if g >= 1)
        out[f"P@{k}"] = hits / k
        out[f"R@{k}"] = (hits / gold_total) if gold_total else float("nan")
    first = next((i for i, g in enumerate(grades) if g >= 1), None)
    out["MRR"] = 1.0 / (first + 1) if first is not None else 0.0
    for k in CUTOFFS:
        idcg = _dcg(sorted([2] * min(gold_total, k), reverse=True), k)
        out[f"nDCG@{k}"] = (_dcg(grades, k) / idcg) if idcg else float("nan")
    return out


def _macro(rows: list[dict[str, float]], key: str) -> float:
    vals = [r[key] for r in rows if not math.isnan(r[key])]
    return sum(vals) / len(vals) if vals else float("nan")


# ───────────────────────── Main flow ─────────────────────────

async def run(dataset: str, use_rerank: bool, report: str) -> None:
    settings = get_settings()
    retriever = build_retriever(settings)
    reranker = None
    if use_rerank:
        from app.retrieval.rerank import build_reranker
        reranker = build_reranker(settings)

    cases = [json.loads(l) for l in Path(dataset).read_text(encoding="utf-8").splitlines() if l.strip()]
    per_query_rows: list[tuple[str, str, dict[str, float], dict[str, float] | None, int]] = []
    intent_mismatches: list[str] = []

    for case in cases:
        cid, q, group = case["id"], case["question"], case.get("user_group", "sre")
        gold_total = len(case["relevant"])

        # Intent consistency (pure heuristic, no LLM)
        if case.get("intent"):
            from app.graph.nodes import IntentCheckNode
            got_intent = IntentCheckNode()({"question": q})["intent"]
            if got_intent != case["intent"]:
                intent_mismatches.append(f"{cid}: expected {case['intent']}, got {got_intent}")
            if case["intent"] == "chitchat":
                per_query_rows.append((cid, q, {}, None, 0))
                continue

        docs = await retriever.hybrid_search(q, group, top_k=settings.hybrid_top_k)
        grades = [_doc_grade(d, case["relevant"]) for d in docs]
        metrics = _per_query(grades, gold_total)

        # Empty gold standard hard negative: correct_empty = no relevant retrieved at all
        if gold_total == 0:
            metrics["correct_empty"] = 1.0 if not any(g >= 1 for g in grades) else 0.0

        # Permission violation
        violations = sum(
            1 for d in docs
            if any((d.get("source") or "").endswith(s) for s in case.get("forbidden_sources", []))
        )

        reranked_metrics = None
        if reranker is not None:
            ranked = await reranker.rerank(q, docs, top_k=settings.final_top_k)
            rg = [_doc_grade(d, case["relevant"]) for d in ranked]
            reranked_metrics = _per_query(rg, gold_total)

        per_query_rows.append((cid, q, metrics, reranked_metrics, violations))

    _write_report(report, per_query_rows, intent_mismatches, use_rerank)

    if reranker is not None and hasattr(reranker, "aclose"):
        await reranker.aclose()


def _write_report(path: str, rows, intent_mismatches, use_rerank) -> None:
    scored = [m for _, _, m, _, _ in rows if m]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    L = ["# Offline Retrieval Evaluation Report\n", f"Samples: {len(rows)}\n"]

    L.append("## Summary (retrieval order" + (" / reranked)\n" if use_rerank else ")\n"))
    L.append("| Metric | Retrieval |" + (" Rerank |" if use_rerank else ""))
    L.append("|---|---:|" + ("---:|" if use_rerank else ""))
    for key in ["P@1", "P@3", "P@5", "R@1", "R@3", "R@5", "MRR", "nDCG@1", "nDCG@3", "nDCG@5"]:
        v = _macro(scored, key)
        line = f"| {key} | {v:.3f} |"
        if use_rerank:
            rv = _macro([rm for _, _, _, rm, _ in rows if rm], key)
            line += f" {rv:.3f} |"
        L.append(line)
    ce = [m.get("correct_empty") for m in scored if "correct_empty" in m]
    if ce:
        L.append(f"\nHard negative accuracy (expect 0 recall): {sum(ce)/len(ce):.0%}")
    total_v = sum(v for _, _, _, _, v in rows)
    L.append(f"\nTotal permission violations: **{total_v}** (expect 0)")
    L.append(f"\nIntent mismatches: **{len(intent_mismatches)}**" +
             ("\n" + "\n".join(f"- {x}" for x in intent_mismatches) if intent_mismatches else ""))

    L.append("\n## Per-sample\n")
    L.append("| ID | Question | R@3 | nDCG@3 | MRR | Violations |")
    L.append("|---|---|---:|---:|---:|---:|")
    for cid, q, m, _, v in rows:
        if not m:
            L.append(f"| {cid} | {q[:28]} | chitchat | - | - | - |")
        else:
            r3 = m.get("R@3"); n3 = m.get("nDCG@3")
            fmt = lambda x: "-" if math.isnan(x) else f"{x:.2f}"
            L.append(f"| {cid} | {q[:28]} | {fmt(r3)} | {fmt(n3)} | {m['MRR']:.2f} | {v} |")

    Path(path).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:20]))
    print(f"\n[eval] Full report written to {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Offline retrieval evaluation (zero Chat LLM cost)")
    p.add_argument("--dataset", default="data/eval/golden.jsonl")
    p.add_argument("--report", default="data/eval/retrieval_report.md")
    p.add_argument("--rerank", action="store_true", help="Also evaluate reranked order")
    a = p.parse_args()
    asyncio.run(run(a.dataset, a.rerank, a.report))


if __name__ == "__main__":
    main()
