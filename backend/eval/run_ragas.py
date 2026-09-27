# @Author: RebeccaZhou
# @Description: Offline answer evaluation via Ragas
#              基于 Ragas 的离线答案评估

"""RAGAS evaluation pipeline (design doc 5.3).

Usage:
  python -m eval.run_ragas --dataset data/eval/qa.jsonl

Metrics: Faithfulness / Answer Relevancy / Context Precision / Context Recall
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.graph.workflow import RAGWorkflow


async def _run_pipeline(workflow: RAGWorkflow, item: dict[str, Any]) -> dict[str, Any]:
    result = await workflow.arun(
        question=item["question"],
        user_group=item.get("user_group", "sre"),
        session_id=f"eval-{item.get('id', '')}",
    )
    item["answer"] = result.get("answer", "")
    item["contexts"] = [c["snippet"] for c in result.get("citations", [])]
    return item


async def run(dataset_path: str, report_path: str = "data/eval/ragas_report.md") -> None:
    settings = get_settings()
    workflow = RAGWorkflow(settings=settings)
    await workflow.a_init()

    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    evaluated: list[dict[str, Any]] = []
    for item in rows:
        try:
            evaluated.append(await _run_pipeline(workflow, item))
        except Exception as exc:  # noqa: BLE001
            print(f"[ragas] Skipping failed sample {item.get('id')}: {exc}", file=sys.stderr)

    metrics: dict[str, float] = {}
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        ds = Dataset.from_list(evaluated)
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        metrics = {k: float(v) for k, v in result.items()}
    except Exception as exc:  # noqa: BLE001
        print(f"[ragas] RAGAS unavailable, skipping deep evaluation: {exc}", file=sys.stderr)
        metrics = {"note": "ragas_unavailable", "samples": len(evaluated)}

    _write_report(report_path, metrics, evaluated)
    await workflow.a_close()


def _write_report(path: str, metrics: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["# RAGAS Evaluation Report\n", f"Samples: {len(samples)}\n", "## Metrics\n"]
    for k, v in metrics.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## Samples\n")
    for s in samples[:5]:
        lines.append(f"### Q: {s.get('question','')}")
        lines.append(f"- Answer: {s.get('answer','')[:200]}")
        lines.append(f"- Ground truth: {s.get('ground_truth','')[:200]}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[ragas] Report written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--dataset", default="data/eval/qa.jsonl")
    parser.add_argument("--report", default="data/eval/ragas_report.md")
    args = parser.parse_args()
    asyncio.run(run(args.dataset, args.report))


if __name__ == "__main__":
    main()
