"""DeepEval 评估流水线（设计文档 5.3）。

用法：
  python -m eval.run_deepeval --dataset data/eval/qa.jsonl

指标：Faithfulness / Answer Relevancy（基于 DeepEval 的 GEval/指标）
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


async def _evaluate_item(workflow: RAGWorkflow, item: dict[str, Any]) -> dict[str, Any]:
    result = await workflow.arun(
        question=item["question"],
        user_group=item.get("user_group", "sre"),
        session_id=f"de-{item.get('id','')}",
    )
    item["answer"] = result.get("answer", "")
    item["contexts"] = [c["snippet"] for c in result.get("citations", [])]
    return item


async def run(dataset_path: str, report_path: str = "data/eval/deepeval_report.md") -> None:
    settings = get_settings()
    workflow = RAGWorkflow(settings=settings)
    await workflow.a_init()

    rows = [json.loads(l) for l in open(dataset_path, "r", encoding="utf-8") if l.strip()]
    evaluated = []
    for item in rows:
        try:
            evaluated.append(await _evaluate_item(workflow, item))
        except Exception as exc:  # noqa: BLE001
            print(f"[deepeval] 跳过 {item.get('id')}: {exc}", file=sys.stderr)

    metrics: dict[str, Any] = {"samples": len(evaluated)}
    try:
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        faith = FaithfulnessMetric(threshold=0.7)
        rel = AnswerRelevancyMetric(threshold=0.7)
        faith_scores: list[float] = []
        rel_scores: list[float] = []
        for s in evaluated:
            tc = LLMTestCase(
                input=s.get("question", ""),
                actual_output=s.get("answer", ""),
                expected_output=s.get("ground_truth", ""),
                retrieval_context=s.get("contexts", []),
            )
            faith.measure(tc)
            rel.measure(tc)
            faith_scores.append(faith.score or 0.0)
            rel_scores.append(rel.score or 0.0)
        metrics["faithfulness"] = sum(faith_scores) / max(len(faith_scores), 1)
        metrics["answer_relevancy"] = sum(rel_scores) / max(len(rel_scores), 1)
    except Exception as exc:  # noqa: BLE001
        print(f"[deepeval] DeepEval 不可用：{exc}", file=sys.stderr)
        metrics["note"] = "deepeval_unavailable"

    _write_report(report_path, metrics, evaluated)
    await workflow.a_close()


def _write_report(path: str, metrics: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["# DeepEval 评估报告\n", f"样本数: {len(samples)}\n", "## 指标\n"]
    for k, v in metrics.items():
        lines.append(f"- **{k}**: {v}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[deepeval] 报告已写入 {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 DeepEval 评估")
    parser.add_argument("--dataset", default="data/eval/qa.jsonl")
    parser.add_argument("--report", default="data/eval/deepeval_report.md")
    args = parser.parse_args()
    asyncio.run(run(args.dataset, args.report))


if __name__ == "__main__":
    main()
