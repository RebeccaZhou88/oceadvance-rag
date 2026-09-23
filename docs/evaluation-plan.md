# RAG Annotated Evaluation Set Plan

> Goal: Use a **reproducible, zero Chat-LLM-cost** offline evaluation to quantify "is retrieval accurate, is ranking correct, are permissions leaking",
> so every change to chunking strategy / embedding / rerank has objective regression evidence rather than relying on gut feeling.

## 1. Evaluation Layers

| Layer | Evaluation Content | Calls LLM | Tool |
|---|---|---|---|
| Retrieval | Whether recall includes gold-standard chunks, ranking quality, permission filtering | No (embedding only) | [run_retrieval_eval.py](../backend/eval/run_retrieval_eval.py) |
| Generation | Faithfulness / Answer Relevancy / Hallucination | Yes | run_ragas.py / run_deepeval.py |
| End-to-end | Intent classification accuracy, latency, tokens | No (heuristic) | Built into run_retrieval_eval.py |

**Run the retrieval layer for daily regression** (cheap, deterministic); run the generation layer once before each release.

## 2. Dataset Design

File: [golden.jsonl](../backend/data/eval/golden.jsonl). One sample per line:

```json
{
  "id": "g01",
  "question": "SharePoint pages load slowly — how to troubleshoot?",
  "user_group": "sre",
  "intent": "query",
  "relevant": [
    {"source": "sharepoint_latency.md", "heading": "Troubleshooting Target", "relevance": 2},
    {"source": "appgw_failures.kql", "contains": ["failedReqRatio"], "relevance": 1}
  ],
  "forbidden_sources": [],
  "notes": "Annotation rationale"
}
```

Annotations **do not reference volatile chunk_id**; instead they use `source + heading / content feature keywords` (chunk content starts with an `[H1 › H2]` path for stable matching). Annotations remain valid after re-indexing or re-chunking, and are resolved to real chunks at runtime.

### Sample Composition (currently 12 items, balanced by type)

| Type | Target Ratio | Description |
|---|---|---|
| Fact/Step (has standard answer) | ~50% | Runbook / Kusto main scenarios |
| Cross-document | ~15% | One question requires both postmortem + icm |
| Hard negative (no answer in index) | ~15% | Expect 0 recall + fallback answer, tests `correct_empty` |
| Permission | ~10% | dev/ops groups, with `forbidden_sources` |
| Chitchat | ~10% | Expect chitchat intent, skip retrieval |

Scale plan: when documents <=50, use **30-50 items**; afterward expand with ">=2 positive items per document + 10% hard negatives across the set", up to ~300 items (a single evaluation run completes within 5 minutes).

## 3. Annotation Guidelines

**Relevance Grading**

| Grade | Definition | Criteria |
|---|---|---|
| 2 (Fully relevant) | Cannot answer completely/correctly without this chunk | Contains the core steps, thresholds, or conclusions asked by the question |
| 1 (Partially relevant) | Provides supporting evidence or supplementary validation | E.g., KQL referenced by a runbook, incident metadata |
| 0 (Irrelevant) | No annotation needed (default 0) | Same topic but doesn't answer this question also counts as 0 |

**Rules**

- Each question must have at least one grade-2 annotation; otherwise either rewrite the question or supplement the document
- Annotations must point to specific sentences in the chunk source text (written into `notes`); guessing from headings is prohibited
- Hard negatives `relevant: []`: choose near-miss questions that "look like they might have an answer but don't" (e.g., asking about Kubernetes when the library only has SharePoint) — these better detect false positives than completely unrelated questions
- Permission cases: positives only annotate chunks visible to that group; additionally list `forbidden_sources`

**Quality Control**

- Each item is annotated independently by 2 annotators; binarize (>=1) and compute Cohen's Kappa; **items < 0.7 are re-aligned**
- At current scale (<50 items), single-annotator + author review is acceptable; switch to double-annotation above 50 items

## 4. Metric Definitions

| Metric | Definition | Focus |
|---|---|---|
| Precision@K | Proportion of relevant items in top-K | Purity of context fed to the LLM |
| Recall@K | Proportion of gold-standard relevant chunks recalled | Whether anything is missed (**retrieval layer first priority**) |
| MRR | Reciprocal rank of the first relevant chunk | Is the most relevant result ranked high |
| nDCG@K | Graded relevance discounted gain | Overall ranking quality (grade-2 should rank above grade-1) |
| correct_empty | Proportion of hard negatives with 0 recall | False-positive control |
| Permission violations | Count of forbidden_sources appearances | **Must be 0**, security red line |
| Intent consistency | Heuristic intent vs gold-standard intent | Whether chitchat bypass is correct |

Aggregation uses **macro-average** (each query weighted equally, avoiding over-weighting samples with larger gold standards). Samples with empty gold standards are excluded from Recall and reported separately as correct_empty.

## 5. How to Run

```powershell
cd backend
$env:PYTHONPATH = "src"
# Retrieval layer regression (embedding cost only)
python -m eval.run_retrieval_eval --dataset data/eval/golden.jsonl
# Also evaluate ranking after cross-encoder rerank
python -m eval.run_retrieval_eval --rerank
# Pre-release: generation layer (requires Chat LLM)
python -m eval.run_ragas --dataset data/eval/qa.jsonl
```

Output: `data/eval/retrieval_report.md` (summary table + per-sample details).

## 6. CI and Iteration

- CI (.github/workflows/ci.yml) adds a step: run the retrieval evaluation script in mock mode to verify the evaluation pipeline itself works (Azure credentials are not in CI)
- Local regression gate recommendation: **R@3 >= 0.8, nDCG@3 >= 0.75, permission violations = 0, hard negative accuracy = 100%**; changes below baseline require PR explanation
- Report files (*_report.md) are not committed; gold standard golden.jsonl is committed
- Triggers for expanding annotations: new documents added, new retrieval strategy deployed, production bad cases of "should-have-hit-but-missed" (collected from the governance board)

## 7. Deliverables

| File | Description |
|---|---|
| [golden.jsonl](../backend/data/eval/golden.jsonl) | 12 seed gold-standard items (covering all 5 types) |
| [run_retrieval_eval.py](../backend/eval/run_retrieval_eval.py) | Retrieval layer evaluator (stdlib implementation, metrics self-tested) |
| This document | Plan and annotation guidelines |
