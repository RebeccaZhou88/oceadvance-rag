# OCECopilotAdvanceRAG

> Advanced RAG prototype for SRE knowledge bases — LangGraph + Azure AI Search + CrewAI governance.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/langgraph-0.2.34-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/fastapi-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Azure Search](https://img.shields.io/badge/azure-ai_search-blue.svg)](https://learn.microsoft.com/en-us/azure/search/)
[![CrewAI](https://img.shields.io/badge/crewai-0.175.0-teal.svg)](https://github.com/crewAIInc/crewAI)

A production-oriented RAG system that goes beyond naive vector search. Combines **intent routing**, **cross-encoder reranking**, **online self-evaluation**, and an asynchronous **CrewAI quality governance loop** that continuously closes the "poor answer → document gap → remediation" feedback cycle.

---

## ✨ Features

| Capability | What it does |
|---|---|
| Intent-aware routing | `chitchat` bypasses retrieval; empty retrieval skips LLM → saves tokens |
| Hybrid retrieval | Pure vector search over Azure AI Search + `RRF` fusion; **embedding source and dimension must match** write path |
| Pluggable rerank | Cross-Encoder / LLM / Semantic / none; before/after rank diff recorded for each result |
| Citation-grounded answers | `[1] [2]` inline citations trace back to the exact document chunk |
| Role-based permission filter | OData filter on `allowed_groups` + post-retrieval assertion — sensitive docs never enter the LLM context |
| Online self-evaluation | Faithfulness / Answer Relevancy / Hallucination scored by LLM self-check |
| **CrewAI quality governance** | **3-agent sequential Crew (Blind Spot → Doc Quality → Remediation)** fires **asynchronously** when answers fall below thresholds; status tracked in SQLite kanban |
| Full observability | Prometheus metrics + structured terminal logs per LangGraph node |

## 🏗️ Architecture

![LangGraph + CrewAI workflow](docs/main-process.png)

CrewAI does **not** live inside the LangGraph graph — it fires after `END` via `asyncio.create_task` in `routes.py`, so governance is always non-blocking.

## 🛠️ Tech Stack

| Layer | Choice |
|---|---|
| Backend | Python 3.11 · FastAPI · LangGraph 0.2.34 |
| Retrieval | Azure AI Search (async SDK), pure vector + RRF |
| Embedding | Azure `text-embedding-ada-002` · DashScope `text-embedding-v2` (both 1536-D; **pick one, stay consistent**) |
| Chat | DashScope qwen / DeepSeek / Azure OpenAI via OpenAI-compatible endpoint |
| Rerank | Cross-Encoder `BAAI/bge-reranker-base` · LLM · Semantic · none |
| Eval | LLM self-check (`EVALUATE_SYSTEM_PROMPT`) |
| Governance | CrewAI 0.175.0 (**optional**; install or leave out, main flow works either way) |
| Storage | SQLite (WAL) for governance task store |
| Metrics | Prometheus `/metrics` |

## 📂 Directory

```
OCECopilotAdvanceRAG/
├── backend/
│   ├── ingestion/          # Chunking, embedding, index create, merge-or-upload
│   │   ├── indexer.py
│   │   ├── create_index.py
│   │   └── chunking.py
│   ├── src/app/
│   │   ├── graph/          # LangGraph 5 nodes + workflow assembly
│   │   ├── retrieval/      # Azure search + rerank (cross-encoder / llm / semantic)
│   │   ├── governance/     # CrewAI crew + manager + SQLite store
│   │   ├── api/            # FastAPI routes (+ governance_routes.py)
│   │   ├── llm/            # Azure / DashScope / Mock chat + embed clients
│   │   ├── security/       # allowed_groups OData filter + assert_no_leak
│   │   └── observability/  # Prometheus metrics + structured logging
│   └── data/raw/           # Runbooks / Postmortems / Kusto / IcM (grouped by permission)
├── frontend/               # index.html (single-page, no build step)
└── docs/
```

## 🚀 Quick Start

### 1. Install

```powershell
# conda env (Windows)
conda create -n oce-rag python=3.11 -y
conda activate oce-rag

cd backend
pip install -r requirements.txt
# optional: governance
pip install crewai==0.175.0 litellm==1.74.9 openai==1.109.1
```

### 2. Configure

Create `backend/.env`:

```env
# Azure AI Search (retrieval backend)
USE_MOCK_BACKEND=false
AZURE_SEARCH_ENDPOINT=https://<name>.search.windows.net
AZURE_SEARCH_INDEX_NAME=rag-index-v2

# Embedding — pick ONE, keep consistent with ingestion
LLM_EMBEDDING_MODEL_NAME=text-embedding-v2
LLM_EMBEDDING_DIMENSIONS=1536

# Chat model (OpenAI-compatible endpoint)
LLM_MODEL_NAME=qwen-plus
LLM_MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Rerank
RERANK_STRATEGY=cross-encoder
HYBRID_TOP_K=20
FINAL_TOP_K=5
RRF_K=60
```

**Secrets**: Put `AZURE_SEARCH_API_KEY` / `LLM_API_KEY` / `AZURE_OPENAI_API_KEY` into Windows user-level environment variables (`[Environment]::SetEnvironmentVariable($name, $value, 'User')`) rather than `.env`. pydantic-settings reads system env vars first.

### 3. Ingest + Run

```powershell
cd backend
$env:PYTHONPATH = "src"

# 1) create index (once)
python -m ingestion.create_index

# 2) ingest docs (run whenever data/raw/ changes)
python -m ingestion.indexer --data data/raw

# 3) start backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 4. Enjoy

![ui-screenshot](docs\ui-screenshot.png)

![governance](docs\governance.png)

## 📜 License

All data included in this repository has been sanitized; no real credentials, customer data, or API keys are present.
