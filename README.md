# LogSage

An LLM-powered log analysis system that ingests unstructured system logs, mines recurring event templates, detects anomalous log lines via embedding distance, and generates root-cause diagnoses and suggested fixes using retrieval-augmented generation (RAG).

Built against the [Loghub HDFS](https://github.com/logpai/loghub) dataset as a realistic test corpus.

## Why this exists

Manually reading through thousands of system log lines to find the handful that actually matter is slow and error-prone. LogSage automates that triage:
- it learns what "normal" log activity looks like
- flags lines that deviate from it
- only spends LLM inference (and its associated cost/latency) on the anomalies.

## Architecture

```
Raw log file
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│   Parsing   │────▶│ Template     │────▶│  Embedding +  │
│ (regex →    │     │ Mining       │     │  FAISS Index  │
│  structured │     │ (Drain3)     │     │  (baseline)   │
│  fields)    │     └──────────────┘     └───────────────┘
└─────────────┘                                  │
                                                 ▼
                                        ┌───────────────────┐
                     New log line ────▶ │ Anomaly Detection │
                                        │ (distance vs.     │
                                        │  baseline index)  │
                                        └───────────────────┘
                                                  │
                                    distance > threshold?
                                         │              │
                                        no             yes
                                         │              │
                                         ▼              ▼
                                  return match     ┌───────────────┐
                                  (no LLM call)    │ RAG Retrieval │
                                                   │ (known fixes) │
                                                   └───────────────┘
                                                          │
                                                          ▼
                                                   ┌──────────────┐
                                                   │ Redaction +  │
                                                   │ LLM Analysis │
                                                   │ (root cause, │
                                                   │  fix)        │
                                                   └──────────────┘
```

### Key design decision: separate baseline vs. analysis ingestion

Anomaly detection only works if new events are scored against a **fixed, stable reference distribution**. If the log you're trying to analyze is included in the same batch used to build that reference index, it trivially matches itself (distance ≈ 0) and never gets flagged.

To prevent this, ingestion is split into two distinct endpoints:

- **`POST /upload/baseline`** — parses a log file, mines templates, and *rebuilds* the FAISS "normal" index from it. Use this with a clean, representative log sample.
- **`POST /upload/logs`** — parses a log file and stores each line for later analysis, **without** touching the baseline index.

This mirrors how anomaly detection is actually done in production systems: the reference distribution is established once (or periodically retrained), and new events are always scored against it.

## Pipeline stages

1. **Parsing** (`app/parsing.py`) — regex-based extraction of `date`, `time`, `pid`, `level`, `component`, and `message` from each HDFS-style log line. Malformed lines are skipped.
2. **Template mining** (`app/embedding.py`) — uses [Drain3](https://github.com/logpai/Drain3) to cluster raw messages into templates (e.g. `"blk_123 terminating"` → `"<*> terminating"`), collapsing thousands of near-duplicate lines into a small set of unique patterns.
3. **Embedding + indexing** (`app/vector_store.py`) — embeds unique templates with `sentence-transformers` (`all-MiniLM-L6-v2`) and stores them in a FAISS `IndexFlatL2` index. Rebuilt fresh on every `/upload/baseline` call to avoid stale/accumulated vectors.
4. **Anomaly detection** (`app/anomaly.py`) — embeds an incoming log line and computes its L2 distance to the nearest baseline template. Distance above a threshold (default `0.6`) triggers the LLM branch; below it, the log is treated as normal and returned immediately with **no LLM call** (this is the cost-control mechanism).
5. **RAG retrieval** (`app/rag.py`) — for anomalous lines, retrieves the top-k most similar entries from a small hand-curated knowledge base of known issues/fixes (`data/knowledge_base.json`).
6. **Redaction** (`app/redact.py`) — strips IPs, emails, and API-key-shaped strings from a log line before it's sent to any external LLM provider.
7. **LLM analysis** (`app/llm_analysis.py`) — sends the redacted anomalous log plus retrieved context to an LLM, requesting a structured JSON response: `root_cause`, `suggested_fix`, `confidence`.

## API Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `POST` | `/upload/baseline` | Builds or rebuilds the normal-pattern reference index from a clean log file. |
| `POST` | `/upload/logs` | Ingests a log file for later analysis (does not affect the baseline index). |
| `GET` | `/analyze/{trace_id}` | Runs anomaly detection (+ RAG + LLM if anomalous) on a specific parsed log line. |
| `GET` | `/logs` | Lists all currently stored trace IDs. |
| `GET` | `/docs` | Serves the interactive Swagger UI documentation. |

## Tech stack

- **Python 3.11+**
- **FastAPI** — async REST API
- **Drain3** — log template mining
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local, free embedding model
- **FAISS** (`faiss-cpu`) — vector similarity search
- **Groq API** (OpenAI-SDK-compatible, `openai/gpt-oss-20b`) — LLM inference for root-cause analysis. Any OpenAI-SDK-compatible provider works by swapping `base_url` and `model` in `app/llm_analysis.py` (the project was originally designed against `gpt-4o-mini`; Groq is used here for free-tier development/testing)
- **Docker** — containerized deployment
- **Loghub HDFS_2k** — test dataset

## Setup

### Prerequisites
- Docker Desktop
- An API key from OpenAI or Groq ([console.groq.com](https://console.groq.com))

### Environment variables

Copy `.env.example` to `.env` and fill in your key:

```
OPENAI_API_KEY=your_openai_api_key_here
MOCK_LLM=false
```

Setting `MOCK_LLM=true` bypasses the real LLM call and returns a canned response — useful for testing the full request/response flow without spending API credits.

### Build and run

```bash
docker build -t logsage .
docker run -p 8000:8000 --env-file .env logsage
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### Usage flow

1. `POST /upload/baseline` with a clean log file (e.g. `HDFS_2k.log`) — establishes what "normal" looks like.
2. `POST /upload/logs` with the file containing the log line(s) you want analyzed.
3. `GET /logs` to find the `trace_id` of the line you want to check (format: `{filename}-{line_index}`).
4. `GET /analyze/{trace_id}` — returns either a normal-match result or a full anomaly analysis with root cause and suggested fix.

## Testing

```bash
pip install pytest faiss-cpu numpy
pytest tests/ -v
```

The test suite covers:
- **`test_parsing.py`** — well-formed/malformed line parsing, batch file parsing, edge cases (empty files, blank lines, malformed entries mixed with valid ones)
- **`test_anomaly.py`** — threshold boundary behavior, empty-index handling, custom threshold sensitivity
- **`test_vector_store.py`** — FAISS index build/query correctness, and a regression test for a fixed bug where rebuilding the index accumulated stale vectors instead of replacing them (previously caused `KeyError` crashes on `/analyze`)

The embedding model is mocked in tests so the suite runs offline in well under a second, with no GPU or model download required.

## Known limitations

- In-memory storage: parsed logs and indices reset on container restart (no persistence layer currently).
- Single-process only — not designed for concurrent multi-user baseline rebuilds.
- The knowledge base (`data/knowledge_base.json`) is a small, hand-written seed set for demonstration.
- Anomaly threshold (`0.6`) was chosen empirically against the HDFS_2k dataset and would need re-tuning for other log formats.

## Possible extensions

- Persist FAISS indices and parsed logs to disk (or a real DB) instead of in-memory dicts.
- CI/CD integration: ingest logs directly from GitHub Actions/Jenkins build failures for "shift-left" defect detection.
- LLM-as-judge evaluation harness to score diagnosis quality against a labeled set.
- Configurable model/provider via environment variable instead of hardcoded in `llm_analysis.py`.