# LogSage

An LLM-powered log analysis system that ingests unstructured system logs, mines recurring event templates, detects anomalous log lines via embedding distance, and generates root-cause diagnoses and suggested fixes using a multi-step agentic RAG pipeline.

Built as three independently deployable microservices, coordinated via Kafka and a shared Redis store, and tested against the [Loghub HDFS](https://github.com/logpai/loghub) dataset.

## Why this exists

Manually reading through thousands of system log lines to find the handful that actually matter is slow and error-prone. LogSage automates that triage:
- it learns what "normal" log activity looks like
- flags lines that deviate from it
- only spends LLM inference (and its associated cost/latency) on the anomalies

## Architecture

```
┌──────────────────────┐        ┌──────────────┐         ┌─────────────────────┐
│  ingestion_service   │───────▶│    Kafka     │───────▶ │   consumer_service  │
│                      │        │ (log topic)  │         │                     │
│ /upload/baseline     │        └──────────────┘         │ writes parsed logs  │
│ /upload/logs         │                                 │ into Redis          │
│                      │                                 └──────────┬──────────┘
│ - parsing            │                                            │
│ - Drain3 template    │                                            ▼
│   mining             │                                    ┌──────────────┐
│ - builds baseline    │                                    │    Redis     │
│   FAISS index        │                                    │ (log store)  │
└──────────┬───────────┘                                    └──────┬───────┘
           │                                                       │
           ▼                                                       │
   ┌───────────────┐                                               │
   │ shared volume │◀───────────────────────┐                      │
   │ (FAISS index) │                        │                      │
   └───────┬───────┘                        │                      │
           │                                │                      │
           ▼                                │                      ▼
   ┌────────────────────────────────────────┴──────────────────────────┐
   │                         analysis_service                          │
   │                                                                   │
   │  GET /analyze/{trace_id}  GET /logs  GET /stats                   │
   │                                                                   │
   │  1. loads baseline index from shared volume                       │
   │  2. anomaly detection (FAISS distance threshold)                  │
   │  3. if anomalous → LangGraph agentic pipeline:                    │
   │       triage node → research node (RAG) → report node             │
   │  4. Pandas/scikit-learn analytics via /stats                      │
   └───────────────────────────────────────────────────────────────────┘
```

### Why three separate services, not one

Each service owns a distinct responsibility and can be deployed, scaled, and reasoned about independently:

- **`ingestion_service`** — parsing, template mining, baseline index construction. Write-heavy, bursty. Kept to a single replica since concurrent baseline rebuilds would race on the shared index volume.
- **`consumer_service`** — a minimal, non-HTTP worker that only consumes from Kafka and writes to Redis. No FastAPI, no torch, no LLM dependencies — its image is intentionally the lightest of the three. Safely scaled to multiple replicas; Kafka's consumer-group protocol handles partition rebalancing.
- **`analysis_service`** — anomaly detection, RAG, the LangGraph agentic pipeline, and analytics. Read-heavy, latency-sensitive, the only service that calls an external LLM. Safely scaled independently of ingestion.

They share one library, **`logsage_common`** (`libs/logsage_common`), installed editable into all three images, so fixes to `VectorStore`, `LogStore`, or `redact()` propagate everywhere without copy-pasted code.

### Key design decision: separate baseline vs. analysis ingestion

Anomaly detection only works if new events are scored against a **fixed, stable reference distribution**. If the log being analyzed is included in the same batch used to build that reference index, it trivially matches itself (distance ≈ 0) and never gets flagged.

- **`POST /upload/baseline`** (ingestion_service) — parses a log file, mines templates via Drain3, and *rebuilds* the FAISS baseline index, persisting it to a shared volume.
- **`POST /upload/logs`** (ingestion_service) — parses a log file and publishes each line to Kafka for asynchronous storage, **without** touching the baseline index.

### Key design decision: the baseline index is shared via disk, not memory

Since `ingestion_service` (writer) and `analysis_service` (reader) are separate processes/containers, the FAISS index can no longer live in a single shared Python object. `VectorStore.save()`/`.load()` persist it to a Docker-managed volume (`vector-index`) mounted into both containers at `/shared`. `analysis_service` reloads the index fresh on every `/analyze` call, so it always reflects the latest baseline without any direct service-to-service coupling.

### Key design decision: agentic diagnosis, not a single prompt call

`analysis_service`'s diagnostic pipeline (`agentic_pipeline.py`) is a 3-node [LangGraph](https://github.com/langchain-ai/langgraph) state graph, not one LLM call:

1. **Triage node** — redacts sensitive data, extracts key entities (component, error keywords, severity) from the raw log line.
2. **Research node** — retrieves the most relevant historical issues/fixes from the RAG knowledge base.
3. **Report node** — synthesizes triage + research into a final `root_cause` / `suggested_fix` / `confidence` verdict.

State accumulates across nodes via a typed dict, matching LangGraph's standard pattern — this is a real loop, not a single prompt relabeled as an "agent."

## Services

| Service | Port | Responsibilities | Depends on |
|---|---|---|---|
| `ingestion_service` | 8001 | Parsing, Drain3 template mining, baseline FAISS index construction, Kafka publishing | Kafka |
| `consumer_service` | — (no HTTP) | Consumes from Kafka, writes parsed logs to Redis | Kafka, Redis |
| `analysis_service` | 8002 | Anomaly detection, RAG, LangGraph agentic diagnosis, Pandas/scikit-learn analytics | Redis, shared FAISS volume, LLM provider |

## API endpoints

**ingestion_service** (`:8001`)

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload/baseline` | Builds/rebuilds the baseline reference index from a clean log file |
| `POST` | `/upload/logs` | Publishes a log file's lines to Kafka for asynchronous storage |
| `GET` | `/health` | Health check |

**analysis_service** (`:8002`)

| Method | Path | Description |
|---|---|---|
| `GET` | `/analyze/{trace_id}` | Runs anomaly detection (+ agentic RAG/LLM diagnosis if anomalous) |
| `GET` | `/logs` | Lists all currently stored trace IDs |
| `GET` | `/stats` | Pandas-based aggregate analytics (log level distribution, top templates) |
| `GET` | `/health` | Health check |

Both services also serve interactive Swagger docs at `/docs`.

## Tech stack

- **Python 3.11+**
- **FastAPI** — async REST APIs for ingestion_service and analysis_service
- **Drain3** — log template mining
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local, free embedding model
- **FAISS** (`faiss-cpu`) — vector similarity search, persisted to a shared Docker volume
- **Apache Kafka** (+ Zookeeper) — asynchronous log ingestion pipeline between ingestion_service and consumer_service
- **Redis** — persistent store for parsed log entries, replacing an earlier in-memory dict
- **LangGraph** — 3-node agentic diagnostic pipeline (triage → research → report)
- **Pandas / scikit-learn** — `/stats` analytics and an `IsolationForest`-based secondary anomaly detector, offered alongside the FAISS distance-threshold approach
- **Groq API** (OpenAI-SDK-compatible, `openai/gpt-oss-20b`) — LLM inference. Any OpenAI-SDK-compatible provider works by swapping `base_url`/`model` in `agentic_pipeline.py`
- **Docker Compose** — local multi-container orchestration
- **Kubernetes manifests** (`k8s/`) — Deployments, Services, and a PersistentVolumeClaim for the shared FAISS index, mirroring the Compose topology for cluster deployment
- **GitHub Actions** — CI running the full test suite on every push
- **Loghub HDFS_2k** — test dataset

## Setup

### Prerequisites
- Docker Desktop
- An API key from Groq ([console.groq.com](https://console.groq.com)) or OpenAI

### Environment variables

Copy `.env.example` to `.env` in the repo root and fill in your key:

```
GROQ_API_KEY=your_key_here
MOCK_LLM=false
```

`.env` is read automatically by Docker Compose and is already covered by `.gitignore` — never commit it. Setting `MOCK_LLM=true` bypasses the real LLM call and returns a canned response, useful for testing the full request/response flow without spending API credits.

### Build and run

The three services share a common base image (`Dockerfile.base`) that installs `uv` and the `logsage_common` package once, rather than each service repeating that work independently. Build it once, then bring up the full stack:

```bash
docker build -f Dockerfile.base -t logsage-base:latest .
docker compose up --build
```

Re-run the first command only when `libs/logsage_common` changes. For everyday iteration, `docker compose up --build` alone is enough — avoid `--no-cache`/`--pull` unless you specifically need to discard Docker's layer cache, since they force every layer (including the slow `sentence-transformers`/torch install) to rebuild from scratch on all three services independently.

This starts six containers: `zookeeper`, `kafka`, `redis`, `ingestion_service`, `analysis_service`, `consumer_service`.

### Usage flow

```bash
curl -X POST http://localhost:8001/upload/baseline -F "file=@HDFS_2k.log"
curl -X POST http://localhost:8001/upload/logs -F "file=@test.log"
sleep 2   # give consumer_service a moment to consume from Kafka and write to Redis
curl http://localhost:8002/logs
curl http://localhost:8002/analyze/test.log-0
curl http://localhost:8002/stats
```

1. `/upload/baseline` establishes what "normal" looks like from a clean log sample.
2. `/upload/logs` publishes the file you want analyzed to Kafka; `consumer_service` picks it up asynchronously and writes it to Redis.
3. `/logs` lists trace IDs (format: `{filename}-{line_index}`) once the consumer has caught up.
4. `/analyze/{trace_id}` returns either a normal-match result or a full agentic diagnosis with root cause and suggested fix.
5. `/stats` returns aggregate analytics across everything ingested so far.

## Testing

Each service defines a top-level package literally named `app`, so importing more than one service's tests in the same Python process causes one to silently shadow the other. Tests are therefore split into per-service folders and **must be run as separate `pytest` invocations**, never combined into a single `pytest tests/` call:

```bash
pip install -r requirements-dev.txt

pytest tests/logsage_common -v
pytest tests/ingestion_service -v
pytest tests/analysis_service -v
pytest tests/consumer_service -v
```

This is exactly how CI runs it (`.github/workflows/test.yml`) — four separate steps, not one.

The suite covers:
- **`tests/logsage_common/`** — FAISS index build/query/persistence (including a cross-process save/load round-trip regression test), and Redis-backed log storage (via `fakeredis`, no real Redis server needed)
- **`tests/ingestion_service/`** — log line parsing edge cases, and Kafka producer message construction/error handling (mocked, no real broker needed)
- **`tests/analysis_service/`** — anomaly threshold boundary behavior, the full 3-node LangGraph pipeline (LLM calls mocked), and Pandas/scikit-learn analytics
- **`tests/consumer_service/`** — Kafka consumer message handling (mocked)

All external dependencies (embedding model, LLM API, Kafka broker, Redis server) are mocked, so the full suite runs offline in well under a second.

## Kubernetes deployment

`k8s/deployment.yaml` and `k8s/service.yaml` mirror the Compose topology: one Deployment per service plus Redis/Kafka/Zookeeper, and a `PersistentVolumeClaim` (`vector-index-pvc`) replacing the Compose named volume for the shared FAISS index.

```bash
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io --docker-username=Tazwar89 \
  --docker-password=<PAT> --docker-email=<email>
kubectl create secret generic logsage-secrets --from-literal=groq-api-key=your_key_here
kubectl apply -f k8s/
```

Two things worth knowing before trying this on a local cluster:
- The PVC requests `ReadWriteMany` access (both ingestion and analysis need concurrent access), which most local provisioners (minikube's default, kind) don't support out of the box — see the comments in `k8s/deployment.yaml` for workarounds.
- All three service images plus the shared base are published to GHCR by CI on every push to main; k8s/deployment.yaml pulls them directly, so no local image build/load step is needed for cluster deployment.

## Known limitations

- The `ReadWriteMany` PVC requirement means the Kubernetes manifests need a compatible storage class or a local workaround to actually run — they aren't a drop-in `kubectl apply` on every cluster.
- `ingestion_service` is pinned to a single replica; scaling it would require coordinating concurrent baseline rebuilds against the shared index, which isn't implemented.
- The knowledge base (`data/knowledge_base.json`) is a small, hand-written seed set for demonstration, not a comprehensive fix database.
- Anomaly threshold (`0.6`) was chosen empirically against the HDFS_2k dataset and would need re-tuning for other log formats.
- There's a small delay between `/upload/logs` and a trace_id becoming queryable via `/analyze`, since ingestion now happens asynchronously through Kafka rather than synchronously in the request/response cycle.

## Possible extensions

- LLM-as-judge evaluation harness to score diagnosis quality against a labeled set.
- Swap the FAISS/shared-volume approach for a dedicated vector database service (e.g. Qdrant or Chroma server), removing the `ReadWriteMany` constraint entirely.
- Configurable model/provider via environment variable instead of hardcoded in `agentic_pipeline.py`.