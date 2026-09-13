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
│   Qdrant collection  │                                    │ (log store)  │
└──────────┬───────────┘                                    └──────┬───────┘
           │                                                       │
           ▼                                                       │
   ┌───────────────┐                                               │
   │    Qdrant     │◀──────────────────────┐                       │
   │ (vector store)│                       │                       │
   └───────┬───────┘                       │                       │
           │                               │                       │
           ▼                               │                       ▼
   ┌───────────────────────────────────────┴──────────────────────────┐
   │                         analysis_service                         │
   │                                                                  │
   │  GET /analyze/{trace_id}  GET /logs  GET /stats                  │
   │                                                                  │
   │  1. queries the baseline collection in Qdrant                    │
   │  2. anomaly detection (Qdrant L2 distance threshold)              │
   │  3. if anomalous → LangGraph agentic pipeline:                   │
   │       triage node → research node (RAG) → report node            │
   │  4. Pandas/scikit-learn analytics via /stats                     │
   └──────────────────────────────────────────────────────────────────┘
```

### Why three application services, not one

Each service owns a distinct responsibility and can be deployed, scaled, and reasoned about independently:

- **`ingestion_service`** — parsing, template mining, baseline index construction. Write-heavy, bursty. Kept to a single replica since concurrent baseline rebuilds recreate the same Qdrant collection.
- **`consumer_service`** — a minimal, non-HTTP worker that only consumes from Kafka and writes to Redis. No FastAPI, no torch, no LLM dependencies — its image is intentionally the lightest of the three. Safely scaled to multiple replicas; Kafka's consumer-group protocol handles partition rebalancing.
- **`analysis_service`** — anomaly detection, RAG, the LangGraph agentic pipeline, and analytics. Read-heavy, latency-sensitive, the only service that calls an external LLM. Safely scaled independently of ingestion.

Alongside these, **Qdrant** runs as its own deployed service (self-hosted, `logsage-qdrant` on Fly.io) rather than an embedded library — both `ingestion_service` (writer) and `analysis_service` (reader) talk to it over HTTP, so there's no shared volume or file-based index to keep in sync.

They also share one library, **`logsage_common`** (`libs/logsage_common`), installed editable into all three application images, so fixes to `QdrantVectorStore`, `LogStore`, or `redact()` propagate everywhere without copy-pasted code.

### Key design decision: separate baseline vs. analysis ingestion

Anomaly detection only works if new events are scored against a **fixed, stable reference distribution**. If the log being analyzed is included in the same batch used to build that reference index, it trivially matches itself (distance ≈ 0) and never gets flagged.

- **`POST /upload/baseline`** (ingestion_service) — parses a log file, mines templates via Drain3, and *recreates* the Qdrant `baseline` collection from scratch.
- **`POST /upload/logs`** (ingestion_service) — parses a log file and publishes each line to Kafka for asynchronous storage, **without** touching the baseline collection.

### Key design decision: Qdrant as a shared server, not a shared volume

`ingestion_service` (writer) and `analysis_service` (reader) are separate processes/containers, so the vector index can't live in a single in-process object. This originally ran on FAISS persisted to a shared Docker/PVC volume (`save()`/`load()` serializing to disk); it's since migrated to **Qdrant**, run as its own server both services connect to over `QDRANT_URL`. `QdrantVectorStore.save()`/`.load()` are now no-ops — every `upsert()` is visible to readers immediately, with no shared volume, no serialization step, and no `ReadWriteMany` PVC requirement. `analysis_service` queries the `baseline` collection fresh on every `/analyze` call, so it always reflects the latest baseline without any direct service-to-service coupling.

Note: Qdrant uses true L2 distance, while the earlier FAISS setup used squared L2 — the anomaly threshold (see Known limitations) was re-tuned accordingly during the migration.

### Key design decision: agentic diagnosis, not a single prompt call

`analysis_service`'s diagnostic pipeline (`agentic_pipeline.py`) is a 3-node [LangGraph](https://github.com/langchain-ai/langgraph) state graph, not one LLM call:

1. **Triage node** — redacts sensitive data, extracts key entities (component, error keywords, severity) from the raw log line.
2. **Research node** — retrieves the most relevant historical issues/fixes from the RAG knowledge base.
3. **Report node** — synthesizes triage + research into a final `root_cause` / `suggested_fix` / `confidence` verdict.

State accumulates across nodes via a typed dict, matching LangGraph's standard pattern — this is a real loop, not a single prompt relabeled as an "agent."

## Services

| Service | Port | Responsibilities | Depends on |
|---|---|---|---|
| `ingestion_service` | 8001 | Parsing, Drain3 template mining, baseline Qdrant collection construction, Kafka publishing | Kafka, Qdrant |
| `consumer_service` | — (no HTTP) | Consumes from Kafka, writes parsed logs to Redis | Kafka, Redis |
| `analysis_service` | 8002 | Anomaly detection, RAG, LangGraph agentic diagnosis, Pandas/scikit-learn analytics | Redis, Qdrant, LLM provider |

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
- **Qdrant** (`qdrant-client`) — vector similarity search, run as its own server (self-hosted on Fly.io in production, via Docker Compose locally); replaced an earlier FAISS + shared-volume approach
- **Apache Kafka** (+ Zookeeper locally / Confluent Cloud in production) — asynchronous log ingestion pipeline between ingestion_service and consumer_service
- **Redis** (self-hosted locally / Upstash in production) — persistent store for parsed log entries, replacing an earlier in-memory dict
- **LangGraph** — 3-node agentic diagnostic pipeline (triage → research → report)
- **Pandas / scikit-learn** — `/stats` analytics and an `IsolationForest`-based secondary anomaly detector, offered alongside the Qdrant distance-threshold approach
- **Groq API** (OpenAI-SDK-compatible) — LLM inference, model configurable via the `LLM_MODEL` environment variable (default `openai/gpt-oss-20b`); any OpenAI-SDK-compatible provider works by setting `LLM_BASE_URL`/`LLM_MODEL`
- **Docker Compose** — local multi-container orchestration
- **Kubernetes manifests** (`k8s/`) — Deployments, Services, and a PersistentVolumeClaim for Qdrant's storage, mirroring the Compose topology for cluster deployment; maintained for portfolio/reference purposes rather than as an actively deployed target
- **Fly.io** — production deployment target (`logsage-ingestion`, `logsage-consumer`, `logsage-analysis`, `logsage-qdrant`)
- **GitHub Actions** — CI running the full test suite on every push, publishing images to GHCR
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
- **`tests/logsage_common/`** — Qdrant-backed vector store build/query behavior (mocked client, no real Qdrant server needed), and Redis-backed log storage (via `fakeredis`, no real Redis server needed)
- **`tests/ingestion_service/`** — log line parsing edge cases, and Kafka producer message construction/error handling (mocked, no real broker needed)
- **`tests/analysis_service/`** — anomaly threshold boundary behavior, the full 3-node LangGraph pipeline (LLM calls mocked), and Pandas/scikit-learn analytics
- **`tests/consumer_service/`** — Kafka consumer message handling (mocked)

All external dependencies (embedding model, LLM API, Kafka broker, Redis server) are mocked, so the full suite runs offline in well under a second.

## Kubernetes deployment

`k8s/deployment.yaml` and `k8s/service.yaml` mirror the Compose topology: one Deployment per service plus Redis/Kafka/Zookeeper/Qdrant, and a `PersistentVolumeClaim` (`qdrant-pvc`) for Qdrant's own storage.

```bash
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io --docker-username=Tazwar89 \
  --docker-password=<PAT> --docker-email=<email>
kubectl create secret generic logsage-secrets --from-literal=groq-api-key=your_key_here
kubectl apply -f k8s/
```

Two things worth knowing before trying this on a local cluster:
- The PVC requests `ReadWriteOnce` access — only Qdrant itself writes to its storage, so this is compatible with local provisioners (minikube's default, kind) unlike the earlier `ReadWriteMany` FAISS setup.
- All three service images plus the shared base are published to GHCR by CI on every push to main; k8s/deployment.yaml pulls them directly, so no local image build/load step is needed for cluster deployment.

## Known limitations

- `ingestion_service` is pinned to a single replica; scaling it would require coordinating concurrent baseline rebuilds against the same Qdrant collection, which isn't implemented.
- The knowledge base (`data/knowledge_base.json`) is a small, hand-written seed set for demonstration, not a comprehensive fix database.
- Anomaly threshold is `sqrt(0.6) ≈ 0.775`, tuned for Qdrant's true L2 distance against the HDFS_2k dataset, and would need re-tuning for other log formats or distance metrics.
- There's a small delay between `/upload/logs` and a trace_id becoming queryable via `/analyze`, since ingestion happens asynchronously through Kafka rather than synchronously in the request/response cycle.

## Possible extensions

- LLM-as-judge evaluation harness to score diagnosis quality against a labeled set.
- Production observability (latency/token-usage tracing via LangSmith or similar) for the agentic pipeline.