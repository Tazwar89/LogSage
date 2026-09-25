# LogSage

[![Build, Push, and Integration Test](https://github.com/Tazwar89/LogSage/actions/workflows/build-push-test.yml/badge.svg)](https://github.com/Tazwar89/LogSage/actions/workflows/build-push-test.yml)

An LLM-powered log analysis system that ingests unstructured system logs, mines recurring event templates, detects anomalies with two complementary detectors (a per-line embedding-distance detector and a DeepLog-style LSTM that scores per-block event sequences), and generates root-cause diagnoses and suggested fixes using a multi-step agentic RAG pipeline.

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
   │  2. anomaly detection (Qdrant L2 distance threshold)             │
   │  3. if anomalous → LangGraph agentic pipeline:                   │
   │       triage node → research node (RAG) → report node            │
   │  4. Pandas/scikit-learn analytics via /stats                     │
   └──────────────────────────────────────────────────────────────────┘
```

Note: The diagram shows the per-line path. The primary HDFS detector is the block-level LSTM behind POST /upload/logs/sequence (see "two detectors" below).

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

### Key design decision: two detectors, because per-line scoring cannot see sequence anomalies

Loghub's HDFS labels are per block, and a block is anomalous because of its event sequence (a missing confirmation, a retry, truncation), not because any single line looks unusual. The per-line embedding detector scored 0% precision and 0% recall on a held-out HDFS_2k split (68 anomalous, 577 normal blocks), so it is kept as a cheap first filter but is not the primary detector for HDFS.

A second line-level comparison (eval/compare_detectors.py, MiniLM embeddings; k-NN vs a TensorFlow autoencoder; results in eval/detector_comparison.json) also failed: each recalled 1 of 14 anomalies at ~5% FPR.

The sequence detector (logsage_common.sequence_anomaly.DeepLogDetector) is a PyTorch LSTM trained on normal blocks only that predicts the next Drain3 template ID; a block's score is its worst next-event surprisal. The threshold is calibrated on held-out normal blocks (target FPR 0.5%), so no anomaly labels are used for tuning.

Held-out results on Loghub HDFS_v1 (575,061 blocks, 45 templates; split 446,578 train / 55,822 val / 55,823 normal test + all 16,838 anomalous blocks as test; seed 42):

| Metric | Value |
| --- | --- |
| Recall | 97.1% (16,346 / 16,838) |
| False-positive rate | 0.45% (253 / 55,823) |
| Precision on this test set | 98.5% (test set is 23% anomalous) |
| Estimated precision at HDFS's natural ~2.9% anomaly rate| ~87% (computed from recall and FPR, not measured) |

Trained with scripts/train_sequence_model.py; artifacts live in models/sequence/ and are baked into the ingestion image. meta.json records 100% Drain3 template-ID agreement between the training and serving parsers on 200k lines.

#### Evaluation

- `eval/run_sequence_eval.py` — samples real labeled HDFS_v1 blocks, runs them through the sequence detector and the agentic pipeline, and has a different judge model grade each diagnosis against the block's own log lines (reference-free, because Loghub has no root-cause labels). Also reports ungrounded-path and destructive-command rates. Results (judge openai/gpt-oss-120b, generator openai/gpt-oss-20b; seeds 7 and 13, 30 anomalous + 30 normal blocks each): 55/57 diagnoses passed (96.5%, Wilson 95% CI 88–99%), 0% ungrounded paths, 0% destructive commands; 91.7% end-to-end counting 3 detector misses. Reference-free judge from the same model family as the generator, so treat as a grounded-diagnosis rate, not accuracy. Per-seed files: eval/sequence_judge_results_seed7.json, _seed13.json.

- `eval/run_eval.py` — legacy per-line harness over a 13-case hand-written golden set (mostly synthetic or near-duplicate lines, self-judged). Kept as a wiring check; its committed result (4/13) reflects the per-line detector missing most cases, not a claim about diagnosis quality.

### Key design decision: Qdrant as a shared server, not a shared volume

`ingestion_service` (writer) and `analysis_service` (reader) are separate processes/containers, so the vector index can't live in a single in-process object. This originally ran on FAISS persisted to a shared Docker/PVC volume (`save()`/`load()` serializing to disk); it's since migrated to **Qdrant**, run as its own server both services connect to over `QDRANT_URL`. `QdrantVectorStore.save()`/`.load()` are now no-ops — every `upsert()` is visible to readers immediately, with no shared volume, no serialization step, and no `ReadWriteMany` PVC requirement. `analysis_service` queries the `baseline` collection fresh on every `/analyze` call, so it always reflects the latest baseline without any direct service-to-service coupling.

Note: Qdrant uses true L2 distance, while the earlier FAISS setup used squared L2 — the anomaly threshold (see Known limitations) was re-tuned accordingly during the migration.

### Key design decision: agentic diagnosis, not a single prompt call

`analysis_service`'s diagnostic pipeline (`agentic_pipeline.py`) is a 3-node [LangGraph](https://github.com/langchain-ai/langgraph) state graph, not one LLM call:

1. **Triage node** — redacts sensitive data, extracts key entities (component, error keywords, severity) from the raw log line.
2. **Research node** — retrieves the most relevant historical issues/fixes from the RAG knowledge base.
3. **Report node** — synthesizes triage + research into a final `root_cause` / `suggested_fix` / `confidence` verdict.

State accumulates across nodes via a typed dict, matching LangGraph's standard pattern — this is a real loop, not a single prompt relabeled as an "agent."

## Example diagnosis

Output of the agentic pipeline for one anomalous HDFS block (`blk_5229240184982419822`, 22 events, sequence score 9.39), taken from `eval/sequence_judge_results_seed13.json`. Fields are `root_cause` and `suggested_fix`.

```json
{
  "root_cause": "The log shows a block that was allocated, received by DataNodes, and added to the block map, yet the NameNode reports it does not belong to any file and deletes it. This pattern indicates the block was orphaned, most likely because the job that created it failed or was killed before the file metadata was updated. The expected event—finalizing the file and associating the block with that file—is missing from the sequence.",
  "suggested_fix": "Investigate the originating job for failures or premature termination that could prevent the file from being finalized. Examine the task logs around the allocation timestamp for errors or abort signals. Verify that block reports are not being duplicated excessively (which can happen after DataNode restarts) and that replication pipelines complete normally."
}
```

The judge scored this 1.0: it identifies the orphaned block and proposes only non-destructive checks.

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
| `POST` | `/upload/logs/sequence` | Groups lines by HDFS BlockId, scores each block's event sequence with the DeepLog-style LSTM, and publishes one entry per anomalous block. Upload complete block sessions; a block split across uploads looks truncated. Response includes anomalous_blocks, anomalous_block_ids (capped, see anomalous_block_ids_truncated) |
| `GET` | `/health` | Health check |
| `GET` | `/sequence/status` | Whether the trained model in models/sequence/ is loaded, plus its training metadata |

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
- **PyTorch** — DeepLog-style long short-term memory (LSTM)
- **Drain3** — log template mining
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local, free embedding model
- **Qdrant** (`qdrant-client`) — vector similarity search, run as its own server (self-hosted on Fly.io in production, via Docker Compose locally); replaced an earlier FAISS + shared-volume approach
- **Apache Kafka** (+ Zookeeper locally / Confluent Cloud in production) — asynchronous log ingestion pipeline between ingestion_service and consumer_service
- **Redis** (self-hosted locally / Upstash in production) — persistent store for parsed log entries, replacing an earlier in-memory dict
- **LangGraph** — 3-node agentic diagnostic pipeline (triage → research → report)
- **Pandas / scikit-learn** — `/stats` analytics and an `IsolationForest`-based secondary anomaly detector, offered alongside the Qdrant distance-threshold approach
- **TensorFlow** — line-level autoencoder baseline
- **Hugging Face Hub** - embedding models via hf_hub, HF_TOKEN
- **Groq API** (OpenAI-SDK-compatible) — LLM inference, model configurable via the `LLM_MODEL` environment variable (default `openai/gpt-oss-20b`); any OpenAI-SDK-compatible provider works by setting `LLM_BASE_URL`/`LLM_MODEL`
- **Docker Compose** — local multi-container orchestration
- **Kubernetes manifests** (`k8s/`) — Deployments, Services, and a PersistentVolumeClaim for Qdrant's storage, mirroring the Compose topology for cluster deployment; maintained for portfolio/reference purposes rather than as an actively deployed target
- **Fly.io** — production deployment target (`logsage-ingestion`, `logsage-consumer`, `logsage-analysis`, `logsage-qdrant`)
- **GitHub Actions** — CI running the full test suite on every push, publishing images to GHCR
- **Loghub HDFS_v1** (575K blocks) for the sequence detector and evals; **HDFS_2k** for the per-line detector and demos.

## Setup

### Prerequisites
- Docker Desktop
- An API key from Groq ([console.groq.com](https://console.groq.com)) or OpenAI

### Environment variables

Copy `.env.example` to `.env` in the repo root and fill in your key:

```
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=openai/gpt-oss-20b
LLM_BASE_URL=https://api.groq.com/openai/v1
JUDGE_MODEL=openai/gpt-oss-120b
MOCK_LLM=false

QDRANT_URL=http://localhost:6333

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password_here
REDIS_TLS=false

KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_SASL_USERNAME=your_kafka_username_here
KAFKA_SASL_PASSWORD=your_kafka_password_here

HF_TOKEN=your_huggingface_token_here
```

`.env` is read automatically by Docker Compose and is already covered by `.gitignore` — never commit it. Setting `MOCK_LLM=true` bypasses the real LLM call and returns a canned response, useful for testing the full request/response flow without spending API credits.

### Build and run

The three services share a common base image (`Dockerfile.base`) that installs `uv` and the `logsage_common` package once, rather than each service repeating that work independently. Build it once, then bring up the full stack:

```bash
docker build -f Dockerfile.base -t logsage-base:latest .
docker compose up --build
```

Re-run the first command only when `libs/logsage_common` changes. For everyday iteration, `docker compose up --build` alone is enough — avoid `--no-cache`/`--pull` unless you specifically need to discard Docker's layer cache, since they force every layer (including the slow `sentence-transformers`/torch install) to rebuild from scratch on all three services independently.

This starts seven containers: `qdrant`, `zookeeper`, `kafka`, `redis`, `ingestion_service`, `analysis_service`, `consumer_service`.

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

Each service defines a top-level package literally named `app`, so importing more than one service's tests in the same Python process causes one to silently shadow the other. Tests are therefore split into per-service folders and are run as separate pytest invocations (as CI does) to keep the per-service app packages isolated.

```bash
pip install -r requirements-dev.txt

pytest tests/logsage_common -v
pytest tests/ingestion_service -v
pytest tests/analysis_service -v
pytest tests/consumer_service -v
pytest eval -v
```

This is exactly how CI runs it (`.github/workflows/test.yml`) — five separate steps.

The suite covers:
- **`tests/logsage_common/`** — Qdrant-backed vector store build/query behavior (mocked client, no real Qdrant server needed), and Redis-backed log storage (via `fakeredis`, no real Redis server needed)
- **`tests/ingestion_service/`** — log line parsing edge cases, and Kafka producer message construction/error handling (mocked, no real broker needed)
- **`tests/analysis_service/`** — anomaly threshold boundary behavior, the full 3-node LangGraph pipeline (LLM calls mocked), and Pandas/scikit-learn analytics
- **`tests/consumer_service/`** — Kafka consumer message handling (mocked)

All external services (embedding API calls, LLM, Kafka, Redis) are mocked, so the suite runs without any infrastructure in under a minute (eval tests load the embedding model, ~40 s).

On CPU-only machines/CI: pip install torch --index-url https://download.pytorch.org/whl/cpu before requirements-dev.txt.

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
- The knowledge base (`data/knowledge_base.json`) is a small, hand-written set covering common HDFS failure families, not a comprehensive fix database. Retrieval drops matches beyond `RAG_MAX_DISTANCE` (default 1.0, uncalibrated), so the report may run with no KB context.
- The sequence model is trained on HDFS_v1 only; other log formats need retraining. Drain3 templates and the block-ID regex are HDFS-specific.
- Splits are random by block, not temporal.
- Anomaly threshold is `sqrt(0.6) ≈ 0.775`, tuned for Qdrant's true L2 distance against the HDFS_2k dataset, and would need re-tuning for other log formats or distance metrics.
- There's a small delay between `/upload/logs` and a trace_id becoming queryable via `/analyze`, since ingestion happens asynchronously through Kafka rather than synchronously in the request/response cycle.
- Per-case diagnosis detail for the seed-7 judge run was not retained; only a terminal-output summary survives (`eval/sequence_judge_results_seed7.json` notes this in its own `note` field). Seed-13 has full per-case detail. The pooled 55/57 result is unaffected, but per-case forensics for seed 7 aren't available.

## Possible extensions

- Add human-verified root-cause references for a subset of anomalous blocks to validate the LLM judge.
- Production observability (latency/token-usage tracing via LangSmith or similar) for the agentic pipeline.