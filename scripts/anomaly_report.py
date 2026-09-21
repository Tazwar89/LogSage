"""
Batch anomaly-detection report (pandas + NumPy + scikit-learn).

Runs the same BaselineAnomalyDetector used by ingestion_service's
/upload/logs endpoint over an arbitrary log file, and produces a pandas
summary: total lines parsed, anomaly rate, and per-template breakdown.
Use this to get a real, reproducible number for how much the anomaly
filter reduces GenAI analysis volume before quoting it anywhere.

Usage (from repo root, with a real baseline already uploaded to Qdrant):

    QDRANT_URL=https://logsage-qdrant.fly.dev:443 \\
    python3 scripts/anomaly_report.py path/to/HDFS_2k.log

The public Loghub HDFS_2k.log sample (2,000 real HDFS log lines) is a good
stand-in for a representative run: https://github.com/logpai/loghub
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # load .env in repo root so QDRANT_URL is available to QdrantVectorStore

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))

from ingestion_service.app.anomaly import (
    BaselineAnomalyDetector,
    fetch_baseline_vectors,
)
from ingestion_service.app.embedding import build_template_miner, deduplicate_logs
from ingestion_service.app.parsing import parse_file
from logsage_common.vector_store_qdrant import QdrantVectorStore


def run_report(log_path: str) -> pd.DataFrame:
    parsed_logs = parse_file(log_path)

    if not parsed_logs:
        raise SystemExit(f"No parseable log lines found in {log_path}")

    template_miner = build_template_miner()
    deduplicate_logs(parsed_logs, template_miner)

    baseline_store = QdrantVectorStore(collection_name="baseline")
    baseline_vectors = fetch_baseline_vectors(baseline_store)
    detector = BaselineAnomalyDetector().fit(baseline_vectors)

    templates = [log.get("template_string", "") for log in parsed_logs]
    vectors = np.array(baseline_store.embed(templates), dtype="float32")
    scores = detector.score(vectors)
    flags = scores > detector.threshold

    df = pd.DataFrame(parsed_logs)
    df["anomaly_score"] = scores
    df["is_anomalous"] = flags

    return df


def print_summary(df: pd.DataFrame):
    total = len(df)
    anomalous = int(df["is_anomalous"].sum())
    rate = anomalous / total if total else 0.0

    print(f"\nParsed {total} log lines")
    print("Baseline templates fitted on: (see detector.threshold below)")
    print(f"Anomalous: {anomalous} ({rate*100:.1f}%)")
    print(f"Suppressed as normal (not forwarded to Kafka/GenAI): {total - anomalous} ({(1-rate)*100:.1f}%)")

    print("\nPer-template breakdown (top 10 by count):")
    breakdown = (
        df.groupby("template_string")
        .agg(count=("template_string", "size"), anomaly_rate=("is_anomalous", "mean"))
        .sort_values("count", ascending=False)
        .head(10)
    )
    print(breakdown.to_string())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python3 {sys.argv[0]} <path-to-log-file>")

    df = run_report(sys.argv[1])
    print_summary(df)

    out_path = _ROOT / "eval" / "anomaly_report.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull per-line report written to {out_path}")