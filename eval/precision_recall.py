"""
Computes precision/recall of the anomaly *detector* (BaselineAnomalyDetector,
via is_anomalous()/scripts/anomaly_report.py) against Loghub's HDFS
block-level ground truth -- a different, and more load-bearing, number than
the LLM-judge diagnostic-quality score in run_eval.py.

IMPORTANT -- read before running:
Loghub's HDFS ground truth (anomaly_label.csv, one row per BlockId with a
Normal/Anomaly label) is NOT included in the 2k-line GitHub sample used
elsewhere in this repo. It ships only with the full ~11.2M-line raw HDFS_1
log, which Loghub gates behind a Zenodo request that asks for your
affiliation (see https://github.com/logpai/loghub -> HDFS_v1 -> Download).
There is no public direct-download URL to fetch programmatically.

So this script does NOT run end-to-end out of the box. It:
  1. Always computes anomaly_report.csv's own summary anomaly rate (works
     today, no external file needed).
  2. Additionally computes precision/recall IF you point --labels at an
     anomaly_label.csv you've obtained from the full HDFS_1 dataset AND
     --log at the *matching* full log file anomaly_report.csv was built
     from (labels are per BlockId across the WHOLE log; the 2k-line
     sample doesn't contain enough of most sessions to label fairly).

Usage:
    # after: python3 scripts/anomaly_report.py <full HDFS log> (writes
    # eval/anomaly_report.csv over the *same* file used below)
    python3 -m eval.precision_recall \
        --report eval/anomaly_report.csv \
        --labels /path/to/anomaly_label.csv
"""
import argparse
import csv
import re
from pathlib import Path

BLOCK_ID_RE = re.compile(r"blk_-?\d+")


def _extract_block_id(message: str) -> str | None:
    m = BLOCK_ID_RE.search(message or "")

    return m.group(0) if m else None


def load_report(report_path: Path) -> dict:
    """
    Returns {block_id: flagged_anomalous} where flagged_anomalous is True
    if ANY line mentioning that block was flagged is_anomalous=True --
    matching Loghub's own convention of labeling anomalies at the
    block/session level, not the individual line level.
    """
    block_flagged = {}

    with open(report_path, newline="") as f:
        for row in csv.DictReader(f):
            block_id = _extract_block_id(row.get("message", ""))

            if block_id is None:
                continue

            is_anom = str(row.get("is_anomalous", "")).strip().lower() == "true"
            block_flagged[block_id] = block_flagged.get(block_id, False) or is_anom

    return block_flagged


def load_ground_truth(labels_path: Path) -> dict:
    """anomaly_label.csv format: BlockId,Label (Label in {Normal, Anomaly})."""
    ground_truth = {}

    with open(labels_path, newline="") as f:
        for row in csv.DictReader(f):
            block_id = row["BlockId"].strip()
            ground_truth[block_id] = row["Label"].strip().lower() == "anomaly"

    return ground_truth


def compute_precision_recall(flagged: dict, ground_truth: dict) -> dict:
    common = set(flagged) & set(ground_truth)

    if not common:
        raise SystemExit(
            "No overlapping BlockIds between the report and the label file -- "
            "they must come from the same underlying log file."
        )

    true_anomalies_flagged = sum(1 for b in common if flagged[b] and ground_truth[b])
    total_flagged = sum(1 for b in common if flagged[b])
    total_true_anomalies = sum(1 for b in common if ground_truth[b])

    precision = true_anomalies_flagged / total_flagged if total_flagged else 0.0
    recall = true_anomalies_flagged / total_true_anomalies if total_true_anomalies else 0.0

    return {
        "blocks_compared": len(common),
        "total_true_anomalies": total_true_anomalies,
        "total_flagged": total_flagged,
        "true_anomalies_flagged": true_anomalies_flagged,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="eval/anomaly_report.csv")
    parser.add_argument("--labels", required=True, help="Loghub anomaly_label.csv (see module docstring)")
    args = parser.parse_args()

    flagged = load_report(Path(args.report))
    ground_truth = load_ground_truth(Path(args.labels))
    result = compute_precision_recall(flagged, ground_truth)

    print(f"Blocks compared: {result['blocks_compared']}")
    print(f"True anomalies in ground truth: {result['total_true_anomalies']}")
    print(f"Flagged by detector: {result['total_flagged']}")
    print(f"True anomalies flagged: {result['true_anomalies_flagged']}")
    print(f"Precision: {result['precision']*100:.1f}%")
    print(f"Recall:    {result['recall']*100:.1f}%")