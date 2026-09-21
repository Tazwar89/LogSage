"""
Benchmarks the scikit-learn k-NN detector against the TensorFlow autoencoder.

Usage (from repo root, with the analysis/eval deps installed):
    python -m eval.compare_detectors \
        --baseline services/analysis_service/data/HDFS_2k.log \
        --anomalous services/analysis_service/data/test_anomalous.log \
        --push-to <hf-username>/logsage-autoencoder   # optional, needs HF_TOKEN

Method: embed baseline lines, hold out 20% as known-normal, embed the
anomalous file as known-anomalous, fit each detector on the 80% split, and
report precision / recall / F1 / false-positive rate on the held-out mix.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "false_positive_rate": fpr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate(train: np.ndarray, normal_holdout: np.ndarray, anomalous: np.ndarray, percentile: float = 95.0) -> dict:
    """Fits both detectors on `train`; scores a labeled holdout. Pure and testable."""
    from logsage_common.anomaly import BaselineAnomalyDetector
    from logsage_common.tf_autoencoder import AutoencoderAnomalyDetector

    x = np.vstack([normal_holdout, anomalous]).astype("float32")
    y = np.concatenate([np.zeros(len(normal_holdout)), np.ones(len(anomalous))]).astype(int)

    knn = BaselineAnomalyDetector(percentile=percentile).fit(train)
    ae = AutoencoderAnomalyDetector(percentile=percentile).fit(train)

    return {
        "knn": _metrics(y, knn.is_anomalous(x).astype(int)),
        "autoencoder": _metrics(y, ae.is_anomalous(x).astype(int)),
        "autoencoder_threshold": ae.threshold,
        "knn_threshold": knn.threshold,
        "n_train": len(train), "n_normal_holdout": len(normal_holdout), "n_anomalous": len(anomalous),
    }


def _read_lines(path: str, limit: int | None = None) -> list[str]:
    with open(path, errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    return lines[:limit] if limit else lines


def main() -> None:
    from logsage_common.hf_hub import load_embedding_model, push_detector
    from logsage_common.tf_autoencoder import AutoencoderAnomalyDetector

    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--anomalous", required=True)
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="eval/detector_comparison.json")
    p.add_argument("--save-dir", default="eval/autoencoder_model")
    p.add_argument("--push-to", default=None, help="HF Hub repo id, e.g. user/logsage-autoencoder")
    args = p.parse_args()

    model = load_embedding_model(args.model)
    baseline = model.encode(_read_lines(args.baseline), convert_to_numpy=True, normalize_embeddings=True)
    anomalous = model.encode(_read_lines(args.anomalous), convert_to_numpy=True, normalize_embeddings=True)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(baseline))
    cut = int(0.8 * len(baseline))
    train, holdout = baseline[order[:cut]], baseline[order[cut:]]

    results = evaluate(train, holdout, anomalous)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))

    if args.push_to:
        detector = AutoencoderAnomalyDetector().fit(train)
        detector.save(args.save_dir)
        print("Pushed to", push_detector(args.save_dir, args.push_to))


if __name__ == "__main__":
    main()