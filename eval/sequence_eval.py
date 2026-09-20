"""
Evaluates sequence-level (per-block) anomaly detectors against Loghub's HDFS
block labels. Detectors are trained on NORMAL blocks only, thresholds are
calibrated on held-out normal blocks (no anomaly labels used for tuning), and
precision / recall / F1 are measured on a test set of the remaining normal
blocks plus every anomalous block.

Usage (from repo root; needs the FULL HDFS.log, not the 2k sample).
--target-fpr accepts a comma-separated sweep (0.005,0.01,0.02); each detector is
trained once and re-calibrated per value.

    python3 -m eval.sequence_eval \\
        --log /path/to/HDFS_v1/HDFS.log \\
        --labels eval/anomaly_label.csv \\
        --max-blocks 100000 \\
        --cache eval/hdfs_sequences.pkl.gz

Parsing the full log with Drain3 is the slow step; --cache stores the
block sequences so re-runs (different detectors/hyperparameters) are fast.
"""
from __future__ import annotations

import argparse, json, random
from pathlib import Path

import numpy as np

from eval.precision_recall import load_ground_truth


def split_blocks(sequences: dict, labels: dict, train_frac: float = 0.5, val_frac: float = 0.1, seed: int = 42) -> dict:
    """Normal blocks -> train / val / test-normal; every anomalous block -> test."""
    labelled = [b for b in sequences if b in labels]
    normal = sorted(b for b in labelled if not labels[b])
    anomalous = sorted(b for b in labelled if labels[b])
    random.Random(seed).shuffle(normal)
    n_train, n_val = int(train_frac * len(normal)), int(val_frac * len(normal))

    return {
        "train": normal[:n_train],
        "val": normal[n_train:n_train + n_val],
        "test_normal": normal[n_train + n_val:],
        "test_anomalous": anomalous,
    }


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "false_positive_rate": round(fpr, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate_detector(detector, sequences: dict, splits: dict, target_fprs: list[float]) -> dict:
    """Fits once, then calibrates + scores at each target FPR. Returns {str(fpr): metrics}."""
    seq = lambda ids: [sequences[b] for b in ids]  # noqa: E731

    detector.fit(seq(splits["train"]))
    test_ids = splits["test_normal"] + splits["test_anomalous"]
    test_seqs = seq(test_ids)
    y_true = np.array([0] * len(splits["test_normal"]) + [1] * len(splits["test_anomalous"]))
    out = {}

    for fpr in target_fprs:
        threshold = detector.calibrate(seq(splits["val"]), fpr)
        y_pred = detector.is_anomalous(test_seqs).astype(int)
        out[str(fpr)] = {"threshold": float(threshold), **confusion_metrics(y_true, y_pred)}

    return out


def make_detectors(names: list[str], steps: int, verbose: bool) -> dict:
    from logsage_common.sequence_anomaly import CountVectorPCADetector, DeepLogDetector

    available = {
        "pca": lambda: CountVectorPCADetector(),
        "deeplog": lambda: DeepLogDetector(steps=steps, verbose=verbose),
        "deeplog-rank": lambda: DeepLogDetector(mode="rank", steps=steps, verbose=verbose),
    }

    return {n: available[n]() for n in names}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="Full HDFS_v1 HDFS.log")
    p.add_argument("--labels", default="eval/anomaly_label.csv")
    p.add_argument("--max-blocks", type=int, default=None)
    p.add_argument("--cache", default=None, help="Path (.pkl.gz) to cache/load parsed block sequences")
    p.add_argument("--detectors", default="pca,deeplog,deeplog-rank")
    p.add_argument("--target-fpr", default="0.01", help="Comma-separated target FPRs, e.g. 0.005,0.01,0.02 (model is trained once)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="eval/sequence_eval_results.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    from eval.hdfs_sessions import build_block_sequences, load_cache, save_cache

    if args.cache and Path(args.cache).exists():
        print(f"Loading cached sequences from {args.cache}")
        sequences, templates = load_cache(args.cache)

    else:
        print(f"Parsing {args.log} (Drain3) ...")
        sequences, templates = build_block_sequences(args.log, max_blocks=args.max_blocks)

        if args.cache:
            save_cache(args.cache, sequences, templates)

    labels = load_ground_truth(Path(args.labels))
    splits = split_blocks(sequences, labels, seed=args.seed)
    lengths = np.array([len(sequences[b]) for b in splits["train"]])

    print(f"{len(sequences):,} blocks, {len(templates)} templates, median {int(np.median(lengths))} events/block (train)")
    print({k: len(v) for k, v in splits.items()})

    if len(splits["test_anomalous"]) == 0:
        raise SystemExit("No labelled anomalous blocks in the parsed sequences -- check --log matches --labels.")

    if float(np.median(lengths)) < 3:
        raise SystemExit("Median <3 events/block: this looks like the 2k sample, not full sessions. Use the full HDFS.log.")

    target_fprs = [float(x) for x in args.target_fpr.split(",")]
    results = {"n_templates": len(templates), "splits": {k: len(v) for k, v in splits.items()}, "target_fprs": target_fprs}

    for name, detector in make_detectors(args.detectors.split(","), args.steps, args.verbose).items():
        print(f"\n== {name} ==")
        results[name] = evaluate_detector(detector, sequences, splits, target_fprs)
        print(json.dumps(results[name], indent=2))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()