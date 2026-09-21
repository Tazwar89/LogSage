"""
Trains the deployable sequence-level anomaly model used by ingestion_service's
POST /upload/logs/sequence, and writes it to a model directory:

    <out-dir>/deeplog.pt              LSTM weights + vocab + calibrated threshold
    <out-dir>/drain_sequence.state    fitted Drain3 miner (template IDs used in training)
    <out-dir>/meta.json               training metadata and held-out metrics

Trains on NORMAL blocks only; the threshold is calibrated on held-out normal
blocks (no anomaly labels used for tuning). Held-out metrics are computed on
unseen normal blocks plus every anomalous block.

Usage (from repo root; needs the FULL HDFS_v1 HDFS.log):

    python3 scripts/train_sequence_model.py \\
        --log ~/Desktop/HDFS_v1/HDFS.log \\
        --labels eval/anomaly_label.csv \\
        --out-dir models/sequence

Then commit models/sequence/ (small) and rebuild the ingestion image.

The final step replays the first lines of the log through the SERVING path
(Drain3 `match()` on the saved state) and reports how often its template IDs
equal the IDs assigned during training. Anything well below ~99% means
serving would see a different vocabulary than the model learned.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "services"))

from logsage_common.sequence_anomaly import DeepLogDetector
from logsage_common.sequence_parsing import (
    UNKNOWN_TEMPLATE_ID,
    build_masking_miner,
    match_template_id,
)

from eval.hdfs_sessions import build_block_sequences, iter_selected_lines
from eval.precision_recall import load_ground_truth
from eval.sequence_eval import confusion_metrics, split_blocks


def parity_check(log_path: str, state_path: str, n_lines: int, max_blocks: int | None) -> dict:
    """Serving-path template IDs (miner.match on saved state) vs training-path IDs (add_log_message)."""
    reference = build_masking_miner(None)
    served = build_masking_miner(state_path)
    total = same = unknown = 0

    for message, _ in iter_selected_lines(log_path, max_blocks, progress_every=0):
        if total >= n_lines:
            break

        train_id = reference.add_log_message(message)["cluster_id"]
        serve_id = match_template_id(served, message)
        total += 1
        same += serve_id == train_id
        unknown += serve_id == UNKNOWN_TEMPLATE_ID

    return {"lines_checked": total, "id_agreement": round(same / total, 5) if total else None,
            "unmatched_at_serving": unknown}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="Full HDFS_v1 HDFS.log")
    p.add_argument("--labels", default="eval/anomaly_label.csv")
    p.add_argument("--out-dir", default="models/sequence")
    p.add_argument("--max-blocks", type=int, default=None)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--parity-lines", type=int, default=200_000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "drain_sequence.state"
    state_path.unlink(missing_ok=True)  # never load a stale miner into a fresh training run

    print(f"Parsing {args.log} ...")
    sequences, templates = build_block_sequences(args.log, max_blocks=args.max_blocks, state_path=str(state_path))

    labels = load_ground_truth(Path(args.labels))
    splits = split_blocks(sequences, labels, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed)
    seq = lambda ids: [sequences[b] for b in ids]
    sizes = {k: len(v) for k, v in splits.items()}
    print(f"{len(sequences):,} blocks, {len(templates)} templates, splits={sizes}")

    if not splits["test_anomalous"]:
        raise SystemExit("No labelled anomalous blocks parsed -- check --log matches --labels.")

    detector = DeepLogDetector(steps=args.steps, seed=args.seed, verbose=True).fit(seq(splits["train"]))
    detector.calibrate(seq(splits["val"]), args.target_fpr)

    test_ids = splits["test_normal"] + splits["test_anomalous"]
    y_true = np.array([0] * len(splits["test_normal"]) + [1] * len(splits["test_anomalous"]))
    metrics = confusion_metrics(y_true, detector.is_anomalous(seq(test_ids)).astype(int))
    print("Held-out metrics:", json.dumps(metrics))

    detector.save(str(out_dir / "deeplog.pt"))

    print(f"Parity check on first {args.parity_lines:,} lines ...")
    parity = parity_check(args.log, str(state_path), args.parity_lines, args.max_blocks)
    print("Parity:", json.dumps(parity))

    meta = {
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "detector": "deeplog-sequence (PyTorch LSTM, surprisal mode)",
        "dataset": "Loghub HDFS_v1",
        "n_blocks_parsed": len(sequences),
        "n_templates": len(templates),
        "templates": {str(k): v for k, v in templates.items()},
        "splits": sizes,
        "target_fpr": args.target_fpr,
        "threshold": detector.threshold,
        "steps": args.steps,
        "seed": args.seed,
        "held_out_metrics": metrics,
        "serving_parity": parity,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_dir}/ (deeplog.pt, drain_sequence.state, meta.json)")


if __name__ == "__main__":
    main()