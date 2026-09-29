"""
RAG_MAX_DISTANCE calibration helper.

Step A -- build a labeling sheet from the unfiltered distance log:
    python -m eval.rag_calibrate make-sheet \
        --log eval/rag_distances.jsonl --out eval/rag_labels.csv --max-dist 1.0

  Dedupes to unique (KB entry, masked query line) pairs, keeps the minimum
  distance seen for each, and writes only pairs at or under --max-dist
  (pairs above the current cutoff are already excluded, so tightening the
  cutoff can only change decisions inside this band). Sorted closest first.
  Fill the `relevant` column with y or n:
    y = the query line is itself an instance of the failure the KB issue
        describes (so the fix text would be a sensible pointer for it)
    n = otherwise (topically nearby wording, benign line, wrong failure)

Step B -- after labeling, sweep cutoffs:
    python -m eval.rag_calibrate sweep --labels eval/rag_labels.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path

# Same masking order as logsage_common.sequence_parsing, so lines that differ
# only in block IDs / IPs / numbers collapse to one labeling decision.
_MASKS = [
    (re.compile(r"blk_-?\d+"), "BLK"),
    (re.compile(r"/\S+"), "PATH"),
    (re.compile(r"(\d{1,3}\.){3}\d{1,3}(:\d+)?"), "IP"),
    (re.compile(r"(?<![\w])-?\d+(?![\w])"), "NUM"),
]


def norm(line: str) -> str:
    for pat, tag in _MASKS:
        line = pat.sub(tag, line)

    return line


def make_sheet(log_path: str, out_path: str, max_dist: float) -> None:
    best: dict = {}

    for raw in Path(log_path).read_text().splitlines():
        if not raw.strip():
            continue

        row = json.loads(raw)
        key = (row["template_id"], norm(row["query_line"]))
        rec = best.get(key)

        if rec is None:
            best[key] = {
                "template_id": row["template_id"],
                "kb_issue": row["kb_issue"],
                "masked_line": key[1],
                "distance": row["distance"],
                "example_line": row["query_line"],
                "cases": {row["case_id"]},
            }

        else:
            rec["cases"].add(row["case_id"])

            if row["distance"] < rec["distance"]:
                rec["distance"] = row["distance"]
                rec["example_line"] = row["query_line"]

    rows = sorted((r for r in best.values() if r["distance"] <= max_dist), key=lambda r: r["distance"])

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distance", "template_id", "kb_issue", "n_cases", "example_line", "relevant"])

        for r in rows:
            w.writerow([f"{r['distance']:.4f}", r["template_id"], r["kb_issue"], len(r["cases"]), r["example_line"], ""])

    print(f"{len(best)} unique (KB entry, masked line) pairs; {len(rows)} at dist <= {max_dist} written to {out_path}")
    print("Fill the `relevant` column with y/n, then run the sweep.")


def sweep(labels_path: str) -> None:
    labeled = []

    with open(labels_path, newline="") as f:
        for r in csv.DictReader(f):
            v = r["relevant"].strip().lower()

            if v in ("y", "n"):
                labeled.append((float(r["distance"]), v == "y"))

    if not labeled:
        raise SystemExit("No labeled rows (relevant must be y or n).")

    total_rel = sum(1 for _, rel in labeled if rel)
    total_irr = len(labeled) - total_rel
    print(f"labeled pairs: {len(labeled)} ({total_rel} relevant, {total_irr} irrelevant)\n")
    print(f"{'cutoff':>7} {'kept_rel':>9} {'kept_irr':>9} {'rel_recall':>11} {'precision':>10}")

    for i in range(20, 105, 5):
        t = i / 100
        kr = sum(1 for d, rel in labeled if rel and d <= t)
        ki = sum(1 for d, rel in labeled if not rel and d <= t)
        recall = kr / total_rel if total_rel else 0.0
        prec = kr / (kr + ki) if kr + ki else 0.0
        print(f"{t:>7.2f} {kr:>9} {ki:>9} {recall:>11.2f} {prec:>10.2f}")

    irr = [d for d, rel in labeled if not rel]

    if irr:
        print(f"\nclosest irrelevant pair: dist={min(irr):.4f} "
              f"(any cutoff below this keeps zero irrelevant matches)")

    rel = [d for d, rel_ in labeled if rel_]

    if rel:
        print(f"farthest relevant pair:  dist={max(rel):.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("make-sheet")
    a.add_argument("--log", default="eval/rag_distances.jsonl")
    a.add_argument("--out", default="eval/rag_labels.csv")
    a.add_argument("--max-dist", type=float, default=1.0)
    b = sub.add_parser("sweep")
    b.add_argument("--labels", default="eval/rag_labels.csv")
    args = p.parse_args()

    if args.cmd == "make-sheet":
        make_sheet(args.log, args.out, args.max_dist)

    else:
        sweep(args.labels)


if __name__ == "__main__":
    main()