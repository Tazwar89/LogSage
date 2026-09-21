# scripts/split_baseline_holdout.py
import random
import re
import sys
from collections import defaultdict

import pandas as pd

BLOCK_RE = re.compile(r"blk_-?\d+")

def split(log_path, labels_path, holdout_frac=0.3, seed=42):
    labels = pd.read_csv(labels_path)
    anomaly_ids = set(labels[labels.Label == "Anomaly"]["BlockId"])

    blocks = defaultdict(list)

    for line in open(log_path):
        m = BLOCK_RE.search(line)
        key = m.group() if m else "_no_block"
        blocks[key].append(line)

    anomaly_present = [k for k in blocks if k in anomaly_ids]
    normal_present = [k for k in blocks if k not in anomaly_ids]

    random.seed(seed)
    random.shuffle(normal_present)
    n = int(len(normal_present) * holdout_frac)
    hold_norm, base_norm = normal_present[:n], normal_present[n:]

    with open("HDFS_2k.baseline.log", "w") as f:
        for k in base_norm:              # normal blocks only
            f.writelines(blocks[k])

    with open("HDFS_2k.holdout.log", "w") as f:
        for k in anomaly_present + hold_norm:   # all anomalies + holdout normals
            f.writelines(blocks[k])

    print(f"baseline: 0 anomaly blocks, {len(base_norm)} normal blocks")
    print(f"holdout:  {len(anomaly_present)} anomaly blocks, {len(hold_norm)} normal blocks")

if __name__ == "__main__":
    split(sys.argv[1], sys.argv[2])