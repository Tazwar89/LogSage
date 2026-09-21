"""
Builds per-block template-ID sequences from a raw HDFS log, for
sequence-level anomaly detection.

Why this exists: Loghub's HDFS labels are per BlockId, and a block's label
depends on its *whole* event sequence. The 2k-line GitHub sample contains
~1 line per block, so there is no sequence to model. This needs the full
HDFS_v1 HDFS.log (~11M lines) from the same Zenodo download as
anomaly_label.csv.

Lines are Drain3-mined into template IDs (in-memory miner with masking, no
state file), then appended to every block ID the line mentions -- the same
session-assignment convention Loglizer uses (one line can belong to several
blocks, e.g. the "ask ... to delete blk_a blk_b blk_c" line).
"""
from __future__ import annotations

import gzip
import pickle
import sys
from pathlib import Path

_SERVICES = Path(__file__).resolve().parent.parent / "services"
sys.path.insert(0, str(_SERVICES))

from ingestion_service.app.parsing import parse_line
from logsage_common.sequence_parsing import BLOCK_ID_RE, build_masking_miner


def iter_selected_lines(log_path: str, max_blocks: int | None = None, progress_cb=None, progress_every: int = 500_000):
    """
    Yields (message, [block_ids]) for every line that belongs to a selected
    block, in log order. max_blocks keeps only the first N distinct blocks by
    first appearance; every later line of those blocks is still yielded, and
    lines that only mention other blocks are skipped entirely (so Drain never
    sees them). Shared by training and the serving-parity check so both
    process exactly the same lines in exactly the same order.
    """
    selected: set[str] = set()
    n_lines = 0

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            n_lines += 1

            if progress_cb and progress_every and n_lines % progress_every == 0:
                progress_cb(n_lines, len(selected))

            parsed = parse_line(raw)

            if parsed is None:
                continue

            message = parsed["message"]
            targets = []

            for b in dict.fromkeys(BLOCK_ID_RE.findall(message)):
                if b in selected:
                    targets.append(b)

                elif max_blocks is None or len(selected) < max_blocks:
                    selected.add(b)
                    targets.append(b)

            if targets:
                yield message, targets


def build_block_sequences(
    log_path: str,
    max_blocks: int | None = None,
    progress_every: int = 500_000,
    state_path: str | None = None,
) -> tuple[dict[str, list[int]], dict[int, str]]:
    """
    Returns ({block_id: [template_id, ...] in log order}, {template_id: template_string}).

    state_path persists the fitted Drain3 miner (needed to reuse the same
    template IDs at serving time). See iter_selected_lines for max_blocks.
    """
    miner = build_masking_miner(state_path)
    sequences: dict[str, list[int]] = {}

    def progress(n_lines, n_blocks):
        print(f"  {n_lines:,} lines, {n_blocks:,} blocks, {len(miner.drain.clusters)} templates")

    for message, targets in iter_selected_lines(log_path, max_blocks, progress, progress_every):
        template_id = miner.add_log_message(message)["cluster_id"]

        for b in targets:
            sequences.setdefault(b, []).append(template_id)

    if state_path:
        miner.save_state("end of training parse")

    templates = {c.cluster_id: c.get_template() for c in miner.drain.clusters}

    return sequences, templates


def save_cache(path: str, sequences: dict, templates: dict) -> None:
    with gzip.open(path, "wb") as f:
        pickle.dump((sequences, templates), f)


def load_cache(path: str) -> tuple[dict, dict]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)