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

import gzip, pickle, re, sys
from pathlib import Path

_SERVICES = Path(__file__).resolve().parent.parent / "services"
sys.path.insert(0, str(_SERVICES))

from ingestion_service.app.parsing import parse_line  # noqa: E402

BLOCK_ID_RE = re.compile(r"blk_-?\d+")


def build_masking_miner():
    from drain3 import TemplateMiner
    from drain3.masking import MaskingInstruction
    from drain3.template_miner_config import TemplateMinerConfig

    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.masking_instructions = [
        MaskingInstruction(r"blk_-?\d+", "BLK"),
        MaskingInstruction(r"/\S+", "PATH"),
        MaskingInstruction(r"(\d{1,3}\.){3}\d{1,3}(:\d+)?", "IP"),
        MaskingInstruction(r"(?<![\w])-?\d+(?![\w])", "NUM"),
    ]

    return TemplateMiner(config=config)


def build_block_sequences(
    log_path: str,
    max_blocks: int | None = None,
    progress_every: int = 500_000,
) -> tuple[dict[str, list[int]], dict[int, str]]:
    """
    Returns ({block_id: [template_id, ...] in log order}, {template_id: template_string}).

    max_blocks keeps only the first N distinct blocks by first appearance;
    every later line of those blocks is still included, and lines that only
    mention other blocks skip Drain entirely.
    """
    miner = build_masking_miner()
    sequences: dict[str, list[int]] = {}
    n_lines = 0

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            n_lines += 1

            if progress_every and n_lines % progress_every == 0:
                print(f"  {n_lines:,} lines, {len(sequences):,} blocks, {len(miner.drain.clusters)} templates")

            parsed = parse_line(raw)

            if parsed is None:
                continue

            message = parsed["message"]
            blocks = set(BLOCK_ID_RE.findall(message))

            if not blocks:
                continue

            targets = []

            for b in blocks:
                if b in sequences:
                    targets.append(b)

                elif max_blocks is None or len(sequences) < max_blocks:
                    sequences[b] = []
                    targets.append(b)

            if not targets:
                continue

            template_id = miner.add_log_message(message)["cluster_id"]

            for b in targets:
                sequences[b].append(template_id)

    templates = {c.cluster_id: c.get_template() for c in miner.drain.clusters}

    return sequences, templates


def save_cache(path: str, sequences: dict, templates: dict) -> None:
    with gzip.open(path, "wb") as f:
        pickle.dump((sequences, templates), f)


def load_cache(path: str) -> tuple[dict, dict]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)