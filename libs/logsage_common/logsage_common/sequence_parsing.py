"""
Shared parsing config for sequence-level anomaly detection.

The Drain3 miner configuration (masking rules) and the block-ID regex MUST be
identical at training time (eval/hdfs_sessions.py, scripts/train_sequence_model.py)
and serving time (ingestion_service), otherwise template IDs drift and the
trained detector sees a different vocabulary than it learned. Keep them here,
in one place.

drain3 is imported lazily so importing this module needs no extra deps.
"""
from __future__ import annotations

import re

BLOCK_ID_RE = re.compile(r"blk_-?\d+")

# Template ID used for lines that match no learned template at serving time.
# It is never in the detector's vocabulary, so it maps to the UNK token, which
# the detector treats as maximally surprising.
UNKNOWN_TEMPLATE_ID = -1


def build_masking_miner(state_path: str | None = None):
    """In-memory Drain3 miner with HDFS-friendly masking. If state_path is
    given, state is loaded from / persisted to that file."""
    from drain3 import TemplateMiner
    from drain3.file_persistence import FilePersistence
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

    # Branch the instantiation to avoid explicitly passing a nullable None down the signature
    if state_path:
        persistence = FilePersistence(state_path)
        return TemplateMiner(persistence_handler=persistence, config=config)

    # Omit persistence_handler entirely so the library implicitly uses its internal default (None)
    return TemplateMiner(config=config)


def match_template_id(miner, message: str) -> int:
    """Template ID for `message` using only already-learned templates (no learning,
    no state change). Unmatched lines get UNKNOWN_TEMPLATE_ID."""
    cluster = miner.match(message, full_search_strategy="fallback")

    return cluster.cluster_id if cluster is not None else UNKNOWN_TEMPLATE_ID


def group_by_block(messages: list[str]) -> dict[str, list[int]]:
    """{block_id: [indices into `messages` that mention it]}, in input order.
    A line mentioning several blocks belongs to each of them."""
    groups: dict[str, list[int]] = {}

    for i, message in enumerate(messages):
        for block in dict.fromkeys(BLOCK_ID_RE.findall(message)):
            groups.setdefault(block, []).append(i)

    return groups