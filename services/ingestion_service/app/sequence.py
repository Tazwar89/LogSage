"""
Sequence-level (per-block) anomaly scoring for ingestion_service.

The per-line detector (anomaly.py) cannot see anomalies where every line is
individually normal but the block's event sequence is wrong. This groups an
uploaded log's lines by BlockId, maps each line to a learned Drain3 template
ID, and scores each block's template sequence with the trained DeepLog-style
LSTM (logsage_common.sequence_anomaly.DeepLogDetector).

Model artifacts (produced by scripts/train_sequence_model.py) live in
SEQUENCE_MODEL_DIR (default /app/models/sequence):
    deeplog.pt              LSTM weights + vocab + calibrated threshold
    drain_sequence.state    fitted Drain3 miner (same template IDs as training)
    meta.json               training metadata / metrics (optional, informational)

Uploaded files must contain COMPLETE block sessions: a block whose lines are
split across two uploads looks truncated (missing END) and will be flagged.
"""
from __future__ import annotations

import json, logging, os, threading
from dataclasses import dataclass
from pathlib import Path

from logsage_common.sequence_parsing import (
    UNKNOWN_TEMPLATE_ID,
    build_masking_miner,
    group_by_block,
    match_template_id,
)

logger = logging.getLogger("ingestion_service")

MODEL_FILE = "deeplog.pt"
STATE_FILE = "drain_sequence.state"
META_FILE = "meta.json"
MAX_CONTEXT_LINES = 30  # lines of an anomalous block sent downstream for LLM diagnosis
DETECTOR_NAME = "deeplog-sequence"


@dataclass
class BlockResult:
    block_id: str
    score: float
    is_anomalous: bool
    line_indices: list[int]
    unknown_templates: int


class SequenceScorer:
    def __init__(self, model_dir: str | None = None):
        self.model_dir = Path(model_dir or os.getenv("SEQUENCE_MODEL_DIR", "/app/models/sequence"))
        self._detector = None
        self._miner = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return (self.model_dir / MODEL_FILE).is_file() and (self.model_dir / STATE_FILE).is_file()


    def _load(self) -> None:
        with self._lock:
            if self._detector is not None:
                return

            from logsage_common.sequence_anomaly import DeepLogDetector

            self._detector = DeepLogDetector.load(str(self.model_dir / MODEL_FILE))
            self._miner = build_masking_miner(str(self.model_dir / STATE_FILE))
            logger.info(f"Loaded sequence model from {self.model_dir} (threshold={self._detector.threshold:.4f})")


    def status(self) -> dict:
        info = {"available": self.available, "model_dir": str(self.model_dir), "detector": DETECTOR_NAME}
        meta_path = self.model_dir / META_FILE

        if meta_path.is_file():
            info["meta"] = json.loads(meta_path.read_text())

        return info


    def score_logs(self, parsed_logs: list[dict]) -> list[BlockResult]:
        """Scores every block mentioned in `parsed_logs` (dicts with a 'message' key)."""
        if not self.available:
            raise FileNotFoundError(f"No sequence model in {self.model_dir}")

        self._load()

        # 1. Capture the detector in local scope and check it immediately.
        # This isolates the variable and guarantees to Pylance it cannot be None downstream.
        detector = self._detector
        if detector is None:
            raise RuntimeError("Model detector failed to initialize or load properly.")

        groups = group_by_block([log["message"] for log in parsed_logs])

        if not groups:
            return []

        template_ids = [match_template_id(self._miner, log["message"]) for log in parsed_logs]
        block_ids = list(groups)
        sequences = [[template_ids[i] for i in groups[b]] for b in block_ids]

        # 2. Reference your clean, non-null local 'detector' variable here
        scores = detector.score(sequences)
        flags = scores > detector.threshold

        return [
            BlockResult(
                block_id=b,
                score=float(scores[k]),
                is_anomalous=bool(flags[k]),
                line_indices=groups[b],
                unknown_templates=sum(1 for t in sequences[k] if t == UNKNOWN_TEMPLATE_ID),
            )
            for k, b in enumerate(block_ids)
        ]


def build_block_entry(parsed_logs: list[dict], result: BlockResult) -> dict:
    """
    One Kafka/Redis entry per anomalous block. Keeps the standard parsed-log
    fields (date/time/pid/level/component from the block's first line) so
    consumers and /stats keep working; `message` becomes the block's event
    context, which is what the LLM diagnostic pipeline reads.
    """
    lines = [parsed_logs[i] for i in result.line_indices]
    fmt = lambda l: f"[{l.get('level', '')}] {l.get('component', '')}: {l['message']}"  # noqa: E731

    if len(lines) > MAX_CONTEXT_LINES:
        half = MAX_CONTEXT_LINES // 2
        omitted = len(lines) - 2 * half
        context = [fmt(l) for l in lines[:half]] + [f"... ({omitted} lines omitted)"] + [fmt(l) for l in lines[-half:]]

    else:
        context = [fmt(l) for l in lines]

    entry = dict(lines[0])
    entry.update(
        message="\n".join(context),
        block_id=result.block_id,
        n_events=len(lines),
        sequence_anomaly_score=round(result.score, 4),
        unknown_templates=result.unknown_templates,
        is_anomalous=True,
        detector=DETECTOR_NAME,
    )

    return entry