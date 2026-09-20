"""
Synthetic-data tests for sequence-level anomaly detection.

The generator mimics the HDFS block lifecycle: allocate -> 3x receiving ->
3x (responder terminating, received) -> 3x addStoredBlock -> optional 3x
delete. Anomalies are built so each detector's blind spot is explicit:
  missing_event / unseen_event  -> change the event COUNTS  (PCA and DeepLog see them)
  reordered                     -> same counts, wrong ORDER (only DeepLog can see it)
"""
import random

import numpy as np
import pytest

from logsage_common.sequence_anomaly import CountVectorPCADetector, DeepLogDetector

ALLOC, RECV, RESP, RCVD, STORED, DELETE, UNSEEN = 1, 2, 3, 4, 5, 6, 99


def normal_seq(rng: random.Random) -> list[int]:
    seq = [ALLOC] + [RECV] * 3

    for _ in range(3):
        pair = [RESP, RCVD]
        rng.shuffle(pair)
        seq += pair

    seq += [STORED] * 3

    if rng.random() < 0.5:
        seq += [DELETE] * 3

    return seq


def anomalous_seq(rng: random.Random, kind: str) -> list[int]:
    seq = normal_seq(rng)

    if kind == "missing_event":
        seq.remove(STORED)

    elif kind == "unseen_event":
        seq.insert(rng.randrange(len(seq)), UNSEEN)

    elif kind == "reordered":
        # Same multiset of events as a normal block, lifecycle order violated.
        seq = [STORED] * 3 + [ALLOC] + [RECV] * 3 + [RESP, RCVD] * 3

    else:
        raise ValueError(kind)

    return seq


@pytest.fixture(scope="module")
def data():
    rng = random.Random(0)

    return {
        "train": [normal_seq(rng) for _ in range(600)],
        "val": [normal_seq(rng) for _ in range(200)],
        "test_normal": [normal_seq(rng) for _ in range(200)],
        **{k: [anomalous_seq(rng, k) for _ in range(50)] for k in ("missing_event", "unseen_event", "reordered")},
    }


@pytest.fixture(scope="module")
def pca(data):
    det = CountVectorPCADetector().fit(data["train"])
    det.calibrate(data["val"], target_fpr=0.01)

    return det


@pytest.fixture(scope="module")
def deeplog(data):
    det = DeepLogDetector(seed=0).fit(data["train"])
    det.calibrate(data["val"], target_fpr=0.01)

    return det


def recall(det, seqs) -> float:
    return float(det.is_anomalous(seqs).mean())


def test_pca_low_false_positives_on_normal(pca, data):
    assert recall(pca, data["test_normal"]) <= 0.05


@pytest.mark.parametrize("kind", ["missing_event", "unseen_event"])
def test_pca_catches_count_anomalies(pca, data, kind):
    assert recall(pca, data[kind]) >= 0.9


def test_pca_is_blind_to_pure_reordering(pca, data):
    # Documents the architectural limit that motivated DeepLog.
    assert recall(pca, data["reordered"]) <= 0.1


def test_deeplog_low_false_positives_on_normal(deeplog, data):
    assert recall(deeplog, data["test_normal"]) <= 0.05


@pytest.mark.parametrize("kind", ["missing_event", "unseen_event", "reordered"])
def test_deeplog_catches_all_anomaly_kinds(deeplog, data, kind):
    assert recall(deeplog, data[kind]) >= 0.9


def test_deeplog_flags_truncated_session_via_end_token(deeplog):
    truncated = [ALLOC] + [RECV] * 3 + [RESP, RCVD]

    assert deeplog.is_anomalous([truncated])[0]


def test_deeplog_scoring_handles_duplicates_and_order(deeplog, data):
    seqs = [data["test_normal"][0], data["reordered"][0], data["test_normal"][0]]
    scores = deeplog.score(seqs)

    assert scores[0] == scores[2]
    assert scores[1] > scores[0]


def test_deeplog_save_load_roundtrip(deeplog, data, tmp_path):
    path = str(tmp_path / "deeplog.pt")
    deeplog.save(path)
    loaded = DeepLogDetector.load(path)
    seqs = data["test_normal"][:20] + data["reordered"][:20]

    np.testing.assert_array_equal(deeplog.score(seqs), loaded.score(seqs))
    assert loaded.threshold == deeplog.threshold


def test_calibrate_respects_target_fpr(deeplog, data):
    deeplog.calibrate(data["val"], target_fpr=0.01)

    assert deeplog.is_anomalous(data["val"]).mean() <= 0.01


def test_deeplog_rank_mode_matches_paper_rule_and_flags_reordering(data):
    det = DeepLogDetector(mode="rank", seed=0).fit(data["train"])
    det.calibrate(data["val"], target_fpr=0.01)

    assert det.is_anomalous(data["test_normal"]).mean() <= 0.05
    assert det.is_anomalous(data["reordered"]).mean() >= 0.9
    assert det.threshold == det.top_g - 1


def test_build_block_sequences_from_hdfs_format_log(tmp_path):
    from eval.hdfs_sessions import build_block_sequences

    lines = [
        "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_111 terminating",
        "081109 203615 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.19.102:50010 is added to blk_111 size 67108864",
        "081109 203807 222 INFO dfs.FSNamesystem: BLOCK* ask 10.250.19.102:50010 to delete blk_111 blk_222",
        "081109 203807 222 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_222 terminating",
        "not a log line",
    ]
    log = tmp_path / "mini.log"
    log.write_text("\n".join(lines) + "\n")

    seqs, templates = build_block_sequences(str(log), progress_every=0)

    assert set(seqs) == {"blk_111", "blk_222"}
    assert len(seqs["blk_111"]) == 3  # responder, addStored, delete-ask
    assert len(seqs["blk_222"]) == 2  # delete-ask (shared line), responder
    assert seqs["blk_111"][0] == seqs["blk_222"][1]  # same masked template
    assert seqs["blk_111"][2] == seqs["blk_222"][0]  # multi-block line -> both sessions


def test_build_block_sequences_max_blocks(tmp_path):
    from eval.hdfs_sessions import build_block_sequences

    lines = [f"081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_{i} terminating" for i in range(10)]
    log = tmp_path / "cap.log"
    log.write_text("\n".join(lines) + "\n")

    seqs, _ = build_block_sequences(str(log), max_blocks=3, progress_every=0)

    assert len(seqs) == 3


def test_split_blocks_keeps_anomalies_out_of_training():
    from eval.sequence_eval import split_blocks

    seqs = {f"b{i}": [1, 2] for i in range(100)}
    labels = {f"b{i}": i % 10 == 0 for i in range(100)}
    splits = split_blocks(seqs, labels)

    assert not set(splits["train"]) & set(splits["test_anomalous"])
    assert len(splits["test_anomalous"]) == 10
    assert all(not labels[b] for b in splits["train"] + splits["val"] + splits["test_normal"])