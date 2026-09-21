import random

import pytest
from ingestion_service.app.parsing import parse_line
from ingestion_service.app.sequence import (
    MAX_CONTEXT_LINES,
    SequenceScorer,
    build_block_entry,
)
from logsage_common.sequence_anomaly import DeepLogDetector
from logsage_common.sequence_parsing import (
    UNKNOWN_TEMPLATE_ID,
    build_masking_miner,
    group_by_block,
    match_template_id,
)


def _line(comp: str, msg: str) -> str:
    return f"081109 203615 148 INFO {comp}: {msg}"


def block_lines(rng: random.Random, blk: str, kind: str = "ok") -> list[str]:
    ips = [f"10.250.{rng.randint(1, 20)}.{rng.randint(1, 254)}" for _ in range(3)]
    alloc = [_line("dfs.FSNamesystem", f"BLOCK* NameSystem.allocateBlock: /user/root/part-{rng.randint(1, 999)}. {blk}")]
    recv = [_line("dfs.DataNode$DataXceiver", f"Receiving block {blk} src: /{ip}:5{rng.randint(1000, 9999)} dest: /{ip}:50010") for ip in ips]
    mid = []

    for i, ip in enumerate(ips):
        pair = [_line("dfs.DataNode$PacketResponder", f"PacketResponder {i} for block {blk} terminating"),
                _line("dfs.DataNode$DataXceiver", f"Received block {blk} of size 67108864 from /{ip}")]
        rng.shuffle(pair)
        mid += pair

    stored = [_line("dfs.FSNamesystem", f"BLOCK* NameSystem.addStoredBlock: blockMap updated: {ip}:50010 is added to {blk} size 67108864") for ip in ips]

    if kind == "missing":
        stored = stored[:2]

    elif kind == "reordered":
        return stored + alloc + recv + mid

    elif kind == "unseen":
        mid.insert(2, _line("dfs.DataNode", f"Unexpected error disk failure while writing {blk}"))

    return alloc + recv + mid + stored


def parse_all(lines):
    return [p for p in (parse_line(l) for l in lines) if p]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("seqmodel")
    rng = random.Random(0)
    miner = build_masking_miner(str(d / "drain_sequence.state"))
    seqs = []

    for i in range(400):
        blk = f"blk_{rng.randint(1, 10**15)}"
        seqs.append([
            miner.add_log_message(p["message"])["cluster_id"] 
            for l in block_lines(rng, blk)
            if (p := parse_line(l)) is not None
        ])

    miner.save_state("test")
    det = DeepLogDetector(steps=800, seed=0).fit(seqs[:320])
    det.calibrate(seqs[320:], target_fpr=0.01)
    det.save(str(d / "deeplog.pt"))

    return d


@pytest.fixture(scope="module")
def scorer(model_dir):
    return SequenceScorer(str(model_dir))


def _score(scorer, kinds, seed=1):
    rng = random.Random(seed)
    lines, blocks = [], {}

    for i, kind in enumerate(kinds):
        blk = f"blk_{-(i + 1)}{rng.randint(1000, 9999)}"
        blocks[blk] = kind
        lines += block_lines(rng, blk, kind)

    parsed = parse_all(lines)

    return parsed, blocks, {r.block_id: r for r in scorer.score_logs(parsed)}


def test_normal_blocks_pass_and_anomalous_blocks_flagged(scorer):
    kinds = ["ok"] * 60 + ["missing"] * 10 + ["reordered"] * 10
    _, blocks, results = _score(scorer, kinds)

    normal_flagged = sum(results[b].is_anomalous for b, k in blocks.items() if k == "ok")

    assert len(results) == len(kinds)
    assert normal_flagged <= 3
    assert all(results[b].is_anomalous for b, k in blocks.items() if k in ("missing", "reordered"))


def test_unseen_template_is_flagged_and_counted(scorer):
    _, blocks, results = _score(scorer, ["unseen"] * 5)

    assert all(r.is_anomalous and r.unknown_templates >= 1 for r in results.values())


def test_build_block_entry_keeps_parsed_fields_and_adds_context(scorer):
    parsed, blocks, results = _score(scorer, ["missing"])
    result = next(iter(results.values()))
    entry = build_block_entry(parsed, result)

    assert {"date", "time", "pid", "level", "component"} <= set(entry)
    assert entry["block_id"] == result.block_id
    assert entry["detector"] == "deeplog-sequence" and entry["is_anomalous"] is True
    assert entry["n_events"] == len(result.line_indices)
    assert result.block_id in entry["message"]
    assert entry["message"].count("\n") == entry["n_events"] - 1


def test_build_block_entry_truncates_long_blocks(scorer):
    parsed, _, results = _score(scorer, ["ok"])
    result = next(iter(results.values()))
    result.line_indices = (result.line_indices * 5)[:60]
    entry = build_block_entry(parsed, result)

    assert "lines omitted" in entry["message"]
    assert entry["message"].count("\n") == MAX_CONTEXT_LINES  # half + marker + half
    assert entry["n_events"] == 60


def test_lines_without_block_ids_are_ignored(scorer):
    parsed = parse_all([_line("dfs.DataNode", "Verification succeeded for something")])

    assert scorer.score_logs(parsed) == []


def test_missing_model_reports_unavailable_and_raises(tmp_path):
    empty = SequenceScorer(str(tmp_path))

    assert not empty.available
    assert empty.status()["available"] is False

    with pytest.raises(FileNotFoundError):
        empty.score_logs([{"message": "x blk_1"}])


def test_group_by_block_assigns_multi_block_lines_to_each_block():
    groups = group_by_block(["a blk_1", "ask to delete blk_1 blk_-2", "no block here", "b blk_-2"])

    assert groups == {"blk_1": [0, 1], "blk_-2": [1, 3]}


def test_match_template_id_never_learns_and_flags_unknown(model_dir):
    miner = build_masking_miner(str(model_dir / "drain_sequence.state"))
    n_before = len(miner.drain.clusters)

    known = match_template_id(miner, "PacketResponder 1 for block blk_5 terminating")
    unknown = match_template_id(miner, "kernel panic: totally unrelated message")

    assert known != UNKNOWN_TEMPLATE_ID
    assert unknown == UNKNOWN_TEMPLATE_ID
    assert len(miner.drain.clusters) == n_before