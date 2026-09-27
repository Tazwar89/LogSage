"""
Quick manual look at RAG retrieval distances against the real KB, before
committing to the full RAG_LOG_ALL_DISTANCES instrumentation pass.

Builds a scratch Qdrant collection from the real knowledge_base.json and
queries it with a handful of representative log lines: some that SHOULD
match a KB entry closely, some that plausibly shouldn't match anything,
so you get an early feel for where relevant/irrelevant distances fall
before designing the full calibration sample.

Needs a reachable Qdrant (QDRANT_URL, default http://localhost:6333) and
the sentence-transformers embedding model (first run downloads it).

Usage (from repo root):
    python -m eval.eyeball_rag
    python -m eval.eyeball_rag --lines path/to/one_line_per_query.txt
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "services"))
sys.path.insert(0, str(_ROOT / "libs" / "logsage_common"))

from analysis_service.app.rag import build_kb_index, load_knowledge_base

# A few hand-picked lines: 3 that should hit a KB entry cleanly, 2 that
# plausibly shouldn't match anything well (routine/benign lines), so you can
# see where "clearly relevant" vs "clearly irrelevant" distances sit before
# doing the full labeled sample.
DEFAULT_LINES = [
    # should match: "Unexpected error trying to delete block. BlockInfo not found in volumeMap"
    "Unexpected error trying to delete block blk_123456. BlockInfo not found in volumeMap.",
    # should match: "writeBlock received exception java.io.IOException: Could not read from stream"
    "writeBlock blk_987654 received exception java.io.IOException: Could not read from stream",
    # should match: "addStoredBlock request received for block on DataNode but it does not belong to any file"
    "BLOCK* NameSystem.addStoredBlock: addStoredBlock request received for blk_555 on 10.0.0.5:50010 But it does not belong to any file.",
    # should NOT match well: routine, non-anomalous line
    "PacketResponder 1 for block blk_222 terminating",
    # should NOT match well: routine, non-anomalous line
    "Received block blk_333 of size 67108864 from /10.0.0.9",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lines", default=None, help="Path to a file, one query line per line")
    p.add_argument("--kb", default=str(_ROOT / "services" / "analysis_service" / "data" / "knowledge_base.json"))
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()

    from logsage_common.vector_store_qdrant import QdrantVectorStore

    kb_entries = load_knowledge_base(args.kb)
    kb_store = QdrantVectorStore(collection_name="eyeball_kb")
    kb_lookup = build_kb_index(kb_store, kb_entries)

    lines = Path(args.lines).read_text().splitlines() if args.lines else DEFAULT_LINES

    for line in lines:
        if not line.strip():
            continue

        results = kb_store.query(line, k=args.k)
        print(f"\nQUERY: {line}")

        if not results:
            print("  (no results -- collection may be empty)")
            continue

        for r in results:
            issue = kb_lookup[r["template_id"]]["issue"]
            print(f"  dist={r['distance']:.4f}  issue={issue!r}")


if __name__ == "__main__":
    main()