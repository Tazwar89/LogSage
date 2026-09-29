import json
import os
import time

# Qdrant collections use EUCLID distance on normalized embeddings (range 0-2; lower = closer).
# Entries farther than this are dropped so unrelated KB items never reach the LLM.
# Calibrated on a manually-labeled sample: 30 unique (KB entry, log line) pairs
# from a 60-block eval run (5 labeled relevant, 25 irrelevant). No cutoff fully
# separates them -- farthest relevant pair was dist 0.94, closest irrelevant
# was dist 0.75 -- so 0.70 trades recall (keeps only 1/5 relevant matches) for
# zero false-positive KB matches on this sample, since a wrong KB match misleads
# the report node while no match safely falls back to log-only reasoning.
# Small sample, single KB, not re-validated since. See eval/rag_labels.csv.
RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.70"))
RAG_MAX_QUERY_LINES = int(os.getenv("RAG_MAX_QUERY_LINES", "40"))

# Set RAG_LOG_ALL_DISTANCES=true to append every candidate (query line, KB
# template, distance) to RAG_DISTANCE_LOG_PATH as JSONL, BEFORE the
# max_distance filter is applied -- rejected matches normally vanish
# silently, so this is the only way to see what the cutoff is excluding.
# Purely additive: never changes retrieve_context's return value.
RAG_LOG_ALL_DISTANCES = os.getenv("RAG_LOG_ALL_DISTANCES", "false").lower() == "true"
RAG_DISTANCE_LOG_PATH = os.getenv("RAG_DISTANCE_LOG_PATH", "eval/rag_distances.jsonl")


def load_knowledge_base(path="data/knowledge_base.json"):
    """
    knowledge_base.json format:
    [{"issue": "OutOfMemoryError in FSNamesystem", "fix": "Increase heap size via -Xmx flag"}, ...]
    """
    with open(path) as f:
        return json.load(f)


def build_kb_index(vector_store, kb_entries):
    templates = {i: entry["issue"] for i, entry in enumerate(kb_entries)}
    vector_store.build_index(templates)

    return {i: entry for i, entry in enumerate(kb_entries)}


def _log_candidates(case_id, query_line, results, kb_lookup):
    """Appends one JSONL row per (query_line, kb_candidate) pair, unfiltered."""
    try:
        with open(RAG_DISTANCE_LOG_PATH, "a") as f:
            for r in results:
                tid = r["template_id"]
                row = {
                    "ts": time.time(),
                    "case_id": case_id,
                    "query_line": query_line,
                    "template_id": tid,
                    "kb_issue": kb_lookup.get(tid, {}).get("issue"),
                    "distance": r["distance"],
                }
                f.write(json.dumps(row) + "\n")

    except OSError:
        pass  # logging is best-effort; never break retrieval over a bad path/permissions


def retrieve_context(anomalous_text, kb_vector_store, kb_lookup, k=3, max_distance=None, case_id=None):
    """
    Queries the KB once per line (a block's context is many lines; embedding the whole
    blob dilutes the one anomalous line), keeps each KB entry's best distance, drops
    anything beyond max_distance, and returns up to k entries, closest first.
    Returns [] when nothing is close enough.

    case_id: optional identifier (e.g. block_id) attached to logged rows when
    RAG_LOG_ALL_DISTANCES=true, so distance samples can be traced back to a case.
    """
    if max_distance is None:
        max_distance = RAG_MAX_DISTANCE

    if case_id is None:
        # research_node() calls this with no case_id (its signature is fixed by
        # LangGraph's node contract). Eval scripts that want traceable logs set
        # this env var per-case instead of threading case_id through the graph.
        case_id = os.getenv("RAG_CURRENT_CASE_ID")

    lines = [l.strip() for l in anomalous_text.splitlines() if l.strip() and not l.startswith("...")]
    queries = lines[:RAG_MAX_QUERY_LINES] or [anomalous_text]

    best: dict = {}
    for q in queries:
        results = kb_vector_store.query(q, k=k)

        if RAG_LOG_ALL_DISTANCES:
            _log_candidates(case_id, q, results, kb_lookup)

        for r in results:
            tid, dist = r["template_id"], r["distance"]

            if tid not in best or dist < best[tid]:
                best[tid] = dist

    ranked = sorted((d, t) for t, d in best.items() if d <= max_distance)

    return [kb_lookup[t] for _, t in ranked[:k]]