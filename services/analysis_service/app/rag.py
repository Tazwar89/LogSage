import json
import os

# Qdrant collections use EUCLID distance on normalized embeddings (range 0-2; lower = closer).
# Entries farther than this are dropped so unrelated KB items never reach the LLM.
# Calibrate against a few known-relevant and known-irrelevant pairs before trusting the default.
RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "1.0"))
RAG_MAX_QUERY_LINES = int(os.getenv("RAG_MAX_QUERY_LINES", "40"))


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


def retrieve_context(anomalous_text, kb_vector_store, kb_lookup, k=3, max_distance=None):
    """
    Queries the KB once per line (a block's context is many lines; embedding the whole
    blob dilutes the one anomalous line), keeps each KB entry's best distance, drops
    anything beyond max_distance, and returns up to k entries, closest first.
    Returns [] when nothing is close enough.
    """
    if max_distance is None:
        max_distance = RAG_MAX_DISTANCE

    lines = [l.strip() for l in anomalous_text.splitlines() if l.strip() and not l.startswith("...")]
    queries = lines[:RAG_MAX_QUERY_LINES] or [anomalous_text]

    best: dict = {}
    for q in queries:
        for r in kb_vector_store.query(q, k=k):
            tid, dist = r["template_id"], r["distance"]
            if tid not in best or dist < best[tid]:
                best[tid] = dist

    ranked = sorted((d, t) for t, d in best.items() if d <= max_distance)

    return [kb_lookup[t] for _, t in ranked[:k]]