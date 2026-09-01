import json

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


def retrieve_context(anomalous_text, kb_vector_store, kb_lookup, k=3):
    results = kb_vector_store.query(anomalous_text, k=k)

    return [kb_lookup[r["template_id"]] for r in results]