def is_anomalous(query_text, vector_store, threshold=0.6):
    results = vector_store.query(query_text, k=1)

    if not results:
        return True, None  # nothing in index yet = treat as anomalous

    nearest = results[0]

    return nearest["distance"] > threshold, nearest