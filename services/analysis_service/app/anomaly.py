"""
Re-checks anomaly status on GET /analyze using the same
BaselineAnomalyDetector and baseline vectors as ingestion_service, so both
services agree on what counts as anomalous. Refit fresh each request to
match main.py's existing "always reflect the latest baseline" behavior.
"""
from logsage_common.anomaly import BaselineAnomalyDetector, fetch_baseline_vectors


def is_anomalous(query_text: str, vector_store, percentile: float = 95.0):
    baseline_vectors = fetch_baseline_vectors(vector_store)
    detector = BaselineAnomalyDetector(percentile=percentile).fit(baseline_vectors)

    results = vector_store.query(query_text, k=1)
    nearest = results[0] if results else None

    vec = vector_store.embed([query_text])
    anomalous = bool(detector.score(vec)[0] > detector.threshold)

    return anomalous, nearest