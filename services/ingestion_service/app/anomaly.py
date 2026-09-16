"""
Ingestion-specific wrapper around the shared BaselineAnomalyDetector:
scores a batch of freshly parsed logs and splits them into anomalous vs
suppressed, annotating each with its score for the /upload/logs response.
"""
from logsage_common.anomaly import BaselineAnomalyDetector, fetch_baseline_vectors

__all__ = ["BaselineAnomalyDetector", "fetch_baseline_vectors", "filter_anomalous_logs"]


def filter_anomalous_logs(parsed_logs: list[dict], vector_store, detector: BaselineAnomalyDetector | None = None):
    if detector is None:
        baseline_vectors = fetch_baseline_vectors(vector_store)
        detector = BaselineAnomalyDetector().fit(baseline_vectors)

    if detector._fitted_on == 0:
        for log in parsed_logs:
            log["anomaly_score"] = None
            log["is_anomalous"] = True

        return parsed_logs, parsed_logs

    templates = [log.get("template_string") or log.get("message", "") for log in parsed_logs]
    vectors = vector_store.embed(templates)
    scores = detector.score(vectors)
    flags = scores > detector.threshold

    anomalous = []

    for log, score, flag in zip(parsed_logs, scores, flags):
        log["anomaly_score"] = float(score)
        log["is_anomalous"] = bool(flag)

        if flag:
            anomalous.append(log)

    return anomalous, parsed_logs