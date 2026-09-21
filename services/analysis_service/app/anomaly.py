"""
Re-checks anomaly status on GET /analyze using the same detector and
baseline vectors as ingestion_service, so both services agree on what counts
as anomalous.

ANOMALY_DETECTOR=knn (default): scikit-learn k-NN + percentile threshold,
refit fresh each request.
ANOMALY_DETECTOR=autoencoder: TensorFlow autoencoder. If AUTOENCODER_DIR
points at a saved detector it is loaded; otherwise one is trained on the
current baseline and cached until the baseline changes.
"""
import hashlib
import os

from logsage_common.anomaly import BaselineAnomalyDetector, fetch_baseline_vectors

_autoencoder_cache = {}


def _baseline_key(vectors, percentile):
    return hashlib.sha1(vectors.tobytes()).hexdigest() + f":{percentile}"


def _autoencoder_detector(baseline_vectors, percentile):
    from logsage_common.tf_autoencoder import AutoencoderAnomalyDetector

    saved_dir = os.getenv("AUTOENCODER_DIR")
    key = saved_dir or _baseline_key(baseline_vectors, percentile)

    if key not in _autoencoder_cache:
        if saved_dir and os.path.isdir(saved_dir):
            _autoencoder_cache[key] = AutoencoderAnomalyDetector.load(saved_dir)

        else:
            _autoencoder_cache[key] = AutoencoderAnomalyDetector(percentile=percentile).fit(baseline_vectors)

    return _autoencoder_cache[key]


def build_detector(baseline_vectors, percentile: float = 95.0):
    if os.getenv("ANOMALY_DETECTOR", "knn").lower() == "autoencoder" and baseline_vectors.shape[0] >= 2:
        return _autoencoder_detector(baseline_vectors, percentile)

    return BaselineAnomalyDetector(percentile=percentile).fit(baseline_vectors)


def is_anomalous(query_text: str, vector_store, percentile: float = 95.0):
    baseline_vectors = fetch_baseline_vectors(vector_store)
    detector = build_detector(baseline_vectors, percentile)

    results = vector_store.query(query_text, k=1)
    nearest = results[0] if results else None

    vec = vector_store.embed([query_text])
    anomalous = bool(detector.score(vec)[0] > detector.threshold)

    return anomalous, nearest