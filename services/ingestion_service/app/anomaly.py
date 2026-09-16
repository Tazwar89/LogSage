"""
Baseline anomaly detection via embedding distance (scikit-learn + NumPy).

Implements the "bonus" feature from the original project spec: store known-
normal log templates as embeddings, and flag new templates that are
"mathematically distant" from that baseline before they're forwarded for
expensive GenAI analysis.

Uses the same 384-dim MiniLM embeddings QdrantVectorStore already computes
(via sentence-transformers/PyTorch), so no extra embedding cost -- just a
scikit-learn NearestNeighbors index fit once per batch and a NumPy
percentile threshold computed from the baseline's own self-distances.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


class BaselineAnomalyDetector:
    """
    Fits a k-NN index over baseline (known-normal) template embeddings, then
    scores new templates by distance to their nearest baseline neighbor.
    A template is anomalous if that distance exceeds a threshold derived
    from the baseline's own internal spread (no new logs are "normal" by
    construction, so the threshold is set from how far baseline templates
    already sit from each other).
    """
    def __init__(self, percentile: float = 95.0, k: int = 1):
        self.percentile = percentile
        self.k = k
        self._nn = None
        self._threshold = None
        self._fitted_on = 0


    def fit(self, baseline_vectors: np.ndarray) -> "BaselineAnomalyDetector":
        """
        baseline_vectors: (n_baseline, dim) array of normalized embeddings
        pulled from the 'baseline' Qdrant collection.
        """
        n = baseline_vectors.shape[0]
        self._fitted_on = n

        if n < 2:
            # Not enough baseline data to establish a spread -- treat
            # everything as anomalous until a real baseline is uploaded.
            self._nn = None
            self._threshold = 0.0

            return self

        self._nn = NearestNeighbors(n_neighbors=min(self.k + 1, n), metric="euclidean")
        self._nn.fit(baseline_vectors)

        # Self-distance of each baseline point to its nearest *other*
        # baseline point defines "how spread out normal already is".
        self_distances, _ = self._nn.kneighbors(baseline_vectors)
        nearest_other = self_distances[:, 1] if self_distances.shape[1] > 1 else self_distances[:, 0]
        self._threshold = float(np.percentile(nearest_other, self.percentile))

        return self


    def score(self, vectors: np.ndarray) -> np.ndarray:
        """Returns distance-to-nearest-baseline-template for each input vector."""
        if self._nn is None:
            return np.full(vectors.shape[0], np.inf)

        distances, _ = self._nn.kneighbors(vectors, n_neighbors=1)

        return distances[:, 0]


    def is_anomalous(self, vectors: np.ndarray) -> np.ndarray:
        """Boolean mask, same length as vectors."""
        return self.score(vectors) > self._threshold


    @property
    def threshold(self) -> float:
        return self._threshold if self._threshold is not None else float("nan")


def fetch_baseline_vectors(vector_store) -> np.ndarray:
    """
    Pulls all stored vectors out of a QdrantVectorStore's 'baseline'
    collection via the underlying qdrant-client, for fitting a detector.
    Returns an (n, dim) float32 array, empty if the collection has no
    points yet (e.g. before the first /upload/baseline call).
    """
    points, _ = vector_store.client.scroll(
        collection_name=vector_store.collection_name,
        with_vectors=True,
        limit=10_000,
    )

    if not points:
        return np.empty((0, vector_store.dim), dtype="float32")

    return np.array([p.vector for p in points], dtype="float32")


def filter_anomalous_logs(parsed_logs: list[dict], vector_store, detector: BaselineAnomalyDetector | None = None):
    """
    Scores each parsed log's mined template against the baseline and
    returns (anomalous_logs, scored_all) where scored_all is every input
    log annotated with 'anomaly_score' and 'is_anomalous' for reporting.

    Falls back to forwarding everything (no filtering) if no baseline has
    been uploaded yet, so /upload/logs never silently drops data before a
    baseline exists.
    """
    if detector is None:
        baseline_vectors = fetch_baseline_vectors(vector_store)
        detector = BaselineAnomalyDetector().fit(baseline_vectors)

    if detector._fitted_on == 0:
        for log in parsed_logs:
            log["anomaly_score"] = None
            log["is_anomalous"] = True  # no baseline yet -- treat all as worth analyzing

        return parsed_logs, parsed_logs

    templates = [log.get("template_string") or log.get("message", "") for log in parsed_logs]
    vectors = vector_store.embed(templates)
    scores = detector.score(np.array(vectors, dtype="float32"))
    flags = scores > detector.threshold

    anomalous = []

    for log, score, flag in zip(parsed_logs, scores, flags):
        log["anomaly_score"] = float(score)
        log["is_anomalous"] = bool(flag)

        if flag:
            anomalous.append(log)

    return anomalous, parsed_logs