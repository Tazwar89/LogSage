"""
Shared baseline anomaly detector (scikit-learn k-NN + NumPy percentile
threshold), used by both ingestion_service (decides what gets published to
Kafka) and analysis_service (re-checks anomaly status before running the
LLM pipeline) against the same baseline Qdrant collection, so both services
agree on what "anomalous" means.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


class BaselineAnomalyDetector:
    def __init__(self, percentile: float = 95.0, k: int = 1):
        self.percentile = percentile
        self.k = k
        self._nn = None
        self._threshold = None
        self._fitted_on = 0


    def fit(self, baseline_vectors: np.ndarray) -> "BaselineAnomalyDetector":
        n = baseline_vectors.shape[0]
        self._fitted_on = n

        if n < 2:
            self._nn = None
            self._threshold = 0.0

            return self

        self._nn = NearestNeighbors(n_neighbors=min(self.k + 1, n), metric="euclidean")
        self._nn.fit(baseline_vectors)

        self_distances, _ = self._nn.kneighbors(baseline_vectors)
        nearest_other = self_distances[:, 1] if self_distances.shape[1] > 1 else self_distances[:, 0]
        self._threshold = float(np.percentile(nearest_other, self.percentile))

        return self


    def score(self, vectors: np.ndarray) -> np.ndarray:
        if self._nn is None:
            return np.full(vectors.shape[0], np.inf)

        distances, _ = self._nn.kneighbors(vectors, n_neighbors=1)

        return distances[:, 0]


    def is_anomalous(self, vectors: np.ndarray) -> np.ndarray:
        return self.score(vectors) > self._threshold


    @property
    def threshold(self) -> float:
        return self._threshold if self._threshold is not None else float("nan")


def fetch_baseline_vectors(vector_store) -> np.ndarray:
    points, _ = vector_store.client.scroll(
        collection_name=vector_store.collection_name,
        with_vectors=True,
        limit=10_000,
    )

    if not points:
        return np.empty((0, vector_store.dim), dtype="float32")

    return np.array([p.vector for p in points], dtype="float32")