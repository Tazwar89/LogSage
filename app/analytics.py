"""
Analytics and a secondary ML-based anomaly detector.

- compute_log_stats(): aggregates parsed log entries with Pandas for the
  /stats endpoint (log level distribution, top templates, counts).
- IsolationForestDetector: an alternative anomaly detector using
  scikit-learn's IsolationForest over embedding vectors, offered alongside
  the FAISS distance-threshold approach for comparison. Two different
  techniques solving the same problem is a stronger portfolio signal than
  one.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np


def compute_log_stats(entries: list[dict]) -> dict:
    """
    entries: list of parsed log dicts (each with at least 'level' and
    optionally 'template_id' / 'template_string' if annotated).
    Returns aggregate stats as plain JSON-serializable types.
    """
    if not entries:
        return {"total_logs": 0, "level_distribution": {}, "top_templates": []}

    df = pd.DataFrame(entries)

    level_distribution = (
        df["level"].value_counts().to_dict() if "level" in df.columns else {}
    )

    top_templates = []

    if "template_string" in df.columns:
        counts = df["template_string"].value_counts().head(10)
        top_templates = [
            {"template": template, "count": int(count)}
            for template, count in counts.items()
        ]

    return {
        "total_logs": len(df),
        "level_distribution": level_distribution,
        "top_templates": top_templates,
    }


class IsolationForestDetector:
    """
    Secondary anomaly detector using scikit-learn's IsolationForest over
    embedding vectors, as an alternative to the FAISS nearest-neighbor
    distance threshold used in anomaly.py.

    Unlike the FAISS approach (distance to nearest known template),
    IsolationForest learns a model of "normal" density across ALL baseline
    vectors at once and flags points that are easy to isolate (i.e. sit in
    sparse regions of the embedding space). Useful for catching anomalies
    that are subtly "off" in aggregate, even if their nearest single
    neighbor isn't that far away.
    """

    def __init__(self, contamination=0.05, random_state=42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self._fitted = False

    def fit(self, baseline_vectors: np.ndarray):
        if len(baseline_vectors) < 2:
            self._fitted = False

            return

        self.model.fit(baseline_vectors)
        self._fitted = True

    def predict(self, vector: np.ndarray) -> bool:
        """Returns True if the vector is flagged as anomalous."""
        if not self._fitted:
            return False  # not enough baseline data to judge -- fail open, not closed

        prediction = self.model.predict(vector.reshape(1, -1))

        return bool(prediction[0] == -1)  # sklearn convention: -1 = anomaly, 1 = normal