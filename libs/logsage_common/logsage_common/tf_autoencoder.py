"""
TensorFlow/Keras autoencoder anomaly detector.

Learns to reconstruct "normal" log-template embeddings from the baseline
collection. Log lines whose embeddings reconstruct poorly (high mean squared
error) are flagged as anomalous. Exposes the same fit/score/is_anomalous/
threshold surface as BaselineAnomalyDetector so the two are interchangeable.
"""
from __future__ import annotations

import os

import numpy as np


def _tf():
    # Imported lazily so services that never use this detector don't pay
    # TensorFlow's import cost or need it installed.
    import tensorflow as tf

    return tf


class AutoencoderAnomalyDetector:
    def __init__(
        self,
        input_dim: int = 384,
        bottleneck: int = 32,
        percentile: float = 95.0,
        epochs: int = 30,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.bottleneck = bottleneck
        self.percentile = percentile
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self._model = None
        self._threshold: float | None = None
        self.history: dict[str, list[float]] = {}


    def _build(self):
        tf = _tf()
        tf.keras.utils.set_random_seed(self.seed)
        inputs = tf.keras.Input(shape=(self.input_dim,))
        x = tf.keras.layers.Dense(128, activation="relu")(inputs)
        x = tf.keras.layers.Dense(self.bottleneck, activation="relu", name="bottleneck")(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        outputs = tf.keras.layers.Dense(self.input_dim, activation="linear")(x)
        model = tf.keras.Model(inputs, outputs, name="logsage_autoencoder")
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

        return model


    def fit(self, baseline_vectors: np.ndarray) -> "AutoencoderAnomalyDetector":
        vectors = np.asarray(baseline_vectors, dtype="float32")

        if vectors.ndim != 2 or vectors.shape[0] < 2:
            raise ValueError("Need at least 2 baseline vectors of shape (n, dim) to fit")

        self.input_dim = vectors.shape[1]
        self._model = self._build()
        fit_history = self._model.fit(
            vectors,
            vectors,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=0,
        )
        self.history = {k: [float(v) for v in vals] for k, vals in fit_history.history.items()}
        self._threshold = float(np.percentile(self.score(vectors), self.percentile))

        return self


    def score(self, vectors: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.full(np.atleast_2d(vectors).shape[0], np.inf)

        vectors = np.atleast_2d(np.asarray(vectors, dtype="float32"))
        reconstructed = self._model.predict(vectors, verbose=0)

        return np.mean(np.square(vectors - reconstructed), axis=1)


    def is_anomalous(self, vectors: np.ndarray) -> np.ndarray:
        return self.score(vectors) > self._threshold


    @property
    def threshold(self) -> float:
        return self._threshold if self._threshold is not None else float("nan")


    def save(self, directory: str) -> None:
        """Writes model.keras and threshold.txt into `directory`."""
        if self._model is None:
            raise RuntimeError("Cannot save an unfitted detector")

        os.makedirs(directory, exist_ok=True)
        self._model.save(os.path.join(directory, "model.keras"))

        with open(os.path.join(directory, "threshold.txt"), "w") as f:
            f.write(repr(self._threshold))


    @classmethod
    def load(cls, directory: str) -> "AutoencoderAnomalyDetector":
        tf = _tf()
        detector = cls()
        detector._model = tf.keras.models.load_model(os.path.join(directory, "model.keras"))
        detector.input_dim = detector._model.input_shape[1]

        with open(os.path.join(directory, "threshold.txt")) as f:
            detector._threshold = float(f.read())

        return detector