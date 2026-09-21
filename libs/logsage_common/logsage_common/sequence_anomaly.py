"""
Sequence-level (session/block-level) anomaly detectors.

The per-line BaselineAnomalyDetector (logsage_common.anomaly) asks "is this
line's *vocabulary* normal?". That cannot see anomalies where every line is
individually normal but the session as a whole is wrong (a missing replica
confirmation, an unexpected retry, events out of order) -- which is how
Loghub's HDFS block labels are defined.

These detectors operate on one *sequence of template IDs per session*
(e.g. per HDFS block) and are trained on normal sessions only:

  CountVectorPCADetector  -- Loglizer-style. Session -> event-count vector,
                             PCA subspace fitted on normal sessions, score =
                             squared prediction error (SPE). Sees count
                             anomalies (missing / extra events, unseen
                             templates); blind to pure reordering.

  DeepLogDetector         -- DeepLog-style (PyTorch LSTM). Learns to predict
                             the next template from the previous `window`
                             templates plus an END token. Session score =
                             worst next-event prediction in the session.
                             mode="surprisal" (default): max -log p(true
                             next event). mode="rank": DeepLog paper rule,
                             max rank of the true event (anomalous if
                             outside top-g). Sees ordering anomalies and
                             truncated sessions. Surprisal is more sensitive
                             when the template vocabulary is small.

Both expose: fit(train_seqs), calibrate(val_seqs, target_fpr),
score(seqs) -> np.ndarray, is_anomalous(seqs) -> np.ndarray[bool].
`calibrate` picks the decision threshold from held-out NORMAL sessions so
no anomaly labels are needed to set it.

torch is imported lazily, so importing this module (and using the PCA
detector) does not require torch.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
from sklearn.decomposition import PCA

Seq = Sequence[int]

_EPS = 1e-6


def _build_vocab(seqs: Iterable[Seq], offset: int) -> dict[int, int]:
    events = sorted({int(e) for s in seqs for e in s})

    return {e: i + offset for i, e in enumerate(events)}


class CountVectorPCADetector:
    def __init__(self, variance: float = 0.95, target_fpr: float = 0.01):
        self.variance = variance
        self.target_fpr = target_fpr
        self._vocab: dict[int, int] = {}
        self._pca: PCA | None = None
        self._threshold: float = float("inf")


    def _counts(self, seqs: Sequence[Seq]) -> np.ndarray:
        unk = len(self._vocab)
        x = np.zeros((len(seqs), unk + 1), dtype="float64")

        for row, s in enumerate(seqs):
            for e in s:
                x[row, self._vocab.get(int(e), unk)] += 1.0

        return x


    def fit(self, train_seqs: Sequence[Seq]) -> CountVectorPCADetector:
        if len(train_seqs) < 2:
            raise ValueError("Need at least 2 training sequences")

        self._vocab = _build_vocab(train_seqs, offset=0)
        x = self._counts(train_seqs)
        self._pca = PCA(n_components=self.variance, svd_solver="full").fit(x)
        self._threshold = max(float(np.quantile(self.score(train_seqs), 1.0 - self.target_fpr)), _EPS)

        return self


    def score(self, seqs: Sequence[Seq]) -> np.ndarray:
        """Squared prediction error: energy outside the normal-behaviour subspace."""
        if self._pca is None:
            raise ValueError("Model must be fitted using fit() before calling score().")

        x = self._counts(seqs)
        recon = self._pca.inverse_transform(self._pca.transform(x))

        return ((x - recon) ** 2).sum(axis=1)


    def calibrate(self, val_normal_seqs: Sequence[Seq], target_fpr: float | None = None) -> float:
        fpr = self.target_fpr if target_fpr is None else target_fpr
        self._threshold = max(float(np.quantile(self.score(val_normal_seqs), 1.0 - fpr)), _EPS)

        return self._threshold


    def is_anomalous(self, seqs: Sequence[Seq]) -> np.ndarray:
        return self.score(seqs) > self._threshold


    @property
    def threshold(self) -> float:
        return self._threshold


    @property
    def n_components(self) -> int:
        if self._pca is None:
            raise AttributeError("PCA model has not been fitted yet.")

        return int(self._pca.n_components_)


# Token layout for DeepLogDetector. Real events start at _OFFSET.
_PAD, _END, _UNK, _OFFSET = 0, 1, 2, 3


class DeepLogDetector:
    def __init__(
        self,
        window: int = 10,
        hidden: int = 64,
        layers: int = 2,
        embed: int = 32,
        top_g: int = 9,
        mode: str = "surprisal",
        steps: int = 3000,
        batch_size: int = 512,
        lr: float = 2e-3,
        seed: int = 42,
        device: str = "cpu",
        verbose: bool = False,
    ):
        self.window, self.hidden, self.layers, self.embed = window, hidden, layers, embed
        if mode not in ("surprisal", "rank"):
            raise ValueError("mode must be 'surprisal' or 'rank'")

        self.top_g, self.mode = top_g, mode
        self._threshold: float = float(top_g - 1) if mode == "rank" else float("inf")
        self.steps, self.batch_size, self.lr = steps, batch_size, lr
        self.seed, self.device, self.verbose = seed, device, verbose
        self._vocab: dict[int, int] = {}
        self._net = None

    # ---- encoding -------------------------------------------------------
    def _encode(self, seq: Seq) -> np.ndarray:
        toks = [self._vocab.get(int(e), _UNK) for e in seq]
        toks.append(_END)

        return np.asarray(toks, dtype="int64")


    def _windows(self, seq: Seq) -> tuple[np.ndarray, np.ndarray]:
        """(contexts[n, window], targets[n]) -- one row per next-event prediction, incl. END."""
        toks = self._encode(seq)
        padded = np.concatenate([np.full(self.window, _PAD, dtype="int64"), toks])
        ctx = np.lib.stride_tricks.sliding_window_view(padded[:-1], self.window)

        return ctx, toks


    def _build_net(self, torch):
        nn = torch.nn
        vocab_size = len(self._vocab) + _OFFSET
        hidden, layers, embed = self.hidden, self.layers, self.embed

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, embed, padding_idx=_PAD)
                self.lstm = nn.LSTM(embed, hidden, layers, batch_first=True)
                self.out = nn.Linear(hidden, vocab_size)


            def forward(self, x):
                h, _ = self.lstm(self.emb(x))

                return self.out(h[:, -1])

        return Net().to(self.device)


    # ---- training -------------------------------------------------------
    def fit(self, train_seqs: Sequence[Seq]) -> DeepLogDetector:
        import torch

        if not train_seqs:
            raise ValueError("Need at least 1 training sequence")

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._vocab = _build_vocab(train_seqs, offset=_OFFSET)

        # Dedupe sequences, then dedupe (context, target) rows, weighting by log(1 + count)
        # so the handful of very common sessions don't dominate training.
        seq_counts = Counter(tuple(int(e) for e in s) for s in train_seqs)
        rows, weights = [], []

        for s, c in seq_counts.items():
            ctx, tgt = self._windows(s)
            rows.append(np.concatenate([ctx, tgt[:, None]], axis=1))
            weights.append(np.full(len(tgt), c, dtype="float64"))

        all_rows = np.concatenate(rows)
        all_w = np.concatenate(weights)
        uniq, inv = np.unique(all_rows, axis=0, return_inverse=True)
        w = np.log1p(np.bincount(inv.ravel(), weights=all_w)).astype("float32")

        x = torch.from_numpy(uniq[:, :-1].astype("int64")).to(self.device)
        y = torch.from_numpy(uniq[:, -1].astype("int64")).to(self.device)
        w_t = torch.from_numpy(w).to(self.device)

        self._net = self._build_net(torch)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        ce = torch.nn.CrossEntropyLoss(reduction="none")
        n = len(y)
        self._net.train()

        # Step-based training: deduped window sets are small, so "epochs" would mean
        # only a handful of optimizer steps. Sample minibatches with replacement instead.
        running = 0.0

        for step in range(self.steps):
            idx = torch.from_numpy(rng.integers(0, n, size=min(self.batch_size, n))).to(self.device)
            loss = (ce(self._net(x[idx]), y[idx]) * w_t[idx]).sum() / w_t[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

            if self.verbose and ((step + 1) % 500 == 0 or step == self.steps - 1):
                print(f"  deeplog step {step + 1}/{self.steps} loss={running / (500 if (step + 1) % 500 == 0 else (step + 1) % 500):.4f} ({n} unique windows)")
                running = 0.0

        self._net.eval()

        if self.mode == "surprisal":
            self.calibrate(train_seqs, target_fpr=0.01)

        return self


    # ---- scoring --------------------------------------------------------
    def score(self, seqs: Sequence[Seq]) -> np.ndarray:
        """Worst next-event prediction in each session (higher = more anomalous).
        surprisal mode: max -log p(true next event). rank mode: max rank of the
        true next event (0 = model's top pick)."""
        import torch

        # 1. Capture the network in local scope and check it immediately.
        # This guarantees it cannot mutate to None and removes the loop check overhead.
        net = self._net

        if net is None:
            raise RuntimeError("The neural network model layer has not been initialized.")

        keys = [tuple(int(e) for e in s) for s in seqs]
        unique = list(dict.fromkeys(keys))
        ctxs, tgts, lens = [], [], []

        for s in unique:
            ctx, tgt = self._windows(s)
            ctxs.append(ctx)
            tgts.append(tgt)
            lens.append(len(tgt))

        x = torch.from_numpy(np.concatenate(ctxs).astype("int64")).to(self.device)
        y = torch.from_numpy(np.concatenate(tgts).astype("int64")).to(self.device)
        per_step = np.empty(len(y), dtype="float64")

        with torch.no_grad():
            for i in range(0, len(y), 8192):
                logits = net(x[i:i + 8192])
                yb = y[i:i + 8192, None]

                if self.mode == "rank":
                    per_step[i:i + 8192] = (logits > logits.gather(1, yb)).sum(dim=1).cpu().numpy()

                else:
                    per_step[i:i + 8192] = -torch.log_softmax(logits, dim=1).gather(1, yb)[:, 0].cpu().numpy()

        offsets = np.concatenate([[0], np.cumsum(lens)[:-1]])
        per_unique = dict(zip(unique, np.maximum.reduceat(per_step, offsets)))

        return np.array([per_unique[k] for k in keys], dtype="float64")


    def calibrate(self, val_normal_seqs: Sequence[Seq], target_fpr: float = 0.01) -> float:
        """Set the threshold so at most `target_fpr` of held-out NORMAL sessions are flagged."""
        scores = self.score(val_normal_seqs)

        if self.mode == "rank":
            g = 1

            while np.mean(scores >= g) > target_fpr:
                g += 1

            self.top_g = g
            self._threshold = float(g - 1)

        else:
            self._threshold = float(np.quantile(scores, 1.0 - target_fpr))

        return self._threshold


    def is_anomalous(self, seqs: Sequence[Seq]) -> np.ndarray:
        return self.score(seqs) > self._threshold


    @property
    def threshold(self) -> float:
        return self._threshold


    # ---- persistence ----------------------------------------------------
    def save(self, path: str) -> None:
        import torch

        # 1. Capture the network in local scope and guard against None
        net = self._net

        if net is None:
            raise RuntimeError("Cannot save model: The neural network model layer has not been initialized.")

        cfg = {k: getattr(self, k) for k in ("window", "hidden", "layers", "embed", "top_g", "mode")}

        # 2. Call .state_dict() on your verified local variable 'net'
        torch.save({"config": json.dumps(cfg),
                    "vocab": json.dumps({str(k): v for k, v in self._vocab.items()}),
                    "threshold": self._threshold,
                    "state": net.state_dict()}, path)


    @classmethod
    def load(cls, path: str, device: str = "cpu") -> DeepLogDetector:
        import torch

        blob = torch.load(path, map_location=device, weights_only=True)
        det = cls(device=device, **json.loads(blob["config"]))
        det._vocab = {int(k): v for k, v in json.loads(blob["vocab"]).items()}
        det._threshold = float(blob["threshold"])
        det._net = det._build_net(torch)
        det._net.load_state_dict(blob["state"])
        det._net.eval()

        return det