"""Uncertainty and open-set flags on top of the frozen-encoder head.

Three signals, each calibrated on validation data and never on test:

- **Margin.** How far the head's logit is from the decision threshold. The
  band that holds `review_rate` of validation windows nearest the threshold
  is marked "low confidence".
- **Ensemble spread.** Heads refitted on bootstrap resamples of whole
  recordings. Their disagreement measures how much the decision depends on
  which recordings happened to be in training. Reported, not used for flags.
- **Familiarity.** Mahalanobis distance of the embedding from the training
  embeddings (PCA, Ledoit-Wolf covariance). Windows farther than
  `familiar_quantile` of validation windows are "unfamiliar": the model was
  never trained on anything like them, so its score is not to be trusted.
  Fitted on drone embeddings only, the same distance is the open-set test:
  a window scored as drone but far from every known drone is an "unknown
  drone".

The head is a scikit-learn pipeline whose last step is a
LogisticRegression, so `sample_weight` is passed as
`logisticregression__sample_weight`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

STATES = ("no drone", "drone", "unknown drone", "low confidence")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


class Familiarity:
    """Distance of an embedding from a reference set of embeddings."""

    def __init__(self, n_components: int = 64):
        self.n_components = n_components
        self.cutoff = np.inf

    def fit(self, E: np.ndarray) -> "Familiarity":
        self.scaler = StandardScaler().fit(E)
        Z = self.scaler.transform(E)
        k = min(self.n_components, Z.shape[0] - 1, Z.shape[1])
        self.pca = PCA(n_components=k, random_state=0).fit(Z)
        self.cov = LedoitWolf().fit(self.pca.transform(Z))
        return self

    def distance(self, E: np.ndarray) -> np.ndarray:
        Z = self.pca.transform(self.scaler.transform(E))
        return np.sqrt(self.cov.mahalanobis(Z))

    def calibrate(self, E_val: np.ndarray, quantile: float = 0.95) -> "Familiarity":
        self.cutoff = float(np.quantile(self.distance(E_val), quantile))
        return self

    def unfamiliar(self, E: np.ndarray) -> np.ndarray:
        return self.distance(E) > self.cutoff


class HeadEnsemble:
    """`n` heads, each fitted on a bootstrap resample of whole recordings,
    drawn separately per class so every member sees both classes."""

    def __init__(self, make_head: Callable, n: int = 20, seed: int = 0):
        self.make_head, self.n, self.seed = make_head, n, seed

    def fit(self, F, y, groups, sample_weight=None) -> "HeadEnsemble":
        y, groups = np.asarray(y), np.asarray(groups)
        rng = np.random.default_rng(self.seed)
        by_class = []
        for c in (0, 1):
            g = np.unique(groups[y == c])
            by_class.append([np.flatnonzero((groups == gi) & (y == c)) for gi in g])
        self.members = []
        for _ in range(self.n):
            idx = np.concatenate([m[i] for m in by_class
                                  for i in rng.integers(0, len(m), len(m))])
            kw = {} if sample_weight is None else \
                {"logisticregression__sample_weight": np.asarray(sample_weight)[idx]}
            self.members.append(self.make_head().fit(F[idx], y[idx], **kw))
        return self

    def __getstate__(self):
        # make_head is often a lambda or a script-level function; the fitted
        # members are all that is needed after fit, and they pickle cleanly.
        return {k: v for k, v in self.__dict__.items() if k != "make_head"}

    def member_logits(self, F) -> np.ndarray:
        return np.stack([logit(m.predict_proba(F)[:, 1]) for m in self.members])


@dataclass
class Detector:
    """Everything needed to turn embeddings into the four proposal states.
    Built by scripts/eval_uncertainty.py and saved with joblib."""
    head: object                       # main head (pipeline)
    threshold: float                   # probability threshold, 1% val FPR
    margin: float                      # half-width of the review band, in logit
    familiar: Familiarity              # vs all training embeddings
    familiar_drone: Familiarity        # vs training drone embeddings
    ensemble: HeadEnsemble | None = None
    info: dict = field(default_factory=dict)

    def scores(self, F: np.ndarray) -> np.ndarray:
        return self.head.predict_proba(F)[:, 1]

    def assess(self, F: np.ndarray) -> dict[str, np.ndarray]:
        p = self.scores(F)
        z = logit(p) - logit(self.threshold)
        out = {
            "p": p,
            "margin": z,
            "distance": self.familiar.distance(F),
            "drone_distance": self.familiar_drone.distance(F),
        }
        out["low_margin"] = np.abs(z) < self.margin
        out["unfamiliar"] = out["distance"] > self.familiar.cutoff
        out["far_from_drones"] = out["drone_distance"] > self.familiar_drone.cutoff
        if self.ensemble is not None:
            out["spread"] = self.ensemble.member_logits(F).std(axis=0)
        out["state"], out["reason"] = decide(p >= self.threshold, out["low_margin"],
                                             out["unfamiliar"], out["far_from_drones"])
        return out


def decide(hit, low_margin, unfamiliar, far_from_drones):
    """Map the flags to the proposal's four states, with a reason.

    Low confidence wins: an input unlike anything in training, or a score
    on the threshold, should go to a human whatever the score says. A
    confident drone call that is far from every known drone is an unknown
    drone rather than a known one."""
    n = len(hit)
    state = np.where(hit, "drone", "no drone").astype(object)
    reason = np.full(n, "", dtype=object)
    unk = hit & far_from_drones
    state[unk], reason[unk] = "unknown drone", "unlike any training drone"
    state[low_margin], reason[low_margin] = "low confidence", "score near threshold"
    state[unfamiliar], reason[unfamiliar] = "low confidence", "input unlike training data"
    return state, reason


def margin_for_review_rate(z_val: np.ndarray, review_rate: float) -> float:
    """Half-width (logit) of the band around the threshold that holds
    `review_rate` of validation windows."""
    return float(np.quantile(np.abs(z_val), review_rate))


def risk_coverage(errors: np.ndarray, uncertainty: np.ndarray,
                  coverages=(1.0, 0.9, 0.8, 0.7, 0.5)) -> dict:
    """Error rate on the most-certain `coverage` share of windows, and the
    area under the full risk-coverage curve (lower is better; equal to the
    plain error rate when uncertainty carries no information)."""
    order = np.argsort(uncertainty, kind="stable")
    e = np.asarray(errors, dtype=float)[order]
    cum = np.cumsum(e) / np.arange(1, len(e) + 1)
    at = {f"{c:.0%}": float(cum[max(int(round(c * len(e))) - 1, 0)]) for c in coverages}
    return {"error_at_coverage": at, "aurc": float(cum.mean())}
