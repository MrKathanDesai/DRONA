"""Audio + RF fusion of calibrated evidence.

No recording in any corpus has audio and RF of the same flight, so a joint
model cannot be trained. Each detector keeps its own score, and fusion works
on calibrated evidence:

1. **Calibrate.** On each modality's own validation windows, a class-balanced
   logistic regression on the score's logit (Platt scaling) maps a score to
   a log-likelihood ratio (LLR): positive = evidence for "drone", negative =
   evidence against, 0 = none. Balanced weights remove the validation set's
   drone share, so the output is a ratio, not a posterior.
2. **Cap.** Each LLR is clipped to +-`cap`, which limits how far one sensor
   can push the total. It does **not** stop one confident sensor from raising
   an alarm on its own: the fused threshold is fitted on validation pairs
   where both sensors are strongly negative, so it sits below zero, and a
   capped positive score from one sensor crosses it while the other is
   neutral. reports/fusion.md ("RF fooled") shows this with the DroneRF
   forest, which calls pure receiver noise "drone".
3. **Combine.** Either the plain sum (the naive-Bayes rule: sound and radio
   are close to independent given drone / no drone), or the proposal's
   learned fusion layer, a logistic regression on the two LLRs. A missing
   sensor contributes 0.
4. **Decide.** A threshold on the fused value at the target false-alarm
   rate on validation pairs, then the same four states as the audio detector.
   There is one threshold per set of available sensors: with a sensor
   missing, the fused value has a different spread, and the both-sensor
   threshold would no longer hold its false-alarm rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from dronedet.eval import metrics as E
from dronedet.models import uncertainty as U

MODALITIES = ("audio", "rf")


class Calibrator:
    """score in (0, 1) -> capped log-likelihood ratio."""

    def __init__(self, cap: float = 5.0):
        self.cap = cap

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        z = U.logit(p)[:, None]
        self.lr = LogisticRegression(class_weight="balanced", C=1.0).fit(z, y)
        return self

    def llr(self, p: np.ndarray) -> np.ndarray:
        z = U.logit(p)[:, None]
        return np.clip(self.lr.decision_function(z), -self.cap, self.cap)


def pair_llrs(llr: dict[str, np.ndarray | None], n: int) -> np.ndarray:
    """(n, 2) matrix of audio and RF LLRs; a missing modality is all zeros."""
    cols = [np.zeros(n) if llr.get(m) is None else np.asarray(llr[m], dtype=float)
            for m in MODALITIES]
    return np.stack(cols, axis=1)


@dataclass
class Fusion:
    calibrators: dict[str, Calibrator]
    method: str = "sum"                     # "sum" or "learned"
    learned: LogisticRegression | None = None
    # available sensors ("audio+rf", "audio", "rf") -> (threshold, review half-width)
    decision: dict[str, tuple[float, float]] = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def llr(self, scores: dict[str, np.ndarray | None]) -> dict[str, np.ndarray | None]:
        return {m: None if scores.get(m) is None else self.calibrators[m].llr(scores[m])
                for m in MODALITIES}

    def combine(self, L: np.ndarray) -> np.ndarray:
        if self.method == "sum":
            return L.sum(axis=1)
        return self.learned.decision_function(L)

    def fused(self, scores: dict[str, np.ndarray | None], n: int) -> np.ndarray:
        return self.combine(pair_llrs(self.llr(scores), n))

    def fit_decision(self, val_scores: dict[str, np.ndarray], y: np.ndarray,
                     target_fpr: float, review_rate: float) -> "Fusion":
        """Threshold and review band for every set of available sensors,
        from the same validation pairs with the other sensor removed."""
        n = len(y)
        for key in ("audio+rf", "audio", "rf"):
            sc = {m: (val_scores[m] if m in key.split("+") else None) for m in MODALITIES}
            f = self.fused(sc, n)
            thr = E.threshold_at_fpr(f[y == 0], target_fpr)
            self.decision[key] = (thr, float(np.quantile(np.abs(f - thr), review_rate)))
        return self

    @staticmethod
    def available(scores: dict) -> str:
        return "+".join(m for m in MODALITIES if scores.get(m) is not None)

    def threshold(self, scores: dict) -> float:
        return self.decision[self.available(scores)][0]

    def assess(self, scores: dict[str, np.ndarray | None], n: int,
               flags: dict[str, dict] | None = None) -> dict[str, np.ndarray]:
        """Fused value, decision and four-state output for n paired windows.

        `flags[m]` may carry the per-window `low_margin`, `unfamiliar` and
        `far_from_drones` arrays from that modality's own detector. A drone
        call becomes "unknown drone" when a sensor arguing *for* a drone says
        its input is unlike every known drone; "low confidence" when the fused
        value sits in the review band, or every available sensor is itself
        unsure."""
        L = pair_llrs(self.llr(scores), n)
        f = self.combine(L)
        thr, margin = self.decision[self.available(scores)]
        hit = f >= thr
        low = np.abs(f - thr) < margin
        far = np.zeros(n, bool)
        unsure_all = np.ones(n, bool)
        any_flags = False
        for i, m in enumerate(MODALITIES):
            fl = (flags or {}).get(m)
            if scores.get(m) is None or fl is None:
                continue
            any_flags = True
            far |= fl["far_from_drones"] & (L[:, i] > 0)
            unsure_all &= fl["low_margin"] | fl["unfamiliar"]
        if any_flags:
            low |= unsure_all
        state, reason = U.decide(hit, low, np.zeros(n, bool), far)
        return {"fused": f, "llr_audio": L[:, 0], "llr_rf": L[:, 1],
                "threshold": thr, "hit": hit, "state": state, "reason": reason}


def fit_learned(L_val: np.ndarray, y_val: np.ndarray) -> LogisticRegression:
    """The proposal's learned fusion layer: logistic regression on the two
    calibrated LLRs, fitted on validation pairs."""
    return LogisticRegression(class_weight="balanced", C=1.0).fit(L_val, y_val)
