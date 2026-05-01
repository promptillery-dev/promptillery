"""Per-student confidence calibration for τ comparison (design §3.7).

We rescale raw model probabilities of the predicted label by fitting a
single temperature T on the held-out validation split via NLL minimization.
The calibrated confidence is comparable across model families because both
students' temperatures are fit against the same labelled split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence


def apply_temperature(
    *,
    logits: Sequence[float],
    temperature: float,
    num_classes: int,
) -> List[float]:
    """Convert per-example top-class logits to softmax probability under T.

    Approximation: we don't know the full logit vector, only the top-class
    raw score. We assume uniform over the remaining `num_classes - 1` classes.
    This matches what the trainers expose (top-1 logit / score).
    """
    probs = []
    for z in logits:
        z_t = z / max(temperature, 1e-6)
        baseline = math.log(max(num_classes - 1, 1))
        denom = math.exp(z_t) + math.exp(baseline - z_t * 0.0)
        probs.append(math.exp(z_t) / denom)
    return probs


def fit_temperature(
    *,
    logits: Sequence[float],
    correctness: Sequence[bool],
    num_classes: int,
    grid: Sequence[float] = tuple(0.5 + 0.1 * i for i in range(46)),
) -> float:
    """Grid-search T ∈ [0.5, 5.0] minimizing NLL on labelled validation."""
    if len(logits) != len(correctness):
        raise ValueError("logits and correctness must align")
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        probs = apply_temperature(
            logits=logits, temperature=T, num_classes=num_classes
        )
        nll = 0.0
        for p, ok in zip(probs, correctness):
            p_label = p if ok else 1.0 - p
            nll -= math.log(max(p_label, 1e-9))
        if nll < best_nll:
            best_T, best_nll = T, nll
    return best_T


@dataclass(frozen=True)
class ConfidenceCalibrator:
    """Maps raw top-class probabilities to calibrated probabilities under T."""

    temperature: float
    num_classes: int

    def calibrate(self, raw_probs: Sequence[float]) -> List[float]:
        T = max(self.temperature, 1e-6)
        out = []
        for p in raw_probs:
            p = min(max(float(p), 1e-9), 1.0 - 1e-9)
            z = math.log(p / (1.0 - p))
            z_t = z / T
            out.append(1.0 / (1.0 + math.exp(-z_t)))
        return out
