"""Linear pure-exploration bandit (Soare et al. 2014, XY-allocation).

We maintain a ridge-regularized least-squares estimate of theta and pull
the arm whose feature most reduces uncertainty in the current best-pair
gap direction. recommend() returns argmax_a phi(a)·theta_hat.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Sequence

from .actions import CoTrainAction


def featurize(action: CoTrainAction, dim: int = 8) -> np.ndarray:
    f = np.zeros(dim, dtype=np.float64)
    if action.is_stop:
        f[0] = 1.0
        return f
    f[1] = 1.0  # bias for non-stop
    f[2] = 1.0 if action.operator == "coverage" else 0.0
    f[3] = 1.0 if action.operator == "boundary" else 0.0
    f[4] = 1.0 if action.operator == "repair" else 0.0
    f[5] = float(action.volume) / 32.0
    f[6] = float(action.tau)
    f[7] = float(action.volume) * float(action.tau) / 32.0
    return f


@dataclass
class _BanditState:
    A: np.ndarray
    b: np.ndarray
    pulls: dict


class LinearBAIController:
    def __init__(
        self,
        *,
        actions: Sequence[CoTrainAction],
        feature_dim: int = 8,
        ridge: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.actions = list(actions)
        self.dim = feature_dim
        self.rng = np.random.default_rng(seed)
        self._features = np.stack(
            [featurize(a, feature_dim) for a in self.actions], axis=0
        )
        self._state = _BanditState(
            A=ridge * np.eye(feature_dim),
            b=np.zeros(feature_dim),
            pulls={a.name: 0 for a in self.actions},
        )

    def _theta_hat(self) -> np.ndarray:
        return np.linalg.solve(self._state.A, self._state.b)

    def select_arm(self) -> CoTrainAction:
        pulls = self._state.pulls
        unpulled = [a for a in self.actions if pulls[a.name] == 0]
        if unpulled:
            return self.rng.choice(np.array(unpulled, dtype=object))

        theta = self._theta_hat()
        scores = self._features @ theta
        order = np.argsort(scores)[::-1]
        x_star = self._features[order[0]]
        x_alt = self._features[order[1]]
        gap = x_star - x_alt
        A_inv = np.linalg.inv(self._state.A)
        scores_explore = []
        for f in self._features:
            denom = float(f.T @ A_inv @ f) + 1e-12
            num = float(gap.T @ A_inv @ f) ** 2 / denom
            scores_explore.append(num)
        return self.actions[int(np.argmax(scores_explore))]

    def update(self, *, arm: CoTrainAction, reward: float) -> None:
        f = featurize(arm, self.dim)
        self._state.A += np.outer(f, f)
        self._state.b += reward * f
        self._state.pulls[arm.name] = self._state.pulls.get(arm.name, 0) + 1

    def recommend(self) -> CoTrainAction:
        theta = self._theta_hat()
        scores = self._features @ theta
        return self.actions[int(np.argmax(scores))]
