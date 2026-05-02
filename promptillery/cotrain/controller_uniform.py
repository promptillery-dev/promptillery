"""Uniform-random controller — sanity floor for the Phase 1 controller ablation.

Purpose: answer "does picking actions intelligently beat picking actions blindly?"
For every cycle, pick a non-STOP action uniformly at random. The engine still
respects budget exhaustion via TokenTracker, so this controller only ever
exposes random *non-terminal* choices.

The return type mirrors FrugalKDCoTrainP.select so the engine dispatch stays
uniform.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from .actions import CoTrainAction
from .controller_p import CoTrainPolicyChoice


class UniformRandomController:
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def select(
        self, state: Mapping[str, Any], actions: Sequence[CoTrainAction]
    ) -> CoTrainPolicyChoice:
        non_stop = [a for a in actions if not a.is_stop]
        if not non_stop:
            stop = next((a for a in actions if a.is_stop), None)
            if stop is None:
                raise ValueError("UniformRandomController received no actions")
            scores = {a.name: 0.0 for a in actions}
            scores[stop.name] = 1.0
            return CoTrainPolicyChoice(
                action=stop,
                action_scores=scores,
                feasible=[a.name for a in actions],
                metadata={
                    "controller": "frugalkd_cotrain_uniform",
                    "forced_stop": True,
                },
            )

        chosen = self._rng.choice(non_stop)
        scores = {a.name: 0.0 for a in actions}
        scores[chosen.name] = 1.0
        return CoTrainPolicyChoice(
            action=chosen,
            action_scores=scores,
            feasible=[a.name for a in actions],
            metadata={"controller": "frugalkd_cotrain_uniform"},
        )
