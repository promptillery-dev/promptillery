"""FrugalKD-CoTrain-P fixed linear scorer (design §3.8).

Hand-tuned weights over a compact feature representation. Justified as
"headline" by Theorem 2 (linear-BAI lower bound at T ≈ 20-30).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

from .actions import CoTrainAction


DEFAULT_FRUGALKD_COTRAIN_P_WEIGHTS: Dict[str, float] = {
    "bias": 0.0,
    "operator_coverage": 0.10,
    "operator_boundary": 0.20,
    "operator_repair": 0.25,
    "volume_frac": 0.10,
    "tau_frac": -0.05,
    "student_a_error_rate_t": 0.20,
    "student_b_error_rate_t": 0.20,
    "agreement_rate_t": -0.10,
    "disagreement_rate_t": 0.15,
    "validation_metric_delta": 0.30,
    "token_budget_remaining_frac": 0.15,
}


@dataclass(frozen=True)
class CoTrainPolicyChoice:
    action: CoTrainAction
    action_scores: Dict[str, float]
    feasible: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> Dict[str, Any]:
        return {
            "action": self.action.model_dump(),
            "action_scores": dict(self.action_scores),
            "feasible_actions": list(self.feasible),
            "metadata": dict(self.metadata),
        }


class FrugalKDCoTrainP:
    """Fixed linear scorer; STOP wins on budget exhaustion or convergence."""

    def __init__(
        self,
        *,
        weights: Mapping[str, float] | None = None,
        stop_accept_rate_threshold: float = 0.95,
        stop_budget_floor: float = 0.05,
        stop_patience_metric_window: int = 3,
    ) -> None:
        self.weights = dict(weights or DEFAULT_FRUGALKD_COTRAIN_P_WEIGHTS)
        self.stop_accept_rate_threshold = stop_accept_rate_threshold
        self.stop_budget_floor = stop_budget_floor
        self.stop_patience_metric_window = stop_patience_metric_window

    def _featurize_action(
        self, action: CoTrainAction, state: Mapping[str, Any]
    ) -> Dict[str, float]:
        feats = {
            "bias": 1.0,
            "operator_coverage": 1.0 if action.operator == "coverage" else 0.0,
            "operator_boundary": 1.0 if action.operator == "boundary" else 0.0,
            "operator_repair": 1.0 if action.operator == "repair" else 0.0,
            "volume_frac": float(action.volume) / 32.0,
            "tau_frac": float(action.tau),
            "student_a_error_rate_t": float(state.get("student_a_error_rate_t", 0.0)),
            "student_b_error_rate_t": float(state.get("student_b_error_rate_t", 0.0)),
            "agreement_rate_t": float(state.get("agreement_rate_t", 0.0)),
            "disagreement_rate_t": float(state.get("disagreement_rate_t", 0.0)),
            "validation_metric_delta": float(state.get("validation_metric_delta", 0.0)),
            "token_budget_remaining_frac": float(
                state.get("token_budget_remaining_frac", 1.0)
            ),
        }
        return feats

    def _score_action(
        self, action: CoTrainAction, state: Mapping[str, Any]
    ) -> float:
        feats = self._featurize_action(action, state)
        return sum(self.weights.get(k, 0.0) * v for k, v in feats.items())

    def _should_force_stop(self, state: Mapping[str, Any]) -> bool:
        if state.get("token_budget_remaining_frac", 1.0) <= self.stop_budget_floor:
            return True
        if state.get("agreement_rate_t", 0.0) >= self.stop_accept_rate_threshold:
            return True
        return False

    def select(
        self, state: Mapping[str, Any], actions: Sequence[CoTrainAction]
    ) -> CoTrainPolicyChoice:
        non_stop = [a for a in actions if not a.is_stop]
        stop = next((a for a in actions if a.is_stop), CoTrainAction(is_stop=True))

        if self._should_force_stop(state) or not non_stop:
            scores = {a.name: 0.0 for a in actions}
            scores["STOP"] = 1.0
            return CoTrainPolicyChoice(
                action=stop, action_scores=scores,
                feasible=[a.name for a in actions],
                metadata={"forced_stop": True},
            )

        scores = {a.name: self._score_action(a, state) for a in actions}
        chosen = max(non_stop, key=lambda a: scores[a.name])
        return CoTrainPolicyChoice(
            action=chosen, action_scores=scores,
            feasible=[a.name for a in actions],
            metadata={},
        )
