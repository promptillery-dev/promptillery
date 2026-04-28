"""Budget-aware policy controller scaffolding."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, Iterable, Mapping, Sequence


DEFAULT_PROMPT_OPERATORS = ("coverage", "boundary", "repair")
DEFAULT_TEACHER_TIERS = ("cheap", "strong")
DEFAULT_BATCH_SIZES = (16, 32, 64)

DEFAULT_FRUGALKD_WEIGHTS = {
    "bias": 0.0,
    "operator_coverage": 0.15,
    "operator_boundary": 0.25,
    "operator_repair": 0.20,
    "tier_strong": 0.05,
    "batch_size_frac": 0.05,
    "eval_error_rate": 0.30,
    "eval_entropy_normalized_mean": 0.20,
    "eval_max_confusion_rate": 0.20,
    "synthetic_ratio": -0.10,
    "token_budget_remaining_frac": 0.10,
}


@dataclass(frozen=True)
class PolicyAction:
    """One teacher-acquisition action or explicit zero-cost STOP."""

    prompt_operator: str | None = None
    teacher_tier: str | None = None
    batch_size: int = 0
    is_stop: bool = False

    @property
    def name(self) -> str:
        """Stable action name for logs and predicted-cost dictionaries."""
        if self.is_stop:
            return "STOP"
        return f"{self.prompt_operator}:{self.teacher_tier}:b{self.batch_size}"

    def model_dump(self) -> Dict[str, Any]:
        """Return a JSON-serializable action dictionary."""
        return {
            "name": self.name,
            "prompt_operator": self.prompt_operator,
            "teacher_tier": self.teacher_tier,
            "batch_size": self.batch_size,
            "is_stop": self.is_stop,
        }


@dataclass(frozen=True)
class PolicyChoice:
    """Selected policy action and the scores used to choose it."""

    policy_name: str
    action: PolicyAction
    action_scores: Dict[str, float]
    predicted_cost: Dict[str, Any]
    feasible_actions: Sequence[str]

    def model_dump(self) -> Dict[str, Any]:
        """Return a JSON-serializable policy choice."""
        return {
            "policy_name": self.policy_name,
            "action": self.action.model_dump(),
            "action_scores": self.action_scores,
            "predicted_cost": self.predicted_cost,
            "feasible_actions": list(self.feasible_actions),
        }


def enumerate_actions(
    prompt_operators: Sequence[str] = DEFAULT_PROMPT_OPERATORS,
    teacher_tiers: Sequence[str] = DEFAULT_TEACHER_TIERS,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    include_stop: bool = True,
) -> list[PolicyAction]:
    """Enumerate the finite action class used in FrugalKD experiments."""
    actions = [
        PolicyAction(
            prompt_operator=operator,
            teacher_tier=tier,
            batch_size=batch_size,
        )
        for operator in prompt_operators
        for tier in teacher_tiers
        for batch_size in batch_sizes
    ]
    if include_stop:
        actions.append(PolicyAction(is_stop=True))
    return actions


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Real):
        return float(value)
    return None


def _state_features(state: Mapping[str, Any]) -> Mapping[str, Any]:
    features = state.get("features")
    if isinstance(features, Mapping):
        return features
    return state


def _budget_state(state: Mapping[str, Any]) -> Mapping[str, Any]:
    budget = state.get("budget")
    if isinstance(budget, Mapping):
        return budget
    return state


def _tokens_remaining(state: Mapping[str, Any]) -> float | None:
    return _as_float(_budget_state(state).get("tokens_remaining"))


def _lookup_cost_value(value: Any) -> float | None:
    if isinstance(value, Mapping):
        if value.get("allowed") is False:
            return math.inf
        for key in (
            "total_tokens",
            "teacher_total_tokens",
            "predicted_total_tokens",
            "predicted_teacher_total_tokens",
        ):
            found = _as_float(value.get(key))
            if found is not None:
                return found
        return None
    return _as_float(value)


def action_cost_tokens(
    action: PolicyAction,
    predicted_costs: Mapping[str, Any] | None = None,
    fallback_tokens_per_example: int = 64,
) -> float:
    """Return the predicted teacher-token cost for an action."""
    if action.is_stop:
        return 0.0
    predicted_costs = predicted_costs or {}
    for key in (
        action.name,
        str(action.prompt_operator),
        str(action.teacher_tier),
    ):
        value = _lookup_cost_value(predicted_costs.get(key))
        if value is not None:
            return value

    tier_multiplier = 2.0 if action.teacher_tier == "strong" else 1.0
    return float(action.batch_size * fallback_tokens_per_example * tier_multiplier)


def feasible_actions(
    actions: Iterable[PolicyAction],
    state: Mapping[str, Any],
    predicted_costs: Mapping[str, Any] | None = None,
) -> list[PolicyAction]:
    """Apply the hard token-budget mask, keeping STOP always feasible."""
    remaining = _tokens_remaining(state)
    feasible = []
    for action in actions:
        cost = action_cost_tokens(action, predicted_costs)
        if action.is_stop or remaining is None or cost <= remaining:
            feasible.append(action)
    return feasible


class PolicyController:
    """Small deterministic controller family for reviewer-facing baselines."""

    def __init__(
        self,
        policy_name: str,
        *,
        lambda_cost: float = 1e-4,
        exploration_bonus: float = 0.0,
        seed: int = 0,
        linear_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.policy_name = policy_name
        self.lambda_cost = lambda_cost
        self.exploration_bonus = exploration_bonus
        self.rng = random.Random(seed)
        self.linear_weights = dict(linear_weights or DEFAULT_FRUGALKD_WEIGHTS)

    def select(
        self,
        state: Mapping[str, Any],
        *,
        actions: Sequence[PolicyAction] | None = None,
        predicted_costs: Mapping[str, Any] | None = None,
    ) -> PolicyChoice:
        """Select one action with hard budget masking and explicit STOP."""
        actions = list(actions or enumerate_actions())
        predicted_costs = predicted_costs or {}
        feasible = feasible_actions(actions, state, predicted_costs)
        stop = next(
            (action for action in actions if action.is_stop),
            PolicyAction(is_stop=True),
        )
        non_stop = [action for action in feasible if not action.is_stop]

        if not non_stop:
            return PolicyChoice(
                policy_name=self.policy_name,
                action=stop,
                action_scores={"STOP": 0.0},
                predicted_cost={"total_tokens": 0},
                feasible_actions=[action.name for action in feasible],
            )

        if self.policy_name == "random_feasible":
            action = self.rng.choice(non_stop)
            scores = {candidate.name: 0.0 for candidate in non_stop}
        elif self.policy_name == "fixed_mixed_teacher":
            action, scores = self._select_mixed_teacher(
                non_stop, state, predicted_costs
            )
        elif self.policy_name.startswith("fixed_"):
            action, scores = self._select_fixed(non_stop, state, predicted_costs)
        elif self.policy_name in {"cheap_only", "strong_only"}:
            action, scores = self._select_tier(non_stop, state, predicted_costs)
        elif self.policy_name == "active_uncertainty":
            action, scores = self._select_scored(
                non_stop, state, predicted_costs, self._active_uncertainty_score
            )
        elif self.policy_name == "student_deficiency":
            action, scores = self._select_scored(
                non_stop, state, predicted_costs, self._student_deficiency_score
            )
        elif self.policy_name == "cost_heuristic":
            action, scores = self._select_scored(
                non_stop, state, predicted_costs, self._heuristic_score
            )
        elif self.policy_name in {"frugalkd_p", "linear_frugalkd_p"}:
            action, scores = self._select_scored(
                non_stop, state, predicted_costs, self._linear_score
            )
        elif self.policy_name in {"STOP", "student_only"}:
            action = stop
            scores = {"STOP": 0.0}
        else:
            raise ValueError(f"Unknown policy_name: {self.policy_name}")

        return PolicyChoice(
            policy_name=self.policy_name,
            action=action,
            action_scores=scores,
            predicted_cost={"total_tokens": action_cost_tokens(action, predicted_costs)},
            feasible_actions=[candidate.name for candidate in feasible],
        )

    def _select_fixed(
        self,
        actions: Sequence[PolicyAction],
        state: Mapping[str, Any],
        predicted_costs: Mapping[str, Any],
    ) -> tuple[PolicyAction, Dict[str, float]]:
        operator = self.policy_name.removeprefix("fixed_")
        matching = [action for action in actions if action.prompt_operator == operator]
        if not matching:
            matching = list(actions)
        return self._select_scored(matching, state, predicted_costs, self._low_cost_score)

    def _select_tier(
        self,
        actions: Sequence[PolicyAction],
        state: Mapping[str, Any],
        predicted_costs: Mapping[str, Any],
    ) -> tuple[PolicyAction, Dict[str, float]]:
        tier = self.policy_name.removesuffix("_only")
        matching = [action for action in actions if action.teacher_tier == tier]
        if not matching:
            matching = list(actions)
        return self._select_scored(matching, state, predicted_costs, self._low_cost_score)

    def _select_mixed_teacher(
        self,
        actions: Sequence[PolicyAction],
        state: Mapping[str, Any],
        predicted_costs: Mapping[str, Any],
    ) -> tuple[PolicyAction, Dict[str, float]]:
        cycle = int(state.get("cycle", 0) or 0)
        tier = "cheap" if cycle % 2 == 0 else "strong"
        matching = [action for action in actions if action.teacher_tier == tier]
        if not matching:
            matching = list(actions)
        return self._select_scored(matching, state, predicted_costs, self._low_cost_score)

    def _select_scored(
        self,
        actions: Sequence[PolicyAction],
        state: Mapping[str, Any],
        predicted_costs: Mapping[str, Any],
        scorer,
    ) -> tuple[PolicyAction, Dict[str, float]]:
        scores = {
            action.name: scorer(action, state, predicted_costs)
            for action in actions
        }
        action = max(
            actions,
            key=lambda candidate: (
                scores[candidate.name],
                -action_cost_tokens(candidate, predicted_costs),
                candidate.name,
            ),
        )
        return action, scores

    def _low_cost_score(
        self,
        action: PolicyAction,
        state: Mapping[str, Any],
        predicted_costs: Mapping[str, Any],
    ) -> float:
        return -action_cost_tokens(action, predicted_costs)

    def _heuristic_score(
        self,
        action: PolicyAction,
        state: Mapping[str, Any],
        predicted_costs: Mapping[str, Any],
    ) -> float:
        features = _state_features(state)
        error_rate = float(features.get("eval_error_rate", 0.0) or 0.0)
        entropy = float(features.get("eval_entropy_normalized_mean", 0.0) or 0.0)
        hard_errors = float(features.get("eval_hard_error_rate", 0.0) or 0.0)
        confusion = float(features.get("eval_max_confusion_rate", 0.0) or 0.0)

        operator_gain = {
            "coverage": 0.02 + entropy,
            "boundary": 0.02 + max(hard_errors, confusion),
            "repair": 0.02 + error_rate,
        }.get(str(action.prompt_operator), 0.02)
        tier_gain = 0.05 if action.teacher_tier == "strong" else 0.0
        batch_gain = math.sqrt(max(action.batch_size, 1) / 16)
        cost = action_cost_tokens(action, predicted_costs)
        return (operator_gain + tier_gain) * batch_gain / max(cost, 1.0)

    def _active_uncertainty_score(
        self,
        action: PolicyAction,
        state: Mapping[str, Any],
        predicted_costs: Mapping[str, Any],
    ) -> float:
        features = _state_features(state)
        entropy = float(features.get("eval_entropy_normalized_mean", 0.0) or 0.0)
        hard_errors = float(features.get("eval_hard_error_rate", 0.0) or 0.0)
        confusion = float(features.get("eval_max_confusion_rate", 0.0) or 0.0)
        operator_gain = {
            "coverage": 0.05,
            "boundary": 0.10 + entropy + max(hard_errors, confusion),
            "repair": 0.05 + hard_errors,
        }.get(str(action.prompt_operator), 0.05)
        batch_gain = math.sqrt(max(action.batch_size, 1) / 16)
        cost = action_cost_tokens(action, predicted_costs)
        return operator_gain * batch_gain / max(cost, 1.0)

    def _student_deficiency_score(
        self,
        action: PolicyAction,
        state: Mapping[str, Any],
        predicted_costs: Mapping[str, Any],
    ) -> float:
        features = _state_features(state)
        error_rate = float(features.get("eval_error_rate", 0.0) or 0.0)
        hard_errors = float(features.get("eval_hard_error_rate", 0.0) or 0.0)
        confusion = float(features.get("eval_max_confusion_rate", 0.0) or 0.0)
        operator_gain = {
            "coverage": 0.05 + 0.25 * error_rate,
            "boundary": 0.05 + confusion,
            "repair": 0.10 + error_rate + hard_errors,
        }.get(str(action.prompt_operator), 0.05)
        batch_gain = math.sqrt(max(action.batch_size, 1) / 16)
        cost = action_cost_tokens(action, predicted_costs)
        return operator_gain * batch_gain / max(cost, 1.0)

    def _linear_score(
        self,
        action: PolicyAction,
        state: Mapping[str, Any],
        predicted_costs: Mapping[str, Any],
    ) -> float:
        features = self._action_features(action, state)
        gain = sum(
            self.linear_weights.get(name, 0.0) * value
            for name, value in features.items()
        )
        cost = action_cost_tokens(action, predicted_costs)
        return gain + self.exploration_bonus - self.lambda_cost * cost

    def _action_features(
        self, action: PolicyAction, state: Mapping[str, Any]
    ) -> Dict[str, float]:
        features = _state_features(state)
        values: Dict[str, float] = {
            "bias": 1.0,
            "batch_size_frac": action.batch_size / max(DEFAULT_BATCH_SIZES),
            "tier_strong": 1.0 if action.teacher_tier == "strong" else 0.0,
        }
        for operator in DEFAULT_PROMPT_OPERATORS:
            values[f"operator_{operator}"] = (
                1.0 if action.prompt_operator == operator else 0.0
            )
        for name in (
            "eval_error_rate",
            "eval_entropy_normalized_mean",
            "eval_max_confusion_rate",
            "synthetic_ratio",
            "token_budget_remaining_frac",
        ):
            values[name] = float(features.get(name, 0.0) or 0.0)
        return values
