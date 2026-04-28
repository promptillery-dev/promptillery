"""Feature extraction for budget-aware distillation policies."""

from __future__ import annotations

import math
from collections import Counter
from numbers import Real
from typing import Any, Dict, Iterable

from .trainers.base import PredictionResult


def _as_float(value: Any) -> float | None:
    """Best-effort numeric conversion for metric dictionaries."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Real):
        return float(value)
    return None


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _prediction_features(
    prefix: str, predictions: PredictionResult, num_classes: int
) -> Dict[str, float]:
    """Summarize prediction uncertainty and errors for a split."""
    total = len(predictions.predicted_labels)
    features: Dict[str, float] = {
        f"{prefix}_available": 1.0 if total else 0.0,
        f"{prefix}_examples": float(total),
        f"{prefix}_confidence_mean": _mean(predictions.confidences),
        f"{prefix}_low_confidence_rate": 0.0,
        f"{prefix}_error_rate": 0.0,
        f"{prefix}_hard_error_rate": 0.0,
        f"{prefix}_entropy_mean": 0.0,
        f"{prefix}_entropy_normalized_mean": 0.0,
        f"{prefix}_max_confusion_rate": 0.0,
    }
    if not total:
        return features

    errors = [
        predicted != true
        for predicted, true in zip(
            predictions.predicted_labels, predictions.true_labels
        )
    ]
    features[f"{prefix}_error_rate"] = sum(errors) / total
    features[f"{prefix}_low_confidence_rate"] = (
        sum(confidence < 0.6 for confidence in predictions.confidences) / total
    )
    features[f"{prefix}_hard_error_rate"] = (
        sum(
            is_error and confidence >= 0.8
            for is_error, confidence in zip(errors, predictions.confidences)
        )
        / total
    )

    if predictions.entropies:
        entropy_mean = _mean(predictions.entropies)
        features[f"{prefix}_entropy_mean"] = entropy_mean
        if num_classes > 1:
            features[f"{prefix}_entropy_normalized_mean"] = entropy_mean / math.log(
                num_classes
            )

    confusion_pairs = Counter(
        (true, predicted)
        for predicted, true in zip(
            predictions.predicted_labels, predictions.true_labels
        )
        if predicted != true
    )
    if confusion_pairs:
        features[f"{prefix}_max_confusion_rate"] = (
            confusion_pairs.most_common(1)[0][1] / total
        )

    metadata = predictions.metadata or {}
    for name in (
        "exact_match",
        "invalid_label_rate",
        "macro_f1",
        "macro_f1_full_canonical",
    ):
        value = _as_float(metadata.get(name))
        if value is not None:
            features[f"{prefix}_{name}"] = value

    return features


def build_policy_features(
    *,
    cycle: int,
    cycles: int,
    metrics: Dict[str, Any],
    previous_metrics: Dict[str, Any] | None,
    train_predictions: PredictionResult,
    eval_predictions: PredictionResult,
    train_size: int | None,
    synthetic_count: int | None,
    budget: Dict[str, Any],
    num_classes: int,
) -> Dict[str, float]:
    """Build a compact numeric state vector for policy logs and controllers."""
    previous_metrics = previous_metrics or {}
    features: Dict[str, float] = {
        "cycle_index": float(cycle),
        "cycle_fraction": float(cycle / max(cycles - 1, 1)),
        "train_size": float(train_size or 0),
        "synthetic_count": float(synthetic_count or 0),
        "synthetic_ratio": 0.0,
        "budget_total_tokens": float(budget.get("total_tokens") or 0),
        "token_budget": float(budget.get("token_budget") or 0),
        "tokens_remaining": float(budget.get("tokens_remaining") or 0),
        "token_budget_remaining_frac": 0.0,
        "usd_spent": float(budget.get("spent_usd") or 0),
        "usd_remaining": float(budget.get("remaining_usd") or 0),
        "usd_budget_remaining_frac": 0.0,
    }

    if train_size:
        features["synthetic_ratio"] = float((synthetic_count or 0) / train_size)

    token_budget = _as_float(budget.get("token_budget"))
    tokens_remaining = _as_float(budget.get("tokens_remaining"))
    if token_budget and tokens_remaining is not None:
        features["token_budget_remaining_frac"] = max(
            0.0, min(1.0, tokens_remaining / token_budget)
        )

    usd_budget = _as_float(budget.get("budget_limit_usd"))
    usd_remaining = _as_float(budget.get("remaining_usd"))
    if usd_budget and usd_remaining is not None:
        features["usd_budget_remaining_frac"] = max(
            0.0, min(1.0, usd_remaining / usd_budget)
        )

    for metric_name, value in metrics.items():
        numeric_value = _as_float(value)
        if numeric_value is None:
            continue
        features[f"metric_{metric_name}"] = numeric_value
        previous_value = _as_float(previous_metrics.get(metric_name))
        features[f"metric_delta_{metric_name}"] = (
            0.0 if previous_value is None else numeric_value - previous_value
        )

    features.update(_prediction_features("train", train_predictions, num_classes))
    features.update(_prediction_features("eval", eval_predictions, num_classes))

    return features
