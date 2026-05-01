"""Cycle-level features for the co-training controller (design §3.8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class StudentEvalSummary:
    error_rate: float
    errors_aligned: List[bool] = field(default_factory=list)


def compute_error_correlation_rho(
    a_errors: List[bool], b_errors: List[bool]
) -> Optional[float]:
    if len(a_errors) != len(b_errors) or not a_errors:
        return None
    a = [1.0 if e else 0.0 for e in a_errors]
    b = [1.0 if e else 0.0 for e in b_errors]
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    var_a = sum((x - mean_a) ** 2 for x in a) / n
    var_b = sum((x - mean_b) ** 2 for x in b) / n
    if var_a == 0.0 or var_b == 0.0:
        return None
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b)) / n
    return cov / ((var_a ** 0.5) * (var_b ** 0.5))


def build_cotrain_features(
    *,
    cycle: int,
    cycles: int,
    student_a: StudentEvalSummary,
    student_b: StudentEvalSummary,
    prev_validation_metric: Optional[float],
    current_validation_metric: Optional[float],
    accepted_counts: Mapping[str, int],
    budget: Mapping[str, Any],
    predicted_cost_next_cycle: float,
) -> Dict[str, float]:
    feats: Dict[str, float] = {
        "cycle_index": float(cycle),
        "cycle_fraction": float(cycle / max(cycles - 1, 1)),
        "student_a_error_rate_t": float(student_a.error_rate),
        "student_b_error_rate_t": float(student_b.error_rate),
    }
    rho = compute_error_correlation_rho(
        student_a.errors_aligned, student_b.errors_aligned
    )
    feats["error_correlation_rho_t"] = float(rho) if rho is not None else 0.0
    feats["error_correlation_rho_t_defined"] = 1.0 if rho is not None else 0.0

    if student_b.error_rate > 0:
        feats["error_ratio"] = student_a.error_rate / max(student_b.error_rate, 1e-9)
    else:
        feats["error_ratio"] = 1.0

    total_accepted = sum(accepted_counts.values()) or 1
    peer = accepted_counts.get("peer_consensus", 0)
    arb_match = accepted_counts.get("strong_teacher_arbitration_match", 0)
    relabel = accepted_counts.get("strong_teacher_relabel", 0)
    reject = accepted_counts.get("ill_posed_reject", 0)
    feats["agreement_rate_t"] = peer / total_accepted
    feats["disagreement_rate_t"] = (arb_match + relabel + reject) / total_accepted
    feats["peer_label_count_t"] = float(peer)
    feats["arbitration_match_count_t"] = float(arb_match)
    feats["relabel_count_t"] = float(relabel)
    feats["reject_count_t"] = float(reject)

    if (
        prev_validation_metric is not None
        and current_validation_metric is not None
    ):
        feats["validation_metric_delta"] = float(
            current_validation_metric - prev_validation_metric
        )
    else:
        feats["validation_metric_delta"] = 0.0

    token_budget = float(budget.get("token_budget") or 0)
    tokens_remaining = float(budget.get("tokens_remaining") or 0)
    feats["token_budget_remaining_frac"] = (
        max(0.0, min(1.0, tokens_remaining / token_budget)) if token_budget else 0.0
    )
    feats["predicted_cost_next_cycle"] = float(predicted_cost_next_cycle)
    return feats
