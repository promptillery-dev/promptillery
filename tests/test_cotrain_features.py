import math
import pytest

from promptillery.cotrain.features import (
    build_cotrain_features,
    compute_error_correlation_rho,
    StudentEvalSummary,
)


def test_error_correlation_zero_when_disjoint_errors():
    a_errors = [True, False, True, False]
    b_errors = [False, True, False, True]
    rho = compute_error_correlation_rho(a_errors, b_errors)
    assert rho == pytest.approx(-1.0)


def test_error_correlation_one_when_identical():
    e = [True, False, True, False]
    assert compute_error_correlation_rho(e, e) == pytest.approx(1.0)


def test_error_correlation_undefined_when_no_variance():
    assert compute_error_correlation_rho([False] * 5, [False] * 5) is None


def test_build_cotrain_features_assembles_expected_keys():
    a = StudentEvalSummary(error_rate=0.3, errors_aligned=[True, False, True])
    b = StudentEvalSummary(error_rate=0.1, errors_aligned=[True, False, False])
    feats = build_cotrain_features(
        cycle=2, cycles=10,
        student_a=a, student_b=b,
        prev_validation_metric=0.6, current_validation_metric=0.65,
        accepted_counts={"peer_consensus": 12, "strong_teacher_arbitration_match": 3,
                         "strong_teacher_relabel": 2, "ill_posed_reject": 1},
        budget={"token_budget": 1_000_000, "tokens_remaining": 700_000},
        predicted_cost_next_cycle=50_000,
    )
    for key in [
        "cycle_index", "cycle_fraction",
        "student_a_error_rate_t", "student_b_error_rate_t",
        "error_correlation_rho_t",
        "agreement_rate_t", "disagreement_rate_t",
        "peer_label_count_t",
        "validation_metric_delta",
        "predicted_cost_next_cycle", "token_budget_remaining_frac",
        "error_ratio",
    ]:
        assert key in feats
    assert feats["error_ratio"] == pytest.approx(3.0)
    total_accepted = 12 + 3 + 2 + 1
    assert feats["agreement_rate_t"] == pytest.approx(12 / total_accepted)
    assert feats["disagreement_rate_t"] == pytest.approx((3 + 2 + 1) / total_accepted)
    assert feats["peer_label_count_t"] == 12
