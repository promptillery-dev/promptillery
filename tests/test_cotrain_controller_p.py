import pytest

from promptillery.cotrain.actions import CoTrainAction, enumerate_cotrain_actions
from promptillery.cotrain.controller_p import (
    FrugalKDCoTrainP,
    DEFAULT_FRUGALKD_COTRAIN_P_WEIGHTS,
)


def _state_with(rho=0.0, error_ratio=1.0, agreement=0.5, budget_frac=1.0,
                a_err=0.3, b_err=0.3, validation_delta=0.01):
    return {
        "cycle_index": 2.0, "cycle_fraction": 0.2,
        "student_a_error_rate_t": a_err, "student_b_error_rate_t": b_err,
        "error_correlation_rho_t": rho, "error_correlation_rho_t_defined": 1.0,
        "error_ratio": error_ratio,
        "agreement_rate_t": agreement, "disagreement_rate_t": 1 - agreement,
        "peer_label_count_t": 10.0,
        "arbitration_match_count_t": 2.0, "relabel_count_t": 1.0, "reject_count_t": 0.0,
        "validation_metric_delta": validation_delta,
        "token_budget_remaining_frac": budget_frac,
        "predicted_cost_next_cycle": 50_000.0,
    }


def test_controller_picks_non_stop_when_budget_healthy_and_progressing():
    actions = enumerate_cotrain_actions()
    ctrl = FrugalKDCoTrainP()
    choice = ctrl.select(_state_with(budget_frac=0.8, validation_delta=0.02), actions)
    assert choice.action.is_stop is False
    assert choice.action.name in (a.name for a in actions if not a.is_stop)


def test_controller_prefers_stop_when_agreement_above_stop_threshold():
    actions = enumerate_cotrain_actions()
    ctrl = FrugalKDCoTrainP(stop_accept_rate_threshold=0.95)
    choice = ctrl.select(_state_with(agreement=0.99, budget_frac=0.5), actions)
    assert choice.action.is_stop is True


def test_controller_prefers_stop_when_budget_exhausted():
    actions = enumerate_cotrain_actions()
    ctrl = FrugalKDCoTrainP()
    choice = ctrl.select(_state_with(budget_frac=0.0), actions)
    assert choice.action.is_stop is True


def test_action_scores_recorded_for_all_candidates():
    actions = enumerate_cotrain_actions()
    ctrl = FrugalKDCoTrainP()
    choice = ctrl.select(_state_with(), actions)
    assert set(choice.action_scores) == {a.name for a in actions}
