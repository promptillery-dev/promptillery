"""Tests for UniformRandomController (controller ablation sanity floor)."""

from promptillery.cotrain.actions import enumerate_cotrain_actions
from promptillery.cotrain.controller_uniform import UniformRandomController


def _actions():
    return enumerate_cotrain_actions(
        operators=["coverage", "boundary", "repair"],
        volumes=[8, 16, 32],
        taus=[0.5, 0.7, 0.9],
        include_stop=True,
    )


def test_returns_action_from_input_list():
    actions = _actions()
    ctrl = UniformRandomController(seed=11)
    choice = ctrl.select(state={}, actions=actions)
    assert choice.action in actions


def test_never_selects_stop():
    actions = _actions()
    ctrl = UniformRandomController(seed=0)
    for _ in range(50):
        choice = ctrl.select(state={}, actions=actions)
        assert choice.action.is_stop is False


def test_same_seed_is_deterministic():
    actions = _actions()
    a = UniformRandomController(seed=42)
    b = UniformRandomController(seed=42)
    seq_a = [a.select(state={}, actions=actions).action.name for _ in range(10)]
    seq_b = [b.select(state={}, actions=actions).action.name for _ in range(10)]
    assert seq_a == seq_b


def test_different_seeds_diverge():
    actions = _actions()
    a = UniformRandomController(seed=1)
    b = UniformRandomController(seed=2)
    seq_a = [a.select(state={}, actions=actions).action.name for _ in range(20)]
    seq_b = [b.select(state={}, actions=actions).action.name for _ in range(20)]
    assert seq_a != seq_b


def test_choice_exposes_action_scores_and_metadata():
    """Mirrors FrugalKDCoTrainP.select return shape so engine code can stay uniform."""
    actions = _actions()
    ctrl = UniformRandomController(seed=7)
    choice = ctrl.select(state={"token_budget_remaining_frac": 0.5}, actions=actions)
    assert hasattr(choice, "action")
    assert hasattr(choice, "action_scores")
    assert hasattr(choice, "feasible")
    assert hasattr(choice, "metadata")
    assert choice.metadata.get("controller") == "frugalkd_cotrain_uniform"
