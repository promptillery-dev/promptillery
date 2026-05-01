import numpy as np
import pytest

from promptillery.cotrain.actions import CoTrainAction, enumerate_cotrain_actions
from promptillery.cotrain.controller_linearbai import LinearBAIController


def test_controller_explores_distinct_arms_in_first_d_rounds():
    actions = enumerate_cotrain_actions(operators=["coverage", "boundary"], volumes=[8], taus=[0.5, 0.7])
    ctrl = LinearBAIController(actions=actions, feature_dim=8, seed=0)
    chosen = []
    for t in range(4):
        chosen.append(ctrl.select_arm())
        ctrl.update(arm=chosen[-1], reward=0.1 * t)
    assert len(set(chosen)) >= 2


def test_recommend_returns_argmax_under_estimated_theta():
    actions = enumerate_cotrain_actions(operators=["coverage", "boundary"], volumes=[8], taus=[0.5])
    ctrl = LinearBAIController(actions=actions, feature_dim=8, seed=0)
    for _ in range(20):
        arm = ctrl.select_arm()
        is_boundary = arm.operator == "boundary"
        ctrl.update(arm=arm, reward=1.0 if is_boundary else 0.0)
    rec = ctrl.recommend()
    assert rec.operator == "boundary"


def test_controller_can_recommend_stop_when_added():
    actions = enumerate_cotrain_actions(operators=["coverage"], volumes=[8], taus=[0.5], include_stop=True)
    ctrl = LinearBAIController(actions=actions, feature_dim=8, seed=0)
    arm = ctrl.select_arm()
    assert arm in actions
