import pytest

from promptillery.cotrain.actions import (
    CoTrainAction,
    enumerate_cotrain_actions,
)


def test_enumerate_default_yields_28():
    actions = enumerate_cotrain_actions(
        operators=["coverage", "boundary", "repair"],
        volumes=[8, 16, 32],
        taus=[0.5, 0.7, 0.9],
        include_stop=True,
    )
    assert len(actions) == 28
    assert sum(1 for a in actions if a.is_stop) == 1


def test_action_name_is_stable_and_unique():
    actions = enumerate_cotrain_actions(
        operators=["coverage", "boundary"],
        volumes=[8, 16],
        taus=[0.5, 0.9],
        include_stop=False,
    )
    names = [a.name for a in actions]
    assert len(set(names)) == len(names)
    assert "coverage:v8:t0.5" in names


def test_stop_action_has_zero_volume():
    [stop] = [a for a in enumerate_cotrain_actions(["coverage"], [8], [0.5]) if a.is_stop]
    assert stop.volume == 0
    assert stop.name == "STOP"


def test_action_model_dump_round_trip():
    a = CoTrainAction(operator="boundary", volume=16, tau=0.7)
    dumped = a.model_dump()
    assert dumped == {
        "name": "boundary:v16:t0.7",
        "operator": "boundary",
        "volume": 16,
        "tau": 0.7,
        "is_stop": False,
    }
