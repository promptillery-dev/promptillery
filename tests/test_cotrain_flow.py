import pytest

from promptillery.cotrain.flow import allocate_volumes, FlowAllocation


def test_balanced_when_error_ratio_in_band():
    out = allocate_volumes(total_volume=24, error_ratio=1.0, low=0.67, high=1.5)
    assert out == FlowAllocation(a_to_b=12, b_to_a=12)


def test_a_lags_routes_more_to_a():
    out = allocate_volumes(total_volume=24, error_ratio=2.0, low=0.67, high=1.5)
    assert out == FlowAllocation(a_to_b=8, b_to_a=16)


def test_b_lags_routes_more_to_b():
    out = allocate_volumes(total_volume=24, error_ratio=0.5, low=0.67, high=1.5)
    assert out == FlowAllocation(a_to_b=16, b_to_a=8)


def test_handles_odd_total_volume():
    out = allocate_volumes(total_volume=10, error_ratio=2.0, low=0.67, high=1.5)
    assert out.a_to_b + out.b_to_a == 10
    assert out.b_to_a > out.a_to_b
