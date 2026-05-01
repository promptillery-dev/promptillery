"""Asymmetric variant flow allocation (design §3.9)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlowAllocation:
    a_to_b: int
    b_to_a: int


def allocate_volumes(
    *, total_volume: int, error_ratio: float, low: float, high: float
) -> FlowAllocation:
    if total_volume < 0:
        raise ValueError("total_volume must be non-negative")
    if error_ratio > high:
        b_to_a = (2 * total_volume + 2) // 3
        a_to_b = total_volume - b_to_a
    elif error_ratio < low:
        a_to_b = (2 * total_volume + 2) // 3
        b_to_a = total_volume - a_to_b
    else:
        a_to_b = total_volume // 2
        b_to_a = total_volume - a_to_b
    return FlowAllocation(a_to_b=a_to_b, b_to_a=b_to_a)
