"""Volume-free deployment frontier for the recommender.

Each row of ``paper_main_results.csv`` -- joined to its student profile -- is one
:class:`Cell`. Dominance runs over four axes that do **not** depend on volume:
accuracy up; p95 latency, distillation spend, and serving cost-per-call down.
Because ``total_cost(V) = distillation + serving_per_call * V`` with both cost
coefficients no worse, a dominated cell loses at *every* volume and is feasible
whenever its dominator is -- so the frontier is computed once and reused across
the whole volume sweep.

Serving cost is priced from a **per-device** table (:class:`HardwarePrices`):
``profiler.py`` pins fasttext students to CPU and everything else to CUDA, so a
scalar GPU rate would bill FastText at GPU rates and destroy the cheap end of the
frontier. The table keys on the profile's ``gpu_name`` stamp, falling back to the
``device``, and hard-fails on a miss rather than silently mispricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


class UnknownHardwareError(KeyError):
    """No price-table entry for a cell's stamped ``gpu_name``/``device``.

    Distinct from ``profiler.ProfileStampError`` (a wrong-GPU *profile*): this is
    a missing *price*, and it hard-fails so a mispriced cell never enters the
    recommender table.
    """


@dataclass(frozen=True)
class Cell:
    """One (dataset, student, approach, budget) candidate configuration.

    ``selection_accuracy`` is what a selector knows at decision time (measured on
    the run's selection split); ``heldout_accuracy`` is the truth, learned only
    after deploying, and is what everyone is graded on.
    """

    dataset: str
    dataset_subset: str
    student_model: str
    student_type: str
    policy_name: str
    control_name: str
    token_budget: int
    selection_accuracy: float  # mean_final_metric   -- chosen on
    heldout_accuracy: float  # mean_heldout_metric -- graded on
    distillation_usd: float  # mean_estimated_cost (teacher labelling spend)
    p95_latency_ms: float  # from profile
    throughput_calls_per_sec: float  # from profile
    device: str  # profile hardware stamp
    gpu_name: Optional[str]
    distillation_usd_std: float = 0.0  # std_estimated_cost -- across-seed cost wobble
    expected_cycles: int = 0  # the budget arm; teacher spend scales with it


@dataclass(frozen=True)
class HardwarePrices:
    """Per-device on-demand hourly rates, keyed by ``gpu_name`` or ``device``."""

    usd_per_hour: Mapping[str, float]

    def rate_for(self, cell: Cell) -> float:
        """Hourly rate for the cell's hardware; raise on an unpriced device."""
        for key in (cell.gpu_name, cell.device):
            if key is not None and key in self.usd_per_hour:
                return self.usd_per_hour[key]
        raise UnknownHardwareError(
            f"no price for gpu_name={cell.gpu_name!r} / device={cell.device!r}; "
            f"known devices: {sorted(self.usd_per_hour)}"
        )


def serving_cost_per_call(cell: Cell, prices: HardwarePrices) -> float:
    """Dollars to serve one call: hourly rate / 3600 / throughput."""
    return prices.rate_for(cell) / 3600.0 / cell.throughput_calls_per_sec


def dominates(a: Cell, b: Cell, prices: HardwarePrices, *, accuracy_key: str) -> bool:
    """``a`` dominates ``b``: no worse on all four axes, strictly better on one.

    ``accuracy_key`` selects ``"selection_accuracy"`` (what selectors see) or
    ``"heldout_accuracy"`` (what the oracle sees) -- two frontiers, same code.
    """
    a_acc, b_acc = getattr(a, accuracy_key), getattr(b, accuracy_key)
    a_serv = serving_cost_per_call(a, prices)
    b_serv = serving_cost_per_call(b, prices)

    no_worse = (
        a_acc >= b_acc
        and a.p95_latency_ms <= b.p95_latency_ms
        and a.distillation_usd <= b.distillation_usd
        and a_serv <= b_serv
    )
    strictly_better = (
        a_acc > b_acc
        or a.p95_latency_ms < b.p95_latency_ms
        or a.distillation_usd < b.distillation_usd
        or a_serv < b_serv
    )
    return no_worse and strictly_better


def pareto_frontier(
    cells: list[Cell], prices: HardwarePrices, *, accuracy_key: str
) -> list[Cell]:
    """Cells not strictly dominated by any other, order preserved."""
    return [
        c
        for i, c in enumerate(cells)
        if not any(
            dominates(other, c, prices, accuracy_key=accuracy_key)
            for j, other in enumerate(cells)
            if j != i
        )
    ]
