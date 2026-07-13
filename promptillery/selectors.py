"""Selectors for the recommender comparison table.

Every selector shares ``select(cells, target, prices, teacher) -> Selection``.
Scored across a grid of deployment targets, they fill the comparison table:

- ``oracle``          -- cheapest cell feasible on *held-out* truth; the ground
                         truth every other selector is graded against (trains all).
- ``expert_default``  -- ship the best encoder, ignore volume/cost/floor (trains 1).
- ``random``          -- uniform pick (trains 1).
- ``volume_threshold``-- the issue's requested heuristic: below a volume, call the
                         teacher; above it, deploy the cheapest-to-serve cell,
                         consulting neither floor nor latency (trains 1).
- ``recommender_rules``         -- cheapest-first at the fixed default budget arm.
- ``recommender_budget_search`` -- cheapest-first over the full budget cross-product.
"""

from __future__ import annotations

import random
from typing import Callable, Iterable

from promptillery.pareto import Cell, HardwarePrices, serving_cost_per_call
from promptillery.recommender import (
    Selection,
    Target,
    Teacher,
    make_selection,
    recommend,
    total_cost,
)

Selector = Callable[[Iterable[Cell], Target, HardwarePrices, Teacher], Selection]

# Encoder classifier students (the expert's default class). Decoder SLMs are
# "slm"; FastText is "fasttext".
_ENCODER_TYPES = {"transformers", "modernbert", "encoder"}


def oracle(
    cells: Iterable[Cell], target: Target, prices: HardwarePrices, teacher: Teacher
) -> Selection:
    """Cheapest cell feasible under *held-out* accuracy; trains all cells."""
    cells = list(cells)
    feasible = [
        c
        for c in cells
        if (target.latency_budget_ms is None or c.p95_latency_ms <= target.latency_budget_ms)
        and c.heldout_accuracy >= target.accuracy_floor
    ]
    if not feasible:
        return make_selection(None, cells, target, prices, teacher, "no_feasible_cell")
    best = min(feasible, key=lambda c: total_cost(c, prices, target.volume))
    return make_selection(best, cells, target, prices, teacher, "cheapest_feasible")


def expert_default(
    cells: Iterable[Cell], target: Target, prices: HardwarePrices, teacher: Teacher
) -> Selection:
    """Highest-selection-accuracy encoder; ignores volume, cost, and the floor."""
    encoders = [c for c in cells if c.student_type in _ENCODER_TYPES]
    if not encoders:
        return make_selection(None, [], target, prices, teacher, "no_encoder")
    best = max(encoders, key=lambda c: c.selection_accuracy)
    return make_selection(best, [best], target, prices, teacher, "expert_default")


def random_selector(seed: int) -> Selector:
    """A uniform pick over the candidates, deterministic under ``seed``."""

    def select(cells, target, prices, teacher) -> Selection:
        choice = random.Random(seed).choice(list(cells))
        return make_selection(choice, [choice], target, prices, teacher, "random")

    return select


def volume_threshold(threshold: float) -> Selector:
    """Below ``threshold`` volume call the teacher; above it deploy the
    cheapest-to-serve cell, consulting neither floor nor latency budget."""

    def select(cells, target, prices, teacher) -> Selection:
        if target.volume < threshold:
            return make_selection(None, [], target, prices, teacher, "below_threshold")
        cheapest = min(cells, key=lambda c: serving_cost_per_call(c, prices))
        return make_selection(
            cheapest, [cheapest], target, prices, teacher, "volume_threshold"
        )

    return select


def recommender_rules(
    cells: Iterable[Cell], target: Target, prices: HardwarePrices, teacher: Teacher
) -> Selection:
    """Cheapest-first search at the fixed default (lowest-cycles) budget arm."""
    return recommend(cells, target, prices, teacher, search_budget=False)


def recommender_budget_search(
    cells: Iterable[Cell], target: Target, prices: HardwarePrices, teacher: Teacher
) -> Selection:
    """Cheapest-first over the full student x approach x budget cross-product."""
    return recommend(cells, target, prices, teacher, search_budget=True)


def standard_selectors(
    *, random_seed: int, volume_threshold_value: float
) -> dict[str, Selector]:
    """The ordered selector set scored in the comparison table (oracle excluded)."""
    return {
        "expert_default": expert_default,
        "random": random_selector(random_seed),
        "volume_threshold": volume_threshold(volume_threshold_value),
        "recommender_rules": recommender_rules,
        "recommender_budget_search": recommender_budget_search,
    }
