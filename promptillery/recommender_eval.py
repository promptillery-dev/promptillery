"""Evaluation metrics for the recommender comparison.

Scored over a grid of deployment targets, per dataset:

    agreement_with_gt   = |{targets : selector.cell == oracle.cell}| / |targets|
                          (both None -- "call the teacher" -- counts as agreement)

    pct_exhaustive_cost = distillation $ of the cells the selector ever trained
                          -------------------------------------------------------
                          distillation $ of training all cells (exhaustive search)

    regret(selector, V) = realized_cost(selector.cell, V) - realized_cost(oracle.cell, V)
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from promptillery.pareto import Cell, HardwarePrices
from promptillery.recommender import (
    Selection,
    Target,
    Teacher,
    realized_cost,
)


def _pick_key(cell: Optional[Cell]):
    """Identity of a chosen cell for agreement (None == teacher fallback)."""
    if cell is None:
        return None
    return (cell.student_model, cell.token_budget, cell.expected_cycles)


def agreement_with_gt(
    selector_selections: Sequence[Selection], oracle_selections: Sequence[Selection]
) -> float:
    """Fraction of targets where the selector's pick equals the oracle's."""
    if len(selector_selections) != len(oracle_selections):
        raise ValueError("selector and oracle selections must align by target")
    if not oracle_selections:
        return 0.0
    agree = sum(
        _pick_key(s.cell) == _pick_key(o.cell)
        for s, o in zip(selector_selections, oracle_selections)
    )
    return agree / len(oracle_selections)


def pct_exhaustive_cost(
    selections: Iterable[Selection], all_cells: Iterable[Cell]
) -> float:
    """Distillation $ of cells the selector trained / distillation $ of all cells.

    Training is one-time and reusable across the target grid, so the numerator is
    the distillation cost of the *union* of cells the selector had to evaluate.
    """
    trained: set[Cell] = set()
    for sel in selections:
        trained.update(sel.cells_evaluated)
    denominator = sum(c.distillation_usd for c in all_cells)
    if denominator <= 0:
        return 0.0
    return sum(c.distillation_usd for c in trained) / denominator


def regret(
    selector_selection: Selection,
    oracle_selection: Selection,
    target: Target,
    prices: HardwarePrices,
    teacher: Teacher,
) -> float:
    """Dollars the selector's pick costs above the oracle's, at this target."""
    return realized_cost(selector_selection, target, prices, teacher) - realized_cost(
        oracle_selection, target, prices, teacher
    )
