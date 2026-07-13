"""Target-conditioned deployment recommender.

Given a deployment target ``(accuracy_floor, latency_budget, volume)``, return
the cheapest deployable student plus a cost receipt and a break-even volume.

Cost model (``V`` = total inference calls over the deployment lifetime)::

    total_cost(cell, V) = distillation_usd + serving_cost_per_call(cell) * V
    teacher_cost(V)     =                    teacher.usd_per_call        * V
    break_even_volume   = distillation_usd / (teacher_rate - serving_rate)
                        = None  when the gap is <= 0 (distilling never pays off)

Selectors *choose* on ``selection_accuracy`` (what a practitioner knows at
decision time); everyone is *graded* by :func:`realized_cost` on
``heldout_accuracy`` (the truth). A cell that clears the floor on selection and
misses it on held-out silently forces a teacher fallback for every call -- the
overfitting-to-selection failure mode the regret curve measures.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

from promptillery.pareto import Cell, HardwarePrices, serving_cost_per_call


@dataclass(frozen=True)
class Teacher:
    """The teacher-API fallback: dollars per call if no student is deployed."""

    usd_per_call: float


@dataclass(frozen=True)
class Target:
    """A deployment target the recommender is conditioned on."""

    accuracy_floor: float
    latency_budget_ms: Optional[float]
    volume: int


@dataclass(frozen=True)
class CostReceipt:
    """The dollar breakdown for a recommendation at the target volume."""

    distillation_usd: float
    serving_usd: float
    total_usd: float
    teacher_total_usd: float
    savings_ratio: Optional[float]  # 1 - total/teacher_total; None if teacher is free
    break_even_volume: Optional[float]


@dataclass(frozen=True)
class Selection:
    """A recommendation: the chosen cell (or None => teacher fallback)."""

    cell: Optional[Cell]
    receipt: CostReceipt
    cells_evaluated: tuple[Cell, ...]  # the prefix a selector had to train
    reason: str  # cheapest_feasible | no_feasible_cell


def total_cost(cell: Cell, prices: HardwarePrices, volume: int) -> float:
    """Upfront distillation plus per-call serving over the whole volume."""
    return cell.distillation_usd + serving_cost_per_call(cell, prices) * volume


def teacher_cost(teacher: Teacher, volume: int) -> float:
    """Cost of answering every call with the teacher API."""
    return teacher.usd_per_call * volume


def break_even_volume(
    cell: Cell, prices: HardwarePrices, teacher: Teacher
) -> Optional[float]:
    """Volume at which distilling this cell overtakes always calling the teacher."""
    gap = teacher.usd_per_call - serving_cost_per_call(cell, prices)
    if gap <= 0:
        return None
    return cell.distillation_usd / gap


def _feasible_latency(cell: Cell, target: Target) -> bool:
    return (
        target.latency_budget_ms is None
        or cell.p95_latency_ms <= target.latency_budget_ms
    )


def base_budget_arm(cells: Iterable[Cell]) -> list[Cell]:
    """Keep only each student's lowest-``expected_cycles`` arm (the fixed default).

    ``recommender_rules`` holds the teacher budget fixed here; only
    ``recommender_budget_search`` is allowed to escalate cycles to buy accuracy.
    """
    by_student: dict[str, list[Cell]] = defaultdict(list)
    for cell in cells:
        by_student[cell.student_model].append(cell)
    kept: list[Cell] = []
    for group in by_student.values():
        floor_cycles = min(c.expected_cycles for c in group)
        kept.extend(c for c in group if c.expected_cycles == floor_cycles)
    return kept


def _receipt(
    cell: Optional[Cell], target: Target, prices: HardwarePrices, teacher: Teacher
) -> CostReceipt:
    teacher_total = teacher_cost(teacher, target.volume)
    if cell is None:
        return CostReceipt(
            distillation_usd=0.0,
            serving_usd=0.0,
            total_usd=teacher_total,
            teacher_total_usd=teacher_total,
            savings_ratio=None,
            break_even_volume=None,
        )
    serving = serving_cost_per_call(cell, prices) * target.volume
    total = cell.distillation_usd + serving
    savings = (1.0 - total / teacher_total) if teacher_total > 0 else None
    return CostReceipt(
        distillation_usd=cell.distillation_usd,
        serving_usd=serving,
        total_usd=total,
        teacher_total_usd=teacher_total,
        savings_ratio=savings,
        break_even_volume=break_even_volume(cell, prices, teacher),
    )


def make_selection(
    cell: Optional[Cell],
    cells_evaluated: Iterable[Cell],
    target: Target,
    prices: HardwarePrices,
    teacher: Teacher,
    reason: str,
) -> Selection:
    """Build a :class:`Selection` with a reconciled receipt (used by selectors)."""
    return Selection(
        cell=cell,
        receipt=_receipt(cell, target, prices, teacher),
        cells_evaluated=tuple(cells_evaluated),
        reason=reason,
    )


def recommend(
    cells: Iterable[Cell],
    target: Target,
    prices: HardwarePrices,
    teacher: Teacher,
    *,
    search_budget: bool,
) -> Selection:
    """Cheapest cell clearing the accuracy floor at the target volume.

    With ``search_budget=False`` the candidate set is first collapsed to each
    student's default (lowest-``expected_cycles``) budget arm; with ``True`` the
    full student x approach x budget cross-product is searched. The cheapest-first
    walk itself is identical either way.
    """
    cells = list(cells)
    if not search_budget:
        cells = base_budget_arm(cells)
    # 1. Latency prune -- free, no training needed (latency is profiled).
    survivors = [c for c in cells if _feasible_latency(c, target)]
    # 2. Cost order at this volume.
    survivors.sort(key=lambda c: total_cost(c, prices, target.volume))
    # 3. Walk; stop at the first cell clearing the floor on selection accuracy.
    evaluated: list[Cell] = []
    for cell in survivors:
        evaluated.append(cell)
        if cell.selection_accuracy >= target.accuracy_floor:
            return Selection(
                cell=cell,
                receipt=_receipt(cell, target, prices, teacher),
                cells_evaluated=tuple(evaluated),
                reason="cheapest_feasible",
            )
    # 4. Nothing clears the floor -> teacher fallback.
    return Selection(
        cell=None,
        receipt=_receipt(None, target, prices, teacher),
        cells_evaluated=tuple(evaluated),
        reason="no_feasible_cell",
    )


@dataclass(frozen=True)
class CrossoverReport:
    """Whether the recommended student changes as volume grows.

    A crossover located entirely by an intercept difference smaller than the
    across-seed cost wobble (``distillation_usd_std``) is not a finding: it would
    move under a different seed. Such a crossover is *suppressed* and reported as
    volume-independent.
    """

    volume_independent: bool
    crossover_suppressed: bool
    pick: Optional[Cell]  # the constant pick when volume-independent, else None
    picks_by_volume: tuple[tuple[int, Optional[Cell]], ...]
    reason: str


def detect_crossover(
    cells: Iterable[Cell],
    target: Target,
    prices: HardwarePrices,
    teacher: Teacher,
    volumes: Iterable[int],
    *,
    search_budget: bool = True,
) -> CrossoverReport:
    """Sweep volume and report whether the pick moves -- guarding on seed noise.

    ``target``'s own ``volume`` is ignored; each value in ``volumes`` is swept.
    """
    cells = list(cells)
    picks_by_volume = tuple(
        (
            v,
            recommend(
                cells,
                Target(target.accuracy_floor, target.latency_budget_ms, v),
                prices,
                teacher,
                search_budget=search_budget,
            ).cell,
        )
        for v in volumes
    )
    distinct = {(c.student_model, c.token_budget) if c else None for _, c in picks_by_volume}

    if len(distinct) <= 1:
        pick = picks_by_volume[0][1] if picks_by_volume else None
        return CrossoverReport(
            volume_independent=True,
            crossover_suppressed=False,
            pick=pick,
            picks_by_volume=picks_by_volume,
            reason="single_pick_across_volumes",
        )

    # Picks move -> is the intercept spread that drives it real, or seed noise?
    picked_cells = [c for _, c in picks_by_volume if c is not None]
    intercepts = [c.distillation_usd for c in picked_cells]
    spread = max(intercepts) - min(intercepts)
    noise = max((c.distillation_usd_std for c in picked_cells), default=0.0)

    if spread <= noise:
        return CrossoverReport(
            volume_independent=True,
            crossover_suppressed=True,
            pick=None,
            picks_by_volume=picks_by_volume,
            reason="intercept_spread_within_seed_variance",
        )

    return CrossoverReport(
        volume_independent=False,
        crossover_suppressed=False,
        pick=None,
        picks_by_volume=picks_by_volume,
        reason="crossover",
    )


def realized_cost(
    selection: Selection, target: Target, prices: HardwarePrices, teacher: Teacher
) -> float:
    """Honest cost after deploying, charged on *held-out* accuracy.

    Distillation is sunk. A cell that misses the floor on held-out forces a
    teacher fallback for every call; a teacher-fallback selection just pays the
    teacher with nothing sunk.
    """
    cell = selection.cell
    if cell is None:
        return teacher_cost(teacher, target.volume)
    if cell.heldout_accuracy < target.accuracy_floor:
        return cell.distillation_usd + teacher_cost(teacher, target.volume)
    return cell.distillation_usd + serving_cost_per_call(cell, prices) * target.volume
