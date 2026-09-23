"""Orchestrate the recommender comparison over a target grid (issue #6, M4).

Shapes the rows the CLI writes:

- ``recommender_table_rows`` -- one row per (dataset, selector): agreement with
  the held-out oracle and the fraction of exhaustive-search cost it paid.
- ``regret_curve_rows``      -- one row per (dataset, selector, volume): mean
  dollars of regret across the accuracy/latency sub-grid at that volume.
- ``single_recommendation``  -- the budget-search receipt for one primary target,
  per dataset.

Everything is *simulated* against frozen fixtures: all accuracies are
precomputed; we only count which cells a selector would have had to train.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional, Union

from promptillery.analyze import (
    RECOMMENDER_FIELDS,
    REGRET_CURVE_FIELDS,
    _write_rows_csv,
)
from promptillery.pareto import Cell, HardwarePrices
from promptillery.recommender import Target, Teacher
from promptillery.recommender_eval import (
    agreement_with_gt,
    pct_exhaustive_cost,
    regret,
)
from promptillery.selectors import (
    oracle,
    recommender_budget_search,
    standard_selectors,
)


def iter_targets(
    accuracy_floors: Iterable[float],
    latency_budgets_ms: Iterable[Optional[float]],
    volumes: Iterable[int],
) -> list[Target]:
    """The full deployment-target cross product."""
    return [
        Target(float(floor), lat, int(volume))
        for floor in accuracy_floors
        for lat in latency_budgets_ms
        for volume in volumes
    ]


def _by_dataset(cells: Iterable[Cell]) -> dict[str, list[Cell]]:
    grouped: dict[str, list[Cell]] = defaultdict(list)
    for cell in cells:
        grouped[cell.dataset].append(cell)
    return grouped


def recommender_table_rows(
    cells: Iterable[Cell],
    prices: HardwarePrices,
    teacher: Teacher,
    *,
    accuracy_floors,
    latency_budgets_ms,
    volumes,
    random_seed: int,
    volume_threshold_value: float,
) -> list[dict]:
    """One row per (dataset, selector): agreement + %-exhaustive-cost."""
    rows: list[dict] = []
    for dataset, dcells in sorted(_by_dataset(cells).items()):
        targets = iter_targets(accuracy_floors, latency_budgets_ms, volumes)
        oracle_sels = [oracle(dcells, t, prices, teacher) for t in targets]
        selectors = standard_selectors(
            random_seed=random_seed, volume_threshold_value=volume_threshold_value
        )
        for name, select in selectors.items():
            sels = [select(dcells, t, prices, teacher) for t in targets]
            rows.append(
                {
                    "dataset": dataset,
                    "selector": name,
                    "agreement_with_gt": agreement_with_gt(sels, oracle_sels),
                    "pct_exhaustive_cost": pct_exhaustive_cost(sels, dcells),
                    "n_targets": len(targets),
                }
            )
    return rows


def regret_curve_rows(
    cells: Iterable[Cell],
    prices: HardwarePrices,
    teacher: Teacher,
    *,
    accuracy_floors,
    latency_budgets_ms,
    volumes,
    random_seed: int,
    volume_threshold_value: float,
) -> list[dict]:
    """One row per (dataset, selector, volume): mean regret over floor x latency."""
    rows: list[dict] = []
    for dataset, dcells in sorted(_by_dataset(cells).items()):
        selectors = standard_selectors(
            random_seed=random_seed, volume_threshold_value=volume_threshold_value
        )
        for name, select in selectors.items():
            for volume in volumes:
                regrets = []
                for floor in accuracy_floors:
                    for lat in latency_budgets_ms:
                        target = Target(float(floor), lat, int(volume))
                        ora = oracle(dcells, target, prices, teacher)
                        sel = select(dcells, target, prices, teacher)
                        regrets.append(regret(sel, ora, target, prices, teacher))
                rows.append(
                    {
                        "dataset": dataset,
                        "selector": name,
                        "volume": int(volume),
                        "regret_usd": statistics.fmean(regrets),
                    }
                )
    return rows


def single_recommendation(
    cells: Iterable[Cell],
    prices: HardwarePrices,
    teacher: Teacher,
    primary_target: Target,
) -> dict:
    """The budget-search recommendation + receipt for one target, per dataset."""
    result: dict = {}
    for dataset, dcells in sorted(_by_dataset(cells).items()):
        sel = recommender_budget_search(dcells, primary_target, prices, teacher)
        result[dataset] = {
            "cell": sel.cell.student_model if sel.cell else None,
            "reason": sel.reason,
            "receipt": asdict(sel.receipt),
        }
    return result


def write_recommender_report(
    cells: Iterable[Cell],
    prices: HardwarePrices,
    config,
    output_dir: Union[str, Path],
) -> dict[str, Path]:
    """Write the three recommender artifacts and return their paths.

    ``config`` is a ``recommender_io.RecommenderConfig``: the pre-registered grid.
    """
    cells = list(cells)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = dict(
        accuracy_floors=config.accuracy_floors,
        latency_budgets_ms=config.latency_budgets_ms,
        volumes=config.volumes,
        random_seed=config.random_seed,
        volume_threshold_value=config.volume_threshold,
    )

    table = recommender_table_rows(cells, prices, config.teacher, **grid)
    regret = regret_curve_rows(cells, prices, config.teacher, **grid)
    recommendation = single_recommendation(
        cells, prices, config.teacher, config.primary_target
    )

    table_path = output_dir / "recommender_table.csv"
    regret_path = output_dir / "regret_curve.csv"
    rec_path = output_dir / "recommendation.json"
    _write_rows_csv(table, table_path, RECOMMENDER_FIELDS)
    _write_rows_csv(regret, regret_path, REGRET_CURVE_FIELDS)
    rec_path.write_text(json.dumps(recommendation, indent=2))

    return {
        "recommender_table": table_path,
        "regret_curve": regret_path,
        "recommendation": rec_path,
    }
