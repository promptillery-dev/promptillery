"""Tests for the recommender report orchestration (fills tab:recommender + figure).

Runs the oracle and every selector across a target grid, per dataset, and shapes
the rows the CLI writes: one recommender-table row per (dataset, selector), and
one regret-curve row per (dataset, selector, volume).
"""

from promptillery.pareto import Cell, HardwarePrices
from promptillery.recommender import Target, Teacher
from promptillery.recommender_report import (
    iter_targets,
    recommender_table_rows,
    regret_curve_rows,
    single_recommendation,
)

PRICES = HardwarePrices({"NVIDIA GeForce RTX 4090": 0.40, "cpu": 0.05})
TEACHER = Teacher(usd_per_call=0.002)
GRID = dict(
    accuracy_floors=[0.85, 0.90],
    latency_budgets_ms=[None, 50.0],
    volumes=[1000, 1_000_000],
    random_seed=0,
    volume_threshold_value=1000,
)


def _cell(**overrides):
    base = dict(
        dataset="agnews", dataset_subset="", student_model="student",
        student_type="transformers", policy_name="uncertainty", control_name="",
        token_budget=20_000_000, selection_accuracy=0.93, heldout_accuracy=0.92,
        distillation_usd=1.0, p95_latency_ms=5.0, throughput_calls_per_sec=200.0,
        device="cuda:0", gpu_name="NVIDIA GeForce RTX 4090", expected_cycles=1,
    )
    base.update(overrides)
    return Cell(**base)


def _two_dataset_trio():
    cells = []
    for ds in ("agnews", "imdb"):
        cells += [
            _cell(dataset=ds, student_model="fasttext", student_type="fasttext",
                  device="cpu", gpu_name=None, selection_accuracy=0.82,
                  heldout_accuracy=0.81, distillation_usd=0.3,
                  throughput_calls_per_sec=100_000.0),
            _cell(dataset=ds, student_model="roberta", selection_accuracy=0.93,
                  heldout_accuracy=0.93, distillation_usd=1.0,
                  throughput_calls_per_sec=246.0, p95_latency_ms=4.0),
            _cell(dataset=ds, student_model="slm", student_type="slm",
                  selection_accuracy=0.94, heldout_accuracy=0.90, distillation_usd=5.0,
                  throughput_calls_per_sec=10.0, p95_latency_ms=95.0),
        ]
    return cells


def test_iter_targets_is_the_full_cross_product():
    targets = iter_targets([0.85, 0.90], [None, 50.0], [1000, 1_000_000])
    assert len(targets) == 2 * 2 * 2
    assert all(isinstance(t, Target) for t in targets)


def test_recommender_table_has_one_row_per_dataset_and_selector():
    rows = recommender_table_rows(_two_dataset_trio(), PRICES, TEACHER, **GRID)

    datasets = {r["dataset"] for r in rows}
    selectors = {r["selector"] for r in rows}
    assert datasets == {"agnews", "imdb"}
    assert "oracle" not in selectors
    assert len(rows) == len(datasets) * len(selectors)
    for r in rows:
        assert 0.0 <= r["agreement_with_gt"] <= 1.0
        assert 0.0 <= r["pct_exhaustive_cost"] <= 1.0


def test_recommender_agrees_with_oracle_at_least_as_often_as_random():
    rows = recommender_table_rows(_two_dataset_trio(), PRICES, TEACHER, **GRID)
    by = {(r["dataset"], r["selector"]): r["agreement_with_gt"] for r in rows}

    for ds in ("agnews", "imdb"):
        assert by[(ds, "recommender_budget_search")] >= by[(ds, "random")]


def test_regret_curve_has_one_row_per_dataset_selector_volume():
    rows = regret_curve_rows(_two_dataset_trio(), PRICES, TEACHER, **GRID)

    assert len(rows) == 2 * 5 * len(GRID["volumes"])  # 2 datasets, 5 selectors
    assert all(r["regret_usd"] >= 0.0 for r in rows)   # no selector beats the oracle


def test_single_recommendation_reports_a_receipt_per_dataset():
    primary = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=1_000_000)
    rec = single_recommendation(_two_dataset_trio(), PRICES, TEACHER, primary)

    assert set(rec) == {"agnews", "imdb"}
    assert rec["agnews"]["receipt"]["total_usd"] >= 0.0
