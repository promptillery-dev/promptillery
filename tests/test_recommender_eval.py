"""Tests for recommender_eval.py -- agreement, %-exhaustive-cost, regret.

These are the columns of ``tab:recommender`` and the y-axis of the regret figure.
The story: the recommender attains near-oracle *agreement* at a small *fraction*
of exhaustive-search cost, and near-zero *regret*.
"""

import pytest

from promptillery.pareto import Cell, HardwarePrices
from promptillery.recommender import Target, Teacher
from promptillery.recommender_eval import agreement_with_gt, pct_exhaustive_cost, regret
from promptillery.selectors import oracle, recommender_budget_search

PRICES = HardwarePrices({"NVIDIA GeForce RTX 4090": 0.40, "cpu": 0.05})
TEACHER = Teacher(usd_per_call=0.002)


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


def test_agreement_counts_both_teacher_fallback_as_agreement():
    cells = [_cell(student_model="roberta", selection_accuracy=0.93, heldout_accuracy=0.93)]
    feasible = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=1000)
    impossible = Target(accuracy_floor=0.999, latency_budget_ms=None, volume=1000)

    sel = [recommender_budget_search(cells, t, PRICES, TEACHER)
           for t in (feasible, impossible)]
    ora = [oracle(cells, t, PRICES, TEACHER) for t in (feasible, impossible)]

    # target 1: both pick roberta; target 2: both fall back to the teacher (None).
    assert sel[1].cell is None and ora[1].cell is None
    assert agreement_with_gt(sel, ora) == 1.0


def test_pct_exhaustive_cost_counts_only_trained_cells():
    below = _cell(student_model="below", selection_accuracy=0.80, distillation_usd=0.2,
                  throughput_calls_per_sec=1000.0)
    pick = _cell(student_model="pick", selection_accuracy=0.92, distillation_usd=1.0)
    dearer = _cell(student_model="dearer", selection_accuracy=0.99, distillation_usd=9.0)
    all_cells = [below, pick, dearer]
    target = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=1000)

    rec = recommender_budget_search(all_cells, target, PRICES, TEACHER)
    ora = oracle(all_cells, target, PRICES, TEACHER)

    # recommender trained only {below, pick}=1.2 of the 10.2 total; oracle trains all.
    assert pct_exhaustive_cost([rec], all_cells) == pytest.approx(1.2 / 10.2)
    assert pct_exhaustive_cost([ora], all_cells) == pytest.approx(1.0)


def test_regret_is_zero_when_the_selector_matches_the_oracle():
    cells = [_cell(student_model="roberta", selection_accuracy=0.93, heldout_accuracy=0.93)]
    target = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=100_000)

    rec = recommender_budget_search(cells, target, PRICES, TEACHER)
    ora = oracle(cells, target, PRICES, TEACHER)

    assert rec.cell == ora.cell
    assert regret(rec, ora, target, PRICES, TEACHER) == pytest.approx(0.0)


def test_regret_is_positive_when_a_pick_secretly_misses_the_floor():
    # `mirage` clears selection but misses held-out -> the selector that ships it
    # eats the teacher fallback; the oracle took the honest cell. Regret > 0.
    mirage = _cell(student_model="mirage", selection_accuracy=0.99, heldout_accuracy=0.80,
                   distillation_usd=0.5, throughput_calls_per_sec=1000.0)
    real = _cell(student_model="real", selection_accuracy=0.92, heldout_accuracy=0.93,
                 distillation_usd=2.0, throughput_calls_per_sec=200.0)
    target = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=100_000)

    rec = recommender_budget_search([mirage, real], target, PRICES, TEACHER)
    ora = oracle([mirage, real], target, PRICES, TEACHER)

    assert rec.cell == mirage and ora.cell == real  # they diverge
    assert regret(rec, ora, target, PRICES, TEACHER) > 0.0
