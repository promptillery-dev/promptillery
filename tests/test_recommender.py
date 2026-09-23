"""Tests for recommender.py -- target-conditioned cheapest deployable student.

The recommender prunes on latency (free -- no training), sorts survivors by
total cost at the target volume, and stops at the first cell clearing the
accuracy floor on its *selection* accuracy. Because the survivors are in cost
order, the first feasible cell is the cheapest feasible cell, and the recommender
returns exactly exhaustive search's pick having "trained" only a prefix.
"""

import pytest

from promptillery.pareto import Cell, HardwarePrices, serving_cost_per_call
from promptillery.recommender import (
    Target,
    Teacher,
    break_even_volume,
    detect_crossover,
    realized_cost,
    recommend,
)

PRICES = HardwarePrices({"NVIDIA GeForce RTX 4090": 0.40, "cpu": 0.05})
TEACHER = Teacher(usd_per_call=0.002)  # ~GPT-4.1 few-shot per-call order of magnitude


def _cell(**overrides):
    base = dict(
        dataset="agnews",
        dataset_subset="",
        student_model="student",
        student_type="transformers",
        policy_name="uncertainty",
        control_name="",
        token_budget=20_000_000,
        selection_accuracy=0.93,
        heldout_accuracy=0.92,
        distillation_usd=1.0,
        p95_latency_ms=5.0,
        throughput_calls_per_sec=200.0,
        device="cuda:0",
        gpu_name="NVIDIA GeForce RTX 4090",
    )
    base.update(overrides)
    return Cell(**base)


def test_recommend_returns_cheapest_cell_clearing_the_floor():
    below = _cell(student_model="fasttext", selection_accuracy=0.85,
                  distillation_usd=0.3, throughput_calls_per_sec=1000.0)
    pick = _cell(student_model="roberta", selection_accuracy=0.92, distillation_usd=1.0)
    dearer = _cell(student_model="big", selection_accuracy=0.95, distillation_usd=3.0)
    target = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=1000)

    sel = recommend([dearer, pick, below], target, PRICES, TEACHER, search_budget=True)

    assert sel.cell == pick
    assert sel.reason == "cheapest_feasible"


def test_recommend_crossover_a_different_cell_wins_as_volume_grows():
    # slm: cheap to train, expensive to serve -> wins at low volume.
    # encoder: expensive to train, cheap to serve -> wins at high volume.
    slm = _cell(student_model="slm", selection_accuracy=0.91,
                distillation_usd=1.0, throughput_calls_per_sec=10.0)
    encoder = _cell(student_model="encoder", selection_accuracy=0.93,
                    distillation_usd=5.0, throughput_calls_per_sec=1000.0)
    floor = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=1000)
    high = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=10_000_000)

    low_pick = recommend([slm, encoder], floor, PRICES, TEACHER, search_budget=True)
    high_pick = recommend([slm, encoder], high, PRICES, TEACHER, search_budget=True)

    assert low_pick.cell == slm
    assert high_pick.cell == encoder


def test_latency_budget_excludes_cheapest_cell_for_free():
    # cheap_slow is the cheapest and clears accuracy, but violates the latency
    # budget -- it must be pruned *before* the walk, so it never enters
    # cells_evaluated (no training dollars spent probing it).
    cheap_slow = _cell(student_model="slm", selection_accuracy=0.95,
                       distillation_usd=0.5, p95_latency_ms=95.0,
                       throughput_calls_per_sec=10.0)
    pick = _cell(student_model="roberta", selection_accuracy=0.92,
                 distillation_usd=1.0, p95_latency_ms=5.0)
    target = Target(accuracy_floor=0.90, latency_budget_ms=50.0, volume=1000)

    sel = recommend([cheap_slow, pick], target, PRICES, TEACHER, search_budget=True)

    assert sel.cell == pick
    assert cheap_slow not in sel.cells_evaluated


def test_no_feasible_cell_falls_back_to_the_teacher():
    below1 = _cell(student_model="a", selection_accuracy=0.80)
    below2 = _cell(student_model="b", selection_accuracy=0.88)
    target = Target(accuracy_floor=0.95, latency_budget_ms=None, volume=1000)

    sel = recommend([below1, below2], target, PRICES, TEACHER, search_budget=True)

    assert sel.cell is None
    assert sel.reason == "no_feasible_cell"
    assert sel.receipt.total_usd == pytest.approx(TEACHER.usd_per_call * 1000)


def test_break_even_volume_is_intercept_over_rate_gap():
    cell = _cell(distillation_usd=2.0, throughput_calls_per_sec=200.0)
    serving = serving_cost_per_call(cell, PRICES)
    expected = 2.0 / (TEACHER.usd_per_call - serving)

    assert break_even_volume(cell, PRICES, TEACHER) == pytest.approx(expected)


def test_break_even_volume_is_none_when_distilling_never_pays_off():
    # A student more expensive to serve than the teacher never breaks even.
    dear_to_serve = _cell(throughput_calls_per_sec=0.02)  # huge serving cost/call
    assert serving_cost_per_call(dear_to_serve, PRICES) > TEACHER.usd_per_call

    assert break_even_volume(dear_to_serve, PRICES, TEACHER) is None


def test_receipt_reconciles_distillation_plus_serving_equals_total():
    cell = _cell(distillation_usd=1.5, throughput_calls_per_sec=200.0)
    target = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=100_000)

    sel = recommend([cell], target, PRICES, TEACHER, search_budget=True)
    r = sel.receipt

    assert r.serving_usd == pytest.approx(serving_cost_per_call(cell, PRICES) * 100_000)
    assert r.distillation_usd + r.serving_usd == pytest.approx(r.total_usd)


def test_cheapest_first_stops_early_and_records_only_the_prefix():
    below = _cell(student_model="cheap-below", selection_accuracy=0.80,
                  distillation_usd=0.2, throughput_calls_per_sec=1000.0)
    pick = _cell(student_model="pick", selection_accuracy=0.92, distillation_usd=1.0)
    dearer = _cell(student_model="dearer", selection_accuracy=0.99, distillation_usd=9.0)
    target = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=1000)

    sel = recommend([dearer, pick, below], target, PRICES, TEACHER, search_budget=True)

    assert sel.cell == pick
    assert below in sel.cells_evaluated   # cheaper, had to be probed
    assert dearer not in sel.cells_evaluated  # costlier than the pick, never probed


def test_selection_pass_heldout_fail_is_charged_the_teacher_fallback():
    # Looks feasible on the selection split, silently misses the floor on
    # held-out -> every call falls back to the teacher; distillation is sunk.
    overfit = _cell(selection_accuracy=0.92, heldout_accuracy=0.80,
                    distillation_usd=1.0, throughput_calls_per_sec=200.0)
    target = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=10_000)
    sel = recommend([overfit], target, PRICES, TEACHER, search_budget=True)

    charged = realized_cost(sel, target, PRICES, TEACHER)
    honest_serving = overfit.distillation_usd + serving_cost_per_call(overfit, PRICES) * 10_000

    assert charged == pytest.approx(
        overfit.distillation_usd + TEACHER.usd_per_call * 10_000
    )
    assert charged > honest_serving  # the fallback is strictly worse than it looked


def test_equal_intercept_candidates_yield_a_volume_independent_pick():
    # Same upfront -> argmin over (upfront + rate*V) = argmin rate, for all V.
    # The cheaper-to-serve student wins at every volume; no crossover exists.
    a = _cell(student_model="a", distillation_usd=2.0,
              throughput_calls_per_sec=10.0, selection_accuracy=0.92)
    b = _cell(student_model="b", distillation_usd=2.0,
              throughput_calls_per_sec=1000.0, selection_accuracy=0.93)
    template = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=0)

    # All swept volumes stay above both students' ~1000-call break-even, so
    # the teacher-below-break-even rule (A.2) never intervenes here -- this
    # test is about intercept-tie noise suppression, not the teacher trade-off.
    report = detect_crossover([a, b], template, PRICES, TEACHER,
                              volumes=[10_000, 100_000, 10_000_000])

    assert report.volume_independent
    assert not report.crossover_suppressed  # genuinely equal, not a noise suppression
    assert report.pick == b                 # the cheaper-to-serve student, at all V


def test_crossover_is_suppressed_within_across_seed_variance():
    # The intercept gap (0.05) that locates the crossover is smaller than the
    # seed-to-seed wobble in distillation cost (std 0.5). A crossover that moves
    # with the seed is not a finding -> suppress it, report volume-independent.
    slm = _cell(student_model="slm", distillation_usd=2.00, distillation_usd_std=0.5,
                throughput_calls_per_sec=10.0, selection_accuracy=0.92)
    encoder = _cell(student_model="encoder", distillation_usd=2.05,
                    distillation_usd_std=0.5, throughput_calls_per_sec=1000.0,
                    selection_accuracy=0.93)
    template = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=0)

    report = detect_crossover([slm, encoder], template, PRICES, TEACHER,
                              volumes=[10, 1000, 10_000_000])

    assert report.crossover_suppressed
    assert report.volume_independent


def test_genuine_crossover_is_reported_when_intercepts_differ_beyond_noise():
    # Intercept gap 4.0 dwarfs the seed std 0.1 -> a real crossover, reported.
    slm = _cell(student_model="slm", distillation_usd=1.0, distillation_usd_std=0.1,
                throughput_calls_per_sec=10.0, selection_accuracy=0.91)
    encoder = _cell(student_model="encoder", distillation_usd=5.0,
                    distillation_usd_std=0.1, throughput_calls_per_sec=1000.0,
                    selection_accuracy=0.93)
    template = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=0)

    report = detect_crossover([slm, encoder], template, PRICES, TEACHER,
                              volumes=[1000, 10_000_000])

    assert not report.volume_independent
    assert not report.crossover_suppressed


def test_recommend_returns_teacher_when_cheaper_than_the_feasible_student():
    # $10 sunk, ~$0 serving; at 100 calls the teacher costs $0.20.
    student = _cell(distillation_usd=10.0)
    target = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=100)

    sel = recommend([student], target, PRICES, TEACHER, search_budget=True)

    assert sel.cell is None
    assert sel.reason == "teacher_cheaper_at_volume"
    assert sel.cells_evaluated == (student,)          # it still had to train it
    assert sel.receipt.total_usd == pytest.approx(0.2)


def test_recommend_returns_the_student_once_volume_passes_break_even():
    student = _cell(distillation_usd=10.0)
    target = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=100_000)

    sel = recommend([student], target, PRICES, TEACHER, search_budget=True)

    assert sel.cell == student
    assert sel.reason == "cheapest_feasible"
