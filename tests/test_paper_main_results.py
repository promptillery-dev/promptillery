"""Tests for the paper main-results table (per-cycle cost column)."""

import csv
import statistics

import pytest

from promptillery.analyze import (
    PAPER_MAIN_RESULT_FIELDS,
    _write_rows_csv,
    summarize_paper_main_results,
)


def _run_row(seed, estimated_cost):
    """A minimal per-run summary row for one (dataset, model, policy) cell."""
    return {
        "dataset": "banking77",
        "dataset_subset": "",
        "student_model": "roberta-base",
        "student_type": "transformers",
        "metric": "macro_f1",
        "mode": "max",
        "token_budget": 100000,
        "policy_name": "uncertainty",
        "control_name": "",
        "seed": seed,
        "estimated_cost": estimated_cost,
    }


def test_main_results_surface_mean_estimated_cost():
    # Teacher $ spend is otherwise only in the appendix budget audit; it belongs
    # on the main table too. Two seeds in one cell spent $2 and $4 -> mean $3.
    rows = [_run_row(seed=1, estimated_cost=2.0), _run_row(seed=2, estimated_cost=4.0)]

    results = summarize_paper_main_results(rows)

    assert len(results) == 1
    assert results[0]["mean_estimated_cost"] == 3.0


def test_mean_estimated_cost_written_to_paper_main_results_csv(tmp_path):
    # The CSV writer's fieldnames are PAPER_MAIN_RESULT_FIELDS, so the cost only
    # becomes a *visible* column if it's declared there (DictWriter would also
    # raise on the extra key otherwise). Read it back to prove it's emitted.
    rows = [_run_row(seed=1, estimated_cost=2.0), _run_row(seed=2, estimated_cost=4.0)]
    main_rows = summarize_paper_main_results(rows)

    out = tmp_path / "paper_main_results.csv"
    _write_rows_csv(main_rows, out, PAPER_MAIN_RESULT_FIELDS)

    with out.open() as f:
        written = list(csv.DictReader(f))
    assert "mean_estimated_cost" in written[0]
    assert written[0]["mean_estimated_cost"] == "3.0"


def test_expected_cycles_arms_stay_distinct_rows():
    # G3 runs three cycle arms of the same cell; the paper key omitted
    # expected_cycles, so they collapsed into one averaged row that looked like
    # three seeds. The recommender's budget axis needs them kept apart.
    rows = [
        {**_run_row(seed=1, estimated_cost=1.0), "expected_cycles": 1,
         "final_metric": 0.80, "heldout_metric": 0.80},
        {**_run_row(seed=1, estimated_cost=5.0), "expected_cycles": 10,
         "final_metric": 0.93, "heldout_metric": 0.93},
    ]

    results = summarize_paper_main_results(rows)

    assert len(results) == 2
    by_cycles = {r["expected_cycles"]: r for r in results}
    assert by_cycles[1]["mean_estimated_cost"] == 1.0
    assert by_cycles[10]["mean_estimated_cost"] == 5.0


def test_std_estimated_cost_is_emitted_for_the_noise_guard():
    # The crossover noise guard compares intercept spread to the
    # across-seed cost wobble, so the table must surface the cost std.
    rows = [_run_row(seed=1, estimated_cost=2.0), _run_row(seed=2, estimated_cost=4.0)]

    results = summarize_paper_main_results(rows)

    assert results[0]["std_estimated_cost"] == pytest.approx(statistics.stdev([2.0, 4.0]))
