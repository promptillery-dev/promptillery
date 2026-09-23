"""scripts/build_cells.py turns run directories into recommender cells."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_cells.py"
spec = importlib.util.spec_from_file_location("build_cells", SCRIPT)
build_cells = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_cells)


def _run_dir(tmp_path, *, run_id, student_type, metric, created_at):
    d = tmp_path / run_id
    d.mkdir()
    heldout = {metric: 0.91, "_selected_cycle": 1}
    (d / "metrics.json").write_text(json.dumps({
        "0": {metric: 0.80}, "1": {metric: 0.88}, "heldout_test": heldout,
    }))
    (d / "run_manifest.json").write_text(json.dumps({
        "run_id": run_id, "status": "completed", "student_model": "org/student",
        "student_type": student_type, "expected_cycles": 2, "seed": 13,
        "token_budget": 20000000, "policy_name": "uncertainty", "control_name": None,
        "dataset_subset": None,
        "reproducibility": {"created_at": created_at},
    }))
    (d / "token_usage.json").write_text(json.dumps({
        "grand_total": {"input_tokens": 1_000_000, "output_tokens": 100_000,
                        "total_tokens": 1_100_000, "estimated_cost": None},
    }))
    return d


def test_price_usd_uses_list_rates():
    assert build_cells.price_usd(1_000_000, 100_000, input_usd_per_m=2.0,
                                 output_usd_per_m=8.0) == pytest.approx(2.8)


def test_metric_key_prefers_accuracy_then_exact_match():
    assert build_cells.metric_key({"accuracy": 1, "f1": 1}) == "accuracy"
    assert build_cells.metric_key({"exact_match": 1, "macro_f1": 1}) == "exact_match"
    with pytest.raises(KeyError):
        build_cells.metric_key({"macro_f1": 1})


def test_cell_row_prices_tokens_and_reconstructs_wall_clock(tmp_path):
    # run id stamp 17:00:00 box-local (UTC+2) -> 15:00 UTC; manifest written 15:10 UTC.
    d = _run_dir(tmp_path, run_id="demo_slm_20260709_170000_000000_s13_ab_cd",
                 student_type="slm", metric="exact_match",
                 created_at="2026-07-09T15:10:00+00:00")
    row = build_cells.cell_row(d, dataset="banking77", utc_offset_hours=2.0,
                               input_usd_per_m=2.0, output_usd_per_m=8.0)
    assert row["dataset"] == "banking77"
    assert row["student_model"] == "org/student"
    assert row["metric"] == "exact_match"
    assert row["mean_heldout_metric"] == 0.91
    assert row["mean_final_metric"] == 0.88          # selected cycle 1
    assert row["mean_estimated_cost"] == pytest.approx(2.8)
    assert row["training_seconds"] == pytest.approx(600.0)
    assert row["training_seconds_source"] == "run_wall_clock_upper_bound"
    assert row["control_name"] == ""


def test_cell_row_prefers_manifest_training_seconds(tmp_path):
    d = _run_dir(tmp_path, run_id="demo_transformers_20260709_170000_000000_s13_ab_cd",
                 student_type="transformers", metric="accuracy",
                 created_at="2026-07-09T15:10:00+00:00")
    m = json.loads((d / "run_manifest.json").read_text())
    m["training_seconds"] = 42.5
    (d / "run_manifest.json").write_text(json.dumps(m))
    row = build_cells.cell_row(d, dataset="x", utc_offset_hours=2.0,
                               input_usd_per_m=2.0, output_usd_per_m=8.0)
    assert row["training_seconds"] == 42.5
    assert row["training_seconds_source"] == "manifest"


def test_main_writes_csv_for_completed_runs_only(tmp_path):
    completed = _run_dir(tmp_path, run_id="a_transformers_20260709_170000_000000_s13_ab_cd",
                         student_type="transformers", metric="accuracy",
                         created_at="2026-07-09T15:10:00+00:00")
    (completed / "model").mkdir()
    (completed / "model" / "profile.json").write_text(json.dumps({"latency_ms": 42}))

    skipped = _run_dir(tmp_path, run_id="b_slm_20260709_170000_000000_s13_ab_cd",
                       student_type="slm", metric="exact_match",
                       created_at="2026-07-09T15:10:00+00:00")
    m = json.loads((skipped / "run_manifest.json").read_text())
    m["status"] = "failed"
    (skipped / "run_manifest.json").write_text(json.dumps(m))
    out = tmp_path / "cells.csv"
    build_cells.main([str(tmp_path), "--dataset", "banking77", "--output", str(out)])
    lines = out.read_text().splitlines()
    assert len(lines) == 2  # header + one completed run
    assert "training_seconds" in lines[0]


def test_main_skips_runs_without_latency_profile(tmp_path):
    # Run with profile.json should be included
    with_profile = _run_dir(tmp_path, run_id="a_transformers_20260709_170000_000000_s13_ab_cd",
                           student_type="transformers", metric="accuracy",
                           created_at="2026-07-09T15:10:00+00:00")
    (with_profile / "model").mkdir()
    (with_profile / "model" / "profile.json").write_text(json.dumps({"latency_ms": 42}))

    # Run without profile.json should be skipped
    _run_dir(tmp_path, run_id="b_slm_20260709_170000_000000_s13_ab_cd",
             student_type="slm", metric="exact_match",
             created_at="2026-07-09T15:10:00+00:00")

    out = tmp_path / "cells.csv"
    build_cells.main([str(tmp_path), "--dataset", "banking77", "--output", str(out)])
    lines = out.read_text().splitlines()
    assert len(lines) == 2  # header + one run with profile
