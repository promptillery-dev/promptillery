import csv
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "recommender_sensitivity", REPO_ROOT / "scripts" / "recommender_sensitivity.py")
sens = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sens)


def _profile(model):
    return {"model": model, "student_type": "transformers",
            "hardware": {"device": "cuda:0", "gpu_name": "NVIDIA GeForce RTX 4090"},
            "student": {"p95_latency_ms": 5.0, "throughput_calls_per_sec": 200.0}}


def _row(student, final, heldout, cost):
    return {"dataset": "b77", "dataset_subset": "", "student_model": student,
            "student_type": "transformers", "metric": "accuracy", "mode": "max",
            "token_budget": "1", "expected_cycles": "10", "policy_name": "u",
            "control_name": "", "mean_final_metric": final,
            "mean_heldout_metric": heldout, "mean_estimated_cost": cost,
            "std_estimated_cost": "0", "training_seconds": "600"}


def _fixture(tmp_path):
    prof = tmp_path / "profiles"
    prof.mkdir()
    (prof / "a.json").write_text(json.dumps(_profile("org/a")))
    (prof / "b.json").write_text(json.dumps(_profile("org/b")))
    cells = tmp_path / "cells.csv"
    rows = [_row("org/a", "0.95", "0.95", "1.0"), _row("org/b", "0.96", "0.96", "5.0")]
    with cells.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return cells, prof


def test_sensitivity_rows_cover_three_factors_times_multipliers(tmp_path):
    cells, prof = _fixture(tmp_path)
    rows = sens.sensitivity_rows(cells, prof, REPO_ROOT / "prices.yaml",
                                 REPO_ROOT / "targets.yaml", multipliers=(0.5, 1.0, 2.0))
    assert len(rows) == 9
    assert {r["factor"] for r in rows} == {"gpu_rate", "cpu_rate", "teacher_price"}
    base = [r for r in rows if r["multiplier"] == 1.0]
    assert len({r["pick"] for r in base}) == 1  # multiplier 1 is the same pick 3x
