"""End-to-end test for `promptillery recommend` and the config loaders.

Also validates that the committed prices.yaml / targets.yaml parse and drive the
pipeline (guarding against a typo in the paper-visible config).
"""

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from promptillery.cli import app
from promptillery.recommender_io import load_prices, load_targets

REPO_ROOT = Path(__file__).resolve().parents[1]


def _profile(model):
    return {
        "model": model,
        "student_type": "transformers",
        "hardware": {"device": "cuda:0", "gpu_name": "NVIDIA GeForce RTX 4090"},
        "student": {"p95_latency_ms": 5.0, "throughput_calls_per_sec": 200.0},
    }


def _row(student_model, student_type, final, heldout, cost, throughput_note=""):
    return {
        "dataset": "agnews", "dataset_subset": "",
        "student_model": student_model, "student_type": student_type,
        "metric": "accuracy", "mode": "max", "token_budget": "20000000",
        "expected_cycles": "1", "policy_name": "uncertainty", "control_name": "",
        "mean_final_metric": final, "mean_heldout_metric": heldout,
        "mean_estimated_cost": cost, "std_estimated_cost": "0.1",
        "training_seconds": "3600",
    }


def test_committed_prices_and_targets_parse():
    prices = load_prices(REPO_ROOT / "prices.yaml")
    config = load_targets(REPO_ROOT / "targets.yaml")

    assert prices.usd_per_hour["NVIDIA GeForce RTX 4090"] == 0.40
    assert config.expect_gpu_name == "NVIDIA GeForce RTX 4090"
    assert None in config.latency_budgets_ms  # null = no latency constraint
    assert config.primary_target.volume == 1_000_000


def test_recommend_cli_writes_the_three_artifacts(tmp_path):
    prof_dir = tmp_path / "profiles"
    prof_dir.mkdir()
    (prof_dir / "p-roberta.json").write_text(json.dumps(_profile("FacebookAI/roberta-base")))
    (prof_dir / "p-slm.json").write_text(json.dumps(_profile("some-org/slm-3b")))

    csv_path = tmp_path / "paper_main_results.csv"
    rows = [
        _row("FacebookAI/roberta-base", "transformers", "0.93", "0.93", "1.0"),
        _row("some-org/slm-3b", "slm", "0.94", "0.90", "5.0"),
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    out = tmp_path / "report"
    result = CliRunner().invoke(
        app,
        [
            "recommend",
            "--main-results", str(csv_path),
            "--profile-dir", str(prof_dir),
            "--prices", str(REPO_ROOT / "prices.yaml"),
            "--targets", str(REPO_ROOT / "targets.yaml"),
            "-o", str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out / "recommender_table.csv").exists()
    assert (out / "regret_curve.csv").exists()
    assert (out / "recommendation.json").exists()

    table = list(csv.DictReader((out / "recommender_table.csv").open()))
    assert {r["selector"] for r in table} == {
        "expert_default", "random", "volume_threshold",
        "recommender_rules", "recommender_budget_search",
    }
    recommendation = json.loads((out / "recommendation.json").read_text())
    assert "agnews" in recommendation

    receipt = recommendation["agnews"]["receipt"]
    assert "training_usd" in receipt and "teacher_labelling_usd" in receipt


def test_committed_prices_and_targets_carry_no_placeholders():
    for name in ("prices.yaml", "targets.yaml"):
        text = (REPO_ROOT / name).read_text()
        assert "TODO" not in text, f"{name} still has a placeholder"
    config = load_targets(REPO_ROOT / "targets.yaml")
    assert config.teacher.usd_per_call == pytest.approx(0.0011)
