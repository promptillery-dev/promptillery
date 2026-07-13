"""Tests for recommender_io.load_cells -- frozen artifacts -> list[Cell].

Joins ``paper_main_results.csv`` rows to their student ``profile.json`` on the
model stamp (both are the HF id, ``config.get("student")``), asserts the profile
was measured on the expected GPU, and hard-fails on a cell missing held-out
accuracy (which would leave the oracle undefined).
"""

import csv
import json

import pytest

from promptillery.profiler import ProfileStampError
from promptillery.recommender_io import load_cells

_PROFILE = {
    "model": "FacebookAI/roberta-base",
    "student_type": "transformers",
    "hardware": {"device": "cuda:0", "gpu_name": "NVIDIA GeForce RTX 4090"},
    "student": {"p95_latency_ms": 4.06, "throughput_calls_per_sec": 246.0},
}

_ROW = {
    "dataset": "banking77", "dataset_subset": "",
    "student_model": "FacebookAI/roberta-base", "student_type": "transformers",
    "metric": "macro_f1", "mode": "max", "token_budget": "20000000",
    "expected_cycles": "10", "policy_name": "uncertainty", "control_name": "",
    "mean_final_metric": "0.9325", "mean_heldout_metric": "0.9301",
    "mean_estimated_cost": "1.5", "std_estimated_cost": "0.1",
}


def _write(tmp_path, profile=_PROFILE, row=_ROW):
    prof_dir = tmp_path / "profiles"
    prof_dir.mkdir(exist_ok=True)
    (prof_dir / "profile-roberta.json").write_text(json.dumps(profile))
    csv_path = tmp_path / "paper_main_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    return csv_path, prof_dir


def test_load_cells_joins_csv_row_to_its_profile(tmp_path):
    csv_path, prof_dir = _write(tmp_path)

    cells = load_cells(csv_path, prof_dir, expect_gpu_name="NVIDIA GeForce RTX 4090")

    assert len(cells) == 1
    c = cells[0]
    assert c.student_model == "FacebookAI/roberta-base"
    assert c.selection_accuracy == 0.9325       # mean_final_metric
    assert c.heldout_accuracy == 0.9301         # mean_heldout_metric
    assert c.distillation_usd == 1.5
    assert c.distillation_usd_std == 0.1
    assert c.p95_latency_ms == 4.06             # from the profile
    assert c.throughput_calls_per_sec == 246.0
    assert c.gpu_name == "NVIDIA GeForce RTX 4090"
    assert c.expected_cycles == 10


def test_load_cells_hard_fails_on_missing_heldout(tmp_path):
    row = {**_ROW, "mean_heldout_metric": ""}
    csv_path, prof_dir = _write(tmp_path, row=row)

    with pytest.raises(ValueError, match="heldout"):
        load_cells(csv_path, prof_dir, expect_gpu_name="NVIDIA GeForce RTX 4090")


def test_load_cells_asserts_the_gpu_stamp(tmp_path):
    wrong = {**_PROFILE, "hardware": {"device": "cuda:0", "gpu_name": "NVIDIA A100-SXM4-80GB"}}
    csv_path, prof_dir = _write(tmp_path, profile=wrong)

    with pytest.raises(ProfileStampError):
        load_cells(csv_path, prof_dir, expect_gpu_name="NVIDIA GeForce RTX 4090")


def test_load_cells_accepts_cpu_fasttext_under_a_gpu_expectation(tmp_path):
    # FastText is pinned to CPU (gpu_name null); the GPU-stamp guard is for the
    # CUDA students and must not reject a legitimate CPU profile.
    cpu_profile = {
        **_PROFILE, "model": "fasttext", "student_type": "fasttext",
        "hardware": {"device": "cpu", "gpu_name": None},
    }
    row = {**_ROW, "student_model": "fasttext", "student_type": "fasttext"}
    csv_path, prof_dir = _write(tmp_path, profile=cpu_profile, row=row)

    cells = load_cells(csv_path, prof_dir, expect_gpu_name="NVIDIA GeForce RTX 4090")

    assert cells[0].gpu_name is None
    assert cells[0].device == "cpu"
