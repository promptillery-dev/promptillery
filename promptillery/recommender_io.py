"""Load frozen artifacts into recommender :class:`Cell`s.

Reads ``paper_main_results.csv`` and joins each row to its student
``profile.json`` on the model stamp, asserting the profile was measured on the
expected GPU -- the guard ``load_profile`` exists for. Everything runs off
frozen artifacts -- no live training, no teacher calls, no GPU.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml

from promptillery.pareto import Cell, HardwarePrices
from promptillery.profiler import ProfileStampError, load_profile
from promptillery.recommender import Target, Teacher


@dataclass(frozen=True)
class RecommenderConfig:
    """The pre-registered target grid + teacher, loaded from ``targets.yaml``."""

    teacher: Teacher
    expect_gpu_name: Optional[str]
    accuracy_floors: list[float]
    latency_budgets_ms: list[Optional[float]]
    volumes: list[int]
    volume_threshold: float
    random_seed: int
    primary_target: Target


def _opt_float(value) -> Optional[float]:
    return None if value is None else float(value)


def load_prices(path: Union[str, Path]) -> HardwarePrices:
    """Load the per-device serving-rate table from ``prices.yaml``."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    rates = data.get("usd_per_hour") or {}
    return HardwarePrices(usd_per_hour={str(k): float(v) for k, v in rates.items()})


def load_targets(path: Union[str, Path]) -> RecommenderConfig:
    """Load the pre-registered deployment-target grid from ``targets.yaml``."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    primary = data["primary_target"]
    return RecommenderConfig(
        teacher=Teacher(usd_per_call=float(data["teacher"]["usd_per_call"])),
        expect_gpu_name=data.get("expect_gpu_name"),
        accuracy_floors=[float(x) for x in data["accuracy_floors"]],
        latency_budgets_ms=[_opt_float(x) for x in data["latency_budgets_ms"]],
        volumes=[int(x) for x in data["volumes"]],
        volume_threshold=float(data["volume_threshold"]),
        random_seed=int(data.get("random_seed", 0)),
        primary_target=Target(
            float(primary["accuracy_floor"]),
            _opt_float(primary.get("latency_budget_ms")),
            int(primary["volume"]),
        ),
    )


def _index_profiles(profile_dir: Path) -> dict[str, Path]:
    """Map each profile's model stamp to its path (latency is seed-independent)."""
    index: dict[str, Path] = {}
    for path in sorted(profile_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        model = data.get("model")
        if model and "student" in data and "hardware" in data:
            index[model] = path
    return index


def _to_float(value, field: str, student: str) -> float:
    if value in (None, ""):
        raise ValueError(f"cell {student!r} missing {field}")
    return float(value)


def load_cells(
    main_results_csv: Union[str, Path],
    profile_dir: Union[str, Path],
    *,
    expect_gpu_name: Optional[str] = None,
) -> list[Cell]:
    """Join ``paper_main_results.csv`` rows to profiles, returning candidate cells.

    ``expect_gpu_name`` asserts every joined profile's hardware stamp, so
    wrong-GPU latency can never reach the recommender table. A cell missing
    ``mean_heldout_metric`` hard-fails: without held-out truth the oracle is
    undefined and silently dropping the cell would change the frontier.
    """
    profile_dir = Path(profile_dir)
    profiles = _index_profiles(profile_dir)

    cells: list[Cell] = []
    with Path(main_results_csv).open(newline="") as f:
        for row in csv.DictReader(f):
            student = row["student_model"]
            heldout = row.get("mean_heldout_metric")
            if heldout in (None, ""):
                raise ValueError(
                    f"cell {student!r} has no mean_heldout_metric -- the oracle "
                    f"is undefined; the run needs paper_mode/report_held_out_test"
                )
            path = profiles.get(student)
            if path is None:
                raise KeyError(
                    f"no profile stamped model={student!r} under {profile_dir}"
                )
            profile = load_profile(path, expect_model=student)
            hardware = profile.get("hardware") or {}
            gpu_name = hardware.get("gpu_name")
            # The GPU-stamp guard applies only to GPU-served students; FastText is
            # pinned to CPU (gpu_name null) and is legitimately off the 4090.
            if expect_gpu_name and gpu_name is not None and gpu_name != expect_gpu_name:
                raise ProfileStampError(
                    f"profile for {student!r} was measured on gpu_name={gpu_name!r}, "
                    f"expected {expect_gpu_name!r}"
                )
            stats = profile["student"]
            cells.append(
                Cell(
                    dataset=row["dataset"],
                    dataset_subset=row.get("dataset_subset", ""),
                    student_model=student,
                    student_type=row.get("student_type", ""),
                    policy_name=row.get("policy_name", ""),
                    control_name=row.get("control_name", ""),
                    token_budget=int(_to_float(row.get("token_budget") or 0, "token_budget", student)),
                    selection_accuracy=_to_float(row.get("mean_final_metric"), "mean_final_metric", student),
                    heldout_accuracy=float(heldout),
                    distillation_usd=_to_float(row.get("mean_estimated_cost") or 0.0, "mean_estimated_cost", student),
                    distillation_usd_std=float(row.get("std_estimated_cost") or 0.0),
                    p95_latency_ms=float(stats["p95_latency_ms"]),
                    throughput_calls_per_sec=float(stats["throughput_calls_per_sec"]),
                    device=hardware.get("device", ""),
                    gpu_name=hardware.get("gpu_name"),
                    expected_cycles=int(float(row.get("expected_cycles") or 0)),
                )
            )
    return cells
