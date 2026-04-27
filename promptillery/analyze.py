"""Analyze Promptillery run artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


PREFERRED_METRICS = (
    "exact_match",
    "accuracy",
    "macro_f1",
    "f1",
    "perplexity",
    "eval_loss",
)
LOWER_IS_BETTER = {"eval_loss", "loss", "perplexity"}
REQUIRED_RUN_FILES = (
    "experiment_config.yaml",
    "metrics.json",
    "token_usage.json",
)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_dirs(path: Path) -> List[Path]:
    """Return run directories under path."""
    if _has_required_run_files(path):
        return [path]
    dirs = {
        candidate.parent
        for candidate in path.rglob("*")
        if candidate.name in REQUIRED_RUN_FILES
        and _has_required_run_files(candidate.parent)
    }
    return sorted(dirs)


def _has_required_run_files(path: Path) -> bool:
    """Return whether path contains the full analysis artifact set."""
    return all((path / filename).exists() for filename in REQUIRED_RUN_FILES)


def _cycle_metrics(metrics: Dict[str, Any]) -> List[tuple[int, Dict[str, Any]]]:
    cycles = []
    for key, value in metrics.items():
        if key.isdigit() and isinstance(value, dict):
            cycles.append((int(key), value))
    return sorted(cycles)


def _choose_metric(
    cycles: List[tuple[int, Dict[str, Any]]], metric: str | None
) -> str | None:
    if metric:
        if any(metric in values for _, values in cycles):
            return metric
        return None
    for candidate in PREFERRED_METRICS:
        if any(candidate in values for _, values in cycles):
            return candidate
    return None


def _resolve_mode(metric_name: str, mode: str) -> str:
    """Resolve auto mode from the selected metric name."""
    if mode != "auto":
        return mode
    return "min" if metric_name in LOWER_IS_BETTER else "max"


def _cycle_token_totals(token_usage: Dict[str, Any]) -> Dict[int, int]:
    totals = {}
    cumulative = 0
    for row in token_usage.get("per_cycle", []):
        cycle = int(row.get("cycle", len(totals)))
        cycle_total = row.get("cycle_total", {})
        cumulative += int(cycle_total.get("total_tokens", 0) or 0)
        totals[cycle] = cumulative
    return totals


def _cycle_auc(points: Iterable[tuple[float, float]]) -> float | None:
    sorted_points = sorted(points)
    if not sorted_points:
        return None
    if sorted_points[0][0] > 0:
        sorted_points.insert(0, (0.0, sorted_points[0][1]))
    max_x = sorted_points[-1][0]
    if max_x <= 0:
        return None

    area = 0.0
    for (x0, y0), (x1, y1) in zip(sorted_points, sorted_points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2
    return area / max_x


def summarize_run(
    run_dir: Path, metric: str | None = None, mode: str = "auto"
) -> Dict[str, Any]:
    """Summarize one Promptillery run directory."""
    missing = [
        filename
        for filename in REQUIRED_RUN_FILES
        if not (run_dir / filename).exists()
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"{run_dir} is missing required artifacts: {missing_list}")

    metrics = _load_json(run_dir / "metrics.json")
    token_usage = _load_json(run_dir / "token_usage.json")
    config = {}
    if (run_dir / "experiment_config.yaml").exists():
        try:
            import yaml

            with (run_dir / "experiment_config.yaml").open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            config = {}

    cycles = _cycle_metrics(metrics)
    metric_name = _choose_metric(cycles, metric)
    if not cycles:
        raise ValueError(f"{run_dir} has no numeric cycle metrics")
    if metric and metric_name is None:
        raise ValueError(f"metric '{metric}' was not found in {run_dir}")
    if metric_name is None:
        raise ValueError(
            f"{run_dir} has no recognized metric; rerun with an explicit --metric"
        )

    resolved_mode = _resolve_mode(metric_name, mode)
    cycle_tokens = _cycle_token_totals(token_usage)
    points = []
    values = []
    for cycle, values_for_cycle in cycles:
        if metric_name in values_for_cycle:
            value = float(values_for_cycle[metric_name])
            values.append((cycle, value))
            points.append((float(cycle_tokens.get(cycle, 0)), value))

    best_cycle = None
    best_value = None
    if values:
        best_cycle, best_value = (
            min(values, key=lambda item: item[1])
            if resolved_mode == "min"
            else max(values, key=lambda item: item[1])
        )

    final_cycle = cycles[-1][0] if cycles else None
    final_value = None
    if final_cycle is not None and metric_name in dict(cycles)[final_cycle]:
        final_value = dict(cycles)[final_cycle][metric_name]

    grand_total = token_usage.get("grand_total", {})
    return {
        "run_dir": str(run_dir),
        "experiment": config.get("name", run_dir.name),
        "metric": metric_name or "",
        "mode": resolved_mode,
        "best_cycle": best_cycle,
        "best_metric": best_value,
        "final_cycle": final_cycle,
        "final_metric": final_value,
        "cycle_quality_cost_auc": _cycle_auc(points),
        "token_budget": config.get("token_budget"),
        "token_budget_overage": (
            max(0, grand_total.get("total_tokens", 0) - int(config["token_budget"]))
            if isinstance(config.get("token_budget"), int)
            else None
        ),
        "teacher_input_tokens": grand_total.get("input_tokens", 0),
        "teacher_output_tokens": grand_total.get("output_tokens", 0),
        "teacher_total_tokens": grand_total.get("total_tokens", 0),
        "estimated_cost": grand_total.get("estimated_cost"),
        "cycles_completed": token_usage.get("cycles_completed", len(cycles)),
    }


def analyze_runs(
    path: Path, metric: str | None = None, mode: str = "auto"
) -> List[Dict[str, Any]]:
    """Analyze all run directories under path."""
    return [summarize_run(run_dir, metric=metric, mode=mode) for run_dir in _run_dirs(path)]


def write_summary_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write analysis rows as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_dir",
        "experiment",
        "metric",
        "mode",
        "best_cycle",
        "best_metric",
        "final_cycle",
        "final_metric",
        "cycle_quality_cost_auc",
        "token_budget",
        "token_budget_overage",
        "teacher_input_tokens",
        "teacher_output_tokens",
        "teacher_total_tokens",
        "estimated_cost",
        "cycles_completed",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
