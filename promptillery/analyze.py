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
    "run_manifest.json",
    "experiment_config.yaml",
    "metrics.json",
    "token_usage.json",
)
POLICY_BEHAVIOR_FIELDS = [
    "run_dir",
    "run_id",
    "experiment",
    "dataset",
    "dataset_subset",
    "student_model",
    "student_type",
    "seed",
    "token_budget",
    "policy_name",
    "action_space_id",
    "cycle",
    "decision_id",
    "action_name",
    "prompt_operator",
    "teacher_tier",
    "batch_size",
    "is_stop",
    "selected_score",
    "score_rank",
    "score_margin_to_second",
    "predicted_total_tokens",
    "feasible_action_count",
    "action_score_count",
    "acquisition_outcome",
    "synthetic_count_before",
    "tokens_remaining_before",
    "tokens_remaining_after",
    "teacher_tokens_this_cycle",
]
TEACHER_CALIBRATION_FIELDS = [
    "run_dir",
    "run_id",
    "experiment",
    "policy_name",
    "cycle",
    "decision_id",
    "attempt_id",
    "status",
    "failure_type",
    "teacher_model",
    "teacher_tier",
    "prompt_operator",
    "batch_size_requested",
    "records_requested",
    "records_parsed",
    "records_accepted",
    "preflight_allowed",
    "preflight_reason",
    "preflight_enforced",
    "estimator",
    "predicted_input_tokens",
    "predicted_max_output_tokens",
    "predicted_total_tokens",
    "realized_input_tokens",
    "realized_output_tokens",
    "realized_total_tokens",
    "realized_over_predicted",
    "over_preflight_bound",
    "over_remaining_budget",
    "tokens_remaining_before",
    "tokens_remaining_after",
]
ORACLE_FRONTIER_FIELDS = [
    "run_dir",
    "run_id",
    "experiment",
    "dataset",
    "dataset_subset",
    "student_model",
    "student_type",
    "seed",
    "token_budget",
    "selection_split",
    "metric",
    "mode",
    "policy_name",
    "frontier_available",
    "final_metric",
    "oracle_final_metric",
    "oracle_final_policy",
    "distance_to_oracle_final",
    "cycle_quality_cost_auc",
    "oracle_auc",
    "oracle_auc_policy",
    "distance_to_oracle_auc",
]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load compact JSONL rows, returning an empty list when absent."""
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_float(value: Any) -> float | None:
    """Best-effort conversion for CSV-friendly numeric summaries."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    """Best-effort integer conversion for optional artifact fields."""
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _rank_score(
    action_scores: Dict[str, Any], action_name: str
) -> tuple[int | None, float | None]:
    """Return 1-indexed score rank and margin to the second-best action."""
    scored = [
        (name, _safe_float(score))
        for name, score in action_scores.items()
        if _safe_float(score) is not None
    ]
    if not scored or action_name not in {name for name, _ in scored}:
        return None, None
    scored.sort(key=lambda item: float(item[1]), reverse=True)
    rank = next(
        index + 1 for index, (name, _) in enumerate(scored) if name == action_name
    )
    selected = _safe_float(action_scores.get(action_name))
    runner_up = None
    for name, score in scored:
        if name != action_name:
            runner_up = score
            break
    margin = (
        selected - runner_up
        if selected is not None and runner_up is not None
        else None
    )
    return rank, margin


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


def _count_by_status(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _teacher_attempt_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    status_counts = _count_by_status(rows)
    calibration_ratios = []
    for row in rows:
        predicted = row.get("predicted_cost", {})
        realized = row.get("realized_cost", {})
        predicted_total = predicted.get("total_tokens")
        realized_total = realized.get("total_tokens")
        if predicted_total and realized_total is not None:
            calibration_ratios.append(float(realized_total) / float(predicted_total))

    return {
        "teacher_attempt_count": len(rows),
        "teacher_success_count": status_counts.get("success", 0),
        "teacher_failure_count": sum(
            count
            for status, count in status_counts.items()
            if status not in {"success", "masked"}
        ),
        "teacher_masked_count": status_counts.get("masked", 0),
        "teacher_budget_violation_count": status_counts.get("budget_violation", 0),
        "max_realized_over_predicted": (
            max(calibration_ratios) if calibration_ratios else None
        ),
    }


def _run_context(run_dir: Path) -> Dict[str, Any]:
    """Return common run metadata used by paper-analysis CSVs."""
    run_manifest = _load_json(run_dir / "run_manifest.json")
    config = {}
    if (run_dir / "experiment_config.yaml").exists():
        try:
            import yaml

            with (run_dir / "experiment_config.yaml").open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            config = {}
    return {
        "run_dir": str(run_dir),
        "run_id": run_manifest.get("run_id", ""),
        "experiment": config.get("name", run_manifest.get("task_name", run_dir.name)),
        "dataset": run_manifest.get("dataset", config.get("dataset", "")),
        "dataset_subset": run_manifest.get("dataset_subset", ""),
        "student_model": run_manifest.get("student_model", config.get("student", "")),
        "student_type": run_manifest.get(
            "student_type", config.get("student_type", "")
        ),
        "seed": run_manifest.get("seed", config.get("seed", "")),
        "token_budget": run_manifest.get(
            "token_budget", config.get("token_budget", "")
        ),
        "policy_name": run_manifest.get(
            "policy_name", config.get("policy_name", "")
        ),
        "action_space_id": run_manifest.get("action_space", {}).get(
            "action_space_id", ""
        ),
    }


def summarize_policy_behavior(run_dir: Path) -> List[Dict[str, Any]]:
    """Return one row per policy decision for behavior plots."""
    context = _run_context(run_dir)
    rows = []
    for decision in _load_jsonl(run_dir / "policy_decisions.jsonl"):
        action = decision.get("action", {}) or {}
        metadata = decision.get("metadata", {}) or {}
        state = decision.get("state", {}) or {}
        budget_before = decision.get("budget_before", {}) or {}
        budget_after = decision.get("budget_after", {}) or {}
        realized = decision.get("realized_cost", {}) or {}
        predicted = decision.get("predicted_cost", {}) or {}
        action_scores = decision.get("action_scores", {}) or {}
        action_name = str(decision.get("action_name") or action.get("name") or "")
        score = action_scores.get(action_name)
        score_rank, score_margin = _rank_score(action_scores, action_name)
        rows.append(
            {
                **context,
                "policy_name": decision.get("policy_name", context["policy_name"]),
                "cycle": decision.get("cycle"),
                "decision_id": decision.get("decision_id", ""),
                "action_name": action_name,
                "prompt_operator": action.get("prompt_operator"),
                "teacher_tier": action.get("teacher_tier"),
                "batch_size": action.get("batch_size"),
                "is_stop": action.get("is_stop", action_name == "STOP"),
                "selected_score": score,
                "score_rank": score_rank,
                "score_margin_to_second": score_margin,
                "predicted_total_tokens": predicted.get("total_tokens"),
                "feasible_action_count": len(metadata.get("feasible_actions", [])),
                "action_score_count": len(action_scores),
                "acquisition_outcome": metadata.get(
                    "acquisition_outcome", action_name
                ),
                "synthetic_count_before": state.get("synthetic_count"),
                "tokens_remaining_before": budget_before.get("tokens_remaining"),
                "tokens_remaining_after": budget_after.get("tokens_remaining"),
                "teacher_tokens_this_cycle": realized.get("total_tokens"),
            }
        )
    return rows


def summarize_teacher_calibration(run_dir: Path) -> List[Dict[str, Any]]:
    """Return one row per teacher attempt for preflight calibration plots."""
    context = _run_context(run_dir)
    rows = []
    for attempt in _load_jsonl(run_dir / "teacher_attempts.jsonl"):
        metadata = attempt.get("metadata", {}) or {}
        predicted = attempt.get("predicted_cost", {}) or {}
        realized = attempt.get("realized_cost", {}) or {}
        budget_before = attempt.get("budget_before", {}) or {}
        budget_after = attempt.get("budget_after", {}) or {}
        predicted_total = _safe_float(predicted.get("total_tokens"))
        realized_total = _safe_float(realized.get("total_tokens"))
        tokens_remaining_before = _safe_float(budget_before.get("tokens_remaining"))
        ratio = (
            realized_total / predicted_total
            if predicted_total and realized_total is not None
            else None
        )
        rows.append(
            {
                "run_dir": context["run_dir"],
                "run_id": attempt.get("run_id", context["run_id"]),
                "experiment": context["experiment"],
                "policy_name": context["policy_name"],
                "cycle": attempt.get("cycle"),
                "decision_id": attempt.get("decision_id"),
                "attempt_id": attempt.get("attempt_id"),
                "status": attempt.get("status"),
                "failure_type": attempt.get("failure_type"),
                "teacher_model": metadata.get(
                    "teacher_model", metadata.get("teacher", predicted.get("teacher_model"))
                ),
                "teacher_tier": metadata.get("teacher_tier"),
                "prompt_operator": metadata.get("prompt_operator"),
                "batch_size_requested": metadata.get("batch_size_requested"),
                "records_requested": metadata.get("records_requested"),
                "records_parsed": metadata.get("records_parsed"),
                "records_accepted": metadata.get("records_accepted"),
                "preflight_allowed": predicted.get("allowed"),
                "preflight_reason": predicted.get("reason"),
                "preflight_enforced": predicted.get("preflight_enforced"),
                "estimator": predicted.get("estimator"),
                "predicted_input_tokens": predicted.get("input_tokens"),
                "predicted_max_output_tokens": predicted.get("max_output_tokens"),
                "predicted_total_tokens": predicted_total,
                "realized_input_tokens": realized.get("input_tokens"),
                "realized_output_tokens": realized.get("output_tokens"),
                "realized_total_tokens": realized_total,
                "realized_over_predicted": ratio,
                "over_preflight_bound": (
                    realized_total > predicted_total
                    if predicted_total is not None and realized_total is not None
                    else None
                ),
                "over_remaining_budget": (
                    realized_total > tokens_remaining_before
                    if tokens_remaining_before is not None
                    and realized_total is not None
                    else None
                ),
                "tokens_remaining_before": tokens_remaining_before,
                "tokens_remaining_after": budget_after.get("tokens_remaining"),
            }
        )
    return rows


def _is_fixed_policy(row: Dict[str, Any]) -> bool:
    policy_name = str(row.get("policy_name") or "")
    return policy_name == "fixed_promptillery" or policy_name.startswith("fixed_")


def _oracle_group_key(row: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("experiment"),
        row.get("dataset"),
        row.get("dataset_subset"),
        row.get("student_model"),
        row.get("student_type"),
        row.get("seed"),
        row.get("token_budget"),
        row.get("selection_split"),
        row.get("metric"),
        row.get("mode"),
    )


def _best_row(rows: List[Dict[str, Any]], value_key: str, mode: str) -> Dict[str, Any] | None:
    candidates = [row for row in rows if _safe_float(row.get(value_key)) is not None]
    if not candidates:
        return None
    chooser = min if mode == "min" else max
    return chooser(candidates, key=lambda row: float(row[value_key]))


def _oracle_distance(value: Any, oracle_value: Any, mode: str) -> float | None:
    value_float = _safe_float(value)
    oracle_float = _safe_float(oracle_value)
    if value_float is None or oracle_float is None:
        return None
    if mode == "min":
        return value_float - oracle_float
    return oracle_float - value_float


def summarize_oracle_frontier(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare each run to the best fixed schedule in its matched group."""
    fixed_by_group: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        if _is_fixed_policy(row):
            fixed_by_group.setdefault(_oracle_group_key(row), []).append(row)

    frontier_rows = []
    for row in rows:
        mode = str(row.get("mode") or "max")
        fixed_rows = fixed_by_group.get(_oracle_group_key(row), [])
        best_final = _best_row(fixed_rows, "final_metric", mode)
        best_auc = _best_row(fixed_rows, "cycle_quality_cost_auc", mode)
        frontier_rows.append(
            {
                "run_dir": row.get("run_dir"),
                "run_id": row.get("run_id"),
                "experiment": row.get("experiment"),
                "dataset": row.get("dataset"),
                "dataset_subset": row.get("dataset_subset"),
                "student_model": row.get("student_model"),
                "student_type": row.get("student_type"),
                "seed": row.get("seed"),
                "token_budget": row.get("token_budget"),
                "selection_split": row.get("selection_split"),
                "metric": row.get("metric"),
                "mode": mode,
                "policy_name": row.get("policy_name"),
                "frontier_available": bool(best_final or best_auc),
                "final_metric": row.get("final_metric"),
                "oracle_final_metric": (
                    best_final.get("final_metric") if best_final else None
                ),
                "oracle_final_policy": (
                    best_final.get("policy_name") if best_final else ""
                ),
                "distance_to_oracle_final": _oracle_distance(
                    row.get("final_metric"),
                    best_final.get("final_metric") if best_final else None,
                    mode,
                ),
                "cycle_quality_cost_auc": row.get("cycle_quality_cost_auc"),
                "oracle_auc": (
                    best_auc.get("cycle_quality_cost_auc") if best_auc else None
                ),
                "oracle_auc_policy": best_auc.get("policy_name") if best_auc else "",
                "distance_to_oracle_auc": _oracle_distance(
                    row.get("cycle_quality_cost_auc"),
                    best_auc.get("cycle_quality_cost_auc") if best_auc else None,
                    mode,
                ),
            }
        )
    return frontier_rows


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
    run_manifest = _load_json(run_dir / "run_manifest.json")
    if run_manifest.get("status") != "completed":
        raise ValueError(
            f"{run_dir} has run_manifest status={run_manifest.get('status')!r}; "
            "only completed runs are summarized"
        )
    policy_decisions = _load_jsonl(run_dir / "policy_decisions.jsonl")
    teacher_attempts = _load_jsonl(run_dir / "teacher_attempts.jsonl")
    teacher_summary = _teacher_attempt_summary(teacher_attempts)
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
    token_budget = run_manifest.get("token_budget", config.get("token_budget"))
    return {
        "run_dir": str(run_dir),
        "run_id": run_manifest.get("run_id", ""),
        "run_status": run_manifest.get("status", ""),
        "selection_split": run_manifest.get("selection_split", ""),
        "paper_mode": run_manifest.get("paper_mode", ""),
        "task_name": run_manifest.get("task_name", config.get("name", run_dir.name)),
        "dataset": run_manifest.get("dataset", config.get("dataset", "")),
        "dataset_subset": run_manifest.get("dataset_subset", ""),
        "student_model": run_manifest.get("student_model", config.get("student", "")),
        "student_type": run_manifest.get(
            "student_type", config.get("student_type", "")
        ),
        "seed": run_manifest.get("seed", config.get("seed", "")),
        "policy_name": run_manifest.get("policy_name", config.get("policy_name", "")),
        "policy_family": run_manifest.get("policy_family", ""),
        "action_space_id": run_manifest.get("action_space", {}).get(
            "action_space_id", ""
        ),
        "experiment": config.get("name", run_dir.name),
        "metric": metric_name or "",
        "mode": resolved_mode,
        "best_cycle": best_cycle,
        "best_metric": best_value,
        "final_cycle": final_cycle,
        "final_metric": final_value,
        "cycle_quality_cost_auc": _cycle_auc(points),
        "token_budget": token_budget,
        "token_budget_overage": (
            max(0, grand_total.get("total_tokens", 0) - int(token_budget))
            if isinstance(token_budget, int)
            else None
        ),
        "teacher_input_tokens": grand_total.get("input_tokens", 0),
        "teacher_output_tokens": grand_total.get("output_tokens", 0),
        "teacher_total_tokens": grand_total.get("total_tokens", 0),
        "estimated_cost": grand_total.get("estimated_cost"),
        "cycles_completed": token_usage.get("cycles_completed", len(cycles)),
        "policy_decision_count": len(policy_decisions),
        "policy_stop_count": sum(
            1 for row in policy_decisions if row.get("action_name") == "STOP"
        ),
        **teacher_summary,
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
        "run_id",
        "run_status",
        "selection_split",
        "paper_mode",
        "task_name",
        "dataset",
        "dataset_subset",
        "student_model",
        "student_type",
        "seed",
        "policy_name",
        "policy_family",
        "action_space_id",
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
        "policy_decision_count",
        "policy_stop_count",
        "teacher_attempt_count",
        "teacher_success_count",
        "teacher_failure_count",
        "teacher_masked_count",
        "teacher_budget_violation_count",
        "max_realized_over_predicted",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_rows_csv(
    rows: List[Dict[str, Any]], output_path: Path, fieldnames: List[str]
) -> None:
    """Write rows with a stable header, even when no rows are present."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_audit_csvs(
    path: Path,
    rows: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Path]:
    """Write reviewer-facing audit CSVs from existing run artifacts."""
    run_dirs = _run_dirs(path)
    policy_rows = [
        row
        for run_dir in run_dirs
        for row in summarize_policy_behavior(run_dir)
    ]
    calibration_rows = [
        row
        for run_dir in run_dirs
        for row in summarize_teacher_calibration(run_dir)
    ]
    oracle_rows = summarize_oracle_frontier(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "policy_actions": output_dir / "policy_actions.csv",
        "teacher_calibration": output_dir / "teacher_calibration.csv",
        "oracle_frontier": output_dir / "oracle_frontier.csv",
    }
    _write_rows_csv(policy_rows, paths["policy_actions"], POLICY_BEHAVIOR_FIELDS)
    _write_rows_csv(
        calibration_rows,
        paths["teacher_calibration"],
        TEACHER_CALIBRATION_FIELDS,
    )
    _write_rows_csv(oracle_rows, paths["oracle_frontier"], ORACLE_FRONTIER_FIELDS)
    return paths


def _normalize_set(values: Iterable[Any]) -> set[str]:
    """Normalize optional CLI/config values for set comparisons."""
    return {str(value) for value in values}


def _gate_check(name: str, passed: bool, **details: Any) -> Dict[str, Any]:
    """Build one machine-readable pilot gate check."""
    return {"name": name, "passed": passed, **details}


def validate_pilot_gate(
    path: Path,
    *,
    metric: str | None = None,
    mode: str = "auto",
    expected_policies: List[str] | None = None,
    expected_seeds: List[str] | None = None,
    expected_budgets: List[str] | None = None,
    require_teacher_attempts: bool = True,
    require_frontier: bool = True,
) -> Dict[str, Any]:
    """Return a pass/fail report for the cheap policy-axis pilot."""
    rows = analyze_runs(path, metric=metric, mode=mode)
    run_dirs = _run_dirs(path)
    policy_rows = [
        row
        for run_dir in run_dirs
        for row in summarize_policy_behavior(run_dir)
    ]
    calibration_rows = [
        row
        for run_dir in run_dirs
        for row in summarize_teacher_calibration(run_dir)
    ]
    frontier_rows = summarize_oracle_frontier(rows)

    policies_found = _normalize_set(row.get("policy_name") for row in rows)
    seeds_found = _normalize_set(row.get("seed") for row in rows)
    budgets_found = _normalize_set(row.get("token_budget") for row in rows)
    expected_policy_set = set(expected_policies or [])
    expected_seed_set = set(expected_seeds or [])
    expected_budget_set = set(expected_budgets or [])

    checks = []
    if expected_policy_set:
        checks.append(
            _gate_check(
                "expected_policies_present",
                expected_policy_set.issubset(policies_found),
                expected=sorted(expected_policy_set),
                found=sorted(policies_found),
                missing=sorted(expected_policy_set - policies_found),
            )
        )
    if expected_seed_set:
        checks.append(
            _gate_check(
                "expected_seeds_present",
                expected_seed_set.issubset(seeds_found),
                expected=sorted(expected_seed_set),
                found=sorted(seeds_found),
                missing=sorted(expected_seed_set - seeds_found),
            )
        )
    if expected_budget_set:
        checks.append(
            _gate_check(
                "expected_budgets_present",
                expected_budget_set.issubset(budgets_found),
                expected=sorted(expected_budget_set),
                found=sorted(budgets_found),
                missing=sorted(expected_budget_set - budgets_found),
            )
        )

    if expected_policy_set and expected_seed_set and expected_budget_set:
        expected_combos = {
            (policy, seed, budget)
            for policy in expected_policy_set
            for seed in expected_seed_set
            for budget in expected_budget_set
        }
        found_combos = {
            (
                str(row.get("policy_name")),
                str(row.get("seed")),
                str(row.get("token_budget")),
            )
            for row in rows
        }
        checks.append(
            _gate_check(
                "expected_policy_seed_budget_grid",
                expected_combos == found_combos,
                expected_count=len(expected_combos),
                found_count=len(found_combos),
                missing=[list(combo) for combo in sorted(expected_combos - found_combos)],
                unexpected=[
                    list(combo) for combo in sorted(found_combos - expected_combos)
                ],
            )
        )

    overage_rows = [
        row for row in rows if (_safe_float(row.get("token_budget_overage")) or 0) > 0
    ]
    checks.append(
        _gate_check(
            "no_token_budget_overage",
            not overage_rows,
            violating_run_ids=[row.get("run_id") for row in overage_rows],
        )
    )

    missing_decision_rows = [
        row for row in rows if (_safe_int(row.get("policy_decision_count")) or 0) <= 0
    ]
    checks.append(
        _gate_check(
            "policy_decisions_present",
            not missing_decision_rows,
            violating_run_ids=[row.get("run_id") for row in missing_decision_rows],
        )
    )

    fixed_operator_failures = []
    for policy_name, expected_operator in (
        ("fixed_coverage", "coverage"),
        ("fixed_boundary", "boundary"),
        ("fixed_repair", "repair"),
    ):
        if expected_policy_set and policy_name not in expected_policy_set:
            continue
        if policy_name not in policies_found:
            continue
        operators = {
            str(row.get("prompt_operator"))
            for row in policy_rows
            if row.get("policy_name") == policy_name and row.get("action_name") != "STOP"
        }
        if expected_operator not in operators:
            fixed_operator_failures.append(
                {
                    "policy_name": policy_name,
                    "expected_operator": expected_operator,
                    "found_operators": sorted(operators),
                }
            )
    checks.append(
        _gate_check(
            "fixed_policies_use_distinct_prompt_operators",
            not fixed_operator_failures,
            failures=fixed_operator_failures,
        )
    )

    decision_ids = {
        str(row.get("decision_id"))
        for row in policy_rows
        if row.get("decision_id")
    }
    orphan_attempts = [
        row
        for row in calibration_rows
        if row.get("decision_id") and str(row.get("decision_id")) not in decision_ids
    ]
    missing_attempt_decisions = [
        row for row in calibration_rows if not row.get("decision_id")
    ]
    checks.append(
        _gate_check(
            "teacher_attempts_join_policy_decisions",
            not orphan_attempts and not missing_attempt_decisions,
            orphan_attempt_ids=[row.get("attempt_id") for row in orphan_attempts],
            missing_decision_attempt_ids=[
                row.get("attempt_id") for row in missing_attempt_decisions
            ],
        )
    )

    if require_teacher_attempts:
        checks.append(
            _gate_check(
                "teacher_attempts_present",
                bool(calibration_rows),
                teacher_attempt_count=len(calibration_rows),
            )
        )

    preflight_violations = [
        row for row in calibration_rows if row.get("over_preflight_bound") is True
    ]
    remaining_violations = [
        row for row in calibration_rows if row.get("over_remaining_budget") is True
    ]
    checks.append(
        _gate_check(
            "no_preflight_or_remaining_budget_violations",
            not preflight_violations and not remaining_violations,
            over_preflight_attempt_ids=[
                row.get("attempt_id") for row in preflight_violations
            ],
            over_remaining_attempt_ids=[
                row.get("attempt_id") for row in remaining_violations
            ],
        )
    )

    if require_frontier:
        missing_frontier = [
            row for row in frontier_rows if row.get("frontier_available") is not True
        ]
        checks.append(
            _gate_check(
                "fixed_policy_frontier_available",
                not missing_frontier,
                violating_run_ids=[row.get("run_id") for row in missing_frontier],
            )
        )

    passed = all(check["passed"] for check in checks)
    return {
        "passed": passed,
        "path": str(path),
        "metric": metric,
        "mode": mode,
        "run_count": len(rows),
        "policy_decision_rows": len(policy_rows),
        "teacher_calibration_rows": len(calibration_rows),
        "frontier_rows": len(frontier_rows),
        "policies_found": sorted(policies_found),
        "seeds_found": sorted(seeds_found),
        "budgets_found": sorted(budgets_found),
        "checks": checks,
    }
