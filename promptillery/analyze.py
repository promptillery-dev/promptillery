"""Analyze Promptillery run artifacts."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PREFERRED_METRICS = (
    "exact_match",
    "accuracy",
    "macro_f1",
    "f1",
    "perplexity",
    "eval_loss",
)
LOWER_IS_BETTER = {"eval_loss", "loss", "perplexity"}
SAME_COUNT_CONTROL_NAMES = {"same_count", "same_synthetic_count"}
SAME_COUNT_CONFIG_IGNORED_KEYS = {
    "base_output_dir",
    "control_name",
    "name",
    "policy_name",
    "synthetic_record_budget",
}
REQUIRED_RUN_FILES = (
    "run_manifest.json",
    "experiment_config.yaml",
    "metrics.json",
    "token_usage.json",
)
POLICY_BEHAVIOR_FIELDS = [
    "run_dir",
    "run_id",
    "control_name",
    "experiment",
    "dataset",
    "dataset_subset",
    "student_model",
    "student_type",
    "seed",
    "token_budget",
    "synthetic_record_budget",
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
    "control_name",
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
    "provider_reported_input_tokens",
    "provider_reported_output_tokens",
    "provider_reported_total_tokens",
    "provider_reported_present",
    "ledger_debit_input_tokens",
    "ledger_debit_output_tokens",
    "ledger_debit_total_tokens",
    "ledger_debit_source",
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
    "control_name",
    "experiment",
    "dataset",
    "dataset_subset",
    "student_model",
    "student_type",
    "seed",
    "token_budget",
    "synthetic_record_budget",
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
PAPER_MAIN_RESULT_FIELDS = [
    "dataset",
    "dataset_subset",
    "student_model",
    "student_type",
    "metric",
    "mode",
    "token_budget",
    "policy_name",
    "control_name",
    "seeds",
    "run_count",
    "mean_cycle_quality_cost_auc",
    "std_cycle_quality_cost_auc",
    "mean_heldout_metric",
    "std_heldout_metric",
    "mean_final_metric",
    "std_final_metric",
    "mean_online_teacher_total_tokens",
    "mean_total_teacher_total_tokens",
    "mean_final_synthetic_count",
]
PAPER_PAIRWISE_DELTA_FIELDS = [
    "dataset",
    "dataset_subset",
    "student_model",
    "student_type",
    "metric",
    "mode",
    "seed",
    "token_budget",
    "success_policy",
    "baseline_policy",
    "baseline_control_name",
    "delta_cycle_quality_cost_auc",
    "delta_heldout_metric",
    "delta_final_metric",
    "auc_win",
    "heldout_win",
    "final_win",
]
PAPER_PAIRWISE_SUMMARY_FIELDS = [
    "dataset",
    "dataset_subset",
    "student_model",
    "student_type",
    "metric",
    "mode",
    "token_budget",
    "success_policy",
    "baseline_policy",
    "baseline_control_name",
    "auc_n",
    "auc_wins",
    "auc_losses",
    "auc_ties",
    "auc_win_rate",
    "auc_mean_delta",
    "auc_sign_test_p",
    "heldout_n",
    "heldout_wins",
    "heldout_losses",
    "heldout_ties",
    "heldout_win_rate",
    "heldout_mean_delta",
    "heldout_sign_test_p",
    "final_n",
    "final_wins",
    "final_losses",
    "final_ties",
    "final_win_rate",
    "final_mean_delta",
    "final_sign_test_p",
]
PAPER_QUALITY_COST_POINT_FIELDS = [
    "run_dir",
    "run_id",
    "dataset",
    "dataset_subset",
    "student_model",
    "student_type",
    "seed",
    "token_budget",
    "synthetic_record_budget",
    "policy_name",
    "control_name",
    "experiment",
    "action_space_id",
    "metric",
    "mode",
    "split",
    "cycle",
    "is_final_cycle",
    "metric_value",
    "cumulative_online_teacher_input_tokens",
    "cumulative_online_teacher_output_tokens",
    "cumulative_online_teacher_total_tokens",
    "seed_teacher_total_tokens",
    "cumulative_total_teacher_total_tokens",
]
PAPER_ACTION_FREQUENCY_FIELDS = [
    "dataset",
    "dataset_subset",
    "student_model",
    "student_type",
    "token_budget",
    "policy_name",
    "control_name",
    "action_name",
    "prompt_operator",
    "teacher_tier",
    "batch_size",
    "count",
    "share",
]
PAPER_BUDGET_AUDIT_FIELDS = [
    "dataset",
    "dataset_subset",
    "student_model",
    "student_type",
    "metric",
    "mode",
    "token_budget",
    "policy_name",
    "control_name",
    "run_count",
    "max_token_budget_overage",
    "max_realized_over_predicted",
    "missing_provider_usage_rows",
    "over_preflight_bound_rows",
    "over_remaining_budget_rows",
    "failed_attempt_rows",
    "mean_online_teacher_total_tokens",
    "mean_total_teacher_total_tokens",
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


def _truthy(value: Any) -> bool:
    """Return whether an artifact field encodes a truthy boolean."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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
        selected - runner_up if selected is not None and runner_up is not None else None
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


def _coerce_paths(paths: Path | Sequence[Path]) -> List[Path]:
    """Normalize one or more analysis roots."""
    if isinstance(paths, Path):
        return [paths]
    return list(paths)


def _run_dirs_for_paths(paths: Path | Sequence[Path]) -> List[Path]:
    """Return run directories under one or more paths without duplicates."""
    seen: set[Path] = set()
    run_dirs: List[Path] = []
    for path in _coerce_paths(paths):
        for run_dir in _run_dirs(path):
            resolved = run_dir.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            run_dirs.append(run_dir)
    return run_dirs


def _cycle_metrics(metrics: Dict[str, Any]) -> List[tuple[int, Dict[str, Any]]]:
    cycles = []
    for key, value in metrics.items():
        if key.isdigit() and isinstance(value, dict):
            cycles.append((int(key), value))
    return sorted(cycles)


def _heldout_test_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return final held-out metrics when a run reported them."""
    heldout = metrics.get("heldout_test")
    return heldout if isinstance(heldout, dict) else {}


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
    return {
        cycle: int(usage.get("total_tokens", 0) or 0)
        for cycle, usage in _cycle_token_cumulative_usage(token_usage).items()
    }


def _cycle_token_cumulative_usage(
    token_usage: Dict[str, Any],
) -> Dict[int, Dict[str, int]]:
    """Return cumulative teacher-token usage at the end of each cycle."""
    totals = {}
    cumulative = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for row in token_usage.get("per_cycle", []):
        cycle = int(row.get("cycle", len(totals)))
        cycle_total = row.get("cycle_total", {})
        for key in cumulative:
            cumulative[key] += int(cycle_total.get(key, 0) or 0)
        totals[cycle] = dict(cumulative)
    return totals


def _charged_seed_usage(token_usage: Dict[str, Any]) -> Dict[str, int]:
    """Return SFT seed tokens debited into token_usage, when present."""
    sft_usage = (token_usage.get("totals") or {}).get("sft_data", {})
    return {
        "input_tokens": int(sft_usage.get("input_tokens", 0) or 0),
        "output_tokens": int(sft_usage.get("output_tokens", 0) or 0),
        "total_tokens": int(sft_usage.get("total_tokens", 0) or 0),
    }


def _charged_seed_total(token_usage: Dict[str, Any]) -> int:
    return _charged_seed_usage(token_usage)["total_tokens"]


def _configured_data_files(config: Dict[str, Any]) -> List[str]:
    """Return dataset file paths from a copied experiment config."""
    data_files = (config.get("dataset_kwargs") or {}).get("data_files")
    if data_files is None:
        return []
    if isinstance(data_files, str):
        return [data_files]
    if isinstance(data_files, list):
        return [str(item) for item in data_files]
    if isinstance(data_files, dict):
        paths: List[str] = []
        for value in data_files.values():
            if isinstance(value, list):
                paths.extend(str(item) for item in value)
            else:
                paths.append(str(value))
        return paths
    return []


def _resolve_existing_path(path: str, run_dir: Path) -> Path | None:
    """Resolve a copied config path against common analysis locations."""
    raw_path = Path(path).expanduser()
    candidates = [raw_path] if raw_path.is_absolute() else []
    if not raw_path.is_absolute():
        candidates.extend(
            [
                Path.cwd() / raw_path,
                run_dir / raw_path,
                run_dir.parent / raw_path,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _seed_materialization_summary(
    run_dir: Path, config: Dict[str, Any], token_usage: Dict[str, Any]
) -> Dict[str, Any]:
    """Summarize fixed seed-data teacher tokens from manifests or token usage."""
    manifest_paths = []
    for data_file in _configured_data_files(config):
        resolved = _resolve_existing_path(data_file, run_dir)
        if resolved is None:
            continue
        manifest = Path(str(resolved) + ".manifest.json")
        if manifest.exists():
            manifest_paths.append(manifest)

    seed_total = 0
    estimated_records = 0
    for manifest_path in sorted(set(manifest_paths)):
        payload = _load_json(manifest_path)
        seed_total += int(payload.get("teacher_total_tokens", 0) or 0)
        estimated_records += int(payload.get("usage_estimated_records", 0) or 0)

    charged_seed_total = _charged_seed_total(token_usage)
    if seed_total <= 0:
        seed_total = charged_seed_total

    grand_total = token_usage.get("grand_total", {})
    online_total = int(grand_total.get("total_tokens", 0) or 0)
    if charged_seed_total > 0:
        online_total = max(0, online_total - charged_seed_total)

    return {
        "seed_materialization_manifest_count": len(set(manifest_paths)),
        "seed_usage_estimated_records": estimated_records,
        "seed_teacher_total_tokens": seed_total,
        "online_teacher_total_tokens": online_total,
        "total_teacher_total_tokens": seed_total + online_total,
    }


def _cycle_auc(
    points: Iterable[tuple[float, float]], *, x_max: float | None = None
) -> float | None:
    sorted_points = sorted(points)
    if not sorted_points:
        return None
    if sorted_points[0][0] > 0:
        sorted_points.insert(0, (0.0, sorted_points[0][1]))

    max_x = sorted_points[-1][0]
    if x_max is not None:
        horizon = float(x_max)
        if horizon <= 0:
            return None
        bounded_points = []
        for point in sorted_points:
            if point[0] <= horizon:
                bounded_points.append(point)
            else:
                break
        if not bounded_points:
            bounded_points = [(0.0, sorted_points[0][1])]
        if bounded_points[-1][0] < horizon:
            bounded_points.append((horizon, bounded_points[-1][1]))
        sorted_points = bounded_points
        max_x = horizon

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
        realized = (
            row.get("provider_reported_cost")
            or row.get("ledger_debit_cost")
            or row.get("realized_cost", {})
        )
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


def _final_synthetic_count(policy_decisions: List[Dict[str, Any]]) -> int | None:
    """Infer final accepted synthetic row count from decision logs."""
    counts: list[int] = []
    for decision in policy_decisions:
        state = decision.get("state", {}) or {}
        metadata = decision.get("metadata", {}) or {}
        before = _safe_int(state.get("synthetic_count"))
        added = _safe_int(
            metadata.get("records_added", metadata.get("articles_added", 0))
        )
        if before is not None:
            counts.append(before + (added or 0))
    return max(counts) if counts else None


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
        "control_name": run_manifest.get(
            "control_name", config.get("control_name", "")
        ),
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
        "synthetic_record_budget": run_manifest.get(
            "synthetic_record_budget", config.get("synthetic_record_budget", "")
        ),
        "policy_name": run_manifest.get("policy_name", config.get("policy_name", "")),
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
                "acquisition_outcome": metadata.get("acquisition_outcome", action_name),
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
        provider_reported_raw = attempt.get("provider_reported_cost") or {}
        provider_reported = provider_reported_raw or attempt.get("realized_cost", {})
        ledger_debit = attempt.get("ledger_debit_cost") or attempt.get(
            "realized_cost", {}
        )
        realized = provider_reported or ledger_debit
        budget_before = attempt.get("budget_before", {}) or {}
        budget_after = attempt.get("budget_after", {}) or {}
        predicted_total = _safe_float(predicted.get("total_tokens"))
        provider_total = _safe_float(provider_reported.get("total_tokens"))
        ledger_total = _safe_float(ledger_debit.get("total_tokens"))
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
                "control_name": context["control_name"],
                "experiment": context["experiment"],
                "policy_name": context["policy_name"],
                "cycle": attempt.get("cycle"),
                "decision_id": attempt.get("decision_id"),
                "attempt_id": attempt.get("attempt_id"),
                "status": attempt.get("status"),
                "failure_type": attempt.get("failure_type"),
                "teacher_model": metadata.get(
                    "teacher_model",
                    metadata.get("teacher", predicted.get("teacher_model")),
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
                "provider_reported_input_tokens": provider_reported.get("input_tokens"),
                "provider_reported_output_tokens": provider_reported.get(
                    "output_tokens"
                ),
                "provider_reported_total_tokens": provider_total,
                "provider_reported_present": bool(provider_reported_raw),
                "ledger_debit_input_tokens": ledger_debit.get("input_tokens"),
                "ledger_debit_output_tokens": ledger_debit.get("output_tokens"),
                "ledger_debit_total_tokens": ledger_total,
                "ledger_debit_source": attempt.get("ledger_debit_source"),
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
    if _is_same_count_control(row):
        return False
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


def _best_row(
    rows: List[Dict[str, Any]], value_key: str, mode: str
) -> Dict[str, Any] | None:
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
                "control_name": row.get("control_name"),
                "experiment": row.get("experiment"),
                "dataset": row.get("dataset"),
                "dataset_subset": row.get("dataset_subset"),
                "student_model": row.get("student_model"),
                "student_type": row.get("student_type"),
                "seed": row.get("seed"),
                "token_budget": row.get("token_budget"),
                "synthetic_record_budget": row.get("synthetic_record_budget"),
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
        filename for filename in REQUIRED_RUN_FILES if not (run_dir / filename).exists()
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
    manifest_final_synthetic_count = run_manifest.get("final_synthetic_count")
    final_synthetic_count = (
        manifest_final_synthetic_count
        if manifest_final_synthetic_count is not None
        else _final_synthetic_count(policy_decisions)
    )
    config = {}
    if (run_dir / "experiment_config.yaml").exists():
        try:
            import yaml

            with (run_dir / "experiment_config.yaml").open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            config = {}
    seed_summary = _seed_materialization_summary(run_dir, config, token_usage)

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
            tokens_at_eval = _safe_float(
                values_for_cycle.get("_teacher_tokens_at_eval")
            )
            if tokens_at_eval is None:
                tokens_at_eval = float(cycle_tokens.get(cycle, 0))
            points.append((tokens_at_eval, value))

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
    heldout = _heldout_test_metrics(metrics)
    heldout_value = (
        _safe_float(heldout.get(metric_name)) if metric_name and heldout else None
    )
    heldout_split = heldout.get("_heldout_split", "") if heldout else ""
    heldout_metric_name = metric_name if heldout_value is not None else ""
    final_cycle_metrics = (
        dict(cycles).get(final_cycle, {}) if final_cycle is not None else {}
    )

    grand_total = token_usage.get("grand_total", {})
    token_budget = run_manifest.get("token_budget", config.get("token_budget"))
    token_budget_int = _safe_int(token_budget)
    teacher_total_tokens = _safe_int(grand_total.get("total_tokens")) or 0
    cycles_completed = _safe_int(
        token_usage.get(
            "cycles_completed", run_manifest.get("cycles_completed", len(cycles))
        )
    )
    expected_cycles = _safe_int(
        run_manifest.get("expected_cycles", config.get("cycles"))
    )
    return {
        "run_dir": str(run_dir),
        "run_id": run_manifest.get("run_id", ""),
        "run_status": run_manifest.get("status", ""),
        "control_name": run_manifest.get(
            "control_name", config.get("control_name", "")
        ),
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
        "same_count_config_hash": _same_count_config_fingerprint(config),
        "experiment": config.get("name", run_dir.name),
        "metric": metric_name or "",
        "mode": resolved_mode,
        "best_cycle": best_cycle,
        "best_metric": best_value,
        "final_cycle": final_cycle,
        "final_metric": final_value,
        "heldout_split": heldout_split,
        "heldout_metric_name": heldout_metric_name,
        "heldout_metric": heldout_value,
        "canonical_label_count": _safe_int(
            final_cycle_metrics.get("canonical_label_count")
        ),
        "observed_gold_label_count": _safe_int(
            final_cycle_metrics.get("observed_gold_label_count")
        ),
        "heldout_canonical_label_count": _safe_int(
            heldout.get("canonical_label_count") if heldout else None
        ),
        "heldout_observed_gold_label_count": _safe_int(
            heldout.get("observed_gold_label_count") if heldout else None
        ),
        "cycle_quality_cost_auc": _cycle_auc(points, x_max=token_budget_int),
        "token_budget": token_budget,
        "synthetic_record_budget": run_manifest.get(
            "synthetic_record_budget", config.get("synthetic_record_budget")
        ),
        "final_synthetic_count": final_synthetic_count,
        "token_budget_overage": (
            max(0, teacher_total_tokens - token_budget_int)
            if token_budget_int is not None
            else None
        ),
        "teacher_input_tokens": grand_total.get("input_tokens", 0),
        "teacher_output_tokens": grand_total.get("output_tokens", 0),
        "teacher_total_tokens": teacher_total_tokens,
        "estimated_cost": grand_total.get("estimated_cost"),
        **seed_summary,
        "cycles_completed": cycles_completed,
        "expected_cycles": expected_cycles,
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
    return [
        summarize_run(run_dir, metric=metric, mode=mode) for run_dir in _run_dirs(path)
    ]


def write_summary_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write analysis rows as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_dir",
        "run_id",
        "run_status",
        "control_name",
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
        "same_count_config_hash",
        "experiment",
        "metric",
        "mode",
        "best_cycle",
        "best_metric",
        "final_cycle",
        "final_metric",
        "heldout_split",
        "heldout_metric_name",
        "heldout_metric",
        "canonical_label_count",
        "observed_gold_label_count",
        "heldout_canonical_label_count",
        "heldout_observed_gold_label_count",
        "cycle_quality_cost_auc",
        "token_budget",
        "synthetic_record_budget",
        "final_synthetic_count",
        "token_budget_overage",
        "teacher_input_tokens",
        "teacher_output_tokens",
        "teacher_total_tokens",
        "estimated_cost",
        "seed_materialization_manifest_count",
        "seed_usage_estimated_records",
        "seed_teacher_total_tokens",
        "online_teacher_total_tokens",
        "total_teacher_total_tokens",
        "cycles_completed",
        "expected_cycles",
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
    path: Path | Sequence[Path],
    rows: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Path]:
    """Write reviewer-facing audit CSVs from existing run artifacts."""
    run_dirs = _run_dirs_for_paths(path)
    policy_rows = [
        row for run_dir in run_dirs for row in summarize_policy_behavior(run_dir)
    ]
    calibration_rows = [
        row for run_dir in run_dirs for row in summarize_teacher_calibration(run_dir)
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


def _mean_std(values: Iterable[Any]) -> tuple[float | None, float | None]:
    """Return mean and sample std for present numeric values."""
    numeric = [
        converted for value in values if (converted := _safe_float(value)) is not None
    ]
    if not numeric:
        return None, None
    if len(numeric) == 1:
        return numeric[0], 0.0
    return statistics.fmean(numeric), statistics.stdev(numeric)


def _paper_result_key(row: Dict[str, Any]) -> tuple[Any, ...]:
    """Group rows by the axes used in the main paper result table."""
    return (
        row.get("dataset"),
        row.get("dataset_subset"),
        row.get("student_model"),
        row.get("student_type"),
        row.get("metric"),
        row.get("mode"),
        row.get("token_budget"),
        row.get("policy_name"),
        row.get("control_name"),
    )


def _paper_match_key(row: Dict[str, Any]) -> tuple[Any, ...]:
    """Match success and baseline rows for paired deltas."""
    return (
        row.get("dataset"),
        row.get("dataset_subset"),
        row.get("student_model"),
        row.get("student_type"),
        row.get("metric"),
        row.get("mode"),
        row.get("seed"),
        row.get("token_budget"),
    )


def _improvement_delta(
    success_value: Any, baseline_value: Any, mode: str
) -> float | None:
    """Return a positive-is-better paired delta."""
    success_float = _safe_float(success_value)
    baseline_float = _safe_float(baseline_value)
    if success_float is None or baseline_float is None:
        return None
    if mode == "min":
        return baseline_float - success_float
    return success_float - baseline_float


def _win_flag(delta: float | None) -> bool | None:
    """Return a strict win flag while preserving missing comparisons."""
    if delta is None:
        return None
    return delta > 0


def summarize_paper_main_results(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate run summaries into paper-table rows."""
    grouped: Dict[tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_paper_result_key(row)].append(row)

    results = []
    for key, group in sorted(
        grouped.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        (
            dataset,
            dataset_subset,
            student_model,
            student_type,
            metric,
            mode,
            token_budget,
            policy_name,
            control_name,
        ) = key
        auc_mean, auc_std = _mean_std(
            row.get("cycle_quality_cost_auc") for row in group
        )
        heldout_mean, heldout_std = _mean_std(
            row.get("heldout_metric") for row in group
        )
        final_mean, final_std = _mean_std(row.get("final_metric") for row in group)
        online_mean, _ = _mean_std(
            row.get("online_teacher_total_tokens") for row in group
        )
        total_mean, _ = _mean_std(
            row.get("total_teacher_total_tokens") for row in group
        )
        synthetic_mean, _ = _mean_std(row.get("final_synthetic_count") for row in group)
        seeds = sorted({str(row.get("seed")) for row in group if row.get("seed") != ""})
        results.append(
            {
                "dataset": dataset,
                "dataset_subset": dataset_subset,
                "student_model": student_model,
                "student_type": student_type,
                "metric": metric,
                "mode": mode,
                "token_budget": token_budget,
                "policy_name": policy_name,
                "control_name": control_name,
                "seeds": ",".join(seeds),
                "run_count": len(group),
                "mean_cycle_quality_cost_auc": auc_mean,
                "std_cycle_quality_cost_auc": auc_std,
                "mean_heldout_metric": heldout_mean,
                "std_heldout_metric": heldout_std,
                "mean_final_metric": final_mean,
                "std_final_metric": final_std,
                "mean_online_teacher_total_tokens": online_mean,
                "mean_total_teacher_total_tokens": total_mean,
                "mean_final_synthetic_count": synthetic_mean,
            }
        )
    return results


def summarize_paper_pairwise_deltas(
    rows: List[Dict[str, Any]],
    *,
    success_policy: str,
    baseline_policies: List[str],
) -> List[Dict[str, Any]]:
    """Return paired success-policy deltas against named baselines."""
    if not success_policy or not baseline_policies:
        return []
    success_by_key = {
        _paper_match_key(row): row
        for row in rows
        if str(row.get("policy_name") or "") == success_policy
        and not _is_same_count_control(row)
    }
    baseline_set = set(baseline_policies)
    delta_rows = []
    for baseline in rows:
        baseline_policy = str(baseline.get("policy_name") or "")
        if baseline_policy not in baseline_set:
            continue
        success = success_by_key.get(_paper_match_key(baseline))
        if not success:
            continue
        mode = str(success.get("mode") or baseline.get("mode") or "max")
        auc_delta = _improvement_delta(
            success.get("cycle_quality_cost_auc"),
            baseline.get("cycle_quality_cost_auc"),
            mode,
        )
        heldout_delta = _improvement_delta(
            success.get("heldout_metric"), baseline.get("heldout_metric"), mode
        )
        final_delta = _improvement_delta(
            success.get("final_metric"), baseline.get("final_metric"), mode
        )
        delta_rows.append(
            {
                "dataset": success.get("dataset"),
                "dataset_subset": success.get("dataset_subset"),
                "student_model": success.get("student_model"),
                "student_type": success.get("student_type"),
                "metric": success.get("metric"),
                "mode": mode,
                "seed": success.get("seed"),
                "token_budget": success.get("token_budget"),
                "success_policy": success_policy,
                "baseline_policy": baseline_policy,
                "baseline_control_name": baseline.get("control_name"),
                "delta_cycle_quality_cost_auc": auc_delta,
                "delta_heldout_metric": heldout_delta,
                "delta_final_metric": final_delta,
                "auc_win": _win_flag(auc_delta),
                "heldout_win": _win_flag(heldout_delta),
                "final_win": _win_flag(final_delta),
            }
        )
    return sorted(
        delta_rows,
        key=lambda row: (
            str(row["dataset"]),
            str(row["token_budget"]),
            str(row["seed"]),
            str(row["baseline_policy"]),
            str(row["baseline_control_name"]),
        ),
    )


def _paper_pairwise_summary_key(row: Dict[str, Any]) -> tuple[Any, ...]:
    """Group paired deltas by axes used for statistical summaries."""
    return (
        row.get("dataset"),
        row.get("dataset_subset"),
        row.get("student_model"),
        row.get("student_type"),
        row.get("metric"),
        row.get("mode"),
        row.get("token_budget"),
        row.get("success_policy"),
        row.get("baseline_policy"),
        row.get("baseline_control_name"),
    )


def _exact_two_sided_sign_test_p(wins: int, losses: int) -> float | None:
    """Return an exact two-sided binomial sign-test p-value."""
    n = wins + losses
    if n == 0:
        return None
    tail = min(wins, losses)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, 2 * probability)


def _paired_delta_stats(
    rows: List[Dict[str, Any]],
    delta_key: str,
) -> Dict[str, Any]:
    """Summarize one paired delta column with ties separated from sign tests."""
    deltas = [
        converted
        for row in rows
        if (converted := _safe_float(row.get(delta_key))) is not None
    ]
    wins = sum(1 for value in deltas if value > 0)
    losses = sum(1 for value in deltas if value < 0)
    ties = sum(1 for value in deltas if value == 0)
    mean_delta, _ = _mean_std(deltas)
    return {
        "n": len(deltas),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / len(deltas) if deltas else None,
        "mean_delta": mean_delta,
        "sign_test_p": _exact_two_sided_sign_test_p(wins, losses),
    }


def summarize_paper_pairwise_summary(
    delta_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate paired deltas into reviewer-facing win-rate summaries."""
    grouped: Dict[tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        grouped[_paper_pairwise_summary_key(row)].append(row)

    summary_rows = []
    for key, group in sorted(
        grouped.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        (
            dataset,
            dataset_subset,
            student_model,
            student_type,
            metric,
            mode,
            token_budget,
            success_policy,
            baseline_policy,
            baseline_control_name,
        ) = key
        auc = _paired_delta_stats(group, "delta_cycle_quality_cost_auc")
        heldout = _paired_delta_stats(group, "delta_heldout_metric")
        final = _paired_delta_stats(group, "delta_final_metric")
        summary_rows.append(
            {
                "dataset": dataset,
                "dataset_subset": dataset_subset,
                "student_model": student_model,
                "student_type": student_type,
                "metric": metric,
                "mode": mode,
                "token_budget": token_budget,
                "success_policy": success_policy,
                "baseline_policy": baseline_policy,
                "baseline_control_name": baseline_control_name,
                "auc_n": auc["n"],
                "auc_wins": auc["wins"],
                "auc_losses": auc["losses"],
                "auc_ties": auc["ties"],
                "auc_win_rate": auc["win_rate"],
                "auc_mean_delta": auc["mean_delta"],
                "auc_sign_test_p": auc["sign_test_p"],
                "heldout_n": heldout["n"],
                "heldout_wins": heldout["wins"],
                "heldout_losses": heldout["losses"],
                "heldout_ties": heldout["ties"],
                "heldout_win_rate": heldout["win_rate"],
                "heldout_mean_delta": heldout["mean_delta"],
                "heldout_sign_test_p": heldout["sign_test_p"],
                "final_n": final["n"],
                "final_wins": final["wins"],
                "final_losses": final["losses"],
                "final_ties": final["ties"],
                "final_win_rate": final["win_rate"],
                "final_mean_delta": final["mean_delta"],
                "final_sign_test_p": final["sign_test_p"],
            }
        )
    return summary_rows


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    """Load the copied experiment config for one completed run."""
    if not (run_dir / "experiment_config.yaml").exists():
        return {}
    try:
        import yaml

        with (run_dir / "experiment_config.yaml").open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _metric_token_at_eval(
    values_for_cycle: Dict[str, Any],
    cumulative_usage: Dict[str, int],
    key: str,
    metric_key: str,
) -> float:
    value = _safe_float(values_for_cycle.get(metric_key))
    if value is not None:
        return value
    return float(cumulative_usage.get(key, 0) or 0)


def summarize_paper_quality_cost_points(
    run_dirs: List[Path],
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return per-cycle quality-cost points for paper figures."""
    summary_by_dir = {str(row.get("run_dir")): row for row in rows}
    point_rows = []
    for run_dir in run_dirs:
        summary = summary_by_dir.get(str(run_dir))
        if not summary:
            continue
        metric_name = str(summary.get("metric") or "")
        if not metric_name:
            continue
        metrics = _load_json(run_dir / "metrics.json")
        token_usage = _load_json(run_dir / "token_usage.json")
        config = _load_run_config(run_dir)
        seed_summary = _seed_materialization_summary(run_dir, config, token_usage)
        seed_total = _safe_int(seed_summary.get("seed_teacher_total_tokens")) or 0
        charged_seed = _charged_seed_usage(token_usage)
        cumulative_by_cycle = _cycle_token_cumulative_usage(token_usage)
        context = _run_context(run_dir)
        final_cycle = _safe_int(summary.get("final_cycle"))

        def make_point(
            *,
            split: str,
            cycle: int | None,
            metric_value: Any,
            cumulative_usage: Dict[str, Any],
        ) -> Dict[str, Any]:
            online_input = max(
                0,
                int(cumulative_usage.get("input_tokens", 0) or 0)
                - charged_seed["input_tokens"],
            )
            online_output = max(
                0,
                int(cumulative_usage.get("output_tokens", 0) or 0)
                - charged_seed["output_tokens"],
            )
            online_total = max(
                0,
                int(cumulative_usage.get("total_tokens", 0) or 0)
                - charged_seed["total_tokens"],
            )
            return {
                **context,
                "metric": metric_name,
                "mode": summary.get("mode"),
                "split": split,
                "cycle": cycle,
                "is_final_cycle": cycle == final_cycle,
                "metric_value": metric_value,
                "cumulative_online_teacher_input_tokens": online_input,
                "cumulative_online_teacher_output_tokens": online_output,
                "cumulative_online_teacher_total_tokens": online_total,
                "seed_teacher_total_tokens": seed_total,
                "cumulative_total_teacher_total_tokens": seed_total + online_total,
            }

        for cycle, values_for_cycle in _cycle_metrics(metrics):
            if metric_name not in values_for_cycle:
                continue
            cumulative_usage = cumulative_by_cycle.get(cycle, {})
            metric_usage = {
                "input_tokens": _metric_token_at_eval(
                    values_for_cycle,
                    cumulative_usage,
                    "input_tokens",
                    "_teacher_input_tokens_at_eval",
                ),
                "output_tokens": _metric_token_at_eval(
                    values_for_cycle,
                    cumulative_usage,
                    "output_tokens",
                    "_teacher_output_tokens_at_eval",
                ),
                "total_tokens": _metric_token_at_eval(
                    values_for_cycle,
                    cumulative_usage,
                    "total_tokens",
                    "_teacher_tokens_at_eval",
                ),
            }
            point_rows.append(
                make_point(
                    split=str(summary.get("selection_split") or "validation"),
                    cycle=cycle,
                    metric_value=values_for_cycle.get(metric_name),
                    cumulative_usage=metric_usage,
                )
            )

        heldout = _heldout_test_metrics(metrics)
        heldout_value = heldout.get(metric_name)
        if heldout_value is not None:
            grand_total = token_usage.get("grand_total", {})
            point_rows.append(
                make_point(
                    split=str(
                        heldout.get("_heldout_split")
                        or summary.get("heldout_split")
                        or "heldout"
                    ),
                    cycle=final_cycle,
                    metric_value=heldout_value,
                    cumulative_usage=grand_total,
                )
            )

    return sorted(
        point_rows,
        key=lambda row: (
            str(row.get("dataset")),
            str(row.get("token_budget")),
            str(row.get("seed")),
            str(row.get("policy_name")),
            str(row.get("control_name")),
            _safe_int(row.get("cycle")) if row.get("cycle") is not None else -1,
            0 if str(row.get("split") or "") in {"validation", "dev"} else 1,
            str(row.get("split")),
        ),
    )


def summarize_paper_action_frequencies(
    policy_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate policy decision logs for action-frequency figures."""
    group_totals: Dict[tuple[Any, ...], int] = defaultdict(int)
    counts: Dict[tuple[Any, ...], int] = defaultdict(int)
    for row in policy_rows:
        group_key = (
            row.get("dataset"),
            row.get("dataset_subset"),
            row.get("student_model"),
            row.get("student_type"),
            row.get("token_budget"),
            row.get("policy_name"),
            row.get("control_name"),
        )
        action_key = group_key + (
            row.get("action_name"),
            row.get("prompt_operator"),
            row.get("teacher_tier"),
            row.get("batch_size"),
        )
        group_totals[group_key] += 1
        counts[action_key] += 1

    frequency_rows = []
    for key, count in sorted(counts.items(), key=lambda item: tuple(map(str, item[0]))):
        group_key = key[:7]
        total = group_totals[group_key]
        frequency_rows.append(
            {
                "dataset": key[0],
                "dataset_subset": key[1],
                "student_model": key[2],
                "student_type": key[3],
                "token_budget": key[4],
                "policy_name": key[5],
                "control_name": key[6],
                "action_name": key[7],
                "prompt_operator": key[8],
                "teacher_tier": key[9],
                "batch_size": key[10],
                "count": count,
                "share": count / total if total else None,
            }
        )
    return frequency_rows


def summarize_paper_budget_audit(
    rows: List[Dict[str, Any]],
    calibration_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate budget-accounting checks for paper audit tables."""
    calibration_by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in calibration_rows:
        calibration_by_run[str(row.get("run_id") or "")].append(row)

    grouped: Dict[tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_paper_result_key(row)].append(row)

    audit_rows = []
    for key, group in sorted(
        grouped.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        (
            dataset,
            dataset_subset,
            student_model,
            student_type,
            metric,
            mode,
            token_budget,
            policy_name,
            control_name,
        ) = key
        group_calibration = [
            attempt
            for row in group
            for attempt in calibration_by_run.get(str(row.get("run_id") or ""), [])
        ]
        overage_values = [
            row.get("token_budget_overage")
            for row in group
            if _safe_int(row.get("token_budget_overage")) is not None
        ]
        realized_ratios = [
            attempt.get("realized_over_predicted") for attempt in group_calibration
        ]
        online_mean, _ = _mean_std(
            row.get("online_teacher_total_tokens") for row in group
        )
        total_mean, _ = _mean_std(
            row.get("total_teacher_total_tokens") for row in group
        )
        audit_rows.append(
            {
                "dataset": dataset,
                "dataset_subset": dataset_subset,
                "student_model": student_model,
                "student_type": student_type,
                "metric": metric,
                "mode": mode,
                "token_budget": token_budget,
                "policy_name": policy_name,
                "control_name": control_name,
                "run_count": len(group),
                "max_token_budget_overage": max(overage_values, default=0),
                "max_realized_over_predicted": max(
                    (
                        value
                        for value in (_safe_float(ratio) for ratio in realized_ratios)
                        if value is not None
                    ),
                    default=None,
                ),
                "missing_provider_usage_rows": sum(
                    1
                    for attempt in group_calibration
                    if not _truthy(attempt.get("provider_reported_present"))
                ),
                "over_preflight_bound_rows": sum(
                    1
                    for attempt in group_calibration
                    if _truthy(attempt.get("over_preflight_bound"))
                ),
                "over_remaining_budget_rows": sum(
                    1
                    for attempt in group_calibration
                    if _truthy(attempt.get("over_remaining_budget"))
                ),
                "failed_attempt_rows": sum(
                    1
                    for attempt in group_calibration
                    if str(attempt.get("status") or "") not in {"", "success"}
                ),
                "mean_online_teacher_total_tokens": online_mean,
                "mean_total_teacher_total_tokens": total_mean,
            }
        )
    return audit_rows


def _win_rate(rows: List[Dict[str, Any]], key: str) -> float | None:
    present = [row[key] for row in rows if row.get(key) is not None]
    if not present:
        return None
    return sum(1 for value in present if value) / len(present)


def _write_paper_report_markdown(
    *,
    source_paths: List[Path],
    output_path: Path,
    rows: List[Dict[str, Any]],
    main_rows: List[Dict[str, Any]],
    delta_rows: List[Dict[str, Any]],
    delta_summary_rows: List[Dict[str, Any]],
    budget_rows: List[Dict[str, Any]],
    point_rows: List[Dict[str, Any]],
    paths: Dict[str, Path],
    success_policy: str,
    baseline_policies: List[str],
) -> None:
    """Write a compact report pointing reviewers to generated tables."""
    policies = sorted({str(row.get("policy_name") or "") for row in rows})
    datasets = sorted({str(row.get("dataset") or "") for row in rows})
    budgets = sorted({str(row.get("token_budget") or "") for row in rows})
    auc_win_rate = _win_rate(delta_rows, "auc_win")
    heldout_win_rate = _win_rate(delta_rows, "heldout_win")
    max_overage = max(
        (_safe_int(row.get("max_token_budget_overage")) or 0 for row in budget_rows),
        default=0,
    )
    missing_usage = sum(
        _safe_int(row.get("missing_provider_usage_rows")) or 0 for row in budget_rows
    )
    over_preflight = sum(
        _safe_int(row.get("over_preflight_bound_rows")) or 0 for row in budget_rows
    )
    lines = [
        "# Paper Readiness Report",
        "",
        "Source run roots: " + ", ".join(f"`{path}`" for path in source_paths),
        "",
        "## Coverage",
        "",
        f"- Runs summarized: {len(rows)}",
        f"- Main-result rows: {len(main_rows)}",
        f"- Datasets: {', '.join(datasets) if datasets else 'none'}",
        f"- Budgets: {', '.join(budgets) if budgets else 'none'}",
        f"- Policies: {', '.join(policies) if policies else 'none'}",
        "",
        "## Paired Success Screen",
        "",
        f"- Success policy: `{success_policy}`",
        f"- Baselines: {', '.join(f'`{name}`' for name in baseline_policies)}",
        f"- Paired comparisons found: {len(delta_rows)}",
        f"- Pairwise summary rows: {len(delta_summary_rows)}",
        f"- Quality-cost points: {len(point_rows)}",
        f"- AUC win rate: {auc_win_rate if auc_win_rate is not None else 'n/a'}",
        (
            f"- Held-out win rate: {heldout_win_rate}"
            if heldout_win_rate is not None
            else "- Held-out win rate: n/a"
        ),
        "",
        "## Budget Audit Screen",
        "",
        f"- Max token-budget overage: {max_overage}",
        f"- Missing provider-usage attempt rows: {missing_usage}",
        f"- Over-preflight attempt rows: {over_preflight}",
        "",
        "## Generated Tables",
        "",
    ]
    for name, path in sorted(paths.items()):
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## Reviewer Use",
            "",
            "Use `paper_main_results.csv` for the main table, "
            "`paper_pairwise_deltas.csv` for paired seed/budget deltas, "
            "`paper_pairwise_summary.csv` for win rates and sign tests, "
            "`paper_quality_cost_points.csv` for quality-cost curves, "
            "`paper_budget_audit.csv` for the accounting table, and "
            "`paper_action_frequencies.csv` for policy-behavior figures.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_paper_report(
    path: Path | Sequence[Path],
    rows: List[Dict[str, Any]],
    output_dir: Path,
    *,
    success_policy: str = "frugalkd_p",
    baseline_policies: List[str] | None = None,
) -> Dict[str, Path]:
    """Write paper-facing tables and a compact readiness report."""
    baseline_policies = baseline_policies or ["cost_heuristic", "random_feasible"]
    source_paths = _coerce_paths(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.csv"
    write_summary_csv(rows, summary_path)
    audit_paths = write_audit_csvs(source_paths, rows, output_dir / "audit")

    run_dirs = _run_dirs_for_paths(source_paths)
    policy_rows = [
        row for run_dir in run_dirs for row in summarize_policy_behavior(run_dir)
    ]
    calibration_rows = [
        row for run_dir in run_dirs for row in summarize_teacher_calibration(run_dir)
    ]
    main_rows = summarize_paper_main_results(rows)
    delta_rows = summarize_paper_pairwise_deltas(
        rows,
        success_policy=success_policy,
        baseline_policies=baseline_policies,
    )
    delta_summary_rows = summarize_paper_pairwise_summary(delta_rows)
    point_rows = summarize_paper_quality_cost_points(run_dirs, rows)
    action_rows = summarize_paper_action_frequencies(policy_rows)
    budget_rows = summarize_paper_budget_audit(rows, calibration_rows)

    paths = {
        "summary": summary_path,
        "audit_policy_actions": audit_paths["policy_actions"],
        "audit_teacher_calibration": audit_paths["teacher_calibration"],
        "audit_oracle_frontier": audit_paths["oracle_frontier"],
        "paper_main_results": output_dir / "paper_main_results.csv",
        "paper_pairwise_deltas": output_dir / "paper_pairwise_deltas.csv",
        "paper_pairwise_summary": output_dir / "paper_pairwise_summary.csv",
        "paper_quality_cost_points": output_dir / "paper_quality_cost_points.csv",
        "paper_action_frequencies": output_dir / "paper_action_frequencies.csv",
        "paper_budget_audit": output_dir / "paper_budget_audit.csv",
        "paper_readiness_report": output_dir / "paper_readiness_report.md",
    }
    _write_rows_csv(
        main_rows,
        paths["paper_main_results"],
        PAPER_MAIN_RESULT_FIELDS,
    )
    _write_rows_csv(
        delta_rows,
        paths["paper_pairwise_deltas"],
        PAPER_PAIRWISE_DELTA_FIELDS,
    )
    _write_rows_csv(
        delta_summary_rows,
        paths["paper_pairwise_summary"],
        PAPER_PAIRWISE_SUMMARY_FIELDS,
    )
    _write_rows_csv(
        point_rows,
        paths["paper_quality_cost_points"],
        PAPER_QUALITY_COST_POINT_FIELDS,
    )
    _write_rows_csv(
        action_rows,
        paths["paper_action_frequencies"],
        PAPER_ACTION_FREQUENCY_FIELDS,
    )
    _write_rows_csv(
        budget_rows,
        paths["paper_budget_audit"],
        PAPER_BUDGET_AUDIT_FIELDS,
    )
    _write_paper_report_markdown(
        source_paths=source_paths,
        output_path=paths["paper_readiness_report"],
        rows=rows,
        main_rows=main_rows,
        delta_rows=delta_rows,
        delta_summary_rows=delta_summary_rows,
        budget_rows=budget_rows,
        point_rows=point_rows,
        paths=paths,
        success_policy=success_policy,
        baseline_policies=baseline_policies,
    )
    return paths


def _normalize_set(values: Iterable[Any]) -> set[str]:
    """Normalize optional CLI/config values for set comparisons."""
    return {str(value) for value in values}


def _coerce_scalar(value: Any) -> Any:
    """Coerce analysis strings back to simple YAML scalars where possible."""
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def _same_count_config_fingerprint(config: Dict[str, Any]) -> str:
    """Hash the config surface that same-count controls must keep identical."""
    comparable = {
        key: value
        for key, value in config.items()
        if key not in SAME_COUNT_CONFIG_IGNORED_KEYS
    }
    payload = json.dumps(comparable, sort_keys=True, default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def _safe_slug(value: Any) -> str:
    """Return a conservative filename slug for generated config paths."""
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value)
    ).strip("_")


def _sort_scalar(value: Any) -> tuple[int, Any]:
    """Sort numeric-looking values numerically and everything else lexically."""
    coerced = _coerce_scalar(value)
    if isinstance(coerced, int):
        return (0, coerced)
    return (1, str(coerced))


def plan_same_count_control_configs(
    pilot_dir: Path,
    base_config_path: Path,
    output_dir: Path,
    *,
    metric: str | None = None,
    mode: str = "auto",
    source_policy: str = "frugalkd_p",
    control_policy: str = "cost_heuristic",
    control_base_output_dir: str | None = None,
) -> List[Dict[str, Any]]:
    """Write matched same_count control configs from completed source runs."""
    import yaml

    rows = analyze_runs(pilot_dir, metric=metric, mode=mode)
    source_rows = [
        row
        for row in rows
        if str(row.get("policy_name") or "") == source_policy
        and not _is_same_count_control(row)
    ]
    if not source_rows:
        raise ValueError(f"no source policy rows found for {source_policy!r}")

    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) or {}
    output_dir.mkdir(parents=True, exist_ok=True)

    planned = []
    seen_pairs: set[tuple[str, str]] = set()
    for row in sorted(
        source_rows,
        key=lambda item: (
            _sort_scalar(item.get("seed")),
            _sort_scalar(item.get("token_budget")),
            str(item.get("run_id")),
        ),
    ):
        seed = str(row.get("seed"))
        token_budget = str(row.get("token_budget"))
        pair = (seed, token_budget)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        synthetic_count = _safe_int(row.get("final_synthetic_count"))
        if synthetic_count is None:
            raise ValueError(
                "source run is missing final_synthetic_count: "
                f"run_id={row.get('run_id')}"
            )

        source_config_path = Path(str(row.get("run_dir"))) / "experiment_config.yaml"
        if source_config_path.exists():
            config_data = (
                yaml.safe_load(source_config_path.read_text(encoding="utf-8")) or {}
            )
        else:
            config_data = dict(base_config)
        config_name = (
            f"{config_data.get('name', 'experiment')}_same_count_"
            f"{_safe_slug(control_policy)}_s{_safe_slug(seed)}_b"
            f"{_safe_slug(token_budget)}"
        )
        config_data.update(
            {
                "name": config_name,
                "policy_name": control_policy,
                "control_name": "same_count",
                "seed": _coerce_scalar(seed),
                "token_budget": _coerce_scalar(token_budget),
                "synthetic_record_budget": synthetic_count,
            }
        )
        if control_base_output_dir:
            config_data["base_output_dir"] = control_base_output_dir

        config_path = output_dir / f"{config_name}.yaml"
        config_path.write_text(
            yaml.safe_dump(config_data, sort_keys=False),
            encoding="utf-8",
        )
        planned.append(
            {
                "config_path": str(config_path),
                "source_run_id": row.get("run_id"),
                "source_policy_name": source_policy,
                "control_policy_name": control_policy,
                "seed": seed,
                "token_budget": token_budget,
                "synthetic_record_budget": synthetic_count,
                "source_final_metric": row.get("final_metric"),
            }
        )

    (output_dir / "same_count_plan.json").write_text(
        json.dumps(planned, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return planned


def _is_same_count_control(row: Dict[str, Any]) -> bool:
    """Return whether an analysis row is tagged as a same-count control."""
    control_name = str(row.get("control_name") or "")
    policy_name = str(row.get("policy_name") or "")
    return (
        control_name in SAME_COUNT_CONTROL_NAMES
        or policy_name in SAME_COUNT_CONTROL_NAMES
    )


def _gate_check(name: str, passed: bool, **details: Any) -> Dict[str, Any]:
    """Build one machine-readable pilot gate check."""
    return {"name": name, "passed": passed, **details}


SUCCESS_MATCH_FIELDS = (
    "dataset",
    "dataset_subset",
    "student_model",
    "student_type",
    "seed",
    "token_budget",
    "selection_split",
    "metric",
    "mode",
)


def _success_match_key(row: Dict[str, Any]) -> tuple[str, ...]:
    """Return the paired-comparison key for scientific-success checks."""
    return tuple(str(row.get(field) or "") for field in SUCCESS_MATCH_FIELDS)


def _comparison_delta(
    policy_value: float | None,
    baseline_value: float | None,
    mode: str,
) -> float | None:
    """Return positive values when the policy beats the baseline."""
    if policy_value is None or baseline_value is None:
        return None
    return (
        baseline_value - policy_value
        if mode == "min"
        else policy_value - baseline_value
    )


def _paired_success_check(
    *,
    rows: List[Dict[str, Any]],
    name: str,
    success_policy: str,
    baseline_rows: List[Dict[str, Any]],
    value_key: str,
    min_win_rate: float,
    min_delta: float,
) -> Dict[str, Any]:
    """Compare one policy against matched baselines for one metric column."""
    success_rows = [
        row
        for row in rows
        if str(row.get("policy_name") or "") == success_policy
        and not _is_same_count_control(row)
    ]
    baseline_names = sorted(
        {
            str(row.get("policy_name") or row.get("control_name") or "")
            for row in baseline_rows
        }
    )
    baselines_by_name_and_key: Dict[
        str, Dict[tuple[str, ...], List[Dict[str, Any]]]
    ] = {}
    for row in baseline_rows:
        baseline_name = str(row.get("policy_name") or row.get("control_name") or "")
        baselines_by_name_and_key.setdefault(baseline_name, {}).setdefault(
            _success_match_key(row), []
        ).append(row)

    comparisons = []
    missing_pairs = []
    per_baseline = []
    threshold = max(min_delta, 0.0)
    for baseline_name in baseline_names:
        matched_for_baseline = baselines_by_name_and_key.get(baseline_name, {})
        baseline_comparisons = []
        baseline_missing = []
        for policy_row in success_rows:
            key = _success_match_key(policy_row)
            matched = matched_for_baseline.get(key, [])
            if not matched:
                baseline_missing.append(
                    {
                        "baseline": baseline_name,
                        "run_id": policy_row.get("run_id"),
                        "key": list(key),
                    }
                )
                continue
            for baseline_row in matched:
                delta = _comparison_delta(
                    _safe_float(policy_row.get(value_key)),
                    _safe_float(baseline_row.get(value_key)),
                    str(policy_row.get("mode") or "max"),
                )
                comparison = {
                    "policy_run_id": policy_row.get("run_id"),
                    "baseline_run_id": baseline_row.get("run_id"),
                    "baseline_policy": baseline_row.get("policy_name"),
                    "baseline_control": baseline_row.get("control_name"),
                    "seed": policy_row.get("seed"),
                    "token_budget": policy_row.get("token_budget"),
                    "delta": delta,
                    "won": delta is not None and delta > threshold,
                }
                baseline_comparisons.append(comparison)
                comparisons.append(comparison)

        valid_for_baseline = [
            row for row in baseline_comparisons if row["delta"] is not None
        ]
        wins_for_baseline = sum(1 for row in valid_for_baseline if row["won"])
        win_rate_for_baseline = (
            wins_for_baseline / len(valid_for_baseline) if valid_for_baseline else 0.0
        )
        per_baseline.append(
            {
                "baseline": baseline_name,
                "comparison_count": len(valid_for_baseline),
                "win_count": wins_for_baseline,
                "win_rate": win_rate_for_baseline,
                "passed": bool(valid_for_baseline)
                and not baseline_missing
                and win_rate_for_baseline >= min_win_rate,
                "missing_pairs": baseline_missing,
            }
        )
        missing_pairs.extend(baseline_missing)

    valid = [row for row in comparisons if row["delta"] is not None]
    win_count = sum(1 for row in valid if row["won"])
    win_rate = (win_count / len(valid)) if valid else 0.0
    return _gate_check(
        name,
        bool(valid)
        and bool(per_baseline)
        and all(row["passed"] for row in per_baseline),
        success_policy=success_policy,
        value_key=value_key,
        min_win_rate=min_win_rate,
        min_delta=min_delta,
        comparison_count=len(valid),
        win_count=win_count,
        win_rate=win_rate,
        per_baseline=per_baseline,
        missing_pairs=missing_pairs,
        failures=[row for row in comparisons if row["delta"] is None or not row["won"]],
    )


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
    require_heldout: bool = False,
    require_full_label_coverage: bool = False,
    require_same_count_control: bool = False,
    same_count_source_policy: str = "frugalkd_p",
    same_count_control_policy: str | None = None,
    require_paper_mode: bool = False,
    require_provider_reported_usage: bool = False,
    success_policy: str | None = None,
    success_baselines: List[str] | None = None,
    min_auc_win_rate: float | None = None,
    min_heldout_win_rate: float | None = None,
    min_final_win_rate: float | None = None,
    min_auc_delta: float = 0.0,
    min_heldout_delta: float = 0.0,
    min_final_delta: float = 0.0,
    require_same_count_success: bool = False,
) -> Dict[str, Any]:
    """Return a pass/fail report for the cheap policy-axis pilot."""
    rows = analyze_runs(path, metric=metric, mode=mode)
    run_dirs = _run_dirs(path)
    policy_rows = [
        row for run_dir in run_dirs for row in summarize_policy_behavior(run_dir)
    ]
    calibration_rows = [
        row for run_dir in run_dirs for row in summarize_teacher_calibration(run_dir)
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
        combo_rows: Dict[tuple[str, str, str], List[Dict[str, Any]]] = {}
        for row in rows:
            if _is_same_count_control(row):
                continue
            combo = (
                str(row.get("policy_name")),
                str(row.get("seed")),
                str(row.get("token_budget")),
            )
            combo_rows.setdefault(combo, []).append(row)
        found_combos = set(combo_rows)
        duplicate_combos = [
            {
                "combo": list(combo),
                "run_ids": [row.get("run_id") for row in combo_group],
            }
            for combo, combo_group in sorted(combo_rows.items())
            if len(combo_group) > 1
        ]
        checks.append(
            _gate_check(
                "expected_policy_seed_budget_grid",
                expected_combos == found_combos and not duplicate_combos,
                expected_count=len(expected_combos),
                found_count=sum(len(group) for group in combo_rows.values()),
                missing=[
                    list(combo) for combo in sorted(expected_combos - found_combos)
                ],
                unexpected=[
                    list(combo) for combo in sorted(found_combos - expected_combos)
                ],
                duplicates=duplicate_combos,
            )
        )

    action_space_failures = []
    rows_by_match: Dict[tuple[str, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        if _is_same_count_control(row):
            continue
        rows_by_match.setdefault(_success_match_key(row), []).append(row)
    for key, group in rows_by_match.items():
        action_space_ids = sorted(
            {str(row.get("action_space_id") or "") for row in group}
        )
        if "" in action_space_ids or len(action_space_ids) != 1:
            action_space_failures.append(
                {
                    "key": list(key),
                    "action_space_ids": action_space_ids,
                    "run_ids": [row.get("run_id") for row in group],
                }
            )
    checks.append(
        _gate_check(
            "action_space_parity",
            not action_space_failures,
            failures=action_space_failures,
        )
    )

    if require_paper_mode:
        non_completed_rows = [
            row for row in rows if str(row.get("run_status") or "") != "completed"
        ]
        checks.append(
            _gate_check(
                "paper_runs_completed",
                not non_completed_rows,
                violating_run_ids=[row.get("run_id") for row in non_completed_rows],
            )
        )

        non_paper_rows = [row for row in rows if not _truthy(row.get("paper_mode"))]
        checks.append(
            _gate_check(
                "paper_mode_enabled",
                not non_paper_rows,
                violating_run_ids=[row.get("run_id") for row in non_paper_rows],
            )
        )

        cycle_failures = []
        for row in rows:
            cycles_completed = _safe_int(row.get("cycles_completed"))
            expected_cycles = _safe_int(row.get("expected_cycles"))
            if (
                cycles_completed is None
                or expected_cycles is None
                or cycles_completed <= 0
                or cycles_completed > expected_cycles
            ):
                cycle_failures.append(
                    {
                        "run_id": row.get("run_id"),
                        "cycles_completed": cycles_completed,
                        "expected_cycles": expected_cycles,
                    }
                )
        checks.append(
            _gate_check(
                "paper_cycle_bounds",
                not cycle_failures,
                failures=cycle_failures,
            )
        )

    if require_provider_reported_usage:
        estimated_seed_rows = [
            row
            for row in rows
            if (_safe_int(row.get("seed_usage_estimated_records")) or 0) > 0
        ]
        checks.append(
            _gate_check(
                "no_estimated_seed_usage",
                not estimated_seed_rows,
                violating_run_ids=[row.get("run_id") for row in estimated_seed_rows],
            )
        )

        failed_teacher_rows = [
            row
            for row in rows
            if (_safe_int(row.get("teacher_failure_count")) or 0) > 0
            or (_safe_int(row.get("teacher_budget_violation_count")) or 0) > 0
        ]
        checks.append(
            _gate_check(
                "teacher_attempts_successful",
                not failed_teacher_rows,
                violating_run_ids=[row.get("run_id") for row in failed_teacher_rows],
            )
        )

        dispatched_attempts = [
            row for row in calibration_rows if str(row.get("status") or "") != "masked"
        ]
        missing_provider_attempts = [
            row
            for row in dispatched_attempts
            if row.get("provider_reported_present") is not True
            or _safe_int(row.get("provider_reported_total_tokens")) is None
        ]
        non_provider_debits = [
            row
            for row in dispatched_attempts
            if str(row.get("ledger_debit_source") or "") != "provider_reported"
        ]
        checks.append(
            _gate_check(
                "provider_reported_usage_present",
                not missing_provider_attempts and not non_provider_debits,
                missing_provider_attempt_ids=[
                    row.get("attempt_id") for row in missing_provider_attempts
                ],
                non_provider_debit_attempt_ids=[
                    row.get("attempt_id") for row in non_provider_debits
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
            if row.get("policy_name") == policy_name
            and row.get("action_name") != "STOP"
        }
        if expected_operator not in operators:
            unexpected_operators = sorted(operators - {expected_operator})
            fixed_operator_failures.append(
                {
                    "policy_name": policy_name,
                    "expected_operator": expected_operator,
                    "found_operators": sorted(operators),
                    "unexpected_operators": unexpected_operators,
                }
            )
        elif operators != {expected_operator}:
            fixed_operator_failures.append(
                {
                    "policy_name": policy_name,
                    "expected_operator": expected_operator,
                    "found_operators": sorted(operators),
                    "unexpected_operators": sorted(operators - {expected_operator}),
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
        str(row.get("decision_id")) for row in policy_rows if row.get("decision_id")
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
    attempt_decision_ids = {
        str(row.get("decision_id"))
        for row in calibration_rows
        if row.get("decision_id")
    }
    acquisition_outcomes = {
        "augment",
        "augment_empty",
        "augment_failed",
        "budget_masked",
    }
    decisions_missing_attempts = [
        row
        for row in policy_rows
        if row.get("decision_id")
        and str(row.get("acquisition_outcome")) in acquisition_outcomes
        and str(row.get("decision_id")) not in attempt_decision_ids
    ]
    checks.append(
        _gate_check(
            "acquisition_decisions_have_teacher_attempts",
            not decisions_missing_attempts,
            missing_decision_ids=[
                row.get("decision_id") for row in decisions_missing_attempts
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

    if require_heldout:
        missing_heldout = [
            row
            for row in rows
            if str(row.get("heldout_split") or "") != "test"
            or _safe_float(row.get("heldout_metric")) is None
        ]
        checks.append(
            _gate_check(
                "heldout_test_metrics_present",
                not missing_heldout,
                violating_run_ids=[row.get("run_id") for row in missing_heldout],
            )
        )

    if require_full_label_coverage:
        missing_label_coverage = []
        for row in rows:
            canonical_count = _safe_int(row.get("canonical_label_count"))
            observed_count = _safe_int(row.get("observed_gold_label_count"))
            if canonical_count is None or observed_count is None:
                missing_label_coverage.append(
                    {
                        "run_id": row.get("run_id"),
                        "split": row.get("selection_split"),
                        "observed_gold_label_count": observed_count,
                        "canonical_label_count": canonical_count,
                    }
                )
            elif observed_count != canonical_count:
                missing_label_coverage.append(
                    {
                        "run_id": row.get("run_id"),
                        "split": row.get("selection_split"),
                        "observed_gold_label_count": observed_count,
                        "canonical_label_count": canonical_count,
                    }
                )

            heldout_canonical_count = _safe_int(
                row.get("heldout_canonical_label_count")
            )
            heldout_observed_count = _safe_int(
                row.get("heldout_observed_gold_label_count")
            )
            if row.get("heldout_split") and (
                heldout_canonical_count is None or heldout_observed_count is None
            ):
                missing_label_coverage.append(
                    {
                        "run_id": row.get("run_id"),
                        "split": row.get("heldout_split"),
                        "observed_gold_label_count": heldout_observed_count,
                        "canonical_label_count": heldout_canonical_count,
                    }
                )
            elif (
                heldout_canonical_count is not None
                and heldout_observed_count != heldout_canonical_count
            ):
                missing_label_coverage.append(
                    {
                        "run_id": row.get("run_id"),
                        "split": row.get("heldout_split"),
                        "observed_gold_label_count": heldout_observed_count,
                        "canonical_label_count": heldout_canonical_count,
                    }
                )

        checks.append(
            _gate_check(
                "full_canonical_label_coverage",
                not missing_label_coverage,
                failures=missing_label_coverage,
            )
        )

    baseline_set = set(success_baselines or [])
    if success_policy and baseline_set:
        non_control_rows = [row for row in rows if not _is_same_count_control(row)]
        baseline_rows = [
            row
            for row in non_control_rows
            if str(row.get("policy_name") or "") in baseline_set
        ]
        if min_auc_win_rate is not None:
            checks.append(
                _paired_success_check(
                    rows=non_control_rows,
                    name="success_policy_beats_baselines_auc",
                    success_policy=success_policy,
                    baseline_rows=baseline_rows,
                    value_key="cycle_quality_cost_auc",
                    min_win_rate=min_auc_win_rate,
                    min_delta=min_auc_delta,
                )
            )
        if min_heldout_win_rate is not None:
            checks.append(
                _paired_success_check(
                    rows=non_control_rows,
                    name="success_policy_beats_baselines_heldout",
                    success_policy=success_policy,
                    baseline_rows=baseline_rows,
                    value_key="heldout_metric",
                    min_win_rate=min_heldout_win_rate,
                    min_delta=min_heldout_delta,
                )
            )
        if min_final_win_rate is not None:
            checks.append(
                _paired_success_check(
                    rows=non_control_rows,
                    name="success_policy_beats_baselines_final",
                    success_policy=success_policy,
                    baseline_rows=baseline_rows,
                    value_key="final_metric",
                    min_win_rate=min_final_win_rate,
                    min_delta=min_final_delta,
                )
            )

    if require_same_count_control:
        same_count_rows = [row for row in rows if _is_same_count_control(row)]
        non_control_rows = [row for row in rows if not _is_same_count_control(row)]
        target_seeds = expected_seed_set or _normalize_set(
            row.get("seed") for row in non_control_rows
        )
        target_budgets = expected_budget_set or _normalize_set(
            row.get("token_budget") for row in non_control_rows
        )
        target_pairs = {
            (str(seed), str(budget))
            for seed in target_seeds
            for budget in target_budgets
        }
        found_pairs = {
            (str(row.get("seed")), str(row.get("token_budget")))
            for row in same_count_rows
        }
        source_counts = {}
        source_action_spaces = {}
        source_config_hashes = {}
        for row in non_control_rows:
            if str(row.get("policy_name") or "") != same_count_source_policy:
                continue
            pair = (str(row.get("seed")), str(row.get("token_budget")))
            source_counts.setdefault(pair, _safe_int(row.get("final_synthetic_count")))
            source_action_spaces.setdefault(pair, str(row.get("action_space_id") or ""))
            source_config_hashes.setdefault(
                pair, str(row.get("same_count_config_hash") or "")
            )

        malformed_controls = []
        wrong_policy_controls = []
        action_space_mismatches = []
        config_mismatches = []
        same_count_rows_by_pair: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for row in same_count_rows:
            pair = (str(row.get("seed")), str(row.get("token_budget")))
            same_count_rows_by_pair.setdefault(pair, []).append(row)
            final_count = _safe_int(row.get("final_synthetic_count"))
            target_count = _safe_int(row.get("synthetic_record_budget"))
            source_count = source_counts.get(pair)
            source_action_space = source_action_spaces.get(pair, "")
            control_action_space = str(row.get("action_space_id") or "")
            source_config_hash = source_config_hashes.get(pair, "")
            control_config_hash = str(row.get("same_count_config_hash") or "")
            if (
                same_count_control_policy
                and str(row.get("policy_name") or "") != same_count_control_policy
            ):
                wrong_policy_controls.append(
                    {
                        "run_id": row.get("run_id"),
                        "policy_name": row.get("policy_name"),
                        "expected_policy_name": same_count_control_policy,
                    }
                )
            if (
                target_count is None
                or final_count != target_count
                or source_count is None
                or target_count != source_count
            ):
                malformed_controls.append(
                    {
                        "run_id": row.get("run_id"),
                        "seed": pair[0],
                        "token_budget": pair[1],
                        "final_synthetic_count": final_count,
                        "synthetic_record_budget": target_count,
                        "source_policy": same_count_source_policy,
                        "source_final_synthetic_count": source_count,
                    }
                )
            if (
                source_action_space or control_action_space
            ) and control_action_space != source_action_space:
                action_space_mismatches.append(
                    {
                        "run_id": row.get("run_id"),
                        "seed": pair[0],
                        "token_budget": pair[1],
                        "action_space_id": control_action_space,
                        "source_policy": same_count_source_policy,
                        "source_action_space_id": source_action_space,
                    }
                )
            if (
                source_config_hash or control_config_hash
            ) and control_config_hash != source_config_hash:
                config_mismatches.append(
                    {
                        "run_id": row.get("run_id"),
                        "seed": pair[0],
                        "token_budget": pair[1],
                        "config_hash": control_config_hash,
                        "source_policy": same_count_source_policy,
                        "source_config_hash": source_config_hash,
                    }
                )
        missing_source_pairs = target_pairs.difference(source_counts.keys())
        duplicate_controls = [
            {
                "seed": pair[0],
                "token_budget": pair[1],
                "run_ids": [row.get("run_id") for row in group],
            }
            for pair, group in sorted(same_count_rows_by_pair.items())
            if len(group) > 1
        ]
        checks.append(
            _gate_check(
                "same_count_controls_present",
                target_pairs.issubset(found_pairs)
                and not missing_source_pairs
                and not wrong_policy_controls
                and not malformed_controls
                and not action_space_mismatches
                and not config_mismatches
                and not duplicate_controls,
                source_policy=same_count_source_policy,
                control_policy=same_count_control_policy,
                expected_pairs=[list(pair) for pair in sorted(target_pairs)],
                found_pairs=[list(pair) for pair in sorted(found_pairs)],
                missing_pairs=[
                    list(pair) for pair in sorted(target_pairs - found_pairs)
                ],
                missing_source_pairs=[
                    list(pair) for pair in sorted(missing_source_pairs)
                ],
                malformed_controls=malformed_controls,
                wrong_policy_controls=wrong_policy_controls,
                action_space_mismatches=action_space_mismatches,
                config_mismatches=config_mismatches,
                duplicate_controls=duplicate_controls,
            )
        )
        if success_policy and require_same_count_success:
            if min_auc_win_rate is not None:
                checks.append(
                    _paired_success_check(
                        rows=rows,
                        name="success_policy_beats_same_count_auc",
                        success_policy=success_policy,
                        baseline_rows=same_count_rows,
                        value_key="cycle_quality_cost_auc",
                        min_win_rate=min_auc_win_rate,
                        min_delta=min_auc_delta,
                    )
                )
            if min_heldout_win_rate is not None:
                checks.append(
                    _paired_success_check(
                        rows=rows,
                        name="success_policy_beats_same_count_heldout",
                        success_policy=success_policy,
                        baseline_rows=same_count_rows,
                        value_key="heldout_metric",
                        min_win_rate=min_heldout_win_rate,
                        min_delta=min_heldout_delta,
                    )
                )
            if min_final_win_rate is not None:
                checks.append(
                    _paired_success_check(
                        rows=rows,
                        name="success_policy_beats_same_count_final",
                        success_policy=success_policy,
                        baseline_rows=same_count_rows,
                        value_key="final_metric",
                        min_win_rate=min_final_win_rate,
                        min_delta=min_final_delta,
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
