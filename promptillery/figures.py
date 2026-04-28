"""Publication figure helpers for paper-report outputs."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError(
            "paper figure generation requires matplotlib; run with "
            "`uv run --extra paper promptillery paper-figures ...`"
        ) from exc
    return plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_")
    return normalized.lower() or "plot"


def _task_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("dataset") or "dataset",
        row.get("dataset_subset") or "",
        row.get("metric") or "metric",
    )


def _task_label(key: tuple[str, str, str]) -> str:
    dataset, subset, metric = key
    task = f"{dataset}/{subset}" if subset else dataset
    return f"{task} ({metric})"


def _task_slug(key: tuple[str, str, str]) -> str:
    dataset, subset, metric = key
    task = "_".join(part for part in (dataset, subset, metric) if part)
    return _slug(task)


def _policy_label(row: dict[str, str]) -> str:
    control = row.get("control_name") or ""
    policy = row.get("policy_name") or row.get("baseline_policy") or "policy"
    return f"{policy}/{control}" if control else policy


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def _save(fig: Any, output_dir: Path, stem: str, fmt: str) -> Path:
    path = output_dir / f"{stem}.{fmt}"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return path


def _plot_quality_cost(
    rows: list[dict[str, str]],
    output_dir: Path,
    fmt: str,
) -> list[Path]:
    plt = _load_pyplot()
    by_task: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if _safe_float(row.get("metric_value")) is not None:
            by_task[_task_key(row)].append(row)

    paths: list[Path] = []
    for key, task_rows in sorted(by_task.items(), key=lambda item: item[0]):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        plotted = False
        by_policy: dict[tuple[str, str], dict[int, list[tuple[float, float]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        heldout: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
        for row in task_rows:
            x = _safe_float(row.get("cumulative_total_teacher_total_tokens"))
            y = _safe_float(row.get("metric_value"))
            if x is None or y is None:
                continue
            policy_key = (_policy_label(row), str(row.get("token_budget") or ""))
            if row.get("split") == "validation":
                cycle = int(_safe_float(row.get("cycle")) or 0)
                by_policy[policy_key][cycle].append((x, y))
            else:
                heldout[policy_key].append((x, y))

        for (label, budget), points_by_cycle in sorted(by_policy.items()):
            points: list[tuple[float, float]] = []
            for cycle in sorted(points_by_cycle):
                xs, ys = zip(*points_by_cycle[cycle])
                mean_x = _mean(xs)
                mean_y = _mean(ys)
                if mean_x is not None and mean_y is not None:
                    points.append((mean_x, mean_y))
            if not points:
                continue
            xs, ys = zip(*points)
            line = ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"{label} {budget}")
            plotted = True
            if heldout.get((label, budget)):
                heldout_x = _mean(x for x, _ in heldout[(label, budget)])
                heldout_y = _mean(y for _, y in heldout[(label, budget)])
                if heldout_x is not None and heldout_y is not None:
                    ax.scatter(
                        [heldout_x],
                        [heldout_y],
                        marker="x",
                        color=line[0].get_color(),
                        s=55,
                    )

        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(f"Quality-cost curve: {_task_label(key)}")
        ax.set_xlabel("Cumulative total teacher tokens")
        ax.set_ylabel(key[2])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        paths.append(_save(fig, output_dir, f"quality_cost_{_task_slug(key)}", fmt))
        plt.close(fig)
    return paths


def _plot_pairwise_summary(
    rows: list[dict[str, str]],
    output_dir: Path,
    fmt: str,
) -> list[Path]:
    plt = _load_pyplot()
    by_task: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("summary_scope") == "all_budgets":
            by_task[_task_key(row)].append(row)

    paths: list[Path] = []
    metrics = [
        ("auc", "AUC delta"),
        ("heldout", "Held-out delta"),
        ("final", "Final validation delta"),
    ]
    for key, task_rows in sorted(by_task.items(), key=lambda item: item[0]):
        labels = [
            (
                f"{row.get('baseline_policy') or 'baseline'}"
                + (
                    f"/{row.get('baseline_control_name')}"
                    if row.get("baseline_control_name")
                    else ""
                )
            )
            for row in task_rows
        ]
        if not labels:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
        plotted = False
        y_positions = list(range(len(labels)))
        for ax, (prefix, title) in zip(axes, metrics):
            values = [_safe_float(row.get(f"{prefix}_mean_delta")) for row in task_rows]
            lows = [
                _safe_float(row.get(f"{prefix}_mean_delta_ci_low")) for row in task_rows
            ]
            highs = [
                _safe_float(row.get(f"{prefix}_mean_delta_ci_high"))
                for row in task_rows
            ]
            xs: list[float] = []
            ys: list[int] = []
            err_low: list[float] = []
            err_high: list[float] = []
            for idx, value in enumerate(values):
                if value is None:
                    continue
                low = lows[idx] if lows[idx] is not None else value
                high = highs[idx] if highs[idx] is not None else value
                xs.append(value)
                ys.append(y_positions[idx])
                err_low.append(max(value - low, 0.0))
                err_high.append(max(high - value, 0.0))
            if xs:
                ax.errorbar(xs, ys, xerr=[err_low, err_high], fmt="o", capsize=3)
                plotted = True
            ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.45)
            ax.set_title(title)
            ax.grid(axis="x", alpha=0.25)
        axes[0].set_yticks(y_positions)
        axes[0].set_yticklabels(labels, fontsize=8)
        fig.suptitle(f"Paired FrugalKD-P deltas: {_task_label(key)}")
        if not plotted:
            plt.close(fig)
            continue
        paths.append(_save(fig, output_dir, f"paired_deltas_{_task_slug(key)}", fmt))
        plt.close(fig)
    return paths


def _plot_policy_behavior(
    rows: list[dict[str, str]],
    output_dir: Path,
    fmt: str,
) -> list[Path]:
    plt = _load_pyplot()
    by_task: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task[
            (
                row.get("dataset") or "dataset",
                row.get("dataset_subset") or "",
                "policy_behavior",
            )
        ].append(row)

    paths: list[Path] = []
    for key, task_rows in sorted(by_task.items(), key=lambda item: item[0]):
        policies = sorted({row.get("policy_name") or "policy" for row in task_rows})
        policies = policies[:6]
        if not policies:
            continue
        fig, axes = plt.subplots(
            len(policies),
            1,
            figsize=(7.0, max(2.2, 1.9 * len(policies))),
            sharex=True,
        )
        if len(policies) == 1:
            axes = [axes]
        plotted = False
        for ax, policy in zip(axes, policies):
            grouped: dict[int, dict[str, float]] = defaultdict(
                lambda: defaultdict(float)
            )
            for row in task_rows:
                if row.get("policy_name") != policy:
                    continue
                cycle = int(_safe_float(row.get("cycle")) or 0)
                operator = row.get("prompt_operator") or ""
                if not operator:
                    operator = "STOP" if row.get("action_name") == "STOP" else "other"
                share = _safe_float(row.get("share")) or 0.0
                grouped[cycle][operator] += share
            cycles = sorted(grouped)
            operators = sorted({op for cycle in grouped.values() for op in cycle})
            bottoms = [0.0 for _ in cycles]
            for operator in operators:
                values = [grouped[cycle].get(operator, 0.0) for cycle in cycles]
                ax.bar(cycles, values, bottom=bottoms, label=operator)
                bottoms = [base + value for base, value in zip(bottoms, values)]
                plotted = True
            ax.set_ylabel(policy, rotation=0, labelpad=45, fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.2)
        axes[-1].set_xlabel("Cycle")
        axes[0].legend(
            fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.45)
        )
        fig.suptitle(f"Policy behavior: {_task_label(key)}")
        if not plotted:
            plt.close(fig)
            continue
        paths.append(_save(fig, output_dir, f"policy_behavior_{_task_slug(key)}", fmt))
        plt.close(fig)
    return paths


def _find_calibration_csv(report_dir: Path) -> Path | None:
    candidates = [
        report_dir / "teacher_calibration.csv",
        report_dir / "audit" / "teacher_calibration.csv",
        report_dir.parent / "audit" / "teacher_calibration.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _plot_token_calibration(report_dir: Path, output_dir: Path, fmt: str) -> list[Path]:
    calibration_path = _find_calibration_csv(report_dir)
    if calibration_path is None:
        return []
    rows = _read_csv(calibration_path)
    points = [
        (
            _safe_float(row.get("predicted_total_tokens")),
            _safe_float(row.get("realized_total_tokens")),
            row.get("over_preflight_bound") == "True",
        )
        for row in rows
    ]
    points = [(x, y, over) for x, y, over in points if x is not None and y is not None]
    if not points:
        return []

    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    colors = ["tab:red" if over else "tab:blue" for _, _, over in points]
    xs = [x for x, _, _ in points]
    ys = [y for _, y, _ in points]
    ax.scatter(xs, ys, c=colors, alpha=0.7)
    upper = max(xs + ys)
    ax.plot([0, upper], [0, upper], color="black", linewidth=1.0, alpha=0.45)
    ax.set_title("Predicted vs realized teacher tokens")
    ax.set_xlabel("Predicted total tokens")
    ax.set_ylabel("Realized total tokens")
    ax.grid(alpha=0.25)
    path = _save(fig, output_dir, "token_calibration", fmt)
    plt.close(fig)
    return [path]


def _plot_budget_audit(
    rows: list[dict[str, str]],
    output_dir: Path,
    fmt: str,
) -> list[Path]:
    plt = _load_pyplot()
    by_task: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task[_task_key(row)].append(row)

    paths: list[Path] = []
    for key, task_rows in sorted(by_task.items(), key=lambda item: item[0]):
        labels: list[str] = []
        seed_tokens: list[float] = []
        online_tokens: list[float] = []
        for row in task_rows:
            seed = _safe_float(row.get("mean_seed_teacher_total_tokens"))
            online = _safe_float(row.get("mean_online_teacher_total_tokens"))
            if seed is None and online is None:
                continue
            labels.append(f"{_policy_label(row)} {row.get('token_budget') or ''}")
            seed_tokens.append(seed or 0.0)
            online_tokens.append(online or 0.0)
        if not labels:
            continue
        fig, ax = plt.subplots(figsize=(max(7.0, 0.45 * len(labels)), 4.2))
        xs = list(range(len(labels)))
        ax.bar(xs, seed_tokens, label="seed teacher tokens")
        ax.bar(xs, online_tokens, bottom=seed_tokens, label="online teacher tokens")
        ax.set_title(f"Teacher-token accounting: {_task_label(key)}")
        ax.set_ylabel("Mean teacher tokens")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        paths.append(_save(fig, output_dir, f"budget_tokens_{_task_slug(key)}", fmt))
        plt.close(fig)
    return paths


def write_paper_figures(
    report_dir: Path,
    output_dir: Path | None = None,
    *,
    fmt: str = "pdf",
) -> dict[str, Any]:
    """Write publication-oriented figures from a paper-report directory."""
    report_dir = Path(report_dir)
    output_dir = Path(output_dir) if output_dir else report_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = fmt.lower().lstrip(".")
    created: list[Path] = []
    skipped: list[str] = []

    plot_specs: list[tuple[str, Iterable[Path]]] = [
        (
            "paper_quality_cost_points.csv",
            [report_dir / "paper_quality_cost_points.csv"],
        ),
        ("paper_pairwise_summary.csv", [report_dir / "paper_pairwise_summary.csv"]),
        (
            "paper_action_cycle_frequencies.csv",
            [report_dir / "paper_action_cycle_frequencies.csv"],
        ),
        ("paper_budget_audit.csv", [report_dir / "paper_budget_audit.csv"]),
    ]
    for name, paths in plot_specs:
        if not any(path.exists() for path in paths):
            skipped.append(name)

    quality_rows = _read_csv(report_dir / "paper_quality_cost_points.csv")
    summary_rows = _read_csv(report_dir / "paper_pairwise_summary.csv")
    behavior_rows = _read_csv(report_dir / "paper_action_cycle_frequencies.csv")
    budget_rows = _read_csv(report_dir / "paper_budget_audit.csv")

    if quality_rows:
        created.extend(_plot_quality_cost(quality_rows, output_dir, fmt))
    if summary_rows:
        created.extend(_plot_pairwise_summary(summary_rows, output_dir, fmt))
    if behavior_rows:
        created.extend(_plot_policy_behavior(behavior_rows, output_dir, fmt))
    if budget_rows:
        created.extend(_plot_budget_audit(budget_rows, output_dir, fmt))

    calibration_paths = _plot_token_calibration(report_dir, output_dir, fmt)
    if calibration_paths:
        created.extend(calibration_paths)
    else:
        skipped.append("teacher_calibration.csv")

    manifest = {
        "report_dir": str(report_dir),
        "output_dir": str(output_dir),
        "format": fmt,
        "created": [str(path) for path in created],
        "skipped": sorted(set(skipped)),
    }
    manifest_path = output_dir / "paper_figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest
