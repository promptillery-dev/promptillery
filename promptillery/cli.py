"""Command line interface for promptillery."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List

import typer
from dotenv import load_dotenv

from .ablation import AblationStudyRunner
from .analyze import (
    analyze_runs,
    plan_same_count_control_configs,
    validate_pilot_gate,
    write_audit_csvs,
    write_summary_csv,
)
from .config import ExperimentConfig
from .engine import DistillationEngine, evaluate_model
from .policy_controller import PolicyController, enumerate_actions
from .sft_materialize import materialize_sft_records
from .utils import setup_logging

app = typer.Typer(add_completion=False)
load_dotenv()

# Default teacher model for baseline evaluation
DEFAULT_BASELINE_TEACHER = "openai/gpt-4.1"


def _csv_option_values(value: str) -> List[str]:
    """Parse a comma-separated CLI option into stable string values."""
    return [item.strip() for item in value.split(",") if item.strip()]


@app.command()
def train(
    config: str,
    base_dir: str = typer.Option(
        ".", "--base-dir", "-d", help="Base directory for experiment outputs"
    ),
) -> None:
    """Run training given a config YAML."""
    setup_logging()
    cfg = ExperimentConfig.from_yaml(config)

    # Override config with CLI parameters if provided
    if base_dir != ".":
        cfg.base_output_dir = base_dir

    engine = DistillationEngine(cfg)
    asyncio.run(engine.run())


@app.command()
def ablation(
    config: str,
    base_dir: str = typer.Option(
        ".", "--base-dir", "-d", help="Base directory for experiment outputs"
    ),
    cleanup: bool = typer.Option(
        True,
        "--cleanup/--no-cleanup",
        help="Clean up after each augmentation_batch_size group, keeping only best performer",
    ),
    cleanup_metric: str = typer.Option(
        None,
        "--cleanup-metric",
        "-m",
        help="Metric to use for selecting best config during cleanup (default: first metric in config)",
    ),
) -> None:
    """Run ablation study with multiple configurations.

    By default, enables cleanup mode which keeps only the best-performing
    configuration for each augmentation_batch_size group. This dramatically
    reduces disk usage (e.g., from 120GB to ~3GB for a 144-config study).

    Use --no-cleanup to keep all experiment results.
    """
    setup_logging()
    cfg = ExperimentConfig.from_yaml(config)

    # Override config with CLI parameters if provided
    if base_dir != ".":
        cfg.base_output_dir = base_dir

    runner = AblationStudyRunner(cfg, cleanup_metric=cleanup_metric)
    asyncio.run(runner.run(cleanup_after_group=cleanup))


@app.command("materialize-sft")
def materialize_sft(
    config: str,
    output: str = typer.Option(
        "generated_sft_records.jsonl",
        "--output",
        "-o",
        help="Output JSONL path for materialized SFT records",
    ),
    split: str = typer.Option("train", "--split", "-s", help="Dataset split to read"),
    mode: str = typer.Option(
        "gold",
        "--mode",
        help="Materialization mode: 'gold' for zero-cost dry runs or 'teacher' for LLM calls",
    ),
    max_samples: int | None = typer.Option(
        None,
        "--max-samples",
        help="Maximum number of source examples to materialize",
    ),
    student_prompt_template: str | None = typer.Option(
        None,
        "--student-prompt-template",
        help=(
            "Override the config's student prompt template; dataset fields are "
            "available"
        ),
    ),
    teacher_prompt_template: str | None = typer.Option(
        None,
        "--teacher-prompt-template",
        help="Jinja template for teacher calls; defaults to the built-in prompt",
    ),
    prompt_operator: str = typer.Option(
        "coverage",
        "--prompt-operator",
        help="Prompt operator label to write into each record",
    ),
    teacher_tier: str = typer.Option(
        "cheap",
        "--teacher-tier",
        help="Teacher tier label to write into each teacher-mode record",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace the output JSONL if it already exists",
    ),
    allow_estimated_usage: bool = typer.Option(
        False,
        "--allow-estimated-usage",
        help="Allow estimated token usage if the provider response omits usage fields",
    ),
    allow_partial: bool = typer.Option(
        False,
        "--allow-partial",
        help="Keep a budget-truncated dataset when preflight stops teacher mode",
    ),
) -> None:
    """Materialize audited SFT JSONL records from a dataset split."""
    setup_logging()
    cfg = ExperimentConfig.from_yaml(config)
    try:
        summary = asyncio.run(
            materialize_sft_records(
                config=cfg,
                output_path=Path(output),
                split=split,
                mode=mode,
                max_samples=max_samples,
                student_prompt_template=student_prompt_template,
                teacher_prompt_template=teacher_prompt_template
                or None,
                prompt_operator=prompt_operator,
                teacher_tier=teacher_tier,
                overwrite=overwrite,
                allow_estimated_usage=allow_estimated_usage,
                allow_partial=allow_partial,
            )
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(
        "Wrote {records} SFT records to {output_path} "
        "({teacher_total_tokens} teacher tokens, stop_reason={stop_reason}; "
        "manifest: {manifest_path})".format(**summary)
    )


@app.command()
def analyze(
    path: str,
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional CSV output path; prints CSV to stdout when omitted",
    ),
    audit_dir: str | None = typer.Option(
        None,
        "--audit-dir",
        help="Optional directory for policy_actions, teacher_calibration, and oracle_frontier CSVs",
    ),
    metric: str | None = typer.Option(
        None,
        "--metric",
        "-m",
        help="Metric to summarize; defaults to the first recognized run metric",
    ),
    mode: str = typer.Option(
        "auto",
        "--mode",
        help="Use 'auto', 'max' for accuracy/F1-like metrics, or 'min' for losses",
    ),
) -> None:
    """Summarize quality-cost artifacts from one run directory or a parent."""
    if mode not in {"auto", "max", "min"}:
        typer.echo("Error: --mode must be 'auto', 'max', or 'min'", err=True)
        raise typer.Exit(code=1)

    try:
        rows = analyze_runs(Path(path), metric=metric, mode=mode)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    if not rows:
        typer.echo(f"Error: no run artifacts found under {path}", err=True)
        raise typer.Exit(code=1)
    if metric and not any(row["metric"] == metric for row in rows):
        typer.echo(f"Error: metric '{metric}' was not found in any run", err=True)
        raise typer.Exit(code=1)

    if output:
        write_summary_csv(rows, Path(output))
        typer.echo(f"Wrote analysis summary to {output}")
        if audit_dir:
            audit_paths = write_audit_csvs(Path(path), rows, Path(audit_dir))
            typer.echo(
                "Wrote audit CSVs to "
                + ", ".join(str(value) for value in audit_paths.values())
            )
        return

    if audit_dir:
        audit_paths = write_audit_csvs(Path(path), rows, Path(audit_dir))
        typer.echo(
            "Wrote audit CSVs to "
            + ", ".join(str(value) for value in audit_paths.values()),
            err=True,
        )

    import csv
    import sys

    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


@app.command("pilot-gate")
def pilot_gate(
    path: str,
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional JSON report output path",
    ),
    metric: str | None = typer.Option(
        None,
        "--metric",
        "-m",
        help="Metric to use for final and AUC checks",
    ),
    mode: str = typer.Option(
        "auto",
        "--mode",
        help="Use 'auto', 'max' for accuracy/F1-like metrics, or 'min' for losses",
    ),
    policies: str = typer.Option(
        "fixed_coverage,fixed_boundary,fixed_repair,random_feasible,cost_heuristic,frugalkd_p",
        "--policies",
        help="Comma-separated expected policy_name values",
    ),
    seeds: str = typer.Option(
        "13,42,101",
        "--seeds",
        help="Comma-separated expected seed values",
    ),
    budgets: str = typer.Option(
        "25000,50000,100000",
        "--budgets",
        help="Comma-separated expected token_budget values",
    ),
    require_teacher_attempts: bool = typer.Option(
        True,
        "--require-teacher-attempts/--no-require-teacher-attempts",
        help="Require at least one teacher attempt row",
    ),
    require_frontier: bool = typer.Option(
        True,
        "--require-frontier/--no-require-frontier",
        help="Require matched fixed-policy frontier rows for every run",
    ),
    require_heldout: bool = typer.Option(
        False,
        "--require-heldout/--no-require-heldout",
        "--require-held-out/--no-require-held-out",
        help="Require final held-out test metrics for every summarized run",
    ),
    require_full_label_coverage: bool = typer.Option(
        False,
        "--require-full-label-coverage/--no-require-full-label-coverage",
        help="Require observed gold labels to cover every canonical label",
    ),
    require_same_count_control: bool = typer.Option(
        False,
        "--require-same-count-controls/--no-require-same-count-controls",
        "--require-same-count-control/--no-require-same-count-control",
        help="Require same_count control rows for every seed/budget pair",
    ),
    same_count_source_policy: str = typer.Option(
        "frugalkd_p",
        "--same-count-source-policy",
        help="Policy whose final synthetic count same_count controls must match",
    ),
) -> None:
    """Validate cheap-pilot artifacts against reviewer-facing gate checks."""
    if mode not in {"auto", "max", "min"}:
        typer.echo("Error: --mode must be 'auto', 'max', or 'min'", err=True)
        raise typer.Exit(code=1)

    try:
        report = validate_pilot_gate(
            Path(path),
            metric=metric,
            mode=mode,
            expected_policies=_csv_option_values(policies),
            expected_seeds=_csv_option_values(seeds),
            expected_budgets=_csv_option_values(budgets),
            require_teacher_attempts=require_teacher_attempts,
            require_frontier=require_frontier,
            require_heldout=require_heldout,
            require_full_label_coverage=require_full_label_coverage,
            require_same_count_control=require_same_count_control,
            same_count_source_policy=same_count_source_policy,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    report_json = json.dumps(report, indent=2, sort_keys=True)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(report_json + "\n", encoding="utf-8")
        typer.echo(f"Wrote pilot gate report to {output}")
    else:
        typer.echo(report_json)

    if not report["passed"]:
        raise typer.Exit(code=1)


@app.command("plan-same-count-controls")
def plan_same_count_controls(
    pilot_dir: str,
    base_config: str,
    output_dir: str = typer.Option(
        "same_count_configs",
        "--output-dir",
        "-o",
        help="Directory for generated same_count control configs",
    ),
    metric: str | None = typer.Option(
        None,
        "--metric",
        "-m",
        help="Metric to read while summarizing source runs",
    ),
    mode: str = typer.Option(
        "auto",
        "--mode",
        help="Use 'auto', 'max' for accuracy/F1-like metrics, or 'min' for losses",
    ),
    source_policy: str = typer.Option(
        "frugalkd_p",
        "--source-policy",
        help="Policy whose final_synthetic_count should define the matched cap",
    ),
    control_policy: str = typer.Option(
        "cost_heuristic",
        "--control-policy",
        help="Policy to rerun under the matched synthetic-record cap",
    ),
    control_base_output_dir: str | None = typer.Option(
        None,
        "--control-base-output-dir",
        help="Optional base_output_dir override for generated control configs",
    ),
) -> None:
    """Generate matched same_count control configs from source pilot runs."""
    if mode not in {"auto", "max", "min"}:
        typer.echo("Error: --mode must be 'auto', 'max', or 'min'", err=True)
        raise typer.Exit(code=1)

    try:
        planned = plan_same_count_control_configs(
            Path(pilot_dir),
            Path(base_config),
            Path(output_dir),
            metric=metric,
            mode=mode,
            source_policy=source_policy,
            control_policy=control_policy,
            control_base_output_dir=control_base_output_dir,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(
        f"Wrote {len(planned)} same_count control config(s) to {output_dir}"
    )
    typer.echo(f"Plan manifest: {Path(output_dir) / 'same_count_plan.json'}")


@app.command("policy-smoke")
def policy_smoke(
    policy: str = typer.Option(
        "cost_heuristic",
        "--policy",
        help=(
            "Policy to smoke test: student_only, random_feasible, "
            "cost_heuristic, frugalkd_p, fixed_mixed_teacher, or fixed_*"
        ),
    ),
    tokens_remaining: int = typer.Option(
        4096,
        "--tokens-remaining",
        help="Synthetic remaining token budget for the hard action mask",
    ),
    seed: int = typer.Option(13, "--seed", help="Random seed for random_feasible"),
) -> None:
    """Smoke test the budget-aware policy action contract."""
    state = {
        "budget": {
            "token_budget": 10000,
            "tokens_remaining": tokens_remaining,
            "total_tokens": 10000 - tokens_remaining,
        },
        "features": {
            "eval_error_rate": 0.35,
            "eval_entropy_normalized_mean": 0.42,
            "eval_hard_error_rate": 0.18,
            "eval_max_confusion_rate": 0.12,
            "synthetic_ratio": 0.20,
            "token_budget_remaining_frac": tokens_remaining / 10000,
        },
    }
    predicted_costs = {
        action.name: {
            "total_tokens": 48 * action.batch_size
            * (2 if action.teacher_tier == "strong" else 1)
        }
        for action in enumerate_actions(include_stop=False)
    }
    controller = PolicyController(policy, seed=seed)
    choice = controller.select(state, predicted_costs=predicted_costs)
    if choice.predicted_cost.get("total_tokens", 0) > tokens_remaining:
        typer.echo("Error: selected action exceeds hard budget mask", err=True)
        raise typer.Exit(code=1)
    typer.echo(json.dumps(choice.model_dump(), indent=2, sort_keys=True))


@app.command()
def eval(
    config: str,
    model_path: str = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Path to trained model checkpoint (auto-detects latest if not provided)",
    ),
    split: str = typer.Option(
        "test", "--split", "-s", help="Dataset split to evaluate (test/validation)"
    ),
    base_dir: str = typer.Option(
        ".", "--base-dir", "-d", help="Base directory to search for experiment outputs"
    ),
) -> None:
    """Evaluate a trained model on the specified dataset split."""
    setup_logging()

    # Load configuration
    cfg = ExperimentConfig.from_yaml(config)

    # Auto-detect model path if not provided
    if model_path is None:
        model_path_obj = _find_latest_model(cfg.name, base_dir)
        if model_path_obj is None:
            typer.echo(
                f"Error: No model found for experiment '{cfg.name}' in '{base_dir}'. "
                "Use --model-path to specify explicitly.",
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(f"Auto-detected model: {model_path_obj}")
    else:
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            typer.echo(f"Error: Model path does not exist: {model_path}", err=True)
            raise typer.Exit(code=1)

    # Run evaluation
    try:
        evaluate_model(cfg, model_path_obj, split)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


def _find_latest_model(experiment_name: str, base_dir: str) -> Path | None:
    """Find the most recent model directory for an experiment.

    Searches for directories matching {experiment_name}_* pattern and returns
    the model/ subdirectory of the most recent one (by directory name timestamp).
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return None

    # Find all matching experiment directories
    # Pattern: {name}_{timestamp} where name might have _transformers etc appended
    matching_dirs = []
    for d in base_path.iterdir():
        if d.is_dir() and d.name.startswith(experiment_name):
            model_dir = d / "model"
            if model_dir.exists():
                matching_dirs.append(model_dir)

    if not matching_dirs:
        return None

    # Sort by directory name (timestamp is in the name) and return most recent
    matching_dirs.sort(key=lambda p: p.parent.name, reverse=True)
    return matching_dirs[0]


@app.command()
def baseline(
    config: str = typer.Argument(
        None, help="Path to experiment config YAML file"
    ),
    dataset: str = typer.Option(
        None, "--dataset", "-d", help="Dataset name (e.g., 'tweet_eval', 'stanfordnlp/imdb') - alternative to config"
    ),
    dataset_config: str = typer.Option(
        None, "--dataset-config", "-dc", help="Dataset config/subset (e.g., 'sentiment', 'plain_text')"
    ),
    text_column: str = typer.Option(
        "text", "--text-column", help="Name of the text column in the dataset"
    ),
    label_column: str = typer.Option(
        "label", "--label-column", help="Name of the label column in the dataset"
    ),
    teacher: str = typer.Option(
        DEFAULT_BASELINE_TEACHER,
        "--teacher",
        "-t",
        help=f"Teacher model to use (default: {DEFAULT_BASELINE_TEACHER})",
    ),
    modes: List[str] = typer.Option(
        ["zero-shot", "few-shot"],
        "--mode",
        "-m",
        help="Evaluation modes to run (can specify multiple)",
    ),
    num_shots: int = typer.Option(
        2, "--num-shots", "-n", help="Number of examples per class for few-shot"
    ),
    max_samples: int = typer.Option(
        None, "--max-samples", help="Maximum samples to evaluate (for testing)"
    ),
    output_dir: str = typer.Option(
        "baseline_results", "--output-dir", "-o", help="Output directory for results"
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    concurrency: int = typer.Option(
        10, "--concurrency", help="Maximum concurrent API calls"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Run baseline zero-shot and few-shot evaluation with teacher models.

    This command evaluates teacher models (e.g., GPT-4.1) directly on classification
    tasks to establish baseline performance for comparison against promptillery
    distillation results.

    Examples:

        # Using an experiment config file (recommended)
        promptillery baseline examples/text_classification_transformers.yaml

        # Using a specific dataset directly (alternative)
        promptillery baseline -d stanfordnlp/imdb -dc plain_text

        # Using a dataset with custom column names
        promptillery baseline -d community-datasets/yahoo_answers_topics \\
            --text-column question_title --label-column topic

        # Run only zero-shot evaluation
        promptillery baseline examples/text_classification_transformers.yaml -m zero-shot

        # Run with a different teacher model
        promptillery baseline examples/text_classification_transformers.yaml -t openai/gpt-4o-mini
    """
    import logging
    from .baseline_eval import run_baseline_evaluation

    setup_logging(level=logging.DEBUG if verbose else logging.INFO)

    # Validate arguments
    if not config and not dataset:
        typer.echo("Error: Either config file or --dataset must be provided", err=True)
        raise typer.Exit(code=1)

    # Validate modes
    valid_modes = {"zero-shot", "few-shot"}
    for mode in modes:
        if mode not in valid_modes:
            typer.echo(f"Error: Invalid mode '{mode}'. Must be one of: {valid_modes}", err=True)
            raise typer.Exit(code=1)

    # Run evaluation
    asyncio.run(run_baseline_evaluation(
        config_path=config,
        dataset_name=dataset,
        dataset_config=dataset_config,
        text_column=text_column,
        label_column=label_column,
        teacher=teacher,
        modes=list(modes),
        num_shots=num_shots,
        max_samples=max_samples,
        output_dir=output_dir,
        seed=seed,
        concurrency=concurrency,
    ))


if __name__ == "__main__":
    app()
