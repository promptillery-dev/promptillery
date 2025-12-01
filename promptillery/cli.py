"""Command line interface for promptillery."""

from __future__ import annotations

import asyncio
from typing import List, Optional
import typer
from pathlib import Path

import typer
from dotenv import load_dotenv

from .ablation import AblationStudyRunner
from .config import ExperimentConfig
from .engine import DistillationEngine, evaluate_model
from .utils import setup_logging

app = typer.Typer(add_completion=False)
load_dotenv()

# Default teacher model for baseline evaluation
DEFAULT_BASELINE_TEACHER = "openai/gpt-4.1"


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
