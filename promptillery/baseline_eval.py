"""Baseline evaluation script for zero-shot and few-shot classification with teacher models.

This script evaluates teacher models (e.g., GPT-4.1) on classification tasks from ablation configs,
providing baseline metrics to compare against promptillery distillation results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import evaluate
import numpy as np
from datasets import DatasetDict, load_dataset
from litellm import acompletion
from pydantic import BaseModel
from rich.console import Console
from rich.logging import RichHandler
import re
from rich.table import Table
from tqdm.asyncio import tqdm_asyncio

from .config import DatasetConfig, ExperimentConfig, SamplingConfig, infer_num_labels
from .engine import ensure_class_label, prepare_dataset
from .token_tracker import _extract_usage_from_response, TokenUsage

logger = logging.getLogger(__name__)
console = Console()


# Default teacher model
DEFAULT_TEACHER = "openai/gpt-4.1"


class ClassificationResponse(BaseModel):
    """Structured response for classification containing the predicted label."""
    class_label: int
    class_name: Optional[str] = None


@dataclass
class TokenStats:
    """Token usage statistics for an evaluation run."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0

    def add(self, usage: TokenUsage) -> None:
        """Add token usage from a single API call."""
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens
        if usage.estimated_cost is not None:
            self.estimated_cost += usage.estimated_cost


@dataclass
class EvalResult:
    """Result of a single evaluation run."""
    dataset: str
    dataset_config: str
    mode: str  # "zero-shot" or "few-shot"
    num_shots: int
    teacher: str
    metrics: Dict[str, float]
    num_samples: int
    predictions: List[int] = field(default_factory=list)
    true_labels: List[int] = field(default_factory=list)
    errors: int = 0
    token_stats: TokenStats = field(default_factory=TokenStats)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure rich logging."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)]
    )


def load_dataset_from_config(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    sampling_config: Optional[SamplingConfig] = None,
    label_column: str = "label",
) -> Tuple[DatasetDict, int]:
    """Load and prepare dataset from config.

    Args:
        dataset_name: HuggingFace dataset name
        dataset_config: Optional dataset subset/config name
        sampling_config: Optional sampling configuration
        label_column: Name of the label column

    Returns:
        Tuple of (prepared dataset, number of classes)
    """
    logger.info(f"Loading dataset: {dataset_name}/{dataset_config or 'default'}")

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        dataset = load_dataset(dataset_name)

    # Ensure label column is ClassLabel type
    dataset = ensure_class_label(dataset, label_column)

    # Apply sampling if configured
    if sampling_config and sampling_config.enabled:
        dataset = prepare_dataset(dataset, sampling_config)

    # Infer number of classes
    num_classes = infer_num_labels(dataset_name, dataset_config)
    if num_classes is None:
        # Fallback: count unique labels in the dataset
        first_split = next(iter(dataset.keys()))
        if label_column in dataset[first_split].column_names:
            num_classes = len(set(dataset[first_split][label_column]))
        else:
            num_classes = 2  # Default fallback

    logger.info(f"Dataset loaded with {num_classes} classes")
    return dataset, num_classes


def extract_few_shot_examples(
    dataset: DatasetDict,
    n_per_class: int = 2,
    text_column: str = "text",
    label_column: str = "label",
    seed: int = 42,
    exclude_indices: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], set, str]:
    """Extract balanced few-shot examples from the training set.

    Args:
        dataset: The dataset to sample from
        n_per_class: Number of examples per class
        text_column: Name of text column
        label_column: Name of label column
        seed: Random seed
        exclude_indices: Indices to exclude from sampling

    Returns:
        Tuple of (list of examples, set of used indices, source split name)
    """
    random.seed(seed)
    exclude_indices = exclude_indices or set()

    # Use train split for few-shot examples
    train_split = "train" if "train" in dataset else next(iter(dataset.keys()))
    train_ds = dataset[train_split]

    # Group samples by label
    samples_by_label: Dict[int, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for i in range(len(train_ds)):
        if i in exclude_indices:
            continue
        label = train_ds[i][label_column]
        samples_by_label[label].append((i, {
            "text": train_ds[i][text_column],
            "label": label,
        }))

    # Select n_per_class from each label
    few_shot = []
    used_indices = set()
    for label in sorted(samples_by_label.keys()):
        samples = samples_by_label[label]
        selected = random.sample(samples, min(n_per_class, len(samples)))
        for idx, sample in selected:
            few_shot.append(sample)
            used_indices.add(idx)

    return few_shot, used_indices, train_split


def format_few_shot_examples(
    examples: List[Dict[str, Any]],
    label_names: Optional[Dict[int, str]] = None,
) -> str:
    """Format few-shot examples for inclusion in prompt.

    Args:
        examples: List of example dicts with 'text' and 'label' keys
        label_names: Optional mapping of label indices to names

    Returns:
        Formatted string of examples
    """
    lines = []
    for i, ex in enumerate(examples, 1):
        label = ex["label"]
        if label_names:
            label_str = label_names.get(label, str(label))
        else:
            label_str = str(label)
        lines.append(f"Example {i}:")
        lines.append(f"Text: {ex['text']}")
        lines.append(f"Label: {label_str}")
        lines.append("")
    return "\n".join(lines)


def build_class_taxonomy(
    num_classes: int,
    label_names: Optional[Dict[int, str]] = None,
) -> str:
    """Build the class taxonomy string for the prompt.

    Args:
        num_classes: Number of classes
        label_names: Optional mapping of label indices to names

    Returns:
        Formatted class taxonomy string
    """
    if label_names:
        lines = [f"{k}: {v}" for k, v in sorted(label_names.items())]
    else:
        lines = [f"{i}: class_{i}" for i in range(num_classes)]
    return "\n".join(lines)


def build_classification_prompt(
    text: str,
    num_classes: int,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    label_names: Optional[Dict[int, str]] = None,
    task_description: Optional[str] = None,
) -> str:
    """Build a classification prompt for the teacher model.

    Uses the standardized prompt format:
    - Instructs model to return JSON with class_label and class_name
    - Includes class taxonomy
    - Optionally includes few-shot examples

    Args:
        text: The text to classify
        num_classes: Number of classes
        few_shot_examples: Optional list of few-shot examples
        label_names: Optional mapping of label indices to names
        task_description: Optional task description (unused, kept for compatibility)

    Returns:
        The formatted prompt
    """
    # Build class taxonomy
    taxonomy = build_class_taxonomy(num_classes, label_names)

    # Build the prompt
    parts = []

    parts.append("Please provide only one category for each text in JSON format. For example:")
    parts.append('{"class_label": 0, "class_name": "example_class"}')
    parts.append("Please do not repeat or return the content back again, just provide the category in the defined format.")
    parts.append("")

    # Add class taxonomy
    parts.append("Categories:")
    parts.append(taxonomy)
    parts.append("")

    # Few-shot examples
    if few_shot_examples:
        parts.append("Examples:")
        for i, ex in enumerate(few_shot_examples, 1):
            label = ex["label"]
            if label_names:
                label_name = label_names.get(label, f"class_{label}")
            else:
                label_name = f"class_{label}"
            parts.append(f"Text: {ex['text']}")
            parts.append(f'{{"class_label": {label}, "class_name": "{label_name}"}}')
            parts.append("")

    # The text to classify
    parts.append("Text-to-classify:")
    parts.append(text)

    return "\n".join(parts)


async def classify_single(
    text: str,
    num_classes: int,
    teacher: str,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    label_names: Optional[Dict[int, str]] = None,
    task_description: Optional[str] = None,
    max_retries: int = 3,
) -> Tuple[Optional[int], Optional[str], Optional[TokenUsage]]:
    """Classify a single text using the teacher model.

    Args:
        text: Text to classify
        num_classes: Number of classes
        teacher: Teacher model name
        few_shot_examples: Optional few-shot examples
        label_names: Optional label name mapping
        task_description: Optional task description
        max_retries: Maximum retry attempts

    Returns:
        Tuple of (predicted label or None, error message or None, token usage or None)
    """
    prompt = build_classification_prompt(
        text=text,
        num_classes=num_classes,
        few_shot_examples=few_shot_examples,
        label_names=label_names,
        task_description=task_description,
    )

    # Track token usage across retries (API calls consume tokens even on parse errors)
    last_token_usage: Optional[TokenUsage] = None

    for attempt in range(max_retries):
        try:
            result = await acompletion(
                model=teacher,
                messages=[{"role": "user", "content": prompt}],
                response_format=ClassificationResponse,
            )

            # Extract token usage from response - capture immediately after API call
            token_usage = _extract_usage_from_response(result)
            last_token_usage = token_usage

            content = result["choices"][0]["message"]["content"]

            # Parse structured response
            try:
                response = ClassificationResponse.model_validate_json(content)
                label = response.class_label
            except Exception:
                # Fallback: try to parse class_label from JSON or plain integer
                content_clean = content.strip()
                # Try to extract class_label from JSON-like string
                class_label_match = re.search(r'"class_label"\s*:\s*(\d+)', content_clean)
                if class_label_match:
                    label = int(class_label_match.group(1))
                else:
                    # Try to extract just any number
                    match = re.search(r'\b(\d+)\b', content_clean)
                    if match:
                        label = int(match.group(1))
                    else:
                        raise ValueError(f"Could not parse label from: {content_clean}")

            # Validate label is in range
            if 0 <= label < num_classes:
                return label, None, token_usage
            else:
                logger.warning(f"Label {label} out of range [0, {num_classes})")
                return None, f"Label out of range: {label}", token_usage

        except Exception as e:
            if attempt < max_retries - 1:
                logger.debug(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
            else:
                # Preserve token usage from last successful API call on final failure
                return None, str(e), last_token_usage

    return None, "Max retries exceeded", last_token_usage


async def evaluate_dataset(
    dataset: DatasetDict,
    num_classes: int,
    teacher: str,
    mode: str = "zero-shot",
    num_shots: int = 2,
    text_column: str = "text",
    label_column: str = "label",
    label_names: Optional[Dict[int, str]] = None,
    task_description: Optional[str] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
    concurrency: int = 10,
) -> EvalResult:
    """Evaluate teacher model on a dataset.

    Args:
        dataset: The dataset to evaluate on
        num_classes: Number of classes
        teacher: Teacher model name
        mode: "zero-shot" or "few-shot"
        num_shots: Number of examples per class for few-shot
        text_column: Name of text column
        label_column: Name of label column
        label_names: Optional label name mapping
        task_description: Optional task description
        max_samples: Maximum samples to evaluate (for testing)
        seed: Random seed
        concurrency: Max concurrent API calls

    Returns:
        EvalResult with metrics and predictions
    """
    # Determine which split to evaluate on
    eval_split = "test" if "test" in dataset else "validation" if "validation" in dataset else "train"
    eval_ds = dataset[eval_split]

    logger.info(f"Evaluating on {eval_split} split with {len(eval_ds)} samples")

    # Sample few-shot examples if needed
    few_shot_examples = None
    used_indices = set()
    few_shot_source_split = None
    if mode == "few-shot":
        few_shot_examples, used_indices, few_shot_source_split = extract_few_shot_examples(
            dataset=dataset,
            n_per_class=num_shots,
            text_column=text_column,
            label_column=label_column,
            seed=seed,
        )
        logger.info(f"Using {len(few_shot_examples)} few-shot examples ({num_shots} per class) from {few_shot_source_split} split")

    # Prepare samples to evaluate
    # Shuffle to avoid class imbalance when using max_samples
    random.seed(seed)
    all_indices = list(range(len(eval_ds)))

    # Exclude few-shot examples from evaluation ONLY when evaluating on the same split
    # they were sampled from, to prevent label leakage.
    # Indices from different splits refer to different samples, so we must not filter
    # when eval_split != few_shot_source_split (would remove unrelated samples).
    if mode == "few-shot" and used_indices and eval_split == few_shot_source_split:
        original_count = len(all_indices)
        all_indices = [i for i in all_indices if i not in used_indices]
        excluded_count = original_count - len(all_indices)
        if excluded_count > 0:
            logger.info(
                f"Excluded {excluded_count} few-shot examples from evaluation "
                f"(eval and few-shot both from {eval_split} split, remaining: {len(all_indices)} samples)"
            )
    elif mode == "few-shot" and eval_split != few_shot_source_split:
        logger.info(
            f"Not excluding few-shot indices: few-shot from {few_shot_source_split} split, "
            f"evaluating on {eval_split} split (different splits, indices refer to different samples)"
        )

    if max_samples and max_samples < len(all_indices):
        random.shuffle(all_indices)
        all_indices = all_indices[:max_samples]

    samples = []
    for i in all_indices:
        samples.append({
            "idx": i,
            "text": eval_ds[i][text_column],
            "label": eval_ds[i][label_column],
        })

    logger.info(f"Evaluating {len(samples)} samples with {mode} mode")

    # Run classification with concurrency control
    semaphore = asyncio.Semaphore(concurrency)
    predictions = []
    true_labels = []
    errors = 0
    token_stats = TokenStats()

    async def classify_with_semaphore(sample: Dict[str, Any]) -> Tuple[int, Optional[int], int, Optional[TokenUsage]]:
        async with semaphore:
            pred, error, usage = await classify_single(
                text=sample["text"],
                num_classes=num_classes,
                teacher=teacher,
                few_shot_examples=few_shot_examples,
                label_names=label_names,
                task_description=task_description,
            )
            return sample["idx"], pred, sample["label"], usage

    # Create tasks
    tasks = [classify_with_semaphore(s) for s in samples]

    # Run with progress bar
    results = await tqdm_asyncio.gather(*tasks, desc=f"Classifying ({mode})")

    # Process results
    for idx, pred, true_label, usage in results:
        true_labels.append(true_label)
        if pred is not None:
            predictions.append(pred)
        else:
            # Use random prediction as fallback to avoid biasing toward any class
            predictions.append(random.randint(0, num_classes - 1))
            errors += 1

        # Accumulate token usage
        if usage is not None:
            token_stats.add(usage)

    if errors > 0:
        logger.warning(f"{errors} classification errors out of {len(samples)} samples")

    # Compute metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    accuracy_result = accuracy_metric.compute(predictions=predictions, references=true_labels)
    accuracy = accuracy_result["accuracy"]

    # Determine F1 averaging
    # For binary, check if both classes are present, otherwise use macro
    unique_classes = set(predictions) | set(true_labels)
    if num_classes == 2 and len(unique_classes) == 2:
        # Binary F1 with pos_label=1
        f1_result = f1_metric.compute(
            predictions=predictions, references=true_labels, average="binary", pos_label=1
        )
    else:
        # Macro averaging for multi-class or degenerate cases
        f1_result = f1_metric.compute(
            predictions=predictions, references=true_labels, average="macro"
        )
    f1 = f1_result["f1"]

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
    }

    return EvalResult(
        dataset="",  # Will be set by caller
        dataset_config="",  # Will be set by caller
        mode=mode,
        num_shots=num_shots if mode == "few-shot" else 0,
        teacher=teacher,
        metrics=metrics,
        num_samples=len(samples),
        predictions=predictions,
        true_labels=true_labels,
        errors=errors,
        token_stats=token_stats,
    )


async def run_baseline_evaluation(
    config_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    text_column: str = "text",
    label_column: str = "label",
    teacher: str = DEFAULT_TEACHER,
    modes: Optional[List[str]] = None,
    num_shots: int = 2,
    max_samples: Optional[int] = None,
    output_dir: str = "baseline_results",
    seed: int = 42,
    concurrency: int = 10,
) -> List[EvalResult]:
    """Run baseline evaluation on one or more datasets.

    Args:
        config_path: Path to experiment config YAML (optional)
        dataset_name: Dataset name (if not using config)
        dataset_config: Dataset config/subset (if not using config)
        text_column: Name of text column (used when not loading from config)
        label_column: Name of label column (used when not loading from config)
        teacher: Teacher model to use
        modes: List of modes to run ("zero-shot", "few-shot")
        num_shots: Number of examples per class for few-shot
        max_samples: Maximum samples per evaluation
        output_dir: Output directory for results
        seed: Random seed
        concurrency: Max concurrent API calls

    Returns:
        List of EvalResult objects
    """
    modes = modes or ["zero-shot", "few-shot"]
    results = []

    # Determine datasets to evaluate
    # Each entry: (dataset_name, subset_name, text_field, label_field, sampling_config)
    datasets_to_eval = []

    if config_path:
        # Load from config
        config = ExperimentConfig.from_yaml(config_path)

        # Get dataset info from config
        base_dataset = config.dataset if isinstance(config.dataset, str) else config.dataset[0]
        sampling_config = config.sampling if hasattr(config, 'sampling') else None

        # Handle dataset_config - can be DatasetConfig, string, or list
        if config.dataset_config:
            if isinstance(config.dataset_config, list):
                # List of configs (ablation case - we'll just use the first one for baseline)
                # For baseline eval, we don't run all ablation configs, just the dataset
                first_dc = config.dataset_config[0]
                if isinstance(first_dc, DatasetConfig):
                    datasets_to_eval.append((
                        base_dataset, first_dc.name, first_dc.text_field,
                        first_dc.label_field, sampling_config
                    ))
                else:
                    datasets_to_eval.append((base_dataset, first_dc, "text", "label", sampling_config))
            elif isinstance(config.dataset_config, DatasetConfig):
                datasets_to_eval.append((
                    base_dataset,
                    config.dataset_config.name,
                    config.dataset_config.text_field,
                    config.dataset_config.label_field,
                    sampling_config,
                ))
            else:
                # Old string format
                datasets_to_eval.append((base_dataset, config.dataset_config, "text", "label", sampling_config))
        else:
            datasets_to_eval.append((base_dataset, None, "text", "label", sampling_config))

    elif dataset_name:
        datasets_to_eval.append((dataset_name, dataset_config, text_column, label_column, None))
    else:
        raise ValueError("Either config_path or dataset_name must be provided")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Evaluate each dataset
    for ds_name, ds_config, text_col, label_col, sampling_cfg in datasets_to_eval:
        console.print(f"\n[bold blue]Evaluating: {ds_name}/{ds_config or 'default'}[/bold blue]")

        # Load dataset
        dataset, num_classes = load_dataset_from_config(
            ds_name, ds_config, sampling_cfg, label_column=label_col
        )

        # Try to get label names from various sources
        label_names = None
        first_split = next(iter(dataset.keys()))
        ds_split = dataset[first_split]

        # Method 1: Check if there's a label_text column (common in many datasets)
        label_text_cols = ['label_text', 'label_name', 'class_name', 'category']
        for label_text_col in label_text_cols:
            if label_text_col in ds_split.column_names and label_col in ds_split.column_names:
                # Build mapping from label to text
                label_to_text = {}
                for i in range(min(1000, len(ds_split))):
                    label = ds_split[i][label_col]
                    text = ds_split[i][label_text_col]
                    if label not in label_to_text:
                        label_to_text[label] = text
                if label_to_text:
                    label_names = {k: v for k, v in sorted(label_to_text.items())}
                    console.print(f"Label names (from {label_text_col}): {label_names}")
                    break

        # Method 2: Check ClassLabel feature names
        if label_names is None and label_col in ds_split.features:
            feature = ds_split.features[label_col]
            if hasattr(feature, "names"):
                names = feature.names
                # Only use if names are meaningful (not just stringified integers)
                if names and not all(name.isdigit() for name in names):
                    label_names = {i: name for i, name in enumerate(names)}
                    console.print(f"Label names (from ClassLabel): {label_names}")

        # Run each mode
        for mode in modes:
            console.print(f"\n[cyan]Running {mode} evaluation...[/cyan]")

            result = await evaluate_dataset(
                dataset=dataset,
                num_classes=num_classes,
                teacher=teacher,
                mode=mode,
                num_shots=num_shots,
                text_column=text_col,
                label_column=label_col,
                label_names=label_names,
                max_samples=max_samples,
                seed=seed,
                concurrency=concurrency,
            )

            # Set dataset info
            result.dataset = ds_name
            result.dataset_config = ds_config or "default"

            results.append(result)

            # Print result
            console.print(f"  Accuracy: {result.metrics['accuracy']:.4f}")
            console.print(f"  F1: {result.metrics['f1']:.4f}")
            console.print(f"  Errors: {result.errors}/{result.num_samples}")
            console.print(f"  Tokens: {result.token_stats.total_tokens:,} "
                         f"(in: {result.token_stats.input_tokens:,}, out: {result.token_stats.output_tokens:,})")
            console.print(f"  Cost: ${result.token_stats.estimated_cost:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"baseline_results_{timestamp}.json"

    # Convert to JSON-serializable format
    results_data = []
    for r in results:
        results_data.append({
            "dataset": r.dataset,
            "dataset_config": r.dataset_config,
            "mode": r.mode,
            "num_shots": r.num_shots,
            "teacher": r.teacher,
            "metrics": r.metrics,
            "num_samples": r.num_samples,
            "errors": r.errors,
            "token_stats": {
                "input_tokens": r.token_stats.input_tokens,
                "output_tokens": r.token_stats.output_tokens,
                "total_tokens": r.token_stats.total_tokens,
                "estimated_cost": r.token_stats.estimated_cost,
            },
        })

    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)

    console.print(f"\n[green]Results saved to: {results_file}[/green]")

    # Print summary table
    print_summary_table(results)

    # Save summary CSV
    save_summary_csv(results, output_path / f"baseline_summary_{timestamp}.csv")

    return results


def print_summary_table(results: List[EvalResult]) -> None:
    """Print a summary table of results."""
    table = Table(title="Baseline Evaluation Results")

    table.add_column("Dataset", style="cyan")
    table.add_column("Config", style="cyan")
    table.add_column("Mode", style="magenta")
    table.add_column("Shots", style="magenta")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("F1", justify="right", style="green")
    table.add_column("Samples", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right", style="yellow")

    for r in results:
        table.add_row(
            r.dataset,
            r.dataset_config,
            r.mode,
            str(r.num_shots) if r.mode == "few-shot" else "-",
            f"{r.metrics['accuracy']:.4f}",
            f"{r.metrics['f1']:.4f}",
            str(r.num_samples),
            f"{r.token_stats.total_tokens:,}",
            f"${r.token_stats.estimated_cost:.4f}",
        )

    console.print(table)

    # Print total cost
    total_cost = sum(r.token_stats.estimated_cost for r in results)
    total_tokens = sum(r.token_stats.total_tokens for r in results)
    console.print(f"\n[bold]Total: {total_tokens:,} tokens, ${total_cost:.4f}[/bold]")


def save_summary_csv(results: List[EvalResult], path: Path) -> None:
    """Save summary to CSV file."""
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "dataset_config", "mode", "num_shots", "teacher",
            "accuracy", "f1", "num_samples", "errors",
            "input_tokens", "output_tokens", "total_tokens", "estimated_cost"
        ])

        for r in results:
            writer.writerow([
                r.dataset,
                r.dataset_config,
                r.mode,
                r.num_shots,
                r.teacher,
                f"{r.metrics['accuracy']:.4f}",
                f"{r.metrics['f1']:.4f}",
                r.num_samples,
                r.errors,
                r.token_stats.input_tokens,
                r.token_stats.output_tokens,
                r.token_stats.total_tokens,
                f"{r.token_stats.estimated_cost:.6f}",
            ])

    console.print(f"[green]Summary saved to: {path}[/green]")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run baseline zero-shot and few-shot evaluation with teacher models"
    )

    parser.add_argument(
        "--config", "-c",
        help="Path to ablation config YAML file",
    )
    parser.add_argument(
        "--dataset", "-d",
        help="Dataset name (e.g., 'tweet_eval')",
    )
    parser.add_argument(
        "--dataset-config", "-dc",
        help="Dataset config/subset (e.g., 'sentiment')",
    )
    parser.add_argument(
        "--teacher", "-t",
        default=DEFAULT_TEACHER,
        help=f"Teacher model to use (default: {DEFAULT_TEACHER})",
    )
    parser.add_argument(
        "--modes", "-m",
        nargs="+",
        default=["zero-shot", "few-shot"],
        choices=["zero-shot", "few-shot"],
        help="Evaluation modes to run",
    )
    parser.add_argument(
        "--num-shots", "-n",
        type=int,
        default=2,
        help="Number of examples per class for few-shot (default: 2)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to evaluate (for testing)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="baseline_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum concurrent API calls (default: 10)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    # Validate arguments
    if not args.config and not args.dataset:
        parser.error("Either --config or --dataset must be provided")

    # Run evaluation
    asyncio.run(run_baseline_evaluation(
        config_path=args.config,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        teacher=args.teacher,
        modes=args.modes,
        num_shots=args.num_shots,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        seed=args.seed,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()
