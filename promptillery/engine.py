"""Core distillation engine."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
)
from litellm import acompletion
from pydantic import BaseModel

from .config import ExperimentConfig, SamplingConfig
from .token_tracker import OperationType, TokenTracker
from .trainers import TrainerFactory
from .utils import (
    create_prompt_environment,
    extract_few_shot_samples,
    extract_hard_negatives,
    extract_high_entropy_samples,
    format_classification_report,
    set_seed,
)

logger = logging.getLogger(__name__)


class AugmentedArticle(BaseModel):
    """Single augmented article with text and label."""

    text: str
    label: int


class AugmentedResponse(BaseModel):
    """Structured response for augmentation containing list of articles."""

    articles: List[AugmentedArticle]


def ensure_class_label(dataset: DatasetDict, label_column: str) -> DatasetDict:
    """Ensure the label column is of ClassLabel type for stratified sampling.

    Some datasets have label columns as plain integers (Value type) instead of
    ClassLabel type. This function casts such columns to ClassLabel to enable
    stratified sampling.

    Args:
        dataset: The dataset dictionary to process
        label_column: Name of the label column to check/cast

    Returns:
        Dataset with label column cast to ClassLabel if needed
    """
    # Check the first available split to determine column type
    first_split = next(iter(dataset.keys()))
    if label_column not in dataset[first_split].column_names:
        return dataset

    feature = dataset[first_split].features[label_column]

    # If already ClassLabel, no conversion needed
    if isinstance(feature, ClassLabel):
        return dataset

    # Get unique labels from all splits to build ClassLabel
    all_labels = set()
    for split in dataset:
        if label_column in dataset[split].column_names:
            all_labels.update(dataset[split].unique(label_column))

    # Sort labels for consistent ordering (assumes integer labels)
    sorted_labels = sorted(all_labels)
    num_classes = len(sorted_labels)

    logger.info(
        f"Converting '{label_column}' from Value to ClassLabel with {num_classes} classes"
    )

    # Create ClassLabel feature
    class_label = ClassLabel(num_classes=num_classes)

    # Cast column in all splits
    for split in dataset:
        if label_column in dataset[split].column_names:
            dataset[split] = dataset[split].cast_column(label_column, class_label)

    return dataset


def prepare_dataset(
    dataset: DatasetDict, sampling_config: SamplingConfig
) -> DatasetDict:
    """Apply stratified sampling to dataset if configured.

    Args:
        dataset: The loaded dataset dictionary
        sampling_config: Configuration for sampling behavior

    Returns:
        Dataset with stratified train/validation split if enabled,
        otherwise returns the original dataset unchanged.
    """
    if not sampling_config.enabled:
        return dataset

    # Find the source split to use for sampling
    # Prefer 'train', but fall back to any available split
    if "train" in dataset:
        source_split = "train"
    else:
        available_splits = list(dataset.keys())
        if not available_splits:
            logger.warning("No splits found in dataset, skipping sampling")
            return dataset
        source_split = available_splits[0]
        logger.info(
            f"No 'train' split found, using '{source_split}' split for sampling"
        )

    train_ds = dataset[source_split]
    sample_size = sampling_config.sample_size
    stratify_col = sampling_config.stratify_column
    seed = sampling_config.seed

    # Check if stratify column exists
    if stratify_col not in train_ds.column_names:
        logger.warning(
            f"Stratify column '{stratify_col}' not found in dataset. "
            f"Available columns: {train_ds.column_names}. Skipping sampling."
        )
        return dataset

    # Sample if dataset is larger than requested size
    if len(train_ds) > sample_size:
        sampled = train_ds.train_test_split(
            train_size=sample_size, stratify_by_column=stratify_col, seed=seed
        )["train"]
        logger.info(f"Sampled {sample_size} from {len(train_ds)} training examples")
    else:
        sampled = train_ds
        logger.info(
            f"Dataset size ({len(train_ds)}) <= sample_size ({sample_size}), "
            "using full dataset"
        )

    # Split into train/validation
    split_data = sampled.train_test_split(
        test_size=1 - sampling_config.train_ratio,
        stratify_by_column=stratify_col,
        seed=seed,
    )

    dataset["train"] = split_data["train"]
    dataset["validation"] = split_data["test"]

    logger.info(
        f"Stratified split: {len(dataset['train'])} train, "
        f"{len(dataset['validation'])} validation samples"
    )

    return dataset


class EarlyStopper:
    """Handles early stopping logic based on metric monitoring."""

    def __init__(self, config, out_dir: Path = None, trainer=None):
        self.enabled = config.enabled
        self.patience = config.patience
        self.metric = config.metric
        self.mode = config.mode
        self.min_delta = config.min_delta
        self.restore_best = config.restore_best

        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.best_cycle = 0
        self.wait_count = 0
        self.stopped = False
        self.best_model_path = None
        self.out_dir = out_dir
        self.trainer = trainer

    def should_stop(
        self, current_metrics: Dict[str, Any], cycle: int, model=None
    ) -> bool:
        """Check if training should stop based on current metrics."""
        if not self.enabled:
            return False

        if self.metric not in current_metrics:
            logger.warning(
                f"Early stopping metric '{self.metric}' not found in metrics: {list(current_metrics.keys())}"
            )
            return False

        current_value = current_metrics[self.metric]

        # Check for improvement
        improved = False
        if self.mode == "min":
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.best_cycle = cycle
            self.wait_count = 0
            if (
                self.restore_best
                and model is not None
                and self.out_dir
                and self.trainer
            ):
                # Save best model checkpoint to disk (prevents memory leak)
                checkpoint_path = self.out_dir / f"checkpoint_best_cycle_{cycle}"
                checkpoint_path.mkdir(parents=True, exist_ok=True)

                # Temporarily change trainer's out_dir to checkpoint path
                original_out_dir = self.trainer.out_dir
                self.trainer.out_dir = checkpoint_path
                self.trainer.save_model(model)
                # Restore original out_dir
                self.trainer.out_dir = original_out_dir

                # Store checkpoint path instead of model object
                self.best_model_path = checkpoint_path
                logger.info(
                    f"Early stopping: saved best model checkpoint to {checkpoint_path}"
                )
            logger.info(
                f"Early stopping: new best {self.metric} = {current_value:.4f} at cycle {cycle}"
            )
        else:
            self.wait_count += 1
            logger.info(
                f"Early stopping: no improvement in {self.metric} for {self.wait_count}/{self.patience} cycles"
            )

        if self.wait_count >= self.patience:
            self.stopped = True
            logger.info(
                f"Early stopping triggered after {self.patience} cycles without improvement"
            )
            logger.info(
                f"Best {self.metric} = {self.best_value:.4f} at cycle {self.best_cycle}"
            )
            return True

        return False

    def get_best_model(self):
        """Load and return the best model from disk if restore_best is enabled."""
        if self.restore_best and self.best_model_path and self.trainer:
            logger.info(f"Loading best model from {self.best_model_path}")
            return self.trainer.load_model(self.best_model_path)
        return None


class DistillationEngine:
    """Run iterative knowledge distillation."""

    def __init__(self, config: ExperimentConfig, dataset: DatasetDict = None) -> None:
        self.cfg = config

        # Validate that ablation parameters are not lists (use 'ablation' command for that)
        list_params = self.cfg.get_list_parameters()
        if list_params:
            raise ValueError(
                f"List values found for parameters: {list_params}. "
                "Use 'promptillery ablation' command for parameter sweeps, "
                "or provide single values for 'promptillery train'."
            )

        set_seed(42)

        if dataset is not None:
            # Use pre-loaded dataset (e.g., from ablation runner)
            # Create a shallow copy to avoid mutations across configs
            self.dataset = DatasetDict(
                {split: ds.select(range(len(ds))) for split, ds in dataset.items()}
            )
            logger.info("Using pre-loaded dataset")
        else:
            # Load dataset from scratch
            dataset_subset = self.cfg.dataset_subset
            if dataset_subset:
                self.dataset = load_dataset(self.cfg.dataset, dataset_subset)
            else:
                self.dataset = load_dataset(self.cfg.dataset)

            # Ensure label column is ClassLabel type for stratified sampling
            if self.cfg.sampling.enabled:
                self.dataset = ensure_class_label(
                    self.dataset, self.cfg.sampling.stratify_column
                )

            # Apply stratified sampling if configured
            self.dataset = prepare_dataset(self.dataset, self.cfg.sampling)

            # keep track of the origin of every row
            for split, ds in self.dataset.items():
                ds = ds.add_column("source_split", [split] * len(ds))
                ds = ds.add_column("source_idx", [-1] * len(ds))
                ds = ds.add_column("origin_cycle", [0] * len(ds))
                self.dataset[split] = ds

        self.out_dir = self.cfg.get_output_dir()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Save experiment configuration copy to output directory
        config_copy_path = self.out_dir / "experiment_config.yaml"
        with open(config_copy_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.cfg.model_dump(), f, default_flow_style=False, sort_keys=False
            )

        # Create trainer using factory
        self.trainer = TrainerFactory.create_trainer(
            self.cfg, self.dataset, self.out_dir
        )

        # Initialize early stopping with trainer and output directory
        self.early_stopper = EarlyStopper(
            self.cfg.early_stopping, out_dir=self.out_dir, trainer=self.trainer
        )

        # Initialize token tracker
        self.token_tracker = TokenTracker(
            experiment_name=self.cfg.name,
            teacher_model=self.cfg.teacher,
            budget_warning=self.cfg.budget_warning,
            budget_stop=self.cfg.budget_stop,
        )

        # Create Jinja2 environment with format_samples_for_prompt available
        self.jinja_env = create_prompt_environment()
        self.prompt_template = (
            self.jinja_env.from_string(self.cfg.prompt) if self.cfg.prompt else None
        )
        self.prompt_vars = self.cfg.prompt_vars or {}
        self.cfg_vars = self.cfg.model_dump()

        # Initialize component flags based on flat config fields
        self.augmentation_enabled = bool(self.cfg.prompt)
        self.dataset_persistence_enabled = self.cfg.persist_datasets

    def _parse_augmented_response(self, content: str) -> List[Dict[str, Any]]:
        """Parse the teacher model's structured JSON response.

        With structured outputs (response_format=AugmentedResponse), the LLM
        returns valid JSON matching our Pydantic schema directly.
        """
        try:
            response = AugmentedResponse.model_validate_json(content)
            return [
                {"text": article.text, "label": article.label}
                for article in response.articles
            ]
        except Exception as e:
            logger.error(f"Failed to parse structured response: {e}")
            logger.error(f"Raw content: {content[:200]}...")
            return []

    def _save_dataset(self, cycle: int) -> None:
        """Persist the current dataset for a given cycle."""
        if not self.dataset_persistence_enabled:
            logger.info("Dataset persistence disabled, skipping")
            return
        path = self.out_dir / f"dataset_cycle_{cycle}"
        self.dataset.save_to_disk(str(path))

    def _prepare_sample_context(self, model) -> Dict[str, Any]:
        """Prepare sample collections for prompt template context.

        Extracts few-shot examples, high-entropy samples, hard negatives,
        and classification report for use in Jinja2 prompt templates.

        Args:
            model: The trained model for generating predictions

        Returns:
            Dictionary with sample collections and metrics:
            - few_shot_samples: Balanced examples per class
            - high_entropy_samples: Most uncertain predictions
            - hard_negative_samples: High-confidence misclassifications
            - classification_report: Per-class precision/recall/F1 report (on validation/test set)
        """
        train_ds = self.dataset["train"]
        text_field = self.cfg.text_field
        label_field = self.cfg.label_field

        # Extract few-shot examples from training data (no predictions needed)
        few_shot = extract_few_shot_samples(
            train_ds,
            n_per_class=2,
            text_column=text_field,
            label_column=label_field,
        )

        # Get detailed predictions on training data for entropy and hard negative extraction
        train_predictions = self.trainer.get_detailed_predictions(model, split="train")

        # Extract high-entropy samples from training data
        high_entropy = extract_high_entropy_samples(
            train_ds,
            train_predictions,
            top_k=5,
            text_column=text_field,
            label_column=label_field,
        )

        # Extract hard negatives from training data (top misclassified samples by confidence)
        hard_negatives = extract_hard_negatives(
            train_ds,
            train_predictions,
            top_k=5,
            text_column=text_field,
            label_column=label_field,
        )

        # Generate classification report on validation/test set for meaningful metrics
        # Using training data would show inflated 100% accuracy due to memorization
        if "validation" in self.dataset:
            eval_split = "validation"
        elif "test" in self.dataset:
            eval_split = "test"
        else:
            # Fallback to train split if no eval split exists (custom datasets)
            eval_split = "train"
            logger.warning(
                "No validation or test split found, using train split for classification report. "
                "Metrics may be inflated due to memorization."
            )
        eval_predictions = self.trainer.get_detailed_predictions(model, split=eval_split)
        classification_report = format_classification_report(eval_predictions)

        logger.info(
            f"Prepared sample context: {len(few_shot)} few-shot, "
            f"{len(high_entropy)} high-entropy, {len(hard_negatives)} hard negatives"
        )

        return {
            "few_shot_samples": few_shot,
            "high_entropy_samples": high_entropy,
            "hard_negative_samples": hard_negatives,
            "classification_report": classification_report,
        }

    async def _augment(self, model, cycle: int) -> None:
        """Augment training data using the teacher model in batch mode.

        Instead of generating samples per misclassified example, this method:
        1. Collects sample context (few-shot, high-entropy, hard negatives)
        2. Makes a single call to the teacher model
        3. Requests augmentation_batch_size new training samples
        """
        if not self.augmentation_enabled or not self.prompt_template:
            logger.info("Augmentation disabled or no prompt provided, skipping")
            return

        ds = self.dataset["train"]

        # Prepare sample collections for prompt context
        sample_context = self._prepare_sample_context(model)

        # Build template context with all sample collections and config
        env: Dict[str, Any] = {}
        env.update(self.cfg_vars)
        env.update(self.prompt_vars)
        env.update(sample_context)
        # augmentation_batch_size is available in cfg_vars from model_dump()

        # Render the prompt template
        msg = self.prompt_template.render(**env)

        # Save rendered prompt for debugging/inspection
        prompt_path = self.out_dir / f"prompt_cycle_{cycle}.txt"
        prompt_path.write_text(msg, encoding="utf-8")
        logger.info(f"Saved rendered prompt to {prompt_path}")

        logger.info(
            f"Requesting {self.cfg.augmentation_batch_size} augmented samples from teacher model"
        )

        try:
            result = await acompletion(
                model=self.cfg.teacher,
                messages=[{"role": "user", "content": msg}],
                response_format=AugmentedResponse,
            )

            content = result["choices"][0]["message"]["content"]
            articles = self._parse_augmented_response(content)

            if not articles:
                logger.warning("No articles parsed from teacher response")
                return

            logger.info(f"Received {len(articles)} augmented samples from teacher")

            # Get field names from config
            text_field = self.cfg.text_field
            label_field = self.cfg.label_field

            # Record token usage from response
            self.token_tracker.record_usage(result, OperationType.AUGMENTATION)

            # Create new rows from augmented articles
            aug_rows: List[Dict[str, Any]] = []
            for article in articles:
                # Create a new row with required columns using configured field names
                row = {
                    text_field: article["text"],
                    label_field: article["label"],
                    "source_split": "augmented",
                    "source_idx": -1,
                    "origin_cycle": cycle,
                }
                # Add any additional columns from the original dataset with default values
                for col in ds.column_names:
                    if col not in row:
                        # Use first row's value type to create appropriate default
                        sample_val = ds[0][col]
                        if isinstance(sample_val, str):
                            row[col] = ""
                        elif isinstance(sample_val, (int, float)):
                            row[col] = 0
                        else:
                            row[col] = None
                aug_rows.append(row)

        except Exception as e:
            logger.error(f"Batch augmentation failed: {e}")
            return

        if not aug_rows:
            return

        columns = aug_rows[0].keys()
        data_dict = {c: [r[c] for r in aug_rows] for c in columns}
        extra = Dataset.from_dict(data_dict)
        extra = extra.cast(ds.features)
        self.dataset["train"] = concatenate_datasets([ds, extra])
        logger.info(
            f"Added {len(aug_rows)} augmented samples to training set "
            f"(total: {len(self.dataset['train'])})"
        )

    async def run(self) -> Dict[str, Any]:
        results = {}
        final_model = None

        try:
            for cycle in range(self.cfg.cycles):
                # Use context manager for robust cycle lifecycle
                with self.token_tracker.cycle(cycle):
                    # if cycle > 0:
                    #     await self._pseudo_label("train")
                    model = self.trainer.train()
                    metrics = self.trainer.evaluate(model)
                    logger.info("Cycle %d metrics: %s", cycle, metrics)
                    results[str(cycle)] = metrics
                    final_model = model

                    # Check for early stopping
                    should_stop = self.early_stopper.should_stop(metrics, cycle, model)
                    if should_stop:
                        logger.info(f"Training stopped early at cycle {cycle}")
                        # Use best model if restore_best is enabled
                        best_model = self.early_stopper.get_best_model()
                        if best_model is not None:
                            final_model = best_model
                            logger.info(
                                f"Restored best model from cycle {self.early_stopper.best_cycle}"
                            )

                    if not should_stop and cycle < self.cfg.cycles - 1:
                        await self._augment(model, cycle)
                    self._save_dataset(cycle)  # save the dataset after the augmentation
                # Print cycle summary after context manager closes
                self.token_tracker.print_cycle_summary()

                # Check for budget stop
                if self.token_tracker.should_stop_for_budget():
                    logger.info(
                        f"Training stopped due to budget limit at cycle {cycle}"
                    )
                    break

                if self.early_stopper.stopped:
                    break

            # Print final token usage summary
            self.token_tracker.print_final_summary()

        finally:
            # Always save token usage, even on error/interruption
            if self.token_tracker.summary.cycles_completed > 0:
                self.token_tracker.save(self.out_dir)

            # Always save metrics.json, even on error/interruption
            # This ensures partial results are preserved if training fails mid-way
            try:
                # Add early stopping info to results
                if self.early_stopper.enabled:
                    results["early_stopping"] = {
                        "triggered": self.early_stopper.stopped,
                        "best_cycle": self.early_stopper.best_cycle,
                        "best_value": self.early_stopper.best_value,
                        "metric": self.early_stopper.metric,
                        "total_cycles": len([k for k in results.keys() if k.isdigit()]),
                    }

                # Add budget stop info to results
                if self.cfg.budget_warning is not None:
                    results["budget_control"] = {
                        "budget_limit": self.cfg.budget_warning,
                        "budget_stop_enabled": self.cfg.budget_stop,
                        "budget_exceeded": self.token_tracker._budget_exceeded,
                        "stopped_for_budget": self.token_tracker.should_stop_for_budget(),
                        "total_cost": self.token_tracker.summary.grand_total.estimated_cost,
                        "total_cycles": len([k for k in results.keys() if k.isdigit()]),
                    }

                if results:  # Only save if we have some results
                    Path(self.out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
            except Exception as e:
                logger.error(f"Failed to save metrics.json: {e}", exc_info=True)

        # Save final model and handle uploading
        if final_model is not None:
            self.trainer.save_model(final_model)
            if self.cfg.output_repo:
                self.trainer.push_to_hub(final_model, self.cfg.output_repo)

        return results


def evaluate_model(
    config: ExperimentConfig, model_path: Path, split: str = "test"
) -> Dict[str, Any]:
    """Evaluate a trained model on a specified dataset split.

    Args:
        config: Experiment configuration
        model_path: Path to the trained model checkpoint
        split: Dataset split to evaluate on (default: "test")

    Returns:
        Dictionary containing evaluation metrics and metadata
    """
    set_seed(42)

    logger.info(f"Loading dataset: {config.dataset}")

    # Load dataset - use dataset_subset to get the string name
    # (handles both old string format and new DatasetConfig format)
    dataset_subset = config.dataset_subset
    if dataset_subset:
        dataset = load_dataset(config.dataset, dataset_subset)
    else:
        dataset = load_dataset(config.dataset)

    # Verify split exists
    if split not in dataset:
        available_splits = list(dataset.keys())
        logger.error(
            f"Split '{split}' not found in dataset. Available splits: {available_splits}"
        )
        raise ValueError(f"Split '{split}' not found. Available: {available_splits}")

    # Create trainer using factory
    trainer = TrainerFactory.create_trainer(config, dataset, model_path)

    # Load the trained model
    logger.info(f"Loading model from {model_path}")
    model = trainer.load_model(model_path)

    # Run evaluation using trainer's evaluate method
    logger.info(f"Evaluating on {split} split...")
    metrics = trainer.evaluate(model, split=split)

    # Display results
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset: {config.dataset} ({split} split)")
    logger.info("-" * 50)

    for metric_name, score in metrics.items():
        logger.info(f"{metric_name}: {score:.4f}")

    logger.info("=" * 50)

    # Save results
    results = {
        "model_path": str(model_path),
        "dataset": config.dataset,
        "dataset_subset": dataset_subset,
        "split": split,
        "metrics": metrics,
    }

    results_path = model_path / f"eval_results_{split}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {results_path}")

    return results
