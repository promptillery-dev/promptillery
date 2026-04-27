"""Core distillation engine."""

from __future__ import annotations

import json
import logging
import shutil
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
from .policy_decisions import PolicyDecision, PolicyDecisionLogger
from .policy_features import build_policy_features
from .token_tracker import OperationType, TokenTracker, TokenUsage
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

try:
    from litellm import token_counter as _token_counter

    _HAS_TOKEN_COUNTER = True
except ImportError:
    _token_counter = None  # type: ignore[assignment]
    _HAS_TOKEN_COUNTER = False


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


def _response_to_jsonable(response: Any) -> Any:
    """Convert a LiteLLM response into data that can be JSON archived."""
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    return str(response)


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


def ensure_validation_split(
    dataset: DatasetDict, config: ExperimentConfig
) -> DatasetDict:
    """Create a validation split from train when experiments require one."""
    if "validation" in dataset or not config.require_validation_split:
        return dataset
    if "train" not in dataset:
        raise ValueError(
            "require_validation_split=true but dataset has no train split to split"
        )
    if len(dataset["train"]) < 2:
        raise ValueError(
            "require_validation_split=true but train split has fewer than two rows"
        )

    seed = config.seed if isinstance(config.seed, int) else config.sampling.seed
    split_data = dataset["train"].train_test_split(
        test_size=1 - config.sampling.train_ratio,
        seed=seed,
    )
    dataset["train"] = split_data["train"]
    dataset["validation"] = split_data["test"]
    logger.info(
        "Created validation split from train: %d train, %d validation samples",
        len(dataset["train"]),
        len(dataset["validation"]),
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

        set_seed(self.cfg.seed)

        if dataset is not None:
            # Use pre-loaded dataset (e.g., from ablation runner)
            # Create a shallow copy to avoid mutations across configs
            self.dataset = DatasetDict(
                {split: ds.select(range(len(ds))) for split, ds in dataset.items()}
            )
            self.dataset = ensure_validation_split(self.dataset, self.cfg)
            logger.info("Using pre-loaded dataset")
        else:
            # Load dataset from scratch
            dataset_subset = self.cfg.dataset_subset
            dataset_kwargs = self.cfg.dataset_kwargs or {}
            if dataset_subset:
                self.dataset = load_dataset(
                    self.cfg.dataset, dataset_subset, **dataset_kwargs
                )
            else:
                self.dataset = load_dataset(self.cfg.dataset, **dataset_kwargs)

            # Ensure label column is ClassLabel type for stratified sampling
            if self.cfg.sampling.enabled:
                self.dataset = ensure_class_label(
                    self.dataset, self.cfg.sampling.stratify_column
                )

            # Apply stratified sampling if configured
            self.dataset = prepare_dataset(self.dataset, self.cfg.sampling)
            self.dataset = ensure_validation_split(self.dataset, self.cfg)

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

        if self.cfg.price_table_path:
            price_table_path = Path(self.cfg.price_table_path)
            if not price_table_path.exists():
                raise FileNotFoundError(
                    f"Configured price_table_path does not exist: {price_table_path}"
                )
            shutil.copy2(price_table_path, self.out_dir / "price_table.yaml")

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
            token_budget=self.cfg.token_budget,
            budget_stop=self.cfg.budget_stop,
        )
        self.policy_decision_logger = PolicyDecisionLogger(
            self.out_dir / "policy_decisions.jsonl"
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
        self._external_sft_usage_recorded = False

    def _budget_snapshot(self, include_current_cycle: bool = True) -> Dict[str, Any]:
        """Return current budget accounting state for decision logs."""
        grand_total = self.token_tracker.summary.grand_total
        current_usage = self.token_tracker.current_cycle_usage()
        spent = grand_total.estimated_cost
        if include_current_cycle and current_usage.estimated_cost is not None:
            spent = (spent or 0.0) + current_usage.estimated_cost

        total_tokens = grand_total.total_tokens
        if include_current_cycle:
            total_tokens += current_usage.total_tokens

        tokens_remaining = None
        if self.cfg.token_budget is not None:
            tokens_remaining = self.cfg.token_budget - total_tokens

        remaining = None
        if self.cfg.budget_warning is not None and spent is not None:
            remaining = self.cfg.budget_warning - spent

        return {
            "budget_limit_usd": self.cfg.budget_warning,
            "token_budget": self.cfg.token_budget,
            "spent_usd": spent,
            "remaining_usd": remaining,
            "budget_stop": self.cfg.budget_stop,
            "total_tokens": total_tokens,
            "tokens_remaining": tokens_remaining,
        }

    def _cycle_eval_split(self) -> str:
        """Return the split used for cycle rewards and policy context."""
        if "validation" in self.dataset:
            return "validation"
        if "test" in self.dataset:
            logger.warning(
                "No validation split found; using test split for cycle evaluation. "
                "Use a validation split for policy rewards and reserve test for final reporting."
            )
            return "test"

        logger.warning(
            "No validation or test split found; using train split for cycle evaluation. "
            "Metrics may be inflated due to memorization."
        )
        return "train"

    def _cycle_state(
        self,
        cycle: int,
        metrics: Dict[str, Any],
        eval_split: str,
        policy_features: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        """Build a compact state snapshot for fixed and learned policies."""
        train_size = len(self.dataset["train"]) if "train" in self.dataset else None
        validation_size = (
            len(self.dataset["validation"]) if "validation" in self.dataset else None
        )
        test_size = len(self.dataset["test"]) if "test" in self.dataset else None
        synthetic_count = None
        if (
            "train" in self.dataset
            and "source_split" in self.dataset["train"].column_names
        ):
            synthetic_count = sum(
                1
                for value in self.dataset["train"]["source_split"]
                if value == "augmented"
            )

        return {
            "cycle": cycle,
            "cycles": self.cfg.cycles,
            "eval_split": eval_split,
            "metrics": metrics,
            "train_size": train_size,
            "validation_size": validation_size,
            "test_size": test_size,
            "synthetic_count": synthetic_count,
            "budget": self._budget_snapshot(),
            "features": policy_features or {},
        }

    def _record_policy_decision(
        self,
        *,
        cycle: int,
        action_name: str,
        state: Dict[str, Any],
        budget_before: Dict[str, Any],
        predicted_cost: Dict[str, Any] | None = None,
        decision_metadata: Dict[str, Any] | None = None,
        reward: float | None = None,
    ) -> None:
        """Record the fixed-loop decision using the policy decision schema."""
        realized_cost = self.token_tracker.current_cycle_usage().model_dump()
        metadata = {
            "seed": self.cfg.seed,
            "student": self.cfg.student,
            "student_type": self.cfg.student_type,
        }
        metadata.update(decision_metadata or {})

        self.policy_decision_logger.record(
            PolicyDecision(
                cycle=cycle,
                policy_name="fixed_promptillery",
                action_name=action_name,
                state=state,
                action={
                    "teacher": self.cfg.teacher,
                    "augmentation_batch_size": self.cfg.augmentation_batch_size,
                    "prompt_enabled": self.augmentation_enabled,
                    "teacher_max_output_tokens": self.cfg.teacher_max_output_tokens,
                },
                predicted_cost=predicted_cost or {},
                realized_cost=realized_cost,
                reward=reward,
                budget_before=budget_before,
                budget_after=self._budget_snapshot(),
                metadata=metadata,
            )
        )

    def _estimate_teacher_call_tokens(
        self, messages: List[Dict[str, str]], budget_before: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Conservatively estimate token use for one teacher call."""
        estimator = "chars_per_3_fallback"
        estimator_error = None

        if _HAS_TOKEN_COUNTER and _token_counter is not None:
            try:
                input_tokens = int(
                    _token_counter(model=self.cfg.teacher, messages=messages)
                )
                estimator = "litellm.token_counter"
            except Exception as exc:
                estimator_error = f"{type(exc).__name__}: {exc}"
                input_tokens = self._estimate_tokens_from_chars(messages)
        else:
            input_tokens = self._estimate_tokens_from_chars(messages)

        max_output_tokens = self.cfg.teacher_max_output_tokens
        predicted_total = None
        if max_output_tokens is not None:
            predicted_total = input_tokens + max_output_tokens

        tokens_remaining = budget_before.get("tokens_remaining")
        token_budget = budget_before.get("token_budget")
        preflight_enforced = token_budget is not None and max_output_tokens is not None
        allowed = True
        reason = None

        if token_budget is not None and max_output_tokens is None:
            allowed = False
            reason = "teacher_max_output_tokens_not_set"
        elif (
            preflight_enforced
            and predicted_total is not None
            and tokens_remaining is not None
            and predicted_total > tokens_remaining
        ):
            allowed = False
            reason = "predicted_tokens_exceed_remaining_budget"

        return {
            "input_tokens": input_tokens,
            "max_output_tokens": max_output_tokens,
            "total_tokens": predicted_total,
            "tokens_remaining": tokens_remaining,
            "token_budget": token_budget,
            "allowed": allowed,
            "preflight_enforced": preflight_enforced,
            "estimator": estimator,
            "estimator_error": estimator_error,
            "reason": reason,
        }

    def _record_external_sft_token_usage(self) -> None:
        """Charge pre-materialized SFT teacher-token usage to the current cycle."""
        if self.cfg.student_type != "causal_lm_sft" or self._external_sft_usage_recorded:
            return

        trainer_config = self.cfg.trainer_config or {}
        budget_splits = trainer_config.get("budget_splits", ["train"])
        required = bool(trainer_config.get("require_teacher_token_fields", True))
        fields = (
            "teacher_input_tokens",
            "teacher_output_tokens",
            "teacher_total_tokens",
        )

        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        charged_rows = 0

        for split in budget_splits:
            if split not in self.dataset:
                if required:
                    raise ValueError(
                        f"SFT budget split '{split}' is missing from the dataset"
                    )
                continue

            ds = self.dataset[split]
            missing = [field for field in fields if field not in ds.column_names]
            if missing:
                if required:
                    raise ValueError(
                        "Pre-materialized SFT data must include teacher token "
                        f"fields {fields}; missing {missing} in split '{split}'"
                    )
                logger.warning(
                    "Skipping SFT token accounting for split %s because fields are missing: %s",
                    split,
                    missing,
                )
                continue

            for row in ds:
                row_input = int(row["teacher_input_tokens"] or 0)
                row_output = int(row["teacher_output_tokens"] or 0)
                row_total = int(row["teacher_total_tokens"] or 0)
                if row_total != row_input + row_output:
                    raise ValueError(
                        "Invalid SFT token accounting: teacher_total_tokens must "
                        "equal teacher_input_tokens + teacher_output_tokens"
                    )
                input_tokens += row_input
                output_tokens += row_output
                total_tokens += row_total
                charged_rows += 1

        if self.cfg.token_budget is not None and total_tokens > self.cfg.token_budget:
            raise ValueError(
                f"Pre-materialized SFT data uses {total_tokens:,} teacher tokens, "
                f"exceeding configured token_budget={self.cfg.token_budget:,}"
            )

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=None,
        )
        self.token_tracker.record_manual_usage(usage, OperationType.SFT_DATA)
        self._external_sft_usage_recorded = True
        logger.info(
            "Charged %d pre-materialized SFT rows to teacher-token budget: %d tokens",
            charged_rows,
            total_tokens,
        )

    @staticmethod
    def _estimate_tokens_from_chars(messages: List[Dict[str, str]]) -> int:
        """Fallback token estimate when the model tokenizer is unavailable."""
        total_chars = sum(len(str(message.get("content", ""))) for message in messages)
        return max(1, (total_chars + 2) // 3)

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

    def _prepare_sample_context(
        self,
        model,
        *,
        cycle: int | None = None,
        metrics: Dict[str, Any] | None = None,
        previous_metrics: Dict[str, Any] | None = None,
        budget: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
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

        # Generate classification report on the same split used for cycle rewards.
        # Using training data would show inflated accuracy due to memorization.
        eval_split = self._cycle_eval_split()
        eval_predictions = self.trainer.get_detailed_predictions(model, split=eval_split)
        classification_report = format_classification_report(eval_predictions)

        logger.info(
            f"Prepared sample context: {len(few_shot)} few-shot, "
            f"{len(high_entropy)} high-entropy, {len(hard_negatives)} hard negatives"
        )

        context = {
            "few_shot_samples": few_shot,
            "high_entropy_samples": high_entropy,
            "hard_negative_samples": hard_negatives,
            "classification_report": classification_report,
        }

        if cycle is not None and metrics is not None:
            synthetic_count = None
            if "source_split" in train_ds.column_names:
                synthetic_count = sum(
                    1 for value in train_ds["source_split"] if value == "augmented"
                )
            context["policy_features"] = build_policy_features(
                cycle=cycle,
                cycles=self.cfg.cycles,
                metrics=metrics,
                previous_metrics=previous_metrics,
                train_predictions=train_predictions,
                eval_predictions=eval_predictions,
                train_size=len(train_ds),
                synthetic_count=synthetic_count,
                budget=budget or self._budget_snapshot(),
                num_classes=self.cfg.num_classes,
            )

        return context

    async def _augment(
        self,
        model,
        cycle: int,
        sample_context: Dict[str, Any] | None = None,
        budget_before: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Augment training data using the teacher model in batch mode.

        Instead of generating samples per misclassified example, this method:
        1. Collects sample context (few-shot, high-entropy, hard negatives)
        2. Makes a single call to the teacher model
        3. Requests augmentation_batch_size new training samples
        """
        if not self.augmentation_enabled or not self.prompt_template:
            logger.info("Augmentation disabled or no prompt provided, skipping")
            return {
                "action_name": "skip",
                "predicted_cost": {},
                "metadata": {"skip_reason": "augmentation_disabled"},
            }

        ds = self.dataset["train"]

        # Prepare sample collections for prompt context
        sample_context = sample_context or self._prepare_sample_context(model)

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

        messages = [{"role": "user", "content": msg}]
        predicted_cost = self._estimate_teacher_call_tokens(
            messages, budget_before or self._budget_snapshot()
        )
        if not predicted_cost["allowed"]:
            logger.warning(
                "Skipping augmentation because budget preflight masked the teacher call: %s",
                predicted_cost["reason"],
            )
            return {
                "action_name": "budget_masked",
                "predicted_cost": predicted_cost,
                "metadata": {"mask_reason": predicted_cost["reason"]},
            }

        try:
            completion_kwargs = {
                "model": self.cfg.teacher,
                "messages": messages,
                "response_format": AugmentedResponse,
            }
            if self.cfg.teacher_max_output_tokens is not None:
                completion_kwargs["max_tokens"] = self.cfg.teacher_max_output_tokens
            result = await acompletion(**completion_kwargs)
            self.token_tracker.record_usage(result, OperationType.AUGMENTATION)
            response_path = self.out_dir / f"teacher_response_cycle_{cycle}.json"
            response_path.write_text(
                json.dumps(_response_to_jsonable(result), indent=2, default=str),
                encoding="utf-8",
            )

            content = result["choices"][0]["message"]["content"]
            articles = self._parse_augmented_response(content)

            if not articles:
                logger.warning("No articles parsed from teacher response")
                return {
                    "action_name": "augment_empty",
                    "predicted_cost": predicted_cost,
                    "metadata": {"articles_parsed": 0, "articles_added": 0},
                }

            logger.info(f"Received {len(articles)} augmented samples from teacher")

            # Get field names from config
            text_field = self.cfg.text_field
            label_field = self.cfg.label_field

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
            return {
                "action_name": "augment_failed",
                "predicted_cost": predicted_cost,
                "metadata": {
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            }

        if not aug_rows:
            return {
                "action_name": "augment_empty",
                "predicted_cost": predicted_cost,
                "metadata": {"articles_parsed": 0, "articles_added": 0},
            }

        columns = aug_rows[0].keys()
        data_dict = {c: [r[c] for r in aug_rows] for c in columns}
        extra = Dataset.from_dict(data_dict)
        extra = extra.cast(ds.features)
        self.dataset["train"] = concatenate_datasets([ds, extra])
        logger.info(
            f"Added {len(aug_rows)} augmented samples to training set "
            f"(total: {len(self.dataset['train'])})"
        )
        return {
            "action_name": "augment",
            "predicted_cost": predicted_cost,
            "metadata": {
                "articles_parsed": len(aug_rows),
                "articles_added": len(aug_rows),
            },
        }

    async def run(self) -> Dict[str, Any]:
        results = {}
        final_model = None
        eval_split = self._cycle_eval_split()

        try:
            for cycle in range(self.cfg.cycles):
                # Use context manager for robust cycle lifecycle
                with self.token_tracker.cycle(cycle):
                    if cycle == 0:
                        self._record_external_sft_token_usage()
                    # if cycle > 0:
                    #     await self._pseudo_label("train")
                    model = self.trainer.train()
                    metrics = self.trainer.evaluate(model, split=eval_split)
                    logger.info(
                        "Cycle %d metrics on %s split: %s",
                        cycle,
                        eval_split,
                        metrics,
                    )
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

                    previous_metrics = results.get(str(cycle - 1), {})
                    budget_before = self._budget_snapshot()
                    sample_context = None
                    policy_features = {}
                    if self.augmentation_enabled:
                        sample_context = self._prepare_sample_context(
                            model,
                            cycle=cycle,
                            metrics=metrics,
                            previous_metrics=previous_metrics,
                            budget=budget_before,
                        )
                        policy_features = sample_context.get("policy_features", {})
                        feature_path = self.out_dir / f"policy_features_cycle_{cycle}.json"
                        feature_path.write_text(
                            json.dumps(policy_features, indent=2),
                            encoding="utf-8",
                        )

                    cycle_state = self._cycle_state(
                        cycle,
                        metrics,
                        eval_split=eval_split,
                        policy_features=policy_features,
                    )
                    action_name = "stop"
                    predicted_cost = {}
                    decision_metadata = {}

                    if not should_stop and cycle < self.cfg.cycles - 1:
                        action_name = "augment" if self.augmentation_enabled else "skip"
                        if self.augmentation_enabled:
                            outcome = await self._augment(
                                model,
                                cycle,
                                sample_context=sample_context,
                                budget_before=budget_before,
                            )
                            action_name = outcome["action_name"]
                            predicted_cost = outcome["predicted_cost"]
                            decision_metadata = outcome["metadata"]
                    elif cycle >= self.cfg.cycles - 1:
                        action_name = "final_cycle"

                    self._record_policy_decision(
                        cycle=cycle,
                        action_name=action_name,
                        state=cycle_state,
                        budget_before=budget_before,
                        predicted_cost=predicted_cost,
                        decision_metadata=decision_metadata,
                    )
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
                if (
                    self.cfg.budget_warning is not None
                    or self.cfg.token_budget is not None
                ):
                    results["budget_control"] = {
                        "budget_limit": self.cfg.budget_warning,
                        "token_budget": self.cfg.token_budget,
                        "budget_stop_enabled": self.cfg.budget_stop,
                        "budget_exceeded": self.token_tracker._budget_exceeded,
                        "stopped_for_budget": self.token_tracker.should_stop_for_budget(),
                        "total_cost": self.token_tracker.summary.grand_total.estimated_cost,
                        "total_tokens": self.token_tracker.summary.grand_total.total_tokens,
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
    if isinstance(config.seed, list):
        raise ValueError("evaluate_model requires a single seed value, not a list")
    set_seed(config.seed)

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
