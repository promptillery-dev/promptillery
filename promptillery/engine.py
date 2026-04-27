"""Core distillation engine."""

from __future__ import annotations

import json
import logging
import shutil
from hashlib import sha256
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
from .policy_controller import PolicyAction, PolicyController, enumerate_actions
from .policy_decisions import PolicyDecision, PolicyDecisionLogger
from .policy_features import build_policy_features
from .token_tracker import (
    OperationType,
    TokenTracker,
    TokenUsage,
    extract_usage_from_response,
)
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


class AugmentedSFTRecord(BaseModel):
    """Single synthetic SFT prompt/response pair."""

    student_prompt: str
    teacher_response: str
    gold_answer: str | None = None
    source_example_id: str | None = None


class AugmentedSFTResponse(BaseModel):
    """Structured response for SFT augmentation."""

    records: List[AugmentedSFTRecord]


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


def require_paper_splits(dataset: DatasetDict) -> None:
    """Require strict split discipline for paper-mode experiments."""
    required = {"train", "validation", "test"}
    missing = sorted(required.difference(dataset.keys()))
    if missing:
        raise ValueError(
            "paper_mode=true requires train, validation, and test splits; "
            f"missing {missing}"
        )


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
            if self.cfg.paper_mode:
                require_paper_splits(self.dataset)
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
            if self.cfg.paper_mode:
                require_paper_splits(self.dataset)

            # keep track of the origin of every row
            for split, ds in self.dataset.items():
                ds = ds.add_column("source_split", [split] * len(ds))
                ds = ds.add_column("source_idx", [-1] * len(ds))
                ds = ds.add_column("origin_cycle", [0] * len(ds))
                self.dataset[split] = ds

        self.out_dir = self.cfg.get_output_dir()
        self.out_dir.mkdir(parents=True, exist_ok=False)
        self.run_id = self.out_dir.name

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
        self.policy_controller = self._build_policy_controller()
        self.action_space = self._build_action_space()
        self.action_space_id = self._action_space_id(self.action_space)
        self._decision_counter = 0
        self._attempt_counter = 0

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

    def _build_policy_controller(self) -> PolicyController | None:
        """Return a controller for policy-driven acquisition when configured."""
        if self.cfg.policy_name == "fixed_promptillery":
            return None
        return PolicyController(
            self.cfg.policy_name,
            lambda_cost=self.cfg.policy_lambda_cost,
            exploration_bonus=self.cfg.policy_exploration_bonus,
            seed=self.cfg.seed,
        )

    def _build_action_space(self) -> List[PolicyAction]:
        """Build the finite action class for acquisition policies."""
        teacher_tiers = list(self.cfg.policy_teacher_tiers.keys()) or ["cheap", "strong"]
        return enumerate_actions(
            prompt_operators=self.cfg.policy_prompt_operators,
            teacher_tiers=teacher_tiers,
            batch_sizes=self.cfg.policy_batch_sizes,
            include_stop=True,
        )

    @staticmethod
    def _action_space_id(actions: List[PolicyAction]) -> str:
        """Stable hash of the action space recorded in paper manifests."""
        payload = [action.model_dump() for action in actions]
        return sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]

    def _next_decision_id(self, cycle: int) -> str:
        self._decision_counter += 1
        return f"{self.run_id}:c{cycle}:d{self._decision_counter}"

    def _next_attempt_id(self, cycle: int) -> str:
        self._attempt_counter += 1
        return f"{self.run_id}:c{cycle}:a{self._attempt_counter}"

    def _teacher_for_action(self, action: PolicyAction | None = None) -> str:
        """Resolve the provider model for a policy action."""
        default_teacher = (
            self.cfg.teacher[0] if isinstance(self.cfg.teacher, list) else self.cfg.teacher
        )
        if action and action.teacher_tier:
            return self.cfg.policy_teacher_tiers.get(action.teacher_tier, default_teacher)
        return default_teacher

    @staticmethod
    def _prompt_focus(prompt_operator: str | None) -> str:
        """Human-readable focus string exposed to prompt templates."""
        return {
            "coverage": "broad coverage of underrepresented classes and intents",
            "boundary": "borderline examples near likely decision boundaries",
            "repair": "student mistakes, confusions, and high-confidence errors",
        }.get(str(prompt_operator), "useful examples for the current student")

    def _render_augmentation_prompt(
        self,
        sample_context: Dict[str, Any],
        action: PolicyAction | None = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Render the augmentation prompt with policy action overrides."""
        batch_size = (
            action.batch_size
            if action and not action.is_stop and action.batch_size
            else self.cfg.augmentation_batch_size
        )
        batch_size = self._effective_acquisition_batch_size(int(batch_size))
        teacher_model = self._teacher_for_action(action)
        prompt_operator = action.prompt_operator if action else None
        teacher_tier = action.teacher_tier if action else None

        env: Dict[str, Any] = {}
        env.update(self.cfg_vars)
        env.update(self.prompt_vars)
        env.update(sample_context)
        env.update(
            {
                "augmentation_batch_size": batch_size,
                "policy_action": action.model_dump() if action else {},
                "prompt_operator": prompt_operator,
                "prompt_focus": self._prompt_focus(prompt_operator),
                "teacher_tier": teacher_tier,
                "teacher_model": teacher_model,
            }
        )
        return self.prompt_template.render(**env), env

    def _policy_predicted_costs(
        self,
        actions: List[PolicyAction],
        sample_context: Dict[str, Any],
        budget_before: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Estimate preflight costs for every action before controller choice."""
        predicted_costs: Dict[str, Any] = {}
        if not self.prompt_template:
            return predicted_costs
        for action in actions:
            if action.is_stop:
                predicted_costs[action.name] = {"total_tokens": 0}
                continue
            prompt, _ = self._render_augmentation_prompt(sample_context, action)
            messages = [{"role": "user", "content": prompt}]
            predicted_costs[action.name] = self._estimate_teacher_call_tokens(
                messages,
                budget_before,
                teacher_model=self._teacher_for_action(action),
            )
        return predicted_costs

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
        if self.cfg.paper_mode:
            raise ValueError(
                "paper_mode=true requires a validation split for cycle rewards"
            )
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

    def _current_synthetic_count(self) -> int | None:
        """Return accepted synthetic rows in the current train split."""
        if (
            "train" not in self.dataset
            or "source_split" not in self.dataset["train"].column_names
        ):
            return None
        return sum(
            1
            for value in self.dataset["train"]["source_split"]
            if value == "augmented"
        )

    def _synthetic_records_remaining(self) -> int | None:
        """Return remaining accepted synthetic row slots, if capped."""
        synthetic_record_budget = getattr(self.cfg, "synthetic_record_budget", None)
        if synthetic_record_budget is None:
            return None
        current_synthetic = self._current_synthetic_count() or 0
        return max(0, int(synthetic_record_budget) - current_synthetic)

    def _effective_acquisition_batch_size(self, requested_batch_size: int) -> int:
        """Cap teacher-requested rows by remaining same-count control slots."""
        remaining_records = self._synthetic_records_remaining()
        if remaining_records is None:
            return requested_batch_size
        return min(requested_batch_size, remaining_records)

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
        synthetic_count = self._current_synthetic_count()

        return {
            "cycle": cycle,
            "cycles": self.cfg.cycles,
            "eval_split": eval_split,
            "metrics": metrics,
            "train_size": train_size,
            "validation_size": validation_size,
            "test_size": test_size,
            "synthetic_count": synthetic_count,
            "synthetic_record_budget": getattr(
                self.cfg, "synthetic_record_budget", None
            ),
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
        decision_id: str,
        policy_name: str | None = None,
        action: Dict[str, Any] | None = None,
        action_scores: Dict[str, float] | None = None,
        predicted_cost: Dict[str, Any] | None = None,
        decision_metadata: Dict[str, Any] | None = None,
        reward: float | None = None,
    ) -> None:
        """Record one acquisition decision using the policy decision schema."""
        realized_cost = self.token_tracker.current_cycle_usage().model_dump()
        metadata = {
            "schema_version": 1,
            "run_id": self.run_id,
            "decision_id": decision_id,
            "seed": self.cfg.seed,
            "student": self.cfg.student,
            "student_type": self.cfg.student_type,
            "action_space_id": self.action_space_id,
        }
        metadata.update(decision_metadata or {})

        self.policy_decision_logger.record(
            PolicyDecision(
                schema_version=1,
                run_id=self.run_id,
                decision_id=decision_id,
                cycle=cycle,
                policy_name=policy_name or self.cfg.policy_name,
                action_name=action_name,
                state=state,
                action=action
                or {
                    "teacher": self.cfg.teacher,
                    "augmentation_batch_size": self.cfg.augmentation_batch_size,
                    "prompt_enabled": self.augmentation_enabled,
                    "teacher_max_output_tokens": self.cfg.teacher_max_output_tokens,
                },
                action_scores=action_scores or {},
                predicted_cost=predicted_cost or {},
                realized_cost=realized_cost,
                reward=reward,
                budget_before=budget_before,
                budget_after=self._budget_snapshot(),
                metadata=metadata,
            )
        )

    def _record_teacher_attempt(
        self,
        *,
        cycle: int,
        status: str,
        predicted_cost: Dict[str, Any],
        budget_before: Dict[str, Any],
        attempt_id: str,
        decision_id: str | None = None,
        realized_cost: TokenUsage | None = None,
        failure_type: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Append an audited teacher-call attempt for budget debugging."""
        row = {
            "schema_version": 1,
            "run_id": self.run_id,
            "attempt_id": attempt_id,
            "decision_id": decision_id,
            "cycle": cycle,
            "status": status,
            "predicted_cost": predicted_cost,
            "realized_cost": realized_cost.model_dump() if realized_cost else {},
            "failure_type": failure_type,
            "budget_before": budget_before,
            "budget_after": self._budget_snapshot(),
            "metadata": metadata or {},
        }
        attempts_path = self.out_dir / "teacher_attempts.jsonl"
        with attempts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    @staticmethod
    def _reserved_usage_from_preflight(predicted_cost: Dict[str, Any]) -> TokenUsage:
        """Return conservative reserved usage for a failed teacher attempt."""
        input_tokens = int(predicted_cost.get("input_tokens") or 0)
        output_tokens = int(predicted_cost.get("max_output_tokens") or 0)
        total_tokens = int(predicted_cost.get("total_tokens") or 0)
        if total_tokens <= 0:
            total_tokens = input_tokens + output_tokens
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=None,
        )

    def _estimate_teacher_call_tokens(
        self,
        messages: List[Dict[str, str]],
        budget_before: Dict[str, Any],
        teacher_model: str | None = None,
    ) -> Dict[str, Any]:
        """Conservatively estimate token use for one teacher call."""
        estimator = "chars_per_3_fallback"
        estimator_error = None
        teacher_model = teacher_model or self.cfg.teacher

        if _HAS_TOKEN_COUNTER and _token_counter is not None:
            try:
                input_tokens = int(
                    _token_counter(model=teacher_model, messages=messages)
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
            "teacher_model": teacher_model,
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
                if row_total < row_input + row_output:
                    raise ValueError(
                        "Invalid SFT token accounting: teacher_total_tokens must "
                        "be at least teacher_input_tokens + teacher_output_tokens"
                    )
                if (
                    self.cfg.paper_mode
                    and "usage_estimated" in ds.column_names
                    and bool(row["usage_estimated"])
                ):
                    raise ValueError(
                        "paper_mode=true rejects SFT rows with usage_estimated=true"
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

    def _augmentation_response_format(self):
        """Return the structured-output schema for the active student type."""
        if self.cfg.student_type == "causal_lm_sft":
            return AugmentedSFTResponse
        return AugmentedResponse

    def _parse_augmented_response(self, content: str) -> List[Dict[str, Any]]:
        """Parse the teacher model's structured JSON response.

        With structured outputs, the LLM returns valid JSON matching the
        response schema for either classifier rows or SFT prompt/response rows.
        """
        try:
            if self.cfg.student_type == "causal_lm_sft":
                response = AugmentedSFTResponse.model_validate_json(content)
                return [
                    {
                        "student_prompt": record.student_prompt,
                        "teacher_response": record.teacher_response,
                        "gold_answer": record.gold_answer,
                        "source_example_id": record.source_example_id,
                    }
                    for record in response.records
                ]
            response = AugmentedResponse.model_validate_json(content)
            return [
                {"text": article.text, "label": article.label}
                for article in response.articles
            ]
        except Exception as e:
            logger.error(f"Failed to parse structured response: {e}")
            logger.error(f"Raw content: {content[:200]}...")
            return []

    @staticmethod
    def _default_augmented_column_value(sample_val: Any) -> Any:
        """Return a cast-friendly default for columns not set by augmentation."""
        if isinstance(sample_val, str):
            return ""
        if isinstance(sample_val, bool):
            return False
        if isinstance(sample_val, int):
            return 0
        if isinstance(sample_val, float):
            return 0.0
        return None

    def _complete_augmented_row(self, row: Dict[str, Any], ds) -> Dict[str, Any]:
        """Fill missing dataset columns before concatenating augmented rows."""
        for col in ds.column_names:
            if col not in row:
                row[col] = self._default_augmented_column_value(ds[0][col])
        return {col: row[col] for col in ds.column_names}

    def _build_augmented_sft_rows(
        self,
        records: List[Dict[str, Any]],
        ds,
        *,
        cycle: int,
        teacher_model: str,
        teacher_tier: str | None,
        prompt_operator: str | None,
    ) -> List[Dict[str, Any]]:
        """Build train rows for causal-LM SFT augmentation."""
        trainer_config = self.cfg.trainer_config or {}
        prompt_field = trainer_config.get("prompt_field", "student_prompt")
        response_field = trainer_config.get("response_field", "teacher_response")
        gold_answer_field = trainer_config.get("gold_answer_field", "gold_answer")
        missing = [
            field
            for field in (prompt_field, response_field)
            if field not in ds.column_names
        ]
        if missing:
            raise ValueError(
                "Online SFT augmentation requires the train dataset schema to "
                f"include prompt/response columns; missing {missing}. "
                "Materialize or load SFT-style rows before enabling augmentation."
            )
        rows = []
        for index, record in enumerate(records):
            student_prompt = str(record.get("student_prompt") or "").strip()
            teacher_response = str(record.get("teacher_response") or "").strip()
            if not student_prompt or not teacher_response:
                continue
            gold_answer = str(record.get("gold_answer") or teacher_response).strip()
            row = {
                "id": f"{self.run_id}/augmented/{cycle}/{index}",
                prompt_field: student_prompt,
                response_field: teacher_response,
                gold_answer_field: gold_answer,
                "source_example_id": str(
                    record.get("source_example_id") or f"augmented/{cycle}/{index}"
                ),
                "source_split": "augmented",
                "source_idx": -1,
                "origin_cycle": cycle,
                "prompt_operator": prompt_operator or "",
                "teacher_tier": teacher_tier or "",
                "teacher_model": teacher_model,
                "teacher_input_tokens": 0,
                "teacher_output_tokens": 0,
                "teacher_total_tokens": 0,
                "cycle": cycle,
                "seed": self.cfg.seed,
                "materialization_mode": "online_policy",
                "usage_estimated": False,
            }
            rows.append(self._complete_augmented_row(row, ds))
        return rows

    def _build_augmented_classification_rows(
        self,
        records: List[Dict[str, Any]],
        ds,
        *,
        cycle: int,
    ) -> List[Dict[str, Any]]:
        """Build train rows for classifier augmentation."""
        text_field = self.cfg.text_field
        label_field = self.cfg.label_field
        rows = []
        for record in records:
            row = {
                text_field: record["text"],
                label_field: record["label"],
                "source_split": "augmented",
                "source_idx": -1,
                "origin_cycle": cycle,
            }
            rows.append(self._complete_augmented_row(row, ds))
        return rows

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
        policy_action: PolicyAction | None = None,
        decision_id: str | None = None,
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

        teacher_model = self._teacher_for_action(policy_action)
        acquisition_batch_size = (
            policy_action.batch_size
            if policy_action and not policy_action.is_stop and policy_action.batch_size
            else self.cfg.augmentation_batch_size
        )
        acquisition_batch_size = self._effective_acquisition_batch_size(
            int(acquisition_batch_size)
        )
        prompt_operator = policy_action.prompt_operator if policy_action else None
        teacher_tier = policy_action.teacher_tier if policy_action else None

        if acquisition_batch_size <= 0:
            synthetic_record_budget = getattr(self.cfg, "synthetic_record_budget", None)
            synthetic_count = self._current_synthetic_count()
            return {
                "action_name": "augment_empty",
                "predicted_cost": {"total_tokens": 0},
                "metadata": {
                    "records_requested": 0,
                    "records_added": 0,
                    "skip_reason": "synthetic_record_budget_exhausted",
                    "synthetic_record_budget": synthetic_record_budget,
                    "synthetic_count_before": synthetic_count,
                },
            }

        # Render the prompt template with action overrides after config values.
        msg, _ = self._render_augmentation_prompt(sample_context, policy_action)

        # Save rendered prompt for debugging/inspection
        prompt_path = self.out_dir / f"prompt_cycle_{cycle}.txt"
        prompt_path.write_text(msg, encoding="utf-8")
        logger.info(f"Saved rendered prompt to {prompt_path}")

        logger.info(
            "Requesting %d augmented samples from teacher model %s",
            acquisition_batch_size,
            teacher_model,
        )

        messages = [{"role": "user", "content": msg}]
        predicted_cost = self._estimate_teacher_call_tokens(
            messages,
            budget_before or self._budget_snapshot(),
            teacher_model=teacher_model,
        )
        prompt_sha256 = sha256(msg.encode("utf-8")).hexdigest()
        attempt_id = self._next_attempt_id(cycle)
        attempt_metadata = {
            "teacher": teacher_model,
            "teacher_tier": teacher_tier,
            "prompt_operator": prompt_operator,
            "policy_action": policy_action.model_dump() if policy_action else {},
            "batch_size_requested": acquisition_batch_size,
            "records_requested": acquisition_batch_size,
            "prompt_sha256": prompt_sha256,
        }
        if not predicted_cost["allowed"]:
            logger.warning(
                "Skipping augmentation because budget preflight masked the teacher call: %s",
                predicted_cost["reason"],
            )
            self._record_teacher_attempt(
                cycle=cycle,
                status="masked",
                predicted_cost=predicted_cost,
                budget_before=budget_before or self._budget_snapshot(),
                attempt_id=attempt_id,
                decision_id=decision_id,
                failure_type=predicted_cost["reason"],
                metadata=attempt_metadata,
            )
            return {
                "action_name": "budget_masked",
                "predicted_cost": predicted_cost,
                "metadata": {
                    "mask_reason": predicted_cost["reason"],
                    **attempt_metadata,
                },
            }

        usage_recorded = False
        try:
            completion_kwargs = {
                "model": teacher_model,
                "messages": messages,
                "response_format": self._augmentation_response_format(),
            }
            if self.cfg.teacher_max_output_tokens is not None:
                completion_kwargs["max_tokens"] = self.cfg.teacher_max_output_tokens
            result = await acompletion(**completion_kwargs)
            usage = extract_usage_from_response(result)
            if usage.total_tokens <= 0:
                raise ValueError(
                    "Teacher response did not include provider token usage; "
                    "paper runs require auditable usage."
                )
            predicted_total = predicted_cost.get("total_tokens")
            if predicted_total is not None and usage.total_tokens > predicted_total:
                self.token_tracker.record_manual_usage(
                    usage, OperationType.AUGMENTATION_FAILURE
                )
                usage_recorded = True
                self._record_teacher_attempt(
                    cycle=cycle,
                    status="budget_violation",
                    predicted_cost=predicted_cost,
                    budget_before=budget_before or self._budget_snapshot(),
                    attempt_id=attempt_id,
                    decision_id=decision_id,
                    realized_cost=usage,
                    failure_type="realized_tokens_exceeded_preflight",
                    metadata=attempt_metadata,
                )
                raise ValueError(
                    "Realized teacher tokens exceeded preflight estimate: "
                    f"realized={usage.total_tokens} predicted={predicted_total}"
                )
            tokens_remaining = predicted_cost.get("tokens_remaining")
            if (
                tokens_remaining is not None
                and usage.total_tokens > int(tokens_remaining)
            ):
                self.token_tracker.record_manual_usage(
                    usage, OperationType.AUGMENTATION_FAILURE
                )
                usage_recorded = True
                self._record_teacher_attempt(
                    cycle=cycle,
                    status="budget_violation",
                    predicted_cost=predicted_cost,
                    budget_before=budget_before or self._budget_snapshot(),
                    attempt_id=attempt_id,
                    decision_id=decision_id,
                    realized_cost=usage,
                    failure_type="realized_tokens_exceeded_remaining_budget",
                    metadata=attempt_metadata,
                )
                raise ValueError(
                    "Realized teacher tokens exceeded remaining token budget: "
                    f"realized={usage.total_tokens} remaining={tokens_remaining}"
                )
            self.token_tracker.record_manual_usage(usage, OperationType.AUGMENTATION)
            usage_recorded = True
            response_path = self.out_dir / f"teacher_response_cycle_{cycle}.json"
            response_path.write_text(
                json.dumps(_response_to_jsonable(result), indent=2, default=str),
                encoding="utf-8",
            )
            response_sha256 = sha256(
                json.dumps(
                    _response_to_jsonable(result), sort_keys=True, default=str
                ).encode("utf-8")
            ).hexdigest()

            content = result["choices"][0]["message"]["content"]
            parsed_records = self._parse_augmented_response(content)
            if len(parsed_records) > acquisition_batch_size:
                parsed_records = parsed_records[:acquisition_batch_size]

            if not parsed_records:
                logger.warning("No records parsed from teacher response")
                self._record_teacher_attempt(
                    cycle=cycle,
                    status="empty_response",
                    predicted_cost=predicted_cost,
                    budget_before=budget_before or self._budget_snapshot(),
                    attempt_id=attempt_id,
                    decision_id=decision_id,
                    realized_cost=usage,
                    failure_type="no_records_parsed",
                    metadata={
                        **attempt_metadata,
                        "records_parsed": 0,
                        "records_accepted": 0,
                        "response_sha256": response_sha256,
                    },
                )
                return {
                    "action_name": "augment_empty",
                    "predicted_cost": predicted_cost,
                    "metadata": {
                        "records_parsed": 0,
                        "records_added": 0,
                        "attempt_id": attempt_id,
                        **attempt_metadata,
                    },
                }

            logger.info(
                f"Received {len(parsed_records)} augmented samples from teacher"
            )

            if self.cfg.student_type == "causal_lm_sft":
                aug_rows = self._build_augmented_sft_rows(
                    parsed_records,
                    ds,
                    cycle=cycle,
                    teacher_model=teacher_model,
                    teacher_tier=teacher_tier,
                    prompt_operator=prompt_operator,
                )
            else:
                aug_rows = self._build_augmented_classification_rows(
                    parsed_records,
                    ds,
                    cycle=cycle,
                )

        except Exception as e:
            logger.error(f"Batch augmentation failed: {e}")
            if not usage_recorded:
                reserved_usage = self._reserved_usage_from_preflight(predicted_cost)
                if reserved_usage.total_tokens > 0:
                    self.token_tracker.record_manual_usage(
                        reserved_usage, OperationType.AUGMENTATION_FAILURE
                    )
                self._record_teacher_attempt(
                    cycle=cycle,
                    status="failed",
                    predicted_cost=predicted_cost,
                    budget_before=budget_before or self._budget_snapshot(),
                    attempt_id=attempt_id,
                    decision_id=decision_id,
                    realized_cost=reserved_usage,
                    failure_type=type(e).__name__,
                    metadata={
                        **attempt_metadata,
                        "error": str(e),
                    },
                )
            return {
                "action_name": "augment_failed",
                "predicted_cost": predicted_cost,
                "metadata": {
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "attempt_id": attempt_id,
                    **attempt_metadata,
                },
            }

        if not aug_rows:
            self._record_teacher_attempt(
                cycle=cycle,
                status="empty_response",
                predicted_cost=predicted_cost,
                budget_before=budget_before or self._budget_snapshot(),
                attempt_id=attempt_id,
                decision_id=decision_id,
                realized_cost=usage,
                failure_type="no_records_accepted",
                metadata={
                    **attempt_metadata,
                    "records_parsed": len(parsed_records),
                    "records_accepted": 0,
                    "response_sha256": response_sha256,
                },
            )
            return {
                "action_name": "augment_empty",
                "predicted_cost": predicted_cost,
                "metadata": {
                    "records_parsed": len(parsed_records),
                    "records_added": 0,
                    "attempt_id": attempt_id,
                    **attempt_metadata,
                },
            }

        synthetic_record_budget = getattr(self.cfg, "synthetic_record_budget", None)
        if synthetic_record_budget is not None:
            current_synthetic = self._current_synthetic_count() or 0
            remaining_records = self._synthetic_records_remaining() or 0
            if remaining_records <= 0:
                self._record_teacher_attempt(
                    cycle=cycle,
                    status="empty_response",
                    predicted_cost=predicted_cost,
                    budget_before=budget_before or self._budget_snapshot(),
                    attempt_id=attempt_id,
                    decision_id=decision_id,
                    realized_cost=usage,
                    failure_type="synthetic_record_budget_exhausted",
                    metadata={
                        **attempt_metadata,
                        "records_parsed": len(parsed_records),
                        "records_accepted": 0,
                        "synthetic_record_budget": synthetic_record_budget,
                        "synthetic_count_before": current_synthetic,
                        "response_sha256": response_sha256,
                    },
                )
                return {
                    "action_name": "augment_empty",
                    "predicted_cost": predicted_cost,
                    "metadata": {
                        "records_parsed": len(parsed_records),
                        "records_added": 0,
                        "synthetic_record_budget": synthetic_record_budget,
                        "synthetic_count_before": current_synthetic,
                        "attempt_id": attempt_id,
                        **attempt_metadata,
                    },
                }
            if len(aug_rows) > remaining_records:
                aug_rows = aug_rows[:remaining_records]

        columns = aug_rows[0].keys()
        data_dict = {c: [r[c] for r in aug_rows] for c in columns}
        extra = Dataset.from_dict(data_dict)
        extra = extra.cast(ds.features)
        self.dataset["train"] = concatenate_datasets([ds, extra])
        logger.info(
            f"Added {len(aug_rows)} augmented samples to training set "
            f"(total: {len(self.dataset['train'])})"
        )
        self._record_teacher_attempt(
            cycle=cycle,
            status="success",
            predicted_cost=predicted_cost,
            budget_before=budget_before or self._budget_snapshot(),
            attempt_id=attempt_id,
            decision_id=decision_id,
            realized_cost=usage,
            metadata={
                **attempt_metadata,
                "records_parsed": len(parsed_records),
                "records_accepted": len(aug_rows),
                "response_sha256": response_sha256,
            },
        )
        return {
            "action_name": "augment",
            "predicted_cost": predicted_cost,
            "metadata": {
                "articles_parsed": len(aug_rows),
                "articles_added": len(aug_rows),
                "records_parsed": len(parsed_records),
                "records_added": len(aug_rows),
                "attempt_id": attempt_id,
                **attempt_metadata,
            },
        }

    async def run(self) -> Dict[str, Any]:
        results = {}
        final_model = None
        eval_split = self._cycle_eval_split()
        run_status = "completed"
        run_error = None

        try:
            try:
                for cycle in range(self.cfg.cycles):
                    with self.token_tracker.cycle(cycle):
                        if cycle == 0:
                            self._record_external_sft_token_usage()

                        model = self.trainer.train()
                        metrics = self.trainer.evaluate(model, split=eval_split)
                        metrics = {
                            **metrics,
                            "_selection_split": eval_split,
                            "_run_id": self.run_id,
                        }
                        logger.info(
                            "Cycle %d metrics on %s split: %s",
                            cycle,
                            eval_split,
                            metrics,
                        )
                        results[str(cycle)] = metrics
                        final_model = model

                        should_stop = self.early_stopper.should_stop(
                            metrics, cycle, model
                        )
                        if should_stop:
                            logger.info(f"Training stopped early at cycle {cycle}")
                            best_model = self.early_stopper.get_best_model()
                            if best_model is not None:
                                final_model = best_model
                                logger.info(
                                    "Restored best model from cycle "
                                    f"{self.early_stopper.best_cycle}"
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
                            feature_path = (
                                self.out_dir / f"policy_features_cycle_{cycle}.json"
                            )
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
                        decision_id = self._next_decision_id(cycle)
                        policy_name = self.cfg.policy_name
                        policy_action = None
                        action_scores = {}
                        policy_stopped = False
                        synthetic_budget = getattr(
                            self.cfg, "synthetic_record_budget", None
                        )
                        synthetic_count = self._current_synthetic_count()
                        synthetic_budget_exhausted = (
                            synthetic_budget is not None
                            and synthetic_count is not None
                            and synthetic_count >= int(synthetic_budget)
                        )

                        if (
                            synthetic_budget_exhausted
                            and cycle < self.cfg.cycles - 1
                        ):
                            action_name = "synthetic_record_budget_stop"
                            policy_stopped = True
                            decision_metadata = {
                                "acquisition_outcome": action_name,
                                "synthetic_record_budget": synthetic_budget,
                                "synthetic_count": synthetic_count,
                            }
                        elif not should_stop and cycle < self.cfg.cycles - 1:
                            action_name = (
                                "augment" if self.augmentation_enabled else "skip"
                            )
                            if self.augmentation_enabled:
                                if self.policy_controller:
                                    all_predicted_costs = self._policy_predicted_costs(
                                        self.action_space,
                                        sample_context or {},
                                        budget_before,
                                    )
                                    choice = self.policy_controller.select(
                                        cycle_state,
                                        actions=self.action_space,
                                        predicted_costs=all_predicted_costs,
                                    )
                                    policy_name = choice.policy_name
                                    policy_action = choice.action.model_dump()
                                    action_scores = choice.action_scores
                                    predicted_cost = all_predicted_costs.get(
                                        choice.action.name, choice.predicted_cost
                                    )
                                    decision_metadata = {
                                        "policy_choice": choice.model_dump(),
                                        "feasible_actions": list(
                                            choice.feasible_actions
                                        ),
                                        "all_predicted_costs": all_predicted_costs,
                                    }
                                    if choice.action.is_stop:
                                        action_name = "STOP"
                                        policy_stopped = True
                                        decision_metadata["acquisition_outcome"] = (
                                            "policy_stop"
                                        )
                                    else:
                                        outcome = await self._augment(
                                            model,
                                            cycle,
                                            sample_context=sample_context,
                                            budget_before=budget_before,
                                            policy_action=choice.action,
                                            decision_id=decision_id,
                                        )
                                        action_name = choice.action.name
                                        decision_metadata.update(outcome["metadata"])
                                        decision_metadata["acquisition_outcome"] = (
                                            outcome["action_name"]
                                        )
                                        predicted_cost = outcome["predicted_cost"]
                                else:
                                    outcome = await self._augment(
                                        model,
                                        cycle,
                                        sample_context=sample_context,
                                        budget_before=budget_before,
                                        decision_id=decision_id,
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
                            decision_id=decision_id,
                            policy_name=policy_name,
                            action=policy_action,
                            action_scores=action_scores,
                            predicted_cost=predicted_cost,
                            decision_metadata=decision_metadata,
                        )
                        self._save_dataset(cycle)

                    self.token_tracker.print_cycle_summary()

                    if self.token_tracker.should_stop_for_budget():
                        logger.info(
                            f"Training stopped due to budget limit at cycle {cycle}"
                        )
                        break

                    if policy_stopped:
                        logger.info(
                            "Training stopped because policy selected STOP at cycle %d",
                            cycle,
                        )
                        break

                    if self.early_stopper.stopped:
                        break

                self.token_tracker.print_final_summary()
            except Exception as exc:
                run_status = "failed"
                run_error = {"type": type(exc).__name__, "message": str(exc)}
                raise

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
                    Path(self.out_dir / "metrics.json").write_text(
                        json.dumps(results, indent=2)
                    )
            except Exception as e:
                logger.error(f"Failed to save metrics.json: {e}", exc_info=True)

            try:
                run_manifest = {
                    "schema_version": 2,
                    "artifact_schema_versions": {
                        "run_manifest": 2,
                        "policy_decisions": 1,
                        "teacher_attempts": 1,
                    },
                    "run_id": self.run_id,
                    "status": run_status,
                    "error": run_error,
                    "method_name": self.cfg.policy_name,
                    "policy_name": self.cfg.policy_name,
                    "policy_family": (
                        "fixed_promptillery"
                        if self.policy_controller is None
                        else "policy_controller"
                    ),
                    "expected_cycles": self.cfg.cycles,
                    "cycles_completed": self.token_tracker.summary.cycles_completed,
                    "selection_split": eval_split,
                    "paper_mode": self.cfg.paper_mode,
                    "control_name": self.cfg.control_name,
                    "task_name": self.cfg.name,
                    "dataset": self.cfg.dataset,
                    "dataset_subset": self.cfg.dataset_subset,
                    "student_model": self.cfg.student,
                    "student_type": self.cfg.student_type,
                    "seed": self.cfg.seed,
                    "token_budget": self.cfg.token_budget,
                    "synthetic_record_budget": self.cfg.synthetic_record_budget,
                    "final_synthetic_count": self._current_synthetic_count(),
                    "action_space": {
                        "prompt_operators": self.cfg.policy_prompt_operators,
                        "teacher_tiers": list(
                            self.cfg.policy_teacher_tiers.keys()
                        )
                        or ["cheap", "strong"],
                        "batch_sizes": self.cfg.policy_batch_sizes,
                        "include_stop": True,
                        "action_space_id": self.action_space_id,
                    },
                    "artifact_paths": {
                        "metrics": "metrics.json",
                        "token_usage": "token_usage.json",
                        "policy_decisions": "policy_decisions.jsonl",
                        "teacher_attempts": "teacher_attempts.jsonl",
                    },
                    "config_hash": sha256(
                        json.dumps(
                            self.cfg.model_dump(mode="json"), sort_keys=True
                        ).encode("utf-8")
                    ).hexdigest(),
                }
                Path(self.out_dir / "run_manifest.json").write_text(
                    json.dumps(run_manifest, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            except Exception as e:
                logger.error(f"Failed to save run_manifest.json: {e}", exc_info=True)

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
