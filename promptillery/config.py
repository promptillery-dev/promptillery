"""Configuration models for promptillery."""

import itertools
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# Shared timestamp format constant
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def infer_num_labels(
    dataset: str, dataset_config: Optional[str] = None
) -> Optional[int]:
    """Infer the number of labels from a dataset's metadata.

    Args:
        dataset: Dataset name (e.g., 'tweet_eval', 'imdb')
        dataset_config: Optional dataset configuration/subset (e.g., 'sentiment', 'emotion')

    Returns:
        Number of labels if successfully inferred, None otherwise
    """
    try:
        import logging

        from datasets import load_dataset_builder

        # Load dataset info without downloading the actual data
        if dataset_config:
            builder = load_dataset_builder(dataset, dataset_config)
        else:
            builder = load_dataset_builder(dataset)

        # Try to get number of classes from the 'label' feature
        if hasattr(builder.info, "features") and builder.info.features:
            features = builder.info.features

            # Common label column names to check
            label_columns = ["label", "labels", "class", "classes"]

            for col in label_columns:
                if col in features:
                    feature = features[col]
                    # Check if it's a ClassLabel feature
                    if hasattr(feature, "num_classes"):
                        logging.debug(
                            f"Inferred num_labels={feature.num_classes} for {dataset}/{dataset_config}"
                        )
                        return feature.num_classes
                    # Check if it has names list (alternative way to get class count)
                    if hasattr(feature, "names") and feature.names:
                        logging.debug(
                            f"Inferred num_labels={len(feature.names)} for {dataset}/{dataset_config}"
                        )
                        return len(feature.names)

        return None
    except Exception as e:
        # If we can't infer, return None and let the user specify manually
        import logging

        logging.debug(f"Could not infer num_labels for {dataset}/{dataset_config}: {e}")
        return None


class DatasetConfig(BaseModel):
    """Configuration for a specific dataset subset.

    Defines the dataset name/subset, number of classes, and which field contains the text.
    """

    name: str = Field(
        ..., description="Dataset subset name (e.g., 'sentiment', 'emoji')"
    )
    num_classes: int = Field(..., ge=2, description="Number of classes in this dataset")
    text_field: str = Field(
        default="text", description="Name of the field containing text data"
    )
    label_field: str = Field(
        default="label", description="Name of the field containing labels"
    )


class SamplingConfig(BaseModel):
    """Configuration for dataset sampling and train/validation splitting."""

    enabled: bool = False
    sample_size: int = Field(
        default=1000, ge=1, description="Number of samples to use from training set"
    )
    train_ratio: float = Field(
        default=0.8,
        gt=0.0,
        lt=1.0,
        description="Ratio of samples for training (rest goes to validation)",
    )
    stratify_column: str = Field(
        default="label", description="Column to use for stratified sampling"
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")


class EarlyStoppingConfig(BaseModel):
    """Configuration for early stopping during training cycles."""

    enabled: bool = False
    patience: int = Field(
        default=2, ge=1, description="Number of cycles to wait before stopping"
    )
    metric: str = Field(
        default="accuracy", description="Metric to monitor for early stopping"
    )
    mode: Literal["min", "max"] = Field(
        default="max", description="Whether to minimize or maximize the metric"
    )
    min_delta: float = Field(
        default=0.0, ge=0.0, description="Minimum change to qualify as improvement"
    )
    restore_best: bool = Field(
        default=True, description="Whether to restore best model when stopping"
    )

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        """Validate that the early stopping metric is in the metrics list."""
        # This will be validated at runtime when we have access to the full config
        return v


class ExperimentConfig(BaseModel):
    """Schema for experiment configuration.

    Values defined here are available as formatting variables in prompts.
    """

    name: str = Field(
        ...,
        description="Name of the experiment (required, alphanumeric, dashes, underscores only)",
    )
    teacher: Union[str, List[str]] = "openai/gpt-4o-mini"
    student: Union[str, List[str]] = "google-bert/bert-base-uncased"
    student_type: str = "transformers"

    # Dataset configuration - can be a string (dataset name) or structured config
    dataset: Union[str, List[str]] = "tweet_eval"
    # New structured dataset_config: list of DatasetConfig objects
    # Old format (str/List[str]) still supported for backwards compatibility
    dataset_config: Union[DatasetConfig, List[DatasetConfig], str, List[str], None] = (
        None
    )
    # Deprecated: use dataset_config.num_classes instead
    num_labels: Union[int, List[int], None] = None

    metrics: List[str] = Field(default_factory=lambda: ["accuracy", "f1"])

    # Core parameters - can be single value or list for ablations
    cycles: Union[int, List[int]] = 3

    # Component configurations
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    output_repo: str | None = None
    base_output_dir: str = "."

    # Augmentation settings - can be single value or list for ablations
    prompt: Optional[str] = None
    prompt_vars: Optional[Dict[str, Any]] = None
    augmentation_batch_size: Union[int, List[int]] = 10

    # Budget control
    budget_warning: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optional budget limit in USD - triggers warning when token costs exceed this amount",
    )
    budget_stop: bool = Field(
        default=False,
        description="If True, automatically stop experiment when budget_warning is exceeded",
    )

    # Training hyperparameters - can be single value or list for ablations
    learning_rate: Union[float, List[float]] = 2e-5
    batch_size: Union[int, List[int]] = 16
    num_train_epochs: Union[int, List[int]] = 3
    warmup_steps: Union[int, List[int]] = 500
    weight_decay: Union[float, List[float]] = 0.01

    # Dataset persistence
    persist_datasets: bool = True

    # Auto-modify name based on student type
    auto_modify_name: bool = Field(
        default=True,
        description="Whether to automatically modify name based on student_type",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate experiment name contains only alphanumeric, dashes, and underscores."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Experiment name must contain only alphanumeric characters, dashes (-), and underscores (_)"
            )
        return v

    @field_validator("student_type")
    @classmethod
    def validate_student_type(cls, v: str) -> str:
        """Validate student_type at runtime by checking with TrainerFactory."""
        # Import here to avoid circular imports
        try:
            from .trainers.factory import TrainerFactory

            available_types = TrainerFactory.get_available_types()
            if v not in available_types:
                raise ValueError(
                    f"Unknown student_type '{v}'. Available types: {available_types}"
                )
        except ImportError:
            # If TrainerFactory is not available during initialization, skip validation
            pass
        return v

    @model_validator(mode="after")
    def validate_early_stopping(self):
        """Validate early stopping configuration against metrics."""
        if self.early_stopping.enabled:
            if self.early_stopping.metric not in self.metrics:
                raise ValueError(
                    f"Early stopping metric '{self.early_stopping.metric}' must be in metrics list: {self.metrics}"
                )
        return self

    @model_validator(mode="after")
    def modify_name_based_on_student_type(self):
        """Automatically modify the experiment name based on student_type if enabled."""
        if not self.auto_modify_name:
            return self

        if self.student_type and self.name:
            modifier = self.student_type

            # Case-insensitive check to avoid duplicates like foo_transformers_transformers
            if modifier.lower() not in self.name.lower():
                self.name = f"{self.name}_{modifier}"

        return self

    def get_dataset_config_obj(self) -> Optional[DatasetConfig]:
        """Get the dataset configuration as a DatasetConfig object.

        Handles both old format (string) and new format (DatasetConfig).
        Returns None if no dataset_config is set.
        """
        if self.dataset_config is None:
            return None

        # If already a DatasetConfig, return it
        if isinstance(self.dataset_config, DatasetConfig):
            return self.dataset_config

        # If it's a list, get the first DatasetConfig
        if isinstance(self.dataset_config, list):
            if len(self.dataset_config) > 0:
                first = self.dataset_config[0]
                if isinstance(first, DatasetConfig):
                    return first
                # Old format: string in list - create DatasetConfig
                return DatasetConfig(
                    name=first,
                    num_classes=self.num_labels or 2,
                    text_field="text",
                    label_field="label",
                )
            return None

        # Old format: string - create DatasetConfig with defaults
        return DatasetConfig(
            name=self.dataset_config,
            num_classes=self.num_labels or 2,
            text_field="text",
            label_field="label",
        )

    @property
    def text_field(self) -> str:
        """Get the text field name from dataset config, defaulting to 'text'."""
        cfg = self.get_dataset_config_obj()
        return cfg.text_field if cfg else "text"

    @property
    def label_field(self) -> str:
        """Get the label field name from dataset config, defaulting to 'label'."""
        cfg = self.get_dataset_config_obj()
        return cfg.label_field if cfg else "label"

    @property
    def num_classes(self) -> int:
        """Get the number of classes from dataset config or num_labels."""
        cfg = self.get_dataset_config_obj()
        if cfg:
            return cfg.num_classes
        return self.num_labels or 2

    @property
    def dataset_subset(self) -> Optional[str]:
        """Get the dataset subset name for loading from HuggingFace."""
        cfg = self.get_dataset_config_obj()
        if cfg:
            return cfg.name
        # Old format: dataset_config is the subset name directly
        if isinstance(self.dataset_config, str):
            return self.dataset_config
        return None

    def get_output_dir(self) -> Path:
        """Generate output directory path based on experiment name and timestamp."""
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        experiment_dir = f"{self.name}_{timestamp}"
        return Path(self.base_output_dir) / experiment_dir

    def get_list_parameters(self) -> List[str]:
        """Return names of parameters that have list values (for ablation)."""
        model_data = self.model_dump()
        list_params = []

        for field_name, field_info in self.model_fields.items():
            field_value = model_data[field_name]

            # Check if this field is Union[T, List[T]] and has a list value
            if (
                hasattr(field_info.annotation, "__origin__")
                and field_info.annotation.__origin__ is Union
            ):
                union_args = field_info.annotation.__args__
                if any(
                    hasattr(arg, "__origin__") and arg.__origin__ is list
                    for arg in union_args
                ):
                    if isinstance(field_value, list):
                        list_params.append(field_name)

        return list_params

    def has_list_parameters(self) -> bool:
        """Check if any Union[T, List[T]] fields have list values (indicating ablation)."""
        return bool(self.get_list_parameters())

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def generate_ablation_configs(self) -> List["ExperimentConfig"]:
        """Generate multiple configurations from list-valued parameters."""
        # Get the actual field values from the model
        model_data = self.model_dump()

        # Identify which fields are list-valued and should be treated as matrix parameters
        matrix_params = {}
        base_data = {}

        # Get field info from the model
        for field_name, field_info in self.model_fields.items():
            field_value = model_data[field_name]

            # Check if this field is Union[T, List[T]] and has a list value
            if (
                hasattr(field_info.annotation, "__origin__")
                and field_info.annotation.__origin__ is Union
            ):
                # Get the union args
                union_args = field_info.annotation.__args__
                # Check if one is a list type and we have a list value
                if any(
                    hasattr(arg, "__origin__") and arg.__origin__ is list
                    for arg in union_args
                ):
                    if isinstance(field_value, list):
                        matrix_params[field_name] = field_value
                    else:
                        base_data[field_name] = field_value
                else:
                    base_data[field_name] = field_value
            else:
                base_data[field_name] = field_value

        # If no matrix parameters, return self
        if not matrix_params:
            return [self]

        # Generate all combinations
        param_names = sorted(matrix_params.keys())  # Sort for consistent ordering
        param_values = [matrix_params[name] for name in param_names]

        configs = []
        for values in itertools.product(*param_values):
            config_data = base_data.copy()

            # Apply the specific values and build name suffix
            name_parts = []
            for param_name, value in zip(param_names, values):
                config_data[param_name] = value
                # Format value for name (handle floats, strings with slashes, etc.)
                if isinstance(value, float):
                    # Scientific notation for small floats, otherwise simple format
                    value_str = f"{value:.0e}" if value < 0.001 else str(value)
                elif isinstance(value, str):
                    # Take last part of path-like strings (e.g., "openai/gpt-4o-mini" -> "gpt-4o-mini")
                    value_str = value.split("/")[-1]
                else:
                    value_str = str(value)

                # Sanitize value_str to only contain alphanumeric, dashes, and underscores
                # Replace periods and other special chars with underscores
                value_str = value_str.replace(".", "_").replace("/", "_")
                # Remove any remaining invalid characters
                value_str = "".join(
                    c if c.isalnum() or c in "-_" else "_" for c in value_str
                )

                name_parts.append(f"{param_name}-{value_str}")

            # Auto-infer num_labels if not set and dataset_config is in the ablation
            if (
                config_data.get("num_labels") is None
                and "dataset_config" in param_names
            ):
                dataset = config_data.get("dataset")
                dataset_config = config_data.get("dataset_config")
                if dataset:
                    inferred = infer_num_labels(dataset, dataset_config)
                    if inferred is not None:
                        config_data["num_labels"] = inferred

            # Create informative name with parameter values
            config_data["name"] = f"{base_data['name']}_{'_'.join(name_parts)}"

            configs.append(ExperimentConfig(**config_data))

        return configs
