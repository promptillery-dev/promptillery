"""Base trainer interface for different model types."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from datasets import Dataset

from ..config import ExperimentConfig
from ..fidelity import agreement_by_index, load_teacher_eval_labels

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results with confidence and entropy."""

    indices: List[int]
    predicted_labels: List[Any]
    true_labels: List[Any]
    confidences: List[float]
    entropies: Optional[List[float]] = None
    label_names: Optional[List[str]] = None
    predicted_texts: Optional[List[str]] = None
    true_texts: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTrainer(ABC):
    """Abstract base class for all model trainers."""

    def __init__(
        self, config: ExperimentConfig, dataset: Dict[str, Dataset], out_dir: Path
    ):
        self.cfg = config
        self.dataset = dataset
        self.out_dir = out_dir
        self.model = None

    @abstractmethod
    def train(self) -> Any:
        """Train the model and return the trained model/trainer."""
        pass

    @abstractmethod
    def evaluate(self, model: Any, split: str = "test") -> Dict[str, Any]:
        """Evaluate the model on the specified split."""
        pass

    @abstractmethod
    def predict_for_augmentation(self, model: Any, split: str = "train") -> List[int]:
        """Get predictions for data augmentation (finding misclassified samples)."""
        pass

    def get_detailed_predictions(
        self, model: Any, split: str = "train"
    ) -> PredictionResult:
        """Get detailed predictions including confidence and entropy.

        Subclasses should override this to provide model-specific implementations.
        Default implementation returns empty result.

        Args:
            model: The trained model
            split: Dataset split to predict on

        Returns:
            PredictionResult with indices, predictions, confidences, and optionally entropies
        """
        return PredictionResult(
            indices=[],
            predicted_labels=[],
            true_labels=[],
            confidences=[],
            entropies=None,
        )

    @abstractmethod
    def save_model(self, model: Any) -> None:
        """Save the trained model."""
        pass

    @abstractmethod
    def load_model(self, model_path: Path) -> Any:
        """Load a trained model from disk."""
        pass

    @abstractmethod
    def push_to_hub(self, model: Any, repo_name: str) -> None:
        """Push model to HuggingFace Hub if supported."""
        pass

    def prepare_data(self, split: str) -> Dataset:
        """Prepare data for training/evaluation. Can be overridden by subclasses."""
        return self.dataset[split]

    def _teacher_fidelity_from_label_ids(
        self, label_ids: Sequence[int], split: str
    ) -> Optional[float]:
        """Top-1 student-vs-teacher agreement for class-id predictions.

        Shared by the classifier-style trainers (Transformers encoder and
        FastText), whose predictions are integer class ids over the split's
        ``ClassLabel`` feature. ``label_ids`` is aligned to the split's row
        order (position i == row i), so it maps to canonical label names and
        joins to the teacher labels by original row index.

        Returns ``None`` when fidelity is not configured for this split, so
        callers only emit ``teacher_fidelity`` when a teacher label file is
        wired in -- experiments that don't set it are unaffected.
        """
        fidelity_cfg = (self.cfg.trainer_config or {}).get("fidelity") or {}
        labels_path = fidelity_cfg.get("teacher_labels_path")
        if not labels_path or split != fidelity_cfg.get("split", "test"):
            return None

        feature = self.dataset[split].features.get(self.cfg.label_field)
        label_names = getattr(feature, "names", None)
        if not label_names:
            logger.warning(
                "Cannot compute teacher_fidelity: label field '%s' is not a "
                "ClassLabel with names on the '%s' split.",
                self.cfg.label_field,
                split,
            )
            return None

        teacher_labels = load_teacher_eval_labels(labels_path)
        student_by_index = {i: label_names[int(p)] for i, p in enumerate(label_ids)}
        return agreement_by_index(student_by_index, teacher_labels)
