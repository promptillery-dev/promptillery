"""Base trainer interface for different model types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset

from ..config import ExperimentConfig


@dataclass
class PredictionResult:
    """Container for prediction results with confidence and entropy."""

    indices: List[int]
    predicted_labels: List[int]
    true_labels: List[int]
    confidences: List[float]
    entropies: Optional[List[float]] = None


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
