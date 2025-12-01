"""FastText trainer implementation."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import fasttext
import numpy as np
from datasets import Dataset
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import accuracy_score, f1_score

from .base import BaseTrainer, PredictionResult

logger = logging.getLogger(__name__)


class FastTextTrainer(BaseTrainer):
    """Trainer for FastText models."""

    def __init__(self, config, dataset, out_dir):
        super().__init__(config, dataset, out_dir)
        # FastText model will be initialized during training
        self.model = None

    def prepare_data(self, split: str) -> Dataset:
        """For FastText, return data as-is since we don't tokenize."""
        return self.dataset[split]

    def train(self) -> fasttext.FastText:
        """Train the FastText model."""
        # Prepare training data in FastText format
        train_data = self.prepare_data("train")
        text_field = self.cfg.text_field
        label_field = self.cfg.label_field

        # Create temporary file for FastText training
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for i in range(len(train_data)):
                text = train_data[i][text_field]
                label = train_data[i][label_field]
                # FastText format: __label__<label> <text>
                f.write(f"__label__{label} {text}\n")
            train_file = f.name

        try:
            # Train FastText model
            # Map config hyperparameters: num_train_epochs -> epoch, learning_rate -> lr
            # Note: FastText typically uses higher learning rates (0.1-1.0) vs transformers (1e-5 to 5e-5)
            # Users should specify appropriate values for FastText in their config

            # Warn if learning rate seems too low for FastText
            if self.cfg.learning_rate < 0.01:
                logger.warning(
                    f"Learning rate {self.cfg.learning_rate} is very low for FastText. "
                    f"FastText typically uses learning rates in range 0.1-1.0. "
                    f"Consider using higher values for better results."
                )

            # Warn if epochs seem too low for FastText
            if self.cfg.num_train_epochs < 5:
                logger.warning(
                    f"Number of epochs {self.cfg.num_train_epochs} is low for FastText. "
                    f"FastText typically needs more epochs (e.g., 10-25). "
                    f"Consider using higher values for better convergence."
                )

            model = fasttext.train_supervised(
                input=train_file,
                epoch=self.cfg.num_train_epochs,
                lr=self.cfg.learning_rate,
                wordNgrams=2,
                dim=100,
                loss="softmax",
            )

            self.model = model
            return model
        finally:
            # Clean up temporary file
            Path(train_file).unlink(missing_ok=True)

    def evaluate(self, model: fasttext.FastText, split: str = "test") -> Dict[str, Any]:
        """Evaluate the FastText model."""
        test_data = self.prepare_data(split)
        text_field = self.cfg.text_field
        label_field = self.cfg.label_field

        texts = [test_data[i][text_field] for i in range(len(test_data))]
        true_labels = [test_data[i][label_field] for i in range(len(test_data))]

        # Get predictions from FastText
        predictions, _ = model.predict(texts)
        pred_labels = [int(pred[0].replace("__label__", "")) for pred in predictions]

        scores = {}
        for metric_name in self.cfg.metrics:
            if metric_name == "accuracy":
                scores[metric_name] = accuracy_score(true_labels, pred_labels)
            elif metric_name == "f1":
                # Use macro average for multiclass, binary for binary
                avg = "macro" if self.cfg.num_classes > 2 else "binary"
                scores[metric_name] = f1_score(true_labels, pred_labels, average=avg)

        return scores

    def predict_for_augmentation(
        self, model: fasttext.FastText, split: str = "train"
    ) -> List[int]:
        """Get misclassified sample indices for augmentation."""
        train_data = self.prepare_data(split)
        text_field = self.cfg.text_field
        label_field = self.cfg.label_field

        texts = [train_data[i][text_field] for i in range(len(train_data))]
        true_labels = [train_data[i][label_field] for i in range(len(train_data))]
        predictions, _ = model.predict(texts)
        pred_labels = [int(pred[0].replace("__label__", "")) for pred in predictions]
        return [
            i
            for i, (p, label) in enumerate(zip(pred_labels, true_labels))
            if p != label
        ]

    def get_detailed_predictions(
        self, model: fasttext.FastText, split: str = "train"
    ) -> PredictionResult:
        """Get detailed predictions with confidence and entropy scores.

        Args:
            model: The trained FastText model
            split: Dataset split to predict on

        Returns:
            PredictionResult with indices, predictions, confidences, and entropies
        """
        data = self.prepare_data(split)
        text_field = self.cfg.text_field
        label_field = self.cfg.label_field

        texts = [data[i][text_field] for i in range(len(data))]
        true_labels = [data[i][label_field] for i in range(len(data))]

        # Get predictions with probabilities for all labels
        predictions, probs = model.predict(texts, k=self.cfg.num_classes)

        pred_labels = []
        confidences = []
        entropies = []

        for pred_list, prob_list in zip(predictions, probs):
            # Extract predicted label (first one is highest confidence)
            pred_label = int(pred_list[0].replace("__label__", ""))
            pred_labels.append(pred_label)

            # Confidence is the probability of the predicted class
            confidences.append(float(prob_list[0]))

            # Create probability distribution for entropy calculation
            # FastText returns probabilities in descending order
            prob_array = np.array(prob_list, dtype=float)
            # Normalize to ensure it sums to 1 (FastText probs should already sum to 1)
            prob_array = prob_array / prob_array.sum()
            entropies.append(float(scipy_entropy(prob_array)))

        return PredictionResult(
            indices=list(range(len(pred_labels))),
            predicted_labels=pred_labels,
            true_labels=true_labels,
            confidences=confidences,
            entropies=entropies,
        )

    def save_model(self, model: fasttext.FastText) -> None:
        """Save the FastText model."""
        model_path = str(self.out_dir / "fasttext_model.bin")
        model.save_model(model_path)

    def load_model(self, model_path: Path) -> fasttext.FastText:
        """Load a trained FastText model from disk.

        Args:
            model_path: Path to the model file or directory containing fasttext_model.bin
        """
        model_path = Path(model_path)

        # If given a directory, look for the model file inside
        if model_path.is_dir():
            model_file = model_path / "fasttext_model.bin"
            if not model_file.exists():
                raise ValueError(
                    f"FastText model file not found at {model_file}. "
                    f"Expected 'fasttext_model.bin' in {model_path}"
                )
            model_path = model_file

        model = fasttext.load_model(str(model_path))
        self.model = model
        return model

    def push_to_hub(self, model: Any, repo_name: str) -> None:
        """FastText models cannot be pushed to HuggingFace Hub directly."""
        logger.warning("FastText models cannot be pushed to HuggingFace Hub directly")
