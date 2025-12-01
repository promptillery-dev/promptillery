"""HuggingFace Transformers trainer implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import evaluate
import numpy as np
from datasets import Dataset
from scipy.special import softmax
from scipy.stats import entropy
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .base import BaseTrainer, PredictionResult

logger = logging.getLogger(__name__)


class TransformersTrainer(BaseTrainer):
    """Trainer for HuggingFace Transformers models."""

    def __init__(self, config, dataset, out_dir):
        super().__init__(config, dataset, out_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.student)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.student, num_labels=self.cfg.num_classes
        )
        # Load metrics
        self.metrics = {}
        for metric_name in self.cfg.metrics:
            self.metrics[metric_name] = evaluate.load(metric_name)

    def prepare_data(self, split: str) -> Dataset:
        """Tokenize the data for Transformers.

        Also renames the label column to 'labels' (plural) as expected by HuggingFace Trainer.
        """
        text_field = self.cfg.text_field
        label_field = self.cfg.label_field

        ds = self.dataset[split]

        # Rename label column to 'labels' if needed (HuggingFace Trainer expects 'labels')
        if label_field != "labels" and label_field in ds.column_names:
            ds = ds.rename_column(label_field, "labels")

        # Tokenize
        ds = ds.map(
            lambda b: self.tokenizer(b[text_field], truncation=True, padding=True),
            batched=True,
        )

        return ds

    def train(self) -> Trainer:
        """Train the Transformers model."""
        args = TrainingArguments(
            output_dir=str(self.out_dir / "training"),
            num_train_epochs=self.cfg.num_train_epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            per_device_eval_batch_size=self.cfg.batch_size,
            warmup_steps=self.cfg.warmup_steps,
            weight_decay=self.cfg.weight_decay,
            learning_rate=self.cfg.learning_rate,
            logging_dir=str(self.out_dir / "logs"),
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["tensorboard"],
        )
        eval_split = "validation" if "validation" in self.dataset else "test"
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.prepare_data("train"),
            eval_dataset=self.prepare_data(eval_split),
            tokenizer=self.tokenizer,
        )
        trainer.train()
        return trainer

    def evaluate(self, trainer: Trainer, split: str = "test") -> Dict[str, Any]:
        """Evaluate the Transformers model."""
        res = trainer.predict(self.prepare_data(split))
        preds = res.predictions.argmax(axis=1)
        labels = res.label_ids

        # Debug logging
        logger.info(f"Evaluation on {split} split:")
        logger.info(f"Predictions shape: {preds.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        logger.info(f"Unique predictions: {np.unique(preds)}")
        logger.info(f"Unique labels: {np.unique(labels)}")
        logger.info(f"Num classes in config: {self.cfg.num_classes}")

        # Check for class imbalance issues
        unique_combined = np.unique(np.concatenate([preds, labels]))
        logger.info(f"All unique classes present: {unique_combined}")

        scores = {}
        for name, metric in self.metrics.items():
            try:
                if name == "accuracy":
                    result = metric.compute(predictions=preds, references=labels)
                    scores[name] = result["accuracy"]
                elif name in ["f1", "precision", "recall"]:
                    # Use HuggingFace evaluate library
                    # Determine the right averaging strategy
                    if self.cfg.num_classes == 2:
                        # Check if we actually have both classes
                        if len(unique_combined) == 2:
                            average = "binary"
                        else:
                            # Only one class present, use macro but expect low scores
                            average = "macro"
                            logger.warning(
                                f"Only {len(unique_combined)} unique classes found for binary classification"
                            )
                    else:
                        # For multi-class, always use macro
                        average = "macro"

                    # Use HuggingFace evaluate library with proper parameters
                    result = metric.compute(
                        predictions=preds, references=labels, average=average
                    )

                    # Handle different return formats
                    if isinstance(result, dict):
                        scores[name] = result.get(name, result.get("score", 0.0))
                    else:
                        scores[name] = result
                else:
                    # For other metrics, use evaluate library
                    result = metric.compute(predictions=preds, references=labels)
                    scores[name] = result.get(name, result.get("score", 0.0))

            except Exception as e:
                logger.warning(f"Failed to compute {name}: {e}")
                logger.warning(f"Predictions: {preds[:10]}")
                logger.warning(f"Labels: {labels[:10]}")
                scores[name] = 0.0

        logger.info(f"Final scores: {scores}")
        return scores

    def predict_for_augmentation(
        self, trainer: Trainer, split: str = "train"
    ) -> List[int]:
        """Get misclassified sample indices for augmentation."""
        preds = trainer.predict(self.prepare_data(split))
        pred = preds.predictions.argmax(axis=1)
        misclassified = [
            i for i, (p, label) in enumerate(zip(pred, preds.label_ids)) if p != label
        ]

        logger.info(
            f"Found {len(misclassified)} misclassified samples out of {len(pred)} total"
        )
        return misclassified

    def get_detailed_predictions(
        self, trainer: Trainer, split: str = "train"
    ) -> PredictionResult:
        """Get detailed predictions with confidence and entropy scores.

        Args:
            trainer: The trained HuggingFace Trainer
            split: Dataset split to predict on

        Returns:
            PredictionResult with indices, predictions, confidences, and entropies
        """
        res = trainer.predict(self.prepare_data(split))
        logits = res.predictions
        labels = res.label_ids

        # Convert logits to probabilities
        probs = softmax(logits, axis=1)

        # Get predictions and confidences
        pred_labels = probs.argmax(axis=1)
        confidences = probs.max(axis=1)

        # Calculate entropy for each prediction
        entropies = entropy(probs, axis=1)

        return PredictionResult(
            indices=list(range(len(pred_labels))),
            predicted_labels=pred_labels.tolist(),
            true_labels=labels.tolist(),
            confidences=confidences.tolist(),
            entropies=entropies.tolist(),
        )

    def save_model(self, trainer: Trainer) -> None:
        """Save the Transformers model."""
        trainer.save_model(str(self.out_dir / "model"))

    def load_model(self, model_path) -> Trainer:
        """Load a trained model from disk and return a Trainer instance."""

        model_path = Path(model_path)
        logger.info(f"Loading model from {model_path}")

        # Load the model and tokenizer from the checkpoint
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        # Create a Trainer instance for evaluation
        args = TrainingArguments(
            output_dir=str(model_path / "eval_output"),
            per_device_eval_batch_size=16,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=args,
            tokenizer=tokenizer,
        )

        return trainer

    def push_to_hub(self, model: Any, repo_name: str) -> None:
        """Push model to HuggingFace Hub."""
        self.model.push_to_hub(repo_name)
