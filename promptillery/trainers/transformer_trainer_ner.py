"""HuggingFace Transformers trainer implementation for Named Entity Recognition."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import evaluate
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from .base import BaseTrainer

logger = logging.getLogger(__name__)


class TransformersTraiNER(BaseTrainer):
    """Trainer for HuggingFace Transformers models for Named Entity Recognition."""

    def __init__(self, config, dataset, out_dir):
        super().__init__(config, dataset, out_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.student)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.cfg.student, num_labels=self.cfg.num_labels
        )
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)
        self.metrics = {m: evaluate.load(m) for m in self.cfg.metrics}

    def prepare_data(self, split: str) -> Dataset:
        """Tokenize and align labels for NER data."""
        dataset = self.dataset[split]

        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(
                examples["tokens"],
                truncation=True,
                padding=False,  # Let data collator handle padding
                is_split_into_words=True,
            )

            labels = []
            for i, label in enumerate(examples["ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        # Special tokens get -100 (ignored in loss)
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        # First subword of a word gets the label
                        try:
                            label_ids.append(label[word_idx])
                        except IndexError:
                            # Handle edge case where word_idx is out of bounds
                            label_ids.append(-100)
                    else:
                        # Subsequent subwords get -100 (ignored in loss)
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        return dataset.map(tokenize_and_align_labels, batched=True)

    def train(self) -> Trainer:
        """Train the NER model."""
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
            data_collator=self.data_collator,
        )
        trainer.train()
        return trainer

    def evaluate(self, trainer: Trainer, split: str = "test") -> Dict[str, Any]:
        """Evaluate the NER model."""
        res = trainer.predict(self.prepare_data(split))

        # Token classification - predictions shape: (batch_size, seq_len, num_labels)
        predictions = res.predictions.argmax(axis=2)
        labels = res.label_ids

        # Flatten predictions and labels, removing -100 (ignored) labels
        true_predictions = []
        true_labels = []

        for pred_seq, label_seq in zip(predictions, labels):
            for pred, label in zip(pred_seq, label_seq):
                if label != -100:  # Only include non-ignored labels
                    true_predictions.append(pred)
                    true_labels.append(label)

        scores = {}
        for name, metric in self.metrics.items():
            try:
                if name in ["precision", "recall", "f1"]:
                    # For token classification, use average='macro' for multi-class
                    result = metric.compute(
                        predictions=true_predictions,
                        references=true_labels,
                        average="macro",
                    )
                    if isinstance(result, dict):
                        scores[name] = result.get(name, result.get("score", 0.0))
                    else:
                        scores[name] = result
                else:
                    scores[name] = metric.compute(
                        predictions=true_predictions, references=true_labels
                    )[name]
            except Exception as e:
                logger.warning(f"Failed to compute {name}: {e}")
                scores[name] = 0.0

        return scores

    def predict_for_augmentation(
        self, trainer: Trainer, split: str = "train"
    ) -> List[int]:
        """Get misclassified sample indices for augmentation."""
        res = trainer.predict(self.prepare_data(split))
        predictions = res.predictions.argmax(axis=2)
        labels = res.label_ids

        misclassified = []
        for idx, (pred_seq, label_seq) in enumerate(zip(predictions, labels)):
            # Check if any token in the sequence is misclassified
            has_error = False
            for pred, label in zip(pred_seq, label_seq):
                if label != -100 and pred != label:
                    has_error = True
                    break
            if has_error:
                misclassified.append(idx)

        return misclassified

    def save_model(self, trainer: Trainer) -> None:
        """Save the NER model."""
        trainer.save_model(str(self.out_dir / "model"))

    def load_model(self, model_path: Path) -> AutoModelForTokenClassification:
        """Load a trained NER model from disk."""

        model = AutoModelForTokenClassification.from_pretrained(str(model_path))
        self.model = model
        return model

    def push_to_hub(self, model: Any, repo_name: str) -> None:
        """Push model to HuggingFace Hub."""
        self.model.push_to_hub(repo_name)
