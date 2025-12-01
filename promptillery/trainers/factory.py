"""Factory for creating trainer instances based on configuration."""

from pathlib import Path
from typing import Dict

from datasets import Dataset

from ..config import ExperimentConfig
from .base import BaseTrainer
from .fasttext_trainer import FastTextTrainer
from .transformer_trainer_ner import TransformersTraiNER
from .transformers_trainer import TransformersTrainer


class TrainerFactory:
    """Factory class for creating appropriate trainer instances."""

    _trainers = {
        "transformers": TransformersTrainer,
        "transformers_ner": TransformersTraiNER,
        "fasttext": FastTextTrainer,
    }

    @classmethod
    def create_trainer(
        cls, config: ExperimentConfig, dataset: Dict[str, Dataset], out_dir: Path
    ) -> BaseTrainer:
        """Create a trainer instance based on the student_type in config."""
        student_type = config.student_type

        if student_type not in cls._trainers:
            available_types = list(cls._trainers.keys())
            raise ValueError(
                f"Unknown student_type '{student_type}'. "
                f"Available types: {available_types}"
            )

        trainer_class = cls._trainers[student_type]
        return trainer_class(config, dataset, out_dir)

    @classmethod
    def register_trainer(cls, name: str, trainer_class: type):
        """Register a new trainer type."""
        if not issubclass(trainer_class, BaseTrainer):
            raise ValueError("Trainer class must inherit from BaseTrainer")
        cls._trainers[name] = trainer_class

    @classmethod
    def get_available_types(cls) -> list[str]:
        """Get list of available trainer types."""
        return list(cls._trainers.keys())
