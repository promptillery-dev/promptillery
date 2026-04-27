"""Factory for creating trainer instances based on configuration."""

from pathlib import Path
from typing import Dict

from datasets import Dataset

from ..config import ExperimentConfig
from .base import BaseTrainer
from .causal_lm_sft_trainer import CausalLMSFTTrainer
from .transformer_trainer_ner import TransformersTraiNER
from .transformers_trainer import TransformersTrainer

try:
    from .fasttext_trainer import FastTextTrainer
except ImportError as exc:
    FastTextTrainer = None  # type: ignore[assignment]
    _FASTTEXT_IMPORT_ERROR = exc
else:
    _FASTTEXT_IMPORT_ERROR = None


class TrainerFactory:
    """Factory class for creating appropriate trainer instances."""

    _trainers = {
        "transformers": TransformersTrainer,
        "transformers_ner": TransformersTraiNER,
        "causal_lm_sft": CausalLMSFTTrainer,
    }
    if FastTextTrainer is not None:
        _trainers["fasttext"] = FastTextTrainer

    @classmethod
    def create_trainer(
        cls, config: ExperimentConfig, dataset: Dict[str, Dataset], out_dir: Path
    ) -> BaseTrainer:
        """Create a trainer instance based on the student_type in config."""
        student_type = config.student_type

        if student_type not in cls._trainers:
            if student_type == "fasttext" and _FASTTEXT_IMPORT_ERROR is not None:
                raise ValueError(
                    "student_type 'fasttext' requires the optional FastText "
                    "dependency. Install with `promptillery[fasttext]`."
                ) from _FASTTEXT_IMPORT_ERROR
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
