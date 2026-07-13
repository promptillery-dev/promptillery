"""Trainer modules for different model types."""

from .base import BaseTrainer
from .causal_lm_sft_trainer import CausalLMSFTTrainer
from .factory import SFT_STUDENT_TYPES, TrainerFactory
from .transformer_trainer_ner import TransformersTraiNER
from .transformers_trainer import TransformersTrainer

try:
    from .fasttext_trainer import FastTextTrainer
except ImportError:
    FastTextTrainer = None  # type: ignore[assignment]

__all__ = [
    "BaseTrainer",
    "CausalLMSFTTrainer",
    "TransformersTrainer",
    "TransformersTraiNER",
    "TrainerFactory",
    "SFT_STUDENT_TYPES",
]
if FastTextTrainer is not None:
    __all__.append("FastTextTrainer")
