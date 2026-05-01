"""Two heterogeneous LoRA students sharing TrainerFactory machinery (design §3.1)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset

from ..config import CoTrainStudentConfig, ExperimentConfig
from ..trainers.base import BaseTrainer, PredictionResult
from ..trainers.factory import TrainerFactory


def _materialize_student_config(
    parent: ExperimentConfig, student: CoTrainStudentConfig, suffix: str
) -> ExperimentConfig:
    payload = parent.model_dump()
    payload["name"] = f"{parent.name}-{suffix}"
    payload["auto_modify_name"] = False
    payload["student"] = student.model
    payload["learning_rate"] = student.learning_rate
    payload["num_train_epochs"] = student.num_train_epochs
    payload["warmup_steps"] = student.warmup_steps
    payload["weight_decay"] = student.weight_decay
    payload["batch_size"] = student.per_device_batch_size
    trainer_cfg = dict(payload.get("trainer_config") or {})
    trainer_cfg.update({
        "use_lora": True,
        "lora_r": student.lora.r,
        "lora_alpha": student.lora.alpha,
        "lora_dropout": student.lora.dropout,
        "max_seq_length": student.max_seq_length,
    })
    if student.lora.target_modules:
        trainer_cfg["lora_target_modules"] = list(student.lora.target_modules)
    payload["trainer_config"] = trainer_cfg
    return ExperimentConfig.model_validate(payload)


class DualStudentTrainer:
    """Maintain a (trainer_a, trainer_b) pair sharing the same dataset map."""

    def __init__(
        self,
        parent_config: ExperimentConfig,
        dataset: Dict[str, Dataset],
        out_dir: Path,
    ) -> None:
        if parent_config.cotrain is None:
            raise ValueError("DualStudentTrainer requires cotrain config")
        self.parent = parent_config
        self.dataset_a: Dict[str, Dataset] = dict(dataset)
        self.dataset_b: Dict[str, Dataset] = dict(dataset)
        self.out_dir = Path(out_dir)
        self.cfg_a = _materialize_student_config(
            parent_config, parent_config.cotrain.student_a, "a"
        )
        self.cfg_b = _materialize_student_config(
            parent_config, parent_config.cotrain.student_b, "b"
        )
        self.trainer_a: Optional[BaseTrainer] = None
        self.trainer_b: Optional[BaseTrainer] = None
        self.model_a: Any = None
        self.model_b: Any = None

    def set_train_split(self, student: str, ds: Dataset) -> None:
        target = self.dataset_a if student == "a" else self.dataset_b
        target["train"] = ds

    def train_a(self) -> Any:
        self.trainer_a = TrainerFactory.create_trainer(
            self.cfg_a, self.dataset_a, self.out_dir / "a"
        )
        self.model_a = self.trainer_a.train()
        return self.model_a

    def train_b(self) -> Any:
        self.trainer_b = TrainerFactory.create_trainer(
            self.cfg_b, self.dataset_b, self.out_dir / "b"
        )
        self.model_b = self.trainer_b.train()
        return self.model_b

    def evaluate(self, *, split: str) -> Dict[str, Dict[str, Any]]:
        if self.trainer_a is None or self.trainer_b is None:
            raise RuntimeError("call train_a() and train_b() first")
        return {
            "a": self.trainer_a.evaluate(self.model_a, split=split),
            "b": self.trainer_b.evaluate(self.model_b, split=split),
        }

    def detailed_predictions(self, *, split: str) -> Dict[str, PredictionResult]:
        if self.trainer_a is None or self.trainer_b is None:
            raise RuntimeError("call train_a() and train_b() first")
        return {
            "a": self.trainer_a.get_detailed_predictions(self.model_a, split=split),
            "b": self.trainer_b.get_detailed_predictions(self.model_b, split=split),
        }
