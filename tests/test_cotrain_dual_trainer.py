import pytest
from datasets import Dataset
from pathlib import Path

from promptillery.config import ExperimentConfig
from promptillery.cotrain.dual_trainer import DualStudentTrainer
from promptillery.trainers.base import BaseTrainer, PredictionResult
from promptillery.trainers.factory import TrainerFactory


class _FakeTrainer(BaseTrainer):
    instances = []

    def __init__(self, cfg, dataset, out_dir):
        super().__init__(cfg, dataset, out_dir)
        _FakeTrainer.instances.append(self)
        self.trained = 0

    def train(self):
        self.trained += 1
        return self

    def evaluate(self, model, split="test"):
        return {"macro_f1": 0.5 + 0.1 * self.trained}

    def predict_for_augmentation(self, model, split="train"):
        return [0]

    def get_detailed_predictions(self, model, split="train"):
        return PredictionResult(
            indices=[0, 1], predicted_labels=["a", "b"], true_labels=["a", "a"],
            confidences=[0.9, 0.5], entropies=[0.1, 0.6],
        )

    def save_model(self, model): pass
    def load_model(self, model_path): return self
    def push_to_hub(self, model, repo_name): pass


@pytest.fixture(autouse=True)
def _register_fake_trainer():
    TrainerFactory.register_trainer("fake_dual", _FakeTrainer)
    _FakeTrainer.instances = []
    yield


def _base_cfg():
    return ExperimentConfig.model_validate({
        "name": "dual",
        "acquisition_mode": "cotrain",
        "student_type": "fake_dual",
        "cotrain": {
            "student_a": {"model": "Qwen/Qwen2.5-3B-Instruct", "operator": "coverage",
                          "learning_rate": 2e-4},
            "student_b": {"model": "microsoft/Phi-3.5-mini-instruct", "operator": "boundary",
                          "learning_rate": 1e-4},
            "strong_teacher": "openai/gpt-4o", "bootstrap_size": 4,
        },
    })


def test_dual_trainer_creates_two_distinct_trainers(tmp_path):
    ds = {"train": Dataset.from_list([{"text": "x", "label": 0}])}
    cfg = _base_cfg()
    dt = DualStudentTrainer(cfg, ds, tmp_path)
    dt.train_a()
    dt.train_b()
    assert len(_FakeTrainer.instances) == 2
    a_cfg = _FakeTrainer.instances[0].cfg
    b_cfg = _FakeTrainer.instances[1].cfg
    assert a_cfg.student == "Qwen/Qwen2.5-3B-Instruct"
    assert b_cfg.student == "microsoft/Phi-3.5-mini-instruct"
    assert a_cfg.learning_rate == 2e-4
    assert b_cfg.learning_rate == 1e-4


def test_dual_trainer_evaluate_returns_per_student_metrics(tmp_path):
    ds = {"train": Dataset.from_list([{"text": "x", "label": 0}]),
          "validation": Dataset.from_list([{"text": "y", "label": 0}])}
    dt = DualStudentTrainer(_base_cfg(), ds, tmp_path)
    dt.train_a(); dt.train_b()
    metrics = dt.evaluate(split="validation")
    assert "a" in metrics and "b" in metrics
    assert "macro_f1" in metrics["a"]
