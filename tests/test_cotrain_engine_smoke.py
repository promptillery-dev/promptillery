import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from datasets import Dataset

from promptillery.config import ExperimentConfig
from promptillery.cotrain.engine import CoTrainEngine
from promptillery.cotrain.arbitration import ArbitrationResult
from promptillery.trainers.base import BaseTrainer, PredictionResult
from promptillery.trainers.factory import TrainerFactory


class _StubTrainer(BaseTrainer):
    def __init__(self, cfg, dataset, out_dir):
        super().__init__(cfg, dataset, out_dir)

    def train(self): return self
    def evaluate(self, model, split="test"): return {"macro_f1": 0.7}
    def predict_for_augmentation(self, model, split="train"): return [0]
    def get_detailed_predictions(self, model, split="train"):
        ds = self.dataset[split]
        true_labels = [r["label"] for r in ds]
        # Flip the first row's prediction so the engine has at least one
        # fault seed per probe — needed so the validation pipeline writes
        # something to the audit ledger in the smoke test.
        predicted_labels = list(true_labels)
        if predicted_labels:
            flipped = "beta" if predicted_labels[0] == "alpha" else "alpha"
            predicted_labels[0] = flipped
        return PredictionResult(
            indices=list(range(len(ds))),
            predicted_labels=predicted_labels,
            true_labels=true_labels,
            confidences=[0.95] * len(ds),
            entropies=[0.05] * len(ds),
            label_names=["alpha", "beta"],
        )
    def save_model(self, model): pass
    def load_model(self, p): return self
    def push_to_hub(self, m, name): pass


@pytest.fixture(autouse=True)
def _stub_trainer():
    TrainerFactory.register_trainer("stub_cotrain", _StubTrainer)
    yield


def _make_dataset():
    rows = [{"text": f"row {i}", "label": "alpha" if i % 2 == 0 else "beta",
             "id": f"r{i}"} for i in range(8)]
    return {"train": Dataset.from_list(rows),
            "validation": Dataset.from_list(rows[:4])}


def _make_cfg(tmp_path):
    return ExperimentConfig.model_validate({
        "name": "cotrain-smoke",
        "acquisition_mode": "cotrain",
        "student_type": "stub_cotrain",
        "cycles": 2,
        "base_output_dir": str(tmp_path),
        "metrics": ["macro_f1"],
        "cotrain": {
            "student_a": {"model": "fake-a", "operator": "coverage"},
            "student_b": {"model": "fake-b", "operator": "boundary"},
            "strong_teacher": "openai/gpt-4o",
            "bootstrap_size": 4,
            "variants_per_seed": 1,
            "operator_choices": ["coverage", "boundary"],
            "volume_choices": [2],
            "tau_choices": [0.5],
            "task_kind": "classification",
            "controller": "frugalkd_cotrain_p",
        },
    })


def test_engine_runs_two_cycles_without_real_models(tmp_path, monkeypatch):
    fake_arb = AsyncMock(return_value={"text": "alpha",
                                        "usage": {"input": 1, "output": 1}})
    fake_generate = lambda prompt, n, temperature: [
        json.dumps({"variants": [{"text": f"variant for {prompt[:8]}"}]})
    ]
    cfg = _make_cfg(tmp_path)
    engine = CoTrainEngine(
        cfg, dataset=_make_dataset(),
        out_dir=tmp_path / "run",
        generate_fn_a=fake_generate, generate_fn_b=fake_generate,
        arbiter_complete=fake_arb,
    )
    asyncio.run(engine.run())
    out = engine.results
    assert len(out["cycles"]) == 2
    assert (tmp_path / "run" / "cotrain_ledger.jsonl").exists()
    summary = engine.audit_ledger.summary_counts()
    assert sum(summary.values()) > 0


def test_engine_stops_when_controller_emits_stop(tmp_path):
    fake_arb = AsyncMock(return_value={"text": "alpha", "usage": {}})
    cfg = _make_cfg(tmp_path)
    cfg = cfg.model_copy(update={"cycles": 5})
    engine = CoTrainEngine(
        cfg, dataset=_make_dataset(),
        out_dir=tmp_path / "run",
        generate_fn_a=lambda p, n, temperature: [json.dumps({"variants": [{"text": "v"}]})],
        generate_fn_b=lambda p, n, temperature: [json.dumps({"variants": [{"text": "v"}]})],
        arbiter_complete=fake_arb,
        force_stop_after_cycle=1,
    )
    asyncio.run(engine.run())
    assert len(engine.results["cycles"]) == 2
