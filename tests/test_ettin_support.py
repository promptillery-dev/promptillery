"""Ettin matched-pair enablement (issue #3).

Ettin ships identically-trained encoder and decoder checkpoints. The decoder
path (SFT trainer) already forwards trust_remote_code; these tests lock the
symmetric behavior for the encoder (classifier) path so an Ettin encoder
config can opt into remote code the same way.
"""

from types import SimpleNamespace

from datasets import Dataset, DatasetDict

import promptillery.trainers.transformers_trainer as tt
from promptillery.config import ExperimentConfig


class _RecordingLoader:
    """Stub for the HuggingFace hub loaders; records from_pretrained kwargs."""

    calls: list = []

    def __init__(self, **_):
        self.config = SimpleNamespace(id2label={0: "a", 1: "b"})

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        cls.calls.append(kwargs)
        return cls()


def _encoder_config(tmp_path, trust_remote_code):
    trainer_config = {}
    if trust_remote_code is not None:
        trainer_config["trust_remote_code"] = trust_remote_code
    return ExperimentConfig(
        name="ettin-encoder-smoke",
        student="jhu-clsp/ettin-encoder-17m",
        student_type="transformers",
        num_labels=2,
        text_field="text",
        label_field="label",
        metrics=[],
        auto_modify_name=False,
        trainer_config=trainer_config,
    )


def _tiny_dataset():
    return DatasetDict(
        {"train": Dataset.from_dict({"text": ["hi"], "label": [0]})}
    )


def test_encoder_forwards_trust_remote_code_to_hub_loaders(tmp_path, monkeypatch):
    _RecordingLoader.calls = []
    monkeypatch.setattr(tt, "AutoTokenizer", _RecordingLoader)
    monkeypatch.setattr(tt, "AutoModelForSequenceClassification", _RecordingLoader)

    tt.TransformersTrainer(
        _encoder_config(tmp_path, trust_remote_code=True),
        _tiny_dataset(),
        tmp_path,
    )

    # Both the tokenizer and the model must be loaded with remote code enabled.
    assert len(_RecordingLoader.calls) == 2
    assert all(call.get("trust_remote_code") is True for call in _RecordingLoader.calls)


def test_encoder_defaults_trust_remote_code_off(tmp_path, monkeypatch):
    _RecordingLoader.calls = []
    monkeypatch.setattr(tt, "AutoTokenizer", _RecordingLoader)
    monkeypatch.setattr(tt, "AutoModelForSequenceClassification", _RecordingLoader)

    tt.TransformersTrainer(
        _encoder_config(tmp_path, trust_remote_code=None),
        _tiny_dataset(),
        tmp_path,
    )

    # Absent config -> remote code stays disabled (safe default).
    assert all(
        call.get("trust_remote_code") is False for call in _RecordingLoader.calls
    )
