"""Evaluation batch size is independent of the training batch size (task A.13)."""
import pytest
from datasets import Dataset, DatasetDict
from pydantic import ValidationError

from promptillery.config import ExperimentConfig
from promptillery.trainers.factory import TrainerFactory

from test_profiler import _tiny_classifier_trainer  # noqa: E402  (reuse fixture pattern)


def _minimal(**overrides):
    cfg = {
        "name": "t",
        "teacher": "openrouter/openai/gpt-4.1",
        "student": "jhu-clsp/ettin-decoder-150m",
        "student_type": "slm",
        "dataset": "json",
        "dataset_config": {
            "name": "default",
            "text_field": "question",
            "label_field": "answer",
        },
    }
    cfg.update(overrides)
    return cfg


def test_eval_batch_size_defaults_to_64():
    cfg = ExperimentConfig(**_minimal())
    assert cfg.eval_batch_size == 64


def test_eval_batch_size_accepts_override():
    cfg = ExperimentConfig(**_minimal(eval_batch_size=128))
    assert cfg.eval_batch_size == 128


def test_eval_batch_size_rejects_zero():
    with pytest.raises(ValidationError):
        ExperimentConfig(**_minimal(eval_batch_size=0))


def test_train_uses_eval_batch_size_independent_of_batch_size(tmp_path):
    """TrainingArguments must decouple per_device_eval_batch_size from batch_size."""
    from transformers import (
        AutoTokenizer,
        BertConfig,
        BertForSequenceClassification,
    )
    from test_profiler import FIXTURE_TOKENIZER

    tokenizer = AutoTokenizer.from_pretrained(str(FIXTURE_TOKENIZER))
    bert_config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=2,
        pad_token_id=tokenizer.pad_token_id,
    )
    model_dir = tmp_path / "tiny_bert"
    BertForSequenceClassification(bert_config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    config = ExperimentConfig(
        name="eval-batch-smoke",
        student=str(model_dir),
        student_type="transformers",
        num_labels=2,
        text_field="text",
        label_field="label",
        metrics=[],
        auto_modify_name=False,
        num_train_epochs=1,
        batch_size=8,
        eval_batch_size=64,
    )
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"text": ["nice product", "awful service"], "label": [1, 0]}
            ),
            "validation": Dataset.from_dict(
                {"text": ["nice product", "awful service"], "label": [1, 0]}
            ),
        }
    )
    trainer_wrapper = TrainerFactory.create_trainer(config, dataset, tmp_path)

    hf_trainer = trainer_wrapper.train()

    assert hf_trainer.args.per_device_train_batch_size == 8
    assert hf_trainer.args.per_device_eval_batch_size == 64


def test_load_model_uses_configured_eval_batch_size(tmp_path):
    trainer_wrapper = _tiny_classifier_trainer(tmp_path)
    trainer_wrapper.dataset["train"] = trainer_wrapper.dataset["validation"]
    trainer_wrapper.cfg.eval_batch_size = 64
    hf_trainer = trainer_wrapper.train()
    model_path = tmp_path / "saved_model"
    hf_trainer.save_model(str(model_path))

    reloaded = trainer_wrapper.load_model(model_path)

    assert reloaded.args.per_device_eval_batch_size == 64
