import pytest
from promptillery.config import ExperimentConfig


def _base(**overrides):
    cfg = {
        "name": "cotrain-test",
        "acquisition_mode": "cotrain",
        "cotrain": {
            "student_a": {
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "operator": "coverage",
                "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
                "learning_rate": 2e-4,
                "num_train_epochs": 2,
            },
            "student_b": {
                "model": "microsoft/Phi-3.5-mini-instruct",
                "operator": "boundary",
                "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
                "learning_rate": 1e-4,
                "num_train_epochs": 2,
            },
            "strong_teacher": "openai/gpt-4o",
            "bootstrap_size": 400,
            "variants_per_seed": 5,
            "tau_choices": [0.5, 0.7, 0.9],
            "volume_choices": [8, 16, 32],
            "operator_choices": ["coverage", "boundary", "repair"],
            "self_consistency_n": 3,
            "self_consistency_temperature": 0.7,
            "task_kind": "classification",
        },
    }
    cfg.update(overrides)
    return cfg


def test_cotrain_config_parses():
    cfg = ExperimentConfig.model_validate(_base())
    assert cfg.acquisition_mode == "cotrain"
    assert cfg.cotrain.student_a.model == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.cotrain.student_b.operator == "boundary"
    assert cfg.cotrain.tau_choices == [0.5, 0.7, 0.9]


def test_cotrain_config_rejects_overlapping_operators():
    bad = _base()
    bad["cotrain"]["student_a"]["operator"] = "boundary"
    with pytest.raises(ValueError, match="distinct operators"):
        ExperimentConfig.model_validate(bad)


def test_acquisition_mode_defaults_to_legacy():
    cfg = ExperimentConfig.model_validate({"name": "legacy"})
    assert cfg.acquisition_mode == "legacy"
    assert cfg.cotrain is None


def test_cotrain_config_required_when_mode_is_cotrain():
    with pytest.raises(ValueError, match="cotrain block is required"):
        ExperimentConfig.model_validate(
            {"name": "x", "acquisition_mode": "cotrain"}
        )
