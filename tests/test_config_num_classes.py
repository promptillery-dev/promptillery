"""num_classes is optional for slm students, required for classifier students."""
import pytest
from pydantic import ValidationError

from promptillery.config import DatasetConfig, ExperimentConfig


def _base(**overrides):
    cfg = {
        "name": "t",
        "teacher": "openrouter/openai/gpt-4.1",
        "student": "jhu-clsp/ettin-decoder-150m",
        "student_type": "slm",
        "dataset": "json",
        "dataset_config": {"name": "default", "text_field": "question", "label_field": "answer"},
    }
    cfg.update(overrides)
    return cfg


def test_slm_config_parses_without_num_classes():
    cfg = ExperimentConfig(**_base())
    assert cfg.dataset_config.num_classes is None


def test_causal_lm_sft_alias_parses_without_num_classes():
    cfg = ExperimentConfig(**_base(student_type="causal_lm_sft"))
    assert cfg.dataset_config.num_classes is None


def test_transformers_config_requires_num_classes():
    with pytest.raises(ValidationError, match="num_classes"):
        ExperimentConfig(
            **_base(student_type="transformers", student="jhu-clsp/ettin-encoder-150m")
        )


def test_num_classes_still_floor_checked_when_set():
    with pytest.raises(ValidationError):
        DatasetConfig(name="d", num_classes=1)


def test_existing_g3_configs_still_parse():
    ExperimentConfig.from_yaml("examples/paper/G3_agnews_ettin_encoder.yaml")
    ExperimentConfig.from_yaml("examples/paper/G3_agnews_ettin_decoder.yaml")


def test_optional_set_matches_sft_student_types():
    """config.py can't import trainers (cycle); this test pins the sync instead."""
    from promptillery.config import NUM_CLASSES_OPTIONAL_STUDENT_TYPES
    from promptillery.trainers.factory import SFT_STUDENT_TYPES

    assert NUM_CLASSES_OPTIONAL_STUDENT_TYPES == SFT_STUDENT_TYPES
