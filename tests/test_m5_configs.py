"""Offline validation of the M5 same-N gold-FT configs.

Parse-only (no GPU/network/teacher key): every config must construct an
ExperimentConfig, carry the same-N protocol invariants, and match the G3
"fine-tuned (1 cycle)" recipe. Decoder prompt-format identity needs no test
here: the prep script builds the gold SFT from examples/paper/G3_<D>_materialize.yaml
itself (identity by construction; covered in tests/test_prep_m5_same_n.py).
"""
from pathlib import Path

import pytest
import yaml

from promptillery.config import ExperimentConfig
from promptillery.trainers.factory import TrainerFactory

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "paper"

ENCODER = "ettin_encoder"
# RoBERTa's same-N cell reuses the ettin_encoder run's N;
# same seed + same N => identical data, so it reads the ettin_encoder train file.
ROBERTA = "roberta_base"
DECODERS = {
    "ettin_decoder": "jhu-clsp/ettin-decoder-150m",
    "gemma3_270m": "google/gemma-3-270m-it",
}
# Grows one dataset per implementation task.
NUM_CLASSES = {"sst2": 2, "agnews": 4, "imdb": 2, "yahoo": 10, "huffpost": 41}

G3_RECIPE_KEYS = ["learning_rate", "num_train_epochs", "batch_size", "seed"]
G3_DECODER_EXTRA_KEYS = ["warmup_steps", "weight_decay"]
G3_DECODER_TRAINER_KEYS = [
    "use_lora", "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules",
    "max_seq_length", "completion_only_loss", "generation_max_new_tokens",
    "generation_batch_size", "answer_extraction", "canonical_label_decoding",
    "task_metric", "prompt_field", "response_field", "gold_answer_field",
]


def _cells():
    for d in sorted(NUM_CLASSES):
        yield d, ENCODER
        yield d, ROBERTA
        for s in sorted(DECODERS):
            yield d, s


def _path(d, s):
    return EXAMPLES / f"M5_{d}_same_n_{s}.yaml"


def _load(path):
    return ExperimentConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("d,s", list(_cells()), ids=[f"{d}-{s}" for d, s in _cells()])
def test_m5_config_parses_and_carries_protocol(d, s):
    cfg = _load(_path(d, s))
    assert cfg.teacher == "openrouter/openai/gpt-4.1"
    assert cfg.seed == 13
    assert cfg.base_output_dir == "out/m5"
    assert cfg.dataset == "json"
    assert cfg.dataset_config.num_classes == NUM_CLASSES[d]
    files = cfg.dataset_kwargs["data_files"]

    # One plain fine-tune, zero teacher calls.
    assert cfg.cycles == 1
    assert cfg.prompt is None, "same-N runs must not augment (no prompt key)"
    assert cfg.sampling.enabled is False, "rows are pre-built by prep_m5_same_n"
    assert cfg.has_list_parameters() is False
    assert cfg.trainer_config["report_held_out_test"] is True
    assert cfg.require_validation_split is True

    if s in (ENCODER, ROBERTA):
        assert cfg.student_type == "transformers"
        expected_student = {
            ENCODER: "jhu-clsp/ettin-encoder-150m",
            ROBERTA: "FacebookAI/roberta-base",
        }[s]
        assert cfg.student == expected_student
        # Both encoder cells read the ettin_encoder train file: RoBERTa's
        # matched N reuses the Ettin-encoder run's N, and same seed + same N
        # produce byte-identical data.
        assert files == {
            "train": f"out/m5/{d}/ettin_encoder/train.jsonl",
            "validation": f"out/m5/{d}/validation.jsonl",
            "test": f"out/m5/{d}/test.jsonl",
        }
        assert cfg.metrics == ["accuracy", "f1"]
    else:
        assert cfg.student_type == "slm"
        assert cfg.student == DECODERS[s]
        assert files == {
            "train": f"out/m5/{d}/{s}/train_sft.jsonl",
            "validation": f"out/g3/{d}/validation_sft.jsonl",
            "test": f"out/g3/{d}/test_sft.jsonl",
        }
        tc = cfg.trainer_config
        assert tc["canonical_labels_path"] == f"out/g3/{d}/canonical_labels.json"
        assert "max_eval_generation_samples" not in tc, "unset = full-test eval"
        assert "fidelity" not in tc


@pytest.mark.parametrize("d", sorted(NUM_CLASSES))
def test_m5_recipe_matches_g3_fine_tuned_row(d):
    g3_enc = _load(EXAMPLES / f"G3_{d}_ettin_encoder.yaml")
    m5_enc = _load(_path(d, ENCODER))
    for key in G3_RECIPE_KEYS:
        assert getattr(m5_enc, key) == getattr(g3_enc, key), key
    m5_rob = _load(_path(d, ROBERTA))
    for key in G3_RECIPE_KEYS:
        assert getattr(m5_rob, key) == getattr(m5_enc, key), key
    for s in DECODERS:
        g3_dec = _load(EXAMPLES / f"G3_{d}_{s}.yaml")
        m5_dec = _load(_path(d, s))
        for key in G3_RECIPE_KEYS + G3_DECODER_EXTRA_KEYS:
            assert getattr(m5_dec, key) == getattr(g3_dec, key), key
        for key in G3_DECODER_TRAINER_KEYS:
            assert m5_dec.trainer_config[key] == g3_dec.trainer_config[key], key


@pytest.mark.parametrize("d,s", list(_cells()), ids=[f"{d}-{s}" for d, s in _cells()])
def test_m5_trainer_instantiates(d, s):
    cfg = _load(_path(d, s))
    if cfg.student_type not in TrainerFactory.get_available_types():
        pytest.skip(f"{cfg.student_type} extras not installed")
    assert isinstance(cfg.cycles, int)
