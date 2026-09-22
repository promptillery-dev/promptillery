"""Shared pytest fixtures: a synthetic audit run directory (issue #7)."""

import json

import pytest
import yaml
from datasets import Dataset, DatasetDict

# 30-word base sentence: one appended word stays above the 0.8 MinHash
# threshold on word 3-gram shingles.
LONG_BASE = (
    "the film opens with a slow deliberate sequence that patiently introduces "
    "each character while the camera lingers on small details of the town "
    "its people and their daily rituals"
)

SEED_ROWS = [
    ("the movie was a complete triumph of imagination and heart", 1),
    ("a dull plodding mess that squanders its promising premise", 0),
    ("warm funny and quietly devastating performances all around", 1),
    ("clumsy dialogue and flat direction sink this hollow thriller", 0),
    (LONG_BASE, 1),
]

# Cycle 1: an exact dup of seed row 0, a near-dup of LONG_BASE, one novel row.
AUG_CYCLE_1 = [
    ("The movie was a complete  TRIUMPH of imagination and heart", 1),
    (LONG_BASE + " lovingly", 1),
    ("an abrasive misfire with no redeeming qualities whatsoever", 0),
]
# The teacher returned a 4th record that acceptance truncated away.
REJECTED_CYCLE_1 = ("a fourth record the engine truncated away entirely", 0)

AUG_CYCLE_2 = [
    ("a gentle meditation on loss that rewards patient viewers", 1),
    ("the screenplay recycles every genre cliche without irony", 0),
]

GOLD_VALIDATION = [
    ("an uplifting and beautifully shot family drama", 1),
    ("tedious pacing undermines the intriguing premise", 0),
    ("a career best performance anchors this thoughtful film", 1),
    ("the jokes land with a thud in this laughless comedy", 0),
]
GOLD_TEST = [
    ("a soaring emotionally rich piece of cinema", 1),
    ("an incoherent script wastes a talented cast", 0),
]

LABEL_TEXTS = {0: "negative", 1: "positive"}

FIXTURE_CONFIG = {
    "name": "audit_fixture_run",
    "teacher": "openrouter/openai/gpt-4.1",
    "student": "google-bert/bert-base-uncased",
    "student_type": "transformers",
    "dataset": "fixture/sst2-tiny",
    "dataset_config": {
        "name": "default",
        "num_classes": 2,
        "text_field": "text",
        "label_field": "label",
    },
    "auto_modify_name": False,
    "cycles": 3,
    "seed": 13,
    "sampling": {
        "enabled": False,
        "sample_size": 1000,
        "train_ratio": 0.8,
        "stratify_column": "label",
        "seed": 42,
    },
}


def _split(rows, source_split, origin_cycle=None):
    """Build a Dataset split from (text, label) pairs."""
    n = len(rows)
    return Dataset.from_dict(
        {
            "text": [text for text, _ in rows],
            "label": [label for _, label in rows],
            "label_text": [LABEL_TEXTS[label] for _, label in rows],
            "source_split": [source_split] * n,
            "source_idx": [-1] * n,
            "origin_cycle": (
                origin_cycle if origin_cycle is not None else [0] * n
            ),
        }
    )


def build_seed_dataset():
    """The seed DatasetDict as the engine would have it before augmentation."""
    return DatasetDict(
        {
            "train": _split(SEED_ROWS, "train"),
            "validation": _split(GOLD_VALIDATION, "validation"),
            "test": _split(GOLD_TEST, "test"),
        }
    )


def build_final_dataset():
    """The final-cycle DatasetDict: seed train rows + all augmented rows."""
    from datasets import concatenate_datasets

    train = concatenate_datasets(
        [
            _split(SEED_ROWS, "train"),
            _split(AUG_CYCLE_1, "augmented", [1] * len(AUG_CYCLE_1)),
            _split(AUG_CYCLE_2, "augmented", [2] * len(AUG_CYCLE_2)),
        ]
    )
    return DatasetDict(
        {
            "train": train,
            "validation": _split(GOLD_VALIDATION, "validation"),
            "test": _split(GOLD_TEST, "test"),
        }
    )


def _raw_response(rows):
    content = json.dumps(
        {"articles": [{"text": text, "label": label} for text, label in rows]}
    )
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": 500,
            "completion_tokens": 200,
            "total_tokens": 700,
        },
    }


ATTEMPTS = [
    {
        "cycle": 1,
        "attempt_id": "audit_fixture_run:c1:a1",
        "status": "success",
        "failure_type": None,
        "metadata": {
            "records_requested": 4,
            "records_parsed": 4,
            "records_accepted": 3,
        },
    },
    {
        "cycle": 2,
        "attempt_id": "audit_fixture_run:c2:a1",
        "status": "success",
        "failure_type": None,
        "metadata": {
            "records_requested": 2,
            "records_parsed": 2,
            "records_accepted": 2,
        },
    },
    {
        "cycle": 2,
        "attempt_id": "audit_fixture_run:c2:a2",
        "status": "failed",
        "failure_type": "Timeout",
        "metadata": {"records_requested": 8},
    },
]


@pytest.fixture
def audit_run_dir(tmp_path):
    """Factory building a synthetic completed run directory."""

    def _build(with_datasets=True, with_raw=True):
        run_dir = tmp_path / "audit_fixture_run"
        run_dir.mkdir(exist_ok=True)
        (run_dir / "experiment_config.yaml").write_text(
            yaml.safe_dump(FIXTURE_CONFIG), encoding="utf-8"
        )
        with (run_dir / "teacher_attempts.jsonl").open("w") as f:
            for row in ATTEMPTS:
                f.write(json.dumps(row) + "\n")
        if with_raw:
            (run_dir / "teacher_response_cycle_1.json").write_text(
                json.dumps(_raw_response(AUG_CYCLE_1 + [REJECTED_CYCLE_1])),
                encoding="utf-8",
            )
            (run_dir / "teacher_response_cycle_2.json").write_text(
                json.dumps(_raw_response(AUG_CYCLE_2)), encoding="utf-8"
            )
        if with_datasets:
            build_final_dataset().save_to_disk(str(run_dir / "dataset_cycle_2"))
        return run_dir

    return _build


@pytest.fixture
def audit_seed_loader():
    """Injectable dataset_loader for reconstruction tests (no HF download)."""
    return lambda config: build_seed_dataset()
