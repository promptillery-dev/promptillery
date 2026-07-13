"""prep_gsm8k: deterministic label-free 1K carve + gold top-up superset."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "prep_gsm8k", Path(__file__).resolve().parents[1] / "scripts" / "prep_gsm8k.py"
)
prep_gsm8k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep_gsm8k)


def _rows(n):
    return [
        {"question": f"Q{i}: how many? ", "answer": f"Step <<{i}*2={i * 2}>>{i * 2}.\n#### {i * 2}"}
        for i in range(n)
    ]


def test_strip_calculator_annotations():
    assert prep_gsm8k.strip_calculator_annotations("a <<48/2=24>>24 b") == "a 24 b"


def test_normalize_schema_and_answer_number():
    rec = prep_gsm8k.normalize_gsm8k(_rows(1))[0]
    assert set(rec) == {"question", "answer", "answer_number"}
    assert "<<" not in rec["answer"]
    assert rec["answer_number"] == "0"
    assert rec["answer"].endswith("#### 0")


def test_normalize_rejects_answer_without_number():
    with pytest.raises(ValueError):
        prep_gsm8k.normalize_gsm8k([{"question": "q", "answer": "no digits"}])


def test_sample_and_split_deterministic_and_sized():
    records = prep_gsm8k.normalize_gsm8k(_rows(50))
    train_a, val_a = prep_gsm8k.sample_and_split(records, sample_size=10, train_ratio=0.8, seed=13)
    train_b, val_b = prep_gsm8k.sample_and_split(records, sample_size=10, train_ratio=0.8, seed=13)
    assert (train_a, val_a) == (train_b, val_b)
    assert len(train_a) == 8 and len(val_a) == 2
    assert not {r["question"] for r in train_a} & {r["question"] for r in val_a}


def test_topup_is_superset_and_deterministic():
    records = prep_gsm8k.normalize_gsm8k(_rows(50))
    seed_train, _ = prep_gsm8k.sample_and_split(records, 10, 0.8, seed=13)
    topped = prep_gsm8k.topup_pool(records, seed_train, target_n=20, seed=13)
    assert len(topped) == 20
    assert {r["question"] for r in seed_train} <= {r["question"] for r in topped}
    assert topped == prep_gsm8k.topup_pool(records, seed_train, target_n=20, seed=13)


def test_topup_excludes_validation_rows():
    records = prep_gsm8k.normalize_gsm8k(_rows(50))
    seed_train, seed_val = prep_gsm8k.sample_and_split(records, 10, 0.8, seed=13)
    topped = prep_gsm8k.topup_pool(records, seed_train, target_n=45, seed=13, exclude=seed_val)
    assert not {r["question"] for r in seed_val} & {r["question"] for r in topped}
    assert len(topped) == 45


def test_topup_rejects_impossible_targets():
    records = prep_gsm8k.normalize_gsm8k(_rows(12))
    seed_train, _ = prep_gsm8k.sample_and_split(records, 10, 0.8, seed=13)
    with pytest.raises(ValueError):
        prep_gsm8k.topup_pool(records, seed_train, target_n=100, seed=13)
    with pytest.raises(ValueError):
        prep_gsm8k.topup_pool(records, seed_train, target_n=3, seed=13)
