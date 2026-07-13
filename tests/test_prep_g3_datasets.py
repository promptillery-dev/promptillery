import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "prep_g3", Path(__file__).resolve().parents[1] / "scripts" / "prep_g3_datasets.py"
)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def _huffpost_rows():
    cats = ["POLITICS", "WELLNESS", "ENTERTAINMENT", "TRAVEL", "STYLE & BEAUTY",
            "PARENTING", "HEALTHY LIVING", "QUEER VOICES", "FOOD & DRINK", "BUSINESS",
            "COMEDY", "SPORTS", "BLACK VOICES", "HOME & LIVING", "PARENTS", "THE WORLDPOST",
            "WEDDINGS", "WOMEN", "IMPACT", "DIVORCE", "CRIME", "MEDIA", "WEIRD NEWS",
            "GREEN", "WORLDPOST", "RELIGION", "STYLE", "SCIENCE", "WORLD NEWS", "TASTE",
            "TECH", "MONEY", "ARTS", "FIFTY", "GOOD NEWS", "ARTS & CULTURE",
            "ENVIRONMENT", "COLLEGE", "LATINO VOICES", "CULTURE & ARTS", "EDUCATION", "LIFESTYLE"]
    rows = []
    for i, c in enumerate(cats * 3):
        rows.append({"category": c, "headline": f"head {i}", "short_description": f"desc {i}"})
    return rows


def test_huffpost_merges_worldpost_to_41_classes():
    records = prep.normalize_huffpost(_huffpost_rows())
    label_texts = {r["label_text"] for r in records}
    assert len(label_texts) == 41
    assert "THE WORLDPOST" not in label_texts
    assert "WORLDPOST" in label_texts


def test_huffpost_schema_and_text_composition():
    records = prep.normalize_huffpost(_huffpost_rows())
    r = records[0]
    assert set(r.keys()) == {"text", "label", "label_text"}
    assert isinstance(r["label"], int) and 0 <= r["label"] <= 40
    assert r["text"]


def test_stratified_split_is_deterministic_and_covers_splits():
    records = prep.normalize_huffpost(_huffpost_rows())
    a = prep.stratified_split(records, {"train": 0.9, "test": 0.1}, seed=13)
    b = prep.stratified_split(records, {"train": 0.9, "test": 0.1}, seed=13)
    assert [r["text"] for r in a["train"]] == [r["text"] for r in b["train"]]
    assert set(a) == {"train", "test"}
    assert len(a["train"]) + len(a["test"]) == len(records)


def test_stratified_split_rejects_more_than_two_splits():
    records = prep.normalize_huffpost(_huffpost_rows())
    with pytest.raises(ValueError, match="stratified_split supports exactly two splits"):
        prep.stratified_split(records, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=13)


def _yahoo_rows():
    names = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference",
             "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music",
             "Family & Relationships", "Politics & Government"]
    rows = [{"topic": i % 10, "question_title": f"title {i}",
             "question_content": f"content {i}"} for i in range(50)]
    return rows, names


def test_yahoo_normalize_schema_and_text():
    rows, names = _yahoo_rows()
    recs = prep.normalize_yahoo(rows, topic_names=names)
    r = recs[0]
    assert set(r.keys()) == {"text", "label", "label_text"}
    assert r["label_text"] == names[r["label"]]
    assert "title 0" in r["text"] and "content 0" in r["text"]


def test_yahoo_pool_caps_stratified_and_deterministic():
    rows, names = _yahoo_rows()
    recs = prep.normalize_yahoo(rows, topic_names=names)
    p1 = prep.stratified_pool(recs, cap=20, seed=13)
    p2 = prep.stratified_pool(recs, cap=20, seed=13)
    assert len(p1) == 20
    assert [r["text"] for r in p1] == [r["text"] for r in p2]
    assert len({r["label"] for r in p1}) == 10
