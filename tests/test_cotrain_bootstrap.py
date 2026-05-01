import pytest
from datasets import Dataset

from promptillery.cotrain.bootstrap import partition_bootstrap, BootstrapPartition


def _toy_classification(n_per_class=20, num_classes=4, seed_offset=0):
    rows = []
    for cls in range(num_classes):
        for i in range(n_per_class):
            rows.append({
                "text": f"class {cls} example {i + seed_offset}",
                "label": cls,
                "id": f"c{cls}-i{i + seed_offset}",
            })
    return Dataset.from_list(rows)


def test_partition_is_disjoint_and_stratified():
    ds = _toy_classification(n_per_class=20, num_classes=4)
    part = partition_bootstrap(
        ds, label_field="label", id_field="id", target_size=40, seed=7
    )
    assert isinstance(part, BootstrapPartition)
    a_ids = {r["id"] for r in part.bootstrap_a}
    b_ids = {r["id"] for r in part.bootstrap_b}
    assert a_ids.isdisjoint(b_ids)
    assert len(a_ids) == 20 and len(b_ids) == 20
    for cls in range(4):
        a_count = sum(1 for r in part.bootstrap_a if r["label"] == cls)
        b_count = sum(1 for r in part.bootstrap_b if r["label"] == cls)
        assert a_count == 5 and b_count == 5


def test_partition_is_deterministic_on_same_seed():
    ds = _toy_classification()
    p1 = partition_bootstrap(ds, label_field="label", id_field="id", target_size=40, seed=42)
    p2 = partition_bootstrap(ds, label_field="label", id_field="id", target_size=40, seed=42)
    assert [r["id"] for r in p1.bootstrap_a] == [r["id"] for r in p2.bootstrap_a]


def test_partition_raises_when_class_too_small_to_stratify():
    ds = Dataset.from_list([
        {"text": "a", "label": 0, "id": "a"},
        {"text": "b", "label": 1, "id": "b"},
    ])
    with pytest.raises(ValueError, match="at least 2 examples"):
        partition_bootstrap(ds, label_field="label", id_field="id", target_size=2, seed=0)


def test_partition_for_generation_uses_random_split():
    ds = Dataset.from_list([
        {"problem": f"p{i}", "answer": str(i), "id": f"g{i}"} for i in range(40)
    ])
    part = partition_bootstrap(
        ds, label_field=None, id_field="id", target_size=40, seed=0
    )
    a_ids = {r["id"] for r in part.bootstrap_a}
    b_ids = {r["id"] for r in part.bootstrap_b}
    assert a_ids.isdisjoint(b_ids)
    assert len(a_ids) == 20 and len(b_ids) == 20
