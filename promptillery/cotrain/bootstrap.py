"""Class-stratified disjoint bootstrap partitioning (design §3.2)."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from datasets import Dataset


@dataclass(frozen=True)
class BootstrapPartition:
    bootstrap_a: Dataset
    bootstrap_b: Dataset
    seed: int

    def probe_for(self, student: str) -> Dataset:
        if student == "a":
            return self.bootstrap_b
        if student == "b":
            return self.bootstrap_a
        raise ValueError(f"student must be 'a' or 'b', got {student!r}")


def partition_bootstrap(
    dataset: Dataset,
    *,
    label_field: Optional[str],
    id_field: str,
    target_size: int,
    seed: int,
) -> BootstrapPartition:
    """Split `dataset` into two disjoint frozen bootstraps of size ~target_size/2.

    For classification (label_field is not None): stratified per class, every class
    appears in both halves. For generation (label_field is None): random split.
    """
    if id_field not in dataset.column_names:
        raise ValueError(f"id_field {id_field!r} not in dataset columns")
    if target_size > len(dataset):
        raise ValueError(
            f"target_size={target_size} exceeds dataset size {len(dataset)}"
        )
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    if label_field is None:
        head = indices[:target_size]
        a_idx = head[: target_size // 2]
        b_idx = head[target_size // 2 : target_size]
    else:
        if label_field not in dataset.column_names:
            raise ValueError(f"label_field {label_field!r} not in dataset columns")
        per_class = defaultdict(list)
        for i in indices:
            per_class[dataset[i][label_field]].append(i)
        num_classes = len(per_class)
        per_class_target = target_size // (2 * num_classes)
        if per_class_target < 1:
            raise ValueError(
                "each class needs at least 2 examples to stratify "
                f"(target_size={target_size}, num_classes={num_classes})"
            )
        a_idx, b_idx = [], []
        for cls, idxs in per_class.items():
            if len(idxs) < 2 * per_class_target:
                raise ValueError(
                    f"class {cls} has {len(idxs)} examples, needs {2 * per_class_target}"
                )
            a_idx.extend(idxs[:per_class_target])
            b_idx.extend(idxs[per_class_target : 2 * per_class_target])

    a_idx.sort()
    b_idx.sort()
    return BootstrapPartition(
        bootstrap_a=dataset.select(a_idx),
        bootstrap_b=dataset.select(b_idx),
        seed=seed,
    )
