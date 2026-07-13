"""Student-vs-teacher fidelity metric.

Fidelity is the top-1 label agreement between a distilled student and the
teacher that supervised it, measured on the held-out test split:

    fidelity = (1/N) * sum_i 1[ normalize(student_i) == normalize(teacher_i) ]

It is distinct from accuracy-vs-gold: it measures how faithfully the student
reproduces the teacher's decisions, following Stanton et al., "Does Knowledge
Distillation Really Work?" (NeurIPS 2021). Because the API teacher is
black-box (top-1 labels only, no distribution), this is raw top-1 agreement
rather than a KL divergence or chance-corrected kappa. Student predictions
that fall outside the shared label space normalize to a value that cannot
match a valid teacher label, so they count as disagreement.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Mapping, Sequence, Union


def normalize_label(value: object) -> str:
    """Normalize a class label into the shared canonical label space.

    Matches the canonical-label normalization used when materializing SFT
    records so student and teacher labels are compared in the same space.
    """
    text = str(value).strip().lower()
    text = " ".join(text.split()).strip(" .,:;")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def teacher_agreement(
    student_labels: Sequence[object],
    teacher_labels: Sequence[object],
) -> float:
    """Return the top-1 student-vs-teacher label agreement on the test split.

    Both sequences are aligned by row index and must be equal length. Agreement
    is the fraction of rows where the normalized student label equals the
    normalized teacher label; invalid or out-of-space student predictions never
    match a valid teacher label and so count as disagreement.

    Raises ``ValueError`` on empty input or a length mismatch: a mismatch is a
    data/alignment error, and an empty comparison has no defined agreement --
    returning 0.0 there would silently launder a missing or empty teacher-label
    file into a plausible-looking (and catastrophically low) fidelity number.
    """
    if len(student_labels) != len(teacher_labels):
        raise ValueError(
            "student_labels and teacher_labels must be equal length; got "
            f"{len(student_labels)} and {len(teacher_labels)}"
        )
    if not teacher_labels:
        raise ValueError("teacher_agreement requires at least one label pair")
    matches = sum(
        1
        for student, teacher in zip(student_labels, teacher_labels)
        if normalize_label(student) == normalize_label(teacher)
    )
    return matches / len(teacher_labels)


def agreement_by_index(
    student_by_index: Mapping[int, object],
    teacher_labels: Mapping[int, object],
) -> float:
    """Top-1 agreement joining student and teacher labels by shared row index.

    The teacher labels define the evaluated rows (and thus the denominator); a
    row for which the student produced no prediction counts as a disagreement.
    Both trainer families reduce to this join once they map their predictions
    to labels keyed by row index.
    """
    indices = sorted(teacher_labels)
    student_aligned = [student_by_index.get(i) for i in indices]
    teacher_aligned = [teacher_labels[i] for i in indices]
    return teacher_agreement(student_aligned, teacher_aligned)


def load_teacher_eval_labels(path: Union[str, Path]) -> Dict[int, str]:
    """Load teacher test-split predictions keyed by original split row index.

    Reads a JSONL file produced by ``materialize-sft --mode teacher
    --split test``. Each record contributes its raw ``teacher_response`` keyed
    by ``source_original_index`` -- the row's true position in the split -- so
    it aligns with the student's per-row predictions. ``source_original_index``
    is preferred over ``source_index`` (the position within the possibly
    subset/reordered materialization selection); the two coincide only under
    full, prefix-order materialization, and diverge whenever the teacher file
    was produced with ``--max-samples`` under a shuffled/stratified strategy.
    The loader falls back to ``source_index`` when the original index is absent
    (older files). Rows the teacher rejected are simply absent from the mapping.
    """
    labels: Dict[int, str] = {}
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            index = record.get("source_original_index", record.get("source_index"))
            labels[int(index)] = record["teacher_response"]
    return labels
