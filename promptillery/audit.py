"""Gold-anchored synthetic-data audit (issue #7).

Retroactive audit over completed run directories: per-cycle duplicate rates,
lexical diversity, label drift, teacher-failure counts with concrete
exemplars, and two-way gold-anchored label consistency (offline gold-only
verifier + teacher-on-gold API probe). Everything except the probe is
offline and free. Gold = the original human labels of the run's validation
and test splits, untouched by the teacher.

Metric functions raise ``ValueError`` on empty reference/label inputs rather
than returning 0.0 (the ``fidelity.py`` principle): silence must never look
like a clean audit.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .config import ExperimentConfig
from .fidelity import normalize_label
from .token_tracker import OperationType, TokenTracker

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    """Normalize text for duplicate detection: lowercase, collapse whitespace."""
    return " ".join(str(text).lower().split())


def exact_duplicate_flags(
    texts: Sequence[str], reference: Sequence[str]
) -> List[bool]:
    """Flag texts whose normalized form occurs in reference or earlier in texts.

    The first occurrence of a text is never a duplicate; later repeats are.
    """
    if not reference:
        raise ValueError(
            "exact_duplicate_flags requires a non-empty reference corpus"
        )
    seen = {_normalize_text(text) for text in reference}
    flags: List[bool] = []
    for text in texts:
        key = _normalize_text(text)
        flags.append(key in seen)
        seen.add(key)
    return flags


def distinct_n(texts: Sequence[str], n: int = 2) -> float:
    """Distinct-n lexical diversity: unique word n-grams / total word n-grams."""
    if not texts:
        raise ValueError("distinct_n requires at least one text")
    unique: set = set()
    total = 0
    for text in texts:
        words = _normalize_text(text).split()
        grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        unique.update(grams)
        total += len(grams)
    if total == 0:
        raise ValueError(
            f"distinct_n found no {n}-grams; every text is shorter than n words"
        )
    return len(unique) / total


def label_drift(
    aug_labels: Sequence[str], seed_labels: Sequence[str]
) -> float:
    """Total-variation distance between augmented and seed label distributions."""
    if not aug_labels or not seed_labels:
        raise ValueError("label_drift requires non-empty label sequences")

    def distribution(labels: Sequence[str]) -> Dict[str, float]:
        counts = Counter(labels)
        return {key: value / len(labels) for key, value in counts.items()}

    p = distribution(aug_labels)
    q = distribution(seed_labels)
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _shingles(text: str, n: int = 3) -> set:
    """Word n-gram shingles; texts shorter than n words fall back to unigrams."""
    words = _normalize_text(text).split()
    if len(words) >= n:
        return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}
    return set(words) if words else {""}


def near_duplicate_flags(
    texts: Sequence[str],
    reference: Sequence[str],
    threshold: float = 0.8,
    num_perm: int = 128,
) -> List[bool]:
    """Flag MinHash near-duplicates (Jaccard >= threshold on word 3-grams).

    Reference = prior corpus plus earlier texts in the same batch, matching
    ``exact_duplicate_flags``. Callers exclude exact duplicates themselves
    (NDup% counts near-but-not-exact duplicates).
    """
    from datasketch import MinHash, MinHashLSH

    if not reference:
        raise ValueError(
            "near_duplicate_flags requires a non-empty reference corpus"
        )

    def _minhash(text: str) -> "MinHash":
        m = MinHash(num_perm=num_perm)
        for shingle in _shingles(text):
            m.update(shingle.encode("utf-8"))
        return m

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, text in enumerate(reference):
        lsh.insert(f"ref/{i}", _minhash(text))
    flags: List[bool] = []
    for i, text in enumerate(texts):
        m = _minhash(text)
        flags.append(bool(lsh.query(m)))
        lsh.insert(f"new/{i}", m)
    return flags


def parse_teacher_records(
    config: ExperimentConfig, content: str, *, with_teacher_response: bool = False
) -> List[Dict[str, Any]]:
    """Re-parse a raw teacher response through the engine's structured models.

    Returns uniform records ``{"text", "label"}``: classification labels stay
    ints; SFT labels prefer ``gold_answer`` and fall back to
    ``teacher_response`` — the trained-on value ``_build_augmented_sft_rows``
    writes to ``config.label_field``. SFT prompts are wrapped with
    ``trainer_config.augmentation_student_prompt_template`` when the run
    configured one, mirroring ``_build_augmented_sft_rows``. Raises pydantic
    ``ValidationError`` on unparseable content.

    ``with_teacher_response=True`` additionally attaches the stripped
    ``teacher_response`` under ``"_teacher_response"`` on each SFT record.
    ``_build_augmented_sft_rows`` (engine.py:1397-1400) drops a record when
    ``student_prompt`` or ``teacher_response`` is empty — independent of
    ``gold_answer`` — so reconstruction needs ``teacher_response`` on hand to
    replay that exact rule; other callers (e.g. ``failure_summary``) don't
    need it and get the plain two-key shape by default. No-op for
    classification records.
    """
    from .engine import (
        AugmentedResponse,
        AugmentedSFTResponse,
        create_prompt_environment,
    )
    from .trainers.factory import SFT_STUDENT_TYPES

    if config.student_type in SFT_STUDENT_TYPES:
        trainer_config = getattr(config, "trainer_config", None) or {}
        wrap = trainer_config.get("augmentation_student_prompt_template")
        template = (
            create_prompt_environment().from_string(wrap) if wrap else None
        )
        records = []
        for record in AugmentedSFTResponse.model_validate_json(content).records:
            text = str(record.student_prompt or "").strip()
            if template is not None and text:
                text = template.render(question=text)
            teacher_response = str(record.teacher_response or "").strip()
            # Prefer gold_answer, fall back to teacher_response: the exact
            # value _build_augmented_sft_rows trains on (engine.py) and
            # writes to the dataset's label_field column.
            label = str(record.gold_answer or teacher_response or "").strip()
            row: Dict[str, Any] = {"text": text, "label": label}
            if with_teacher_response:
                row["_teacher_response"] = teacher_response
            records.append(row)
        return records
    response = AugmentedResponse.model_validate_json(content)
    return [{"text": a.text, "label": a.label} for a in response.articles]


@dataclass
class FailureExemplar:
    """One concrete recovered failure case (JSON output only)."""

    attempt_id: str
    cycle: int
    kind: str  # "attempt_failure" | "rejected_record"
    failure_type: Optional[str]
    text: str
    label: Optional[str] = None


@dataclass
class FailureSummary:
    """Per-cycle teacher-failure accounting from teacher_attempts.jsonl."""

    cycle: int
    n_attempts: int
    records_requested: int
    records_accepted: int
    n_failed_attempts: int
    n_fail: int
    exemplars: List[FailureExemplar] = field(default_factory=list)


def failure_summary(
    attempts: Sequence[Dict[str, Any]],
    cycle: int,
    *,
    raw_responses: Optional[Dict[int, Dict[str, Any]]] = None,
    accepted_texts: Optional[set] = None,
    config: Optional[ExperimentConfig] = None,
    max_exemplars: int = 10,
) -> FailureSummary:
    """#Fail per cycle plus concrete exemplars.

    n_fail = max(0, requested - accepted) + attempts with failure_type set.
    Exemplars: rejected records recovered by diffing the re-parsed raw
    response against the accepted texts; unparseable responses contribute a
    bounded raw snippet.
    """
    cycle_attempts = [a for a in attempts if a.get("cycle") == cycle]
    requested = accepted = n_failed = 0
    exemplars: List[FailureExemplar] = []

    raw = (raw_responses or {}).get(cycle)
    raw_content: Optional[str] = None
    parsed_records: Optional[List[Dict[str, Any]]] = None
    if raw is not None:
        raw_content = str(raw["choices"][0]["message"]["content"])
        if config is not None:
            try:
                parsed_records = parse_teacher_records(config, raw_content)
            except Exception as exc:
                logger.debug("cycle %s raw response unparseable: %s", cycle, exc)

    for attempt in cycle_attempts:
        meta = attempt.get("metadata") or {}
        requested += int(meta.get("records_requested") or 0)
        accepted += int(meta.get("records_accepted") or 0)
        failure_type = attempt.get("failure_type")
        if failure_type:
            n_failed += 1
            snippet = ""
            if parsed_records is None and raw_content is not None:
                snippet = raw_content[:300]
            exemplars.append(
                FailureExemplar(
                    attempt_id=str(attempt.get("attempt_id")),
                    cycle=cycle,
                    kind="attempt_failure",
                    failure_type=str(failure_type),
                    text=snippet,
                )
            )

    if parsed_records is not None and accepted_texts is not None:
        anchor = next(
            (a for a in cycle_attempts if a.get("status") == "success"),
            cycle_attempts[0] if cycle_attempts else None,
        )
        anchor_id = (
            str(anchor.get("attempt_id")) if anchor else f"cycle_{cycle}"
        )
        for record in parsed_records:
            key = _normalize_text(str(record.get("text") or ""))
            if key not in accepted_texts:
                exemplars.append(
                    FailureExemplar(
                        attempt_id=anchor_id,
                        cycle=cycle,
                        kind="rejected_record",
                        failure_type=None,
                        text=str(record.get("text") or "")[:300],
                        label=str(record.get("label")),
                    )
                )

    return FailureSummary(
        cycle=cycle,
        n_attempts=len(cycle_attempts),
        records_requested=requested,
        records_accepted=accepted,
        n_failed_attempts=n_failed,
        n_fail=max(0, requested - accepted) + n_failed,
        exemplars=exemplars[:max_exemplars],
    )


@dataclass
class RunData:
    """Everything the audit needs from one completed run directory."""

    run_dir: Path
    config: ExperimentConfig
    label_names: Optional[Dict[int, str]]
    seed_rows: List[Dict[str, str]]
    augmented: Dict[int, List[Dict[str, str]]]
    gold_validation: List[Dict[str, str]]
    gold_test: List[Dict[str, str]]
    attempts: List[Dict[str, Any]]
    raw_responses: Dict[int, Dict[str, Any]]
    reconstructed: bool


def _load_attempts(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "teacher_attempts.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


_RESPONSE_RE = re.compile(r"teacher_response_cycle_(\d+)\.json$")


def _load_raw_responses(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    responses: Dict[int, Dict[str, Any]] = {}
    for path in run_dir.glob("teacher_response_cycle_*.json"):
        match = _RESPONSE_RE.search(path.name)
        if match:
            responses[int(match.group(1))] = json.loads(
                path.read_text(encoding="utf-8")
            )
    return responses


def _detect_label_names(dataset, config: ExperimentConfig):
    """index -> label-name mapping from ClassLabel features or label_text."""
    from datasets import ClassLabel

    for split in dataset.values():
        feature = split.features.get(config.label_field)
        if isinstance(feature, ClassLabel):
            return dict(enumerate(feature.names))
    mapping: Dict[int, str] = {}
    for split in dataset.values():
        if "label_text" not in split.column_names:
            continue
        for value, text in zip(split[config.label_field], split["label_text"]):
            if isinstance(value, int) and str(text):
                mapping.setdefault(value, str(text))
    return mapping or None


def _label_name(value: Any, label_names: Optional[Dict[int, str]]) -> str:
    if label_names is not None and not isinstance(value, str):
        try:
            return label_names[int(value)]
        except (KeyError, TypeError, ValueError):
            pass
    return str(value)


def _rows_from_split(
    split, config: ExperimentConfig, label_names
) -> List[Dict[str, str]]:
    return [
        {"text": str(text), "label": _label_name(label, label_names)}
        for text, label in zip(
            split[config.text_field], split[config.label_field]
        )
    ]


def load_run_data(
    run_dir: Union[str, Path],
    dataset_loader: Optional[Callable[[ExperimentConfig], Any]] = None,
    force_reconstruction: bool = False,
) -> RunData:
    """Load a completed run's audit inputs.

    Prefers the final ``dataset_cycle_*`` DatasetDict. When it is absent (or
    ``force_reconstruction`` is set) and raw ``teacher_response_cycle_*.json``
    files exist, reconstructs the augmented rows offline (Task 5). Raises
    ``FileNotFoundError`` naming what was found otherwise.
    """
    run_dir = Path(run_dir)
    config_path = run_dir / "experiment_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{run_dir} has no experiment_config.yaml; not a run directory"
        )
    config = ExperimentConfig.from_yaml(str(config_path))
    attempts = _load_attempts(run_dir)
    raw_responses = _load_raw_responses(run_dir)

    dataset_dirs = sorted(
        (p for p in run_dir.glob("dataset_cycle_*") if p.is_dir()),
        key=lambda p: int(p.name.rsplit("_", 1)[1]),
    )
    if dataset_dirs and not force_reconstruction:
        from datasets import load_from_disk

        dataset = load_from_disk(str(dataset_dirs[-1]))
        label_names = _detect_label_names(dataset, config)
        train = dataset["train"]
        columns = set(train.column_names)
        source_split = (
            train["source_split"]
            if "source_split" in columns
            else ["train"] * len(train)
        )
        origin_cycle = (
            train["origin_cycle"]
            if "origin_cycle" in columns
            else [0] * len(train)
        )
        seed_rows: List[Dict[str, str]] = []
        augmented: Dict[int, List[Dict[str, str]]] = {}
        for text, label, source, cycle in zip(
            train[config.text_field],
            train[config.label_field],
            source_split,
            origin_cycle,
        ):
            row = {"text": str(text), "label": _label_name(label, label_names)}
            if source == "augmented":
                augmented.setdefault(int(cycle), []).append(row)
            else:
                seed_rows.append(row)
        reconstructed = False
    elif raw_responses:
        dataset, label_names, seed_rows, augmented = _reconstruct(
            config, attempts, raw_responses, dataset_loader
        )
        reconstructed = True
    else:
        raise FileNotFoundError(
            f"{run_dir} has neither dataset_cycle_* directories nor "
            "teacher_response_cycle_*.json files; nothing to audit"
        )

    gold_validation = (
        _rows_from_split(dataset["validation"], config, label_names)
        if "validation" in dataset
        else []
    )
    gold_test = (
        _rows_from_split(dataset["test"], config, label_names)
        if "test" in dataset
        else []
    )
    return RunData(
        run_dir=run_dir,
        config=config,
        label_names=label_names,
        seed_rows=seed_rows,
        augmented=augmented,
        gold_validation=gold_validation,
        gold_test=gold_test,
        attempts=attempts,
        raw_responses=raw_responses,
        reconstructed=reconstructed,
    )


class ReconstructionError(RuntimeError):
    """Reconstructed rows disagree with the run's own attempts ledger."""


def _load_seed_dataset(config: ExperimentConfig):
    """Replay the engine's dataset init exactly as DistillationEngine.__init__.

    Deterministic given the run's own experiment_config.yaml: load_dataset ->
    ensure_class_label (only when sampling is enabled) -> prepare_dataset ->
    ensure_validation_split -> ensure_origin_columns.
    """
    from datasets import load_dataset

    from .engine import (
        ensure_class_label,
        ensure_origin_columns,
        ensure_validation_split,
        prepare_dataset,
    )
    from .reproducibility import dataset_load_kwargs

    kwargs = dataset_load_kwargs(config)
    subset = config.dataset_subset
    if subset:
        dataset = load_dataset(config.dataset, subset, **kwargs)
    else:
        dataset = load_dataset(config.dataset, **kwargs)
    if config.sampling.enabled:
        dataset = ensure_class_label(dataset, config.sampling.stratify_column)
    dataset = prepare_dataset(dataset, config.sampling)
    dataset = ensure_validation_split(dataset, config)
    return ensure_origin_columns(dataset)


def _reconstruct_augmented(
    config: ExperimentConfig,
    attempts: Sequence[Dict[str, Any]],
    raw_responses: Dict[int, Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """Rebuild per-cycle accepted records by re-parsing raw teacher responses.

    Engine acceptance is a prefix truncation of the parsed records (batch-size
    cap, then synthetic-record-budget cap), plus an empty-prompt/response drop
    for SFT runs — all replayable offline. Self-consistency screening is not,
    so screened runs hard-fail here.
    """
    from .trainers.factory import SFT_STUDENT_TYPES

    trainer_config = getattr(config, "trainer_config", None)
    if isinstance(trainer_config, dict) and (
        trainer_config.get("augmentation_screening") or {}
    ).get("enabled"):
        raise ReconstructionError(
            "run enabled self-consistency screening "
            "(trainer_config.augmentation_screening.enabled), which cannot "
            "be replayed offline; audit this run from its dataset_cycle_* "
            "directories instead"
        )
    if any("screening_mode" in (a.get("metadata") or {}) for a in attempts):
        raise ReconstructionError(
            "teacher_attempts.jsonl carries screening_mode metadata: run "
            "used self-consistency screening, which cannot be replayed "
            "offline; audit this run from its dataset_cycle_* directories "
            "instead"
        )

    expected_by_cycle: Dict[int, int] = {}
    for attempt in attempts:
        cycle = attempt.get("cycle")
        if cycle is None:
            continue
        meta = attempt.get("metadata") or {}
        expected_by_cycle[int(cycle)] = expected_by_cycle.get(
            int(cycle), 0
        ) + int(meta.get("records_accepted") or 0)
    missing = sorted(
        cycle
        for cycle, expected in expected_by_cycle.items()
        if expected > 0 and cycle not in raw_responses
    )
    if missing:
        names = ", ".join(
            f"teacher_response_cycle_{cycle}.json" for cycle in missing
        )
        raise ReconstructionError(
            f"cycles {missing}: teacher_attempts.jsonl records accepted "
            f"rows but the raw response file is missing ({names}); "
            "refusing to silently drop cycles from the reconstruction"
        )

    is_sft = config.student_type in SFT_STUDENT_TYPES
    augmented: Dict[int, List[Dict[str, Any]]] = {}
    for cycle in sorted(raw_responses):
        cycle_metas = [
            a.get("metadata") or {}
            for a in attempts
            if a.get("cycle") == cycle
        ]
        expected = expected_by_cycle.get(cycle, 0)
        if any(
            int(meta.get("records_rejected") or 0)
            or int(meta.get("records_failed") or 0)
            for meta in cycle_metas
        ):
            raise ReconstructionError(
                f"cycle {cycle}: run used self-consistency screening, which "
                "cannot be replayed offline; audit this run from its "
                "dataset_cycle_* directories instead"
            )
        content = raw_responses[cycle]["choices"][0]["message"]["content"]
        try:
            records = parse_teacher_records(
                config, str(content), with_teacher_response=is_sft
            )
        except Exception as exc:
            if expected == 0:
                logger.info(
                    "cycle %s: unparseable raw response and zero accepted "
                    "records; skipping",
                    cycle,
                )
                continue
            raise ReconstructionError(
                f"cycle {cycle}: raw teacher response failed to parse but "
                f"teacher_attempts.jsonl records {expected} accepted rows"
            ) from exc
        if is_sft:
            # Mirror _build_augmented_sft_rows' drop rule exactly
            # (engine.py:1397-1400): drop when student_prompt or
            # teacher_response is empty — not when the trained-on label
            # (gold_answer or teacher_response) is empty. A record can carry
            # a non-empty gold_answer alongside a whitespace-only
            # teacher_response; the engine drops it, so reconstruction must
            # too. Strip the helper key so returned rows keep the uniform
            # {"text", "label"} shape.
            records = [
                {"text": r["text"], "label": r["label"]}
                for r in records
                if str(r["text"] or "").strip()
                and str(r.get("_teacher_response") or "").strip()
            ]
        if len(records) < expected:
            raise ReconstructionError(
                f"cycle {cycle}: reconstructed only {len(records)} rows but "
                f"teacher_attempts.jsonl records {expected} accepted; "
                "refusing to audit inconsistent data"
            )
        if expected:
            augmented[cycle] = records[:expected]
    return augmented


def _reconstruct(config, attempts, raw_responses, dataset_loader):
    loader = dataset_loader or _load_seed_dataset
    dataset = loader(config)
    label_names = _detect_label_names(dataset, config)
    seed_rows = _rows_from_split(dataset["train"], config, label_names)
    augmented = {
        cycle: [
            {
                "text": str(record["text"]),
                "label": _label_name(record["label"], label_names),
            }
            for record in records
        ]
        for cycle, records in _reconstruct_augmented(
            config, attempts, raw_responses
        ).items()
    }
    return dataset, label_names, seed_rows, augmented


def verifier_agreement(
    predict_fn: Callable[[List[str]], List[str]],
    rows: Sequence[Dict[str, str]],
) -> float:
    """Fraction of rows where the verifier's label matches the row's label.

    ``predict_fn`` maps texts to label-name strings; comparison happens in
    the normalized label space (fidelity.normalize_label).
    """
    if not rows:
        raise ValueError("verifier_agreement requires at least one row")
    texts = [row["text"] for row in rows]
    predictions = predict_fn(texts)
    if len(predictions) != len(rows):
        raise ValueError(
            "predict_fn returned "
            f"{len(predictions)} predictions for {len(rows)} rows"
        )
    matches = sum(
        1
        for prediction, row in zip(predictions, rows)
        if normalize_label(prediction) == normalize_label(row["label"])
    )
    return matches / len(rows)


def make_hf_predict_fn(
    model_dir: str,
    label_names: Union[Dict[int, str], List[str], None] = None,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> Callable[[List[str]], List[str]]:
    """Load a gold-only HF classifier checkpoint as a predict_fn.

    ``label_names`` maps class index -> label name. When omitted, the
    checkpoint's own id2label is used — unless it is the meaningless
    LABEL_0/LABEL_1 default, which raises so agreement is never computed in
    the wrong label space.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    if label_names is not None:
        id2label = (
            dict(enumerate(label_names))
            if isinstance(label_names, list)
            else {int(k): str(v) for k, v in label_names.items()}
        )
    else:
        configured = model.config.id2label or {}
        id2label = {int(k): str(v) for k, v in configured.items()}
        if not id2label or all(
            name.startswith("LABEL_") for name in id2label.values()
        ):
            raise ValueError(
                f"checkpoint {model_dir} has no meaningful id2label mapping; "
                "pass label_names explicitly"
            )

    def predict(texts: List[str]) -> List[str]:
        predictions: List[str] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(**encoded).logits
            predictions.extend(
                id2label[int(index)] for index in logits.argmax(dim=-1)
            )
        return predictions

    return predict


PROBE_PROMPT_TEMPLATE = """You are labeling examples for a text-classification task.
Reply with exactly one label from this list and nothing else:
{labels}

Text: {text}
Label:"""

_PROVIDER_KEYS = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def require_probe_credentials(teacher: str) -> None:
    """Refuse to probe without the teacher provider's API key in the env."""
    provider = str(teacher).split("/", 1)[0].lower()
    key_name = _PROVIDER_KEYS.get(provider)
    if key_name is not None:
        if not os.environ.get(key_name):
            raise RuntimeError(
                f"the probe calls teacher {teacher!r} and requires "
                f"{key_name} in the environment"
            )
        return
    if not any(
        name.endswith("_API_KEY") and value
        for name, value in os.environ.items()
    ):
        raise RuntimeError(
            f"the probe calls teacher {teacher!r} and requires a provider "
            "API key (…_API_KEY) in the environment"
        )


@dataclass
class ProbeResult:
    """Teacher-on-gold probe outcome (one number per run)."""

    k: int
    agreement: float
    teacher_model: str
    rows: List[Dict[str, str]]
    usage: Dict[str, Any]


def teacher_gold_probe(
    config: ExperimentConfig,
    gold_rows: Sequence[Dict[str, str]],
    k: int = 100,
    seed: int = 13,
    tracker: Optional[TokenTracker] = None,
) -> ProbeResult:
    """Label K seeded gold rows with the run's teacher at temperature 0.

    Reports agreement with the gold labels in the normalized label space.
    Callers pass a TokenTracker with an open cycle so usage lands in the
    audit ledger; the caller owns preflight (require_probe_credentials).
    """
    import litellm

    if not gold_rows:
        raise ValueError("teacher_gold_probe requires at least one gold row")
    rng = random.Random(seed)
    rows = list(gold_rows)
    sample = rows if len(rows) <= k else rng.sample(rows, k)
    label_space = sorted({row["label"] for row in gold_rows})
    labels_block = "\n".join(f"- {name}" for name in label_space)

    estimated_input = sum(
        (len(PROBE_PROMPT_TEMPLATE) + len(labels_block) + len(row["text"])) // 4
        for row in sample
    )
    logger.info(
        "probe preflight: %d calls to %s, ~%d input tokens",
        len(sample),
        config.teacher,
        estimated_input,
    )

    results: List[Dict[str, str]] = []
    matches = 0
    for row in sample:
        response = litellm.completion(
            model=config.teacher,
            messages=[
                {
                    "role": "user",
                    "content": PROBE_PROMPT_TEMPLATE.format(
                        labels=labels_block, text=row["text"]
                    ),
                }
            ],
            temperature=0.0,
        )
        if tracker is not None:
            tracker.record_usage(response, OperationType.AUDIT_PROBE)
        raw = str(response["choices"][0]["message"]["content"])
        predicted = normalize_label(raw)
        gold = normalize_label(row["label"])
        if predicted == gold:
            matches += 1
        results.append(
            {
                "text": row["text"],
                "gold_label": gold,
                "predicted_label": predicted,
                "raw_response": raw,
            }
        )
    usage = (
        tracker.current_cycle_usage().model_dump() if tracker is not None else {}
    )
    return ProbeResult(
        k=len(sample),
        agreement=matches / len(sample),
        teacher_model=str(config.teacher),
        rows=results,
        usage=usage,
    )


@dataclass
class CycleAudit:
    """One tab:audit row (a cycle, or the 'all' cumulative row)."""

    cycle: Union[int, str]
    n_rows: int
    dup_pct: float
    ndup_pct: float
    distinct_1: Optional[float]
    distinct_2: Optional[float]
    mean_token_len: Optional[float]
    label_drift: Optional[float]
    lblc_verifier: Optional[float]
    n_fail: int
    failures: FailureSummary


@dataclass
class RunAuditResult:
    """Full audit outcome for one run directory."""

    run_dir: str
    experiment: str
    dataset: str
    reconstructed: bool
    cycles: List[CycleAudit]
    cumulative: CycleAudit
    verifier_model: Optional[str]
    verifier_test_accuracy: Optional[float]
    probe: Optional[ProbeResult]


def _cycle_metrics(
    cycle: Union[int, str],
    rows: List[Dict[str, str]],
    reference_texts: List[str],
    seed_labels: List[str],
    failures: FailureSummary,
    predict_fn: Optional[Callable[[List[str]], List[str]]],
) -> CycleAudit:
    texts = [row["text"] for row in rows]
    if texts:
        exact = exact_duplicate_flags(texts, reference_texts)
        near = near_duplicate_flags(texts, reference_texts)
        near_only = [n and not e for n, e in zip(near, exact)]
        dup_pct = sum(exact) / len(texts)
        ndup_pct = sum(near_only) / len(texts)
        try:
            d1 = distinct_n(texts, 1)
            d2 = distinct_n(texts, 2)
        except ValueError:
            d1 = d2 = None
        mean_len = sum(len(t.split()) for t in texts) / len(texts)
        drift = label_drift([row["label"] for row in rows], seed_labels)
        lblc = verifier_agreement(predict_fn, rows) if predict_fn else None
    else:
        dup_pct = ndup_pct = 0.0
        d1 = d2 = mean_len = drift = lblc = None
    return CycleAudit(
        cycle=cycle,
        n_rows=len(rows),
        dup_pct=dup_pct,
        ndup_pct=ndup_pct,
        distinct_1=d1,
        distinct_2=d2,
        mean_token_len=mean_len,
        label_drift=drift,
        lblc_verifier=lblc,
        n_fail=failures.n_fail,
        failures=failures,
    )


def audit_run(
    run_dir: Union[str, Path],
    *,
    verifier_model: Optional[str] = None,
    predict_fn: Optional[Callable[[List[str]], List[str]]] = None,
    probe: bool = False,
    probe_k: int = 100,
    probe_split: str = "validation",
    seed: int = 13,
    output_dir: Optional[str] = None,
    dataset_loader: Optional[Callable[[ExperimentConfig], Any]] = None,
    force_reconstruction: bool = False,
) -> RunAuditResult:
    """Audit one completed run directory and write audit.json / audit.csv."""
    if probe_split not in {"validation", "test"}:
        raise ValueError(
            f"probe_split must be 'validation' or 'test', got {probe_split!r}"
        )
    run_dir = Path(run_dir)
    data = load_run_data(
        run_dir,
        dataset_loader=dataset_loader,
        force_reconstruction=force_reconstruction,
    )
    config = data.config

    if probe:
        require_probe_credentials(str(config.teacher))
    if predict_fn is None and verifier_model is not None:
        predict_fn = make_hf_predict_fn(
            verifier_model, label_names=data.label_names
        )

    seed_texts = [row["text"] for row in data.seed_rows]
    seed_labels = [row["label"] for row in data.seed_rows]
    if not seed_texts:
        raise ValueError(
            f"{run_dir}: no seed rows found; cannot build a duplicate "
            "reference corpus"
        )

    cycle_ids = sorted(
        set(data.augmented)
        | {
            int(a["cycle"])
            for a in data.attempts
            if a.get("cycle") is not None
        }
    )
    cycles: List[CycleAudit] = []
    reference = list(seed_texts)
    for cycle in cycle_ids:
        rows = data.augmented.get(cycle, [])
        failures = failure_summary(
            data.attempts,
            cycle,
            raw_responses=data.raw_responses,
            accepted_texts={_normalize_text(row["text"]) for row in rows},
            config=config,
        )
        cycles.append(
            _cycle_metrics(cycle, rows, reference, seed_labels, failures, predict_fn)
        )
        reference.extend(row["text"] for row in rows)

    all_rows = [
        row for cycle in sorted(data.augmented) for row in data.augmented[cycle]
    ]
    total_failures = FailureSummary(
        cycle=-1,
        n_attempts=sum(c.failures.n_attempts for c in cycles),
        records_requested=sum(c.failures.records_requested for c in cycles),
        records_accepted=sum(c.failures.records_accepted for c in cycles),
        n_failed_attempts=sum(c.failures.n_failed_attempts for c in cycles),
        n_fail=sum(c.failures.n_fail for c in cycles),
        exemplars=[e for c in cycles for e in c.failures.exemplars][:10],
    )
    cumulative = _cycle_metrics(
        "all", all_rows, seed_texts, seed_labels, total_failures, predict_fn
    )

    verifier_test_accuracy = None
    if predict_fn is not None and data.gold_test:
        verifier_test_accuracy = verifier_agreement(predict_fn, data.gold_test)

    out_dir = (
        Path(output_dir) / run_dir.name if output_dir else run_dir / "audit"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_result: Optional[ProbeResult] = None
    if probe:
        gold_rows = (
            data.gold_validation
            if probe_split == "validation"
            else data.gold_test
        )
        if not gold_rows:
            raise ValueError(
                f"{run_dir}: no gold rows in the {probe_split!r} split for "
                "the probe"
            )
        tracker = TokenTracker(
            experiment_name=f"audit_{config.name}",
            teacher_model=str(config.teacher),
            quiet=True,
        )
        # Tokens are spent call-by-call inside teacher_gold_probe; if it
        # raises mid-loop (e.g. a litellm API error), the calls already made
        # must still land in the ledger instead of vanishing with the
        # exception.
        try:
            with tracker.cycle(0):
                probe_result = teacher_gold_probe(
                    config, gold_rows, k=probe_k, seed=seed, tracker=tracker
                )
        finally:
            tracker.save(out_dir, "audit_usage.json")

    result = RunAuditResult(
        run_dir=str(run_dir),
        experiment=config.name,
        dataset=str(config.dataset),
        reconstructed=data.reconstructed,
        cycles=cycles,
        cumulative=cumulative,
        verifier_model=str(verifier_model) if verifier_model else None,
        verifier_test_accuracy=verifier_test_accuracy,
        probe=probe_result,
    )
    _write_outputs(result, out_dir)
    return result


CSV_FIELDS = [
    "cycle",
    "n_rows",
    "dup_pct",
    "ndup_pct",
    "lblc_verifier",
    "lblc_teacher",
    "distinct_1",
    "distinct_2",
    "mean_token_len",
    "label_drift",
    "n_fail",
    "probe_total_tokens",
    "probe_cost_usd",
]


def _fmt(value: Optional[float]):
    return "" if value is None else round(value, 4)


def csv_rows(result: RunAuditResult) -> List[Dict[str, Any]]:
    """audit.csv rows: one per cycle plus the cumulative 'all' row.

    Lbl-c(t) is one number per run; it appears on the 'all' row only.
    """
    rows: List[Dict[str, Any]] = []
    for audit in [*result.cycles, result.cumulative]:
        is_all = audit.cycle == "all"
        probe_usage = result.probe.usage if result.probe is not None else {}
        rows.append(
            {
                "cycle": audit.cycle,
                "n_rows": audit.n_rows,
                "dup_pct": _fmt(audit.dup_pct),
                "ndup_pct": _fmt(audit.ndup_pct),
                "lblc_verifier": _fmt(audit.lblc_verifier),
                "lblc_teacher": (
                    _fmt(result.probe.agreement)
                    if result.probe is not None and is_all
                    else ""
                ),
                "distinct_1": _fmt(audit.distinct_1),
                "distinct_2": _fmt(audit.distinct_2),
                "mean_token_len": _fmt(audit.mean_token_len),
                "label_drift": _fmt(audit.label_drift),
                "n_fail": audit.n_fail,
                "probe_total_tokens": (
                    probe_usage.get("total_tokens", "") if is_all else ""
                ),
                "probe_cost_usd": (
                    _fmt(probe_usage.get("estimated_cost")) if is_all else ""
                ),
            }
        )
    return rows


def _write_outputs(result: RunAuditResult, out_dir: Path) -> None:
    (out_dir / "audit.json").write_text(
        json.dumps(dataclasses.asdict(result), indent=2, default=str),
        encoding="utf-8",
    )
    with (out_dir / "audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows(result))
    logger.info("audit outputs written to %s", out_dir)


def _pct(value: Optional[float]) -> str:
    return f"{value * 100:.1f}" if value is not None else "--"


def _num(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else "--"


def latex_rows(result: RunAuditResult) -> str:
    """tab:audit-ready rows: Data & Cyc & Dup% & NDup% & Lbl-c(v) & Lbl-c(t) & Div. & #Fail.

    Lbl-c(t) is per-run and appears on the first row of the dataset block.
    """
    name = str(result.dataset).split("/")[-1]
    lines = []
    for i, cycle in enumerate(result.cycles):
        data_cell = name if i == 0 else ""
        lblc_t = (
            _pct(result.probe.agreement)
            if result.probe is not None and i == 0
            else "--"
        )
        lines.append(
            f"{data_cell} & {cycle.cycle} & {_pct(cycle.dup_pct)} & "
            f"{_pct(cycle.ndup_pct)} & {_pct(cycle.lblc_verifier)} & {lblc_t} & "
            f"{_num(cycle.distinct_2)} & {cycle.n_fail} \\\\"
        )
    return "\n".join(lines)


def print_audit_table(result: RunAuditResult, console) -> None:
    """Rich stdout table mirroring tab:audit, plus the cumulative row."""
    from rich.table import Table

    title = f"Audit — {result.experiment}"
    if result.reconstructed:
        title += " (reconstructed from raw teacher responses)"
    table = Table(title=title)
    for column in (
        "Cyc", "Rows", "Dup%", "NDup%", "Lbl-c(v)", "Lbl-c(t)", "Div.", "#Fail",
    ):
        table.add_column(column, justify="right")
    entries = [*result.cycles, result.cumulative]
    for i, cycle in enumerate(entries):
        lblc_t = (
            _pct(result.probe.agreement)
            if result.probe is not None and i == 0
            else "--"
        )
        table.add_row(
            str(cycle.cycle),
            str(cycle.n_rows),
            _pct(cycle.dup_pct),
            _pct(cycle.ndup_pct),
            _pct(cycle.lblc_verifier),
            lblc_t,
            _num(cycle.distinct_2),
            str(cycle.n_fail),
        )
    console.print(table)
    if result.verifier_test_accuracy is not None:
        console.print(
            f"verifier gold-test accuracy: {_pct(result.verifier_test_accuracy)}%"
        )
