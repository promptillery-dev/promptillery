"""Materialize audited SFT JSONL records from a dataset split."""

from __future__ import annotations

import json
import logging
import re
from contextlib import nullcontext
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import DatasetDict, load_dataset
from jinja2 import StrictUndefined
from litellm import acompletion
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .config import ExperimentConfig
from .engine import ensure_class_label, ensure_validation_split, prepare_dataset
from .reproducibility import build_reproducibility_manifest, dataset_load_kwargs
from .utils import create_prompt_environment

logger = logging.getLogger(__name__)

try:
    from litellm import token_counter as _token_counter

    _HAS_TOKEN_COUNTER = True
except ImportError:
    _token_counter = None  # type: ignore[assignment]
    _HAS_TOKEN_COUNTER = False


DEFAULT_STUDENT_PROMPT_TEMPLATE = "{{ text }}"
DEFAULT_TEACHER_PROMPT_TEMPLATE = """\
Answer the student prompt below. Return only the target response.

Student prompt:
{{ student_prompt }}
"""
SOURCE_INDEX_LIST_LIMIT = 5000


def _materialization_progress(enabled: bool) -> Progress | None:
    """Return a compact progress bar for CLI materialization runs."""
    if not enabled:
        return None
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("att={task.fields[attempted]}"),
        TextColumn("rej={task.fields[rejected]}"),
        TextColumn("tok={task.fields[tokens]}"),
        TextColumn("{task.fields[status]}"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=Console(stderr=True),
    )


def _stable_hash(value: Any) -> str:
    """Return a stable short hash for configs and templates."""
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return sha256(payload).hexdigest()[:16]


def load_materialization_dataset(config: ExperimentConfig) -> DatasetDict:
    """Load and prepare a dataset for SFT materialization."""
    dataset_kwargs = dataset_load_kwargs(config)
    if config.dataset_subset:
        dataset = load_dataset(config.dataset, config.dataset_subset, **dataset_kwargs)
    else:
        dataset = load_dataset(config.dataset, **dataset_kwargs)

    if config.sampling.enabled:
        dataset = ensure_class_label(dataset, config.sampling.stratify_column)

    dataset = prepare_dataset(dataset, config.sampling)
    return ensure_validation_split(dataset, config)


def _label_text(
    row: Dict[str, Any], config: ExperimentConfig, dataset, field: str | None = None
) -> Tuple[str, str]:
    """Return a string label/gold answer and the field it came from."""
    label_field = field or config.label_field
    if label_field not in row:
        available = ", ".join(sorted(row.keys()))
        raise ValueError(
            f"Gold answer field '{label_field}' not found. Available fields: {available}"
        )

    label = row[label_field]
    feature = dataset.features.get(label_field)
    if hasattr(feature, "int2str") and isinstance(label, int):
        try:
            return str(feature.int2str(label)), label_field
        except ValueError:
            return str(label), label_field
    return str(label), label_field


def _normalize_canonical_label(value: Any) -> str:
    """Normalize a class label the same way SFT canonical-label evaluation does."""
    text = str(value).strip().lower()
    text = " ".join(text.split()).strip(" .,:;")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _class_label_names(dataset, label_field: str) -> List[str]:
    """Return canonical ClassLabel names for a dataset field when available."""
    feature = dataset.features.get(label_field)
    names = getattr(feature, "names", None)
    if names:
        return [str(name) for name in names]
    num_classes = getattr(feature, "num_classes", None)
    int2str = getattr(feature, "int2str", None)
    if num_classes is not None and int2str is not None:
        labels = []
        for index in range(int(num_classes)):
            try:
                labels.append(str(int2str(index)))
            except ValueError:
                return []
        return labels
    return []


def _field_label_values(dataset, label_field: str) -> List[str]:
    """Return sorted string labels observed in a dataset field."""
    if label_field not in dataset.column_names:
        return []
    values = dataset.unique(label_field)
    return sorted(str(value) for value in values if value is not None)


def _write_canonical_labels_artifact(
    *,
    config: ExperimentConfig,
    output_path: Path,
    source,
    label_field: str,
    canonical_labels: List[str] | None = None,
    canonical_labels_field: str | None = None,
) -> str | None:
    """Write the canonical label schema next to materialized SFT JSONL files."""
    label_source = "config.trainer_config.materialize_sft.canonical_labels"
    if not canonical_labels:
        canonical_labels = _class_label_names(source, label_field)
        label_source = "datasets.ClassLabel.names"
    if not canonical_labels and canonical_labels_field:
        canonical_labels = _field_label_values(source, canonical_labels_field)
        label_source = f"dataset field {canonical_labels_field}"
    if not canonical_labels:
        return None

    normalized_labels = [
        _normalize_canonical_label(label) for label in canonical_labels
    ]
    artifact_path = output_path.parent / "canonical_labels.json"
    temp_path = artifact_path.with_name(f".{artifact_path.name}.tmp")
    payload = {
        "schema_version": 1,
        "dataset": config.dataset,
        "dataset_subset": config.dataset_subset,
        "label_field": label_field,
        "source": label_source,
        "normalization": "canonical_label",
        "canonical_label_count": len(canonical_labels),
        "canonical_labels": canonical_labels,
        "normalized_canonical_labels": normalized_labels,
    }
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    temp_path.replace(artifact_path)
    return str(artifact_path)


def _configured_canonical_labels(config: ExperimentConfig) -> List[str]:
    """Return explicit canonical labels configured for materialization."""
    materialize_config = config.trainer_config.get("materialize_sft", {})
    labels = materialize_config.get("canonical_labels") or config.trainer_config.get(
        "canonical_labels"
    )
    if not labels:
        return []
    return [str(label) for label in labels]


def _temporary_source_index_column(source) -> str:
    """Return a temporary source-index column name absent from the dataset."""
    base_name = "__promptillery_source_index"
    column = base_name
    suffix = 1
    while column in source.column_names:
        suffix += 1
        column = f"{base_name}_{suffix}"
    return column


def _drop_temporary_source_index(source, column: str):
    """Remove the temporary source-index column from a selected dataset."""
    if column in source.column_names:
        return source.remove_columns(column)
    return source


def _selection_metadata(
    *,
    strategy: str,
    seed: int,
    source_records: int,
    selected_indices: List[int],
) -> Dict[str, Any]:
    """Build an auditable manifest block for source-example selection."""
    encoded_indices = json.dumps(selected_indices, separators=(",", ":")).encode(
        "utf-8"
    )
    metadata: Dict[str, Any] = {
        "selection_strategy": strategy,
        "selection_seed": seed,
        "source_records_before_selection": source_records,
        "selected_source_indices_count": len(selected_indices),
        "selected_source_indices_sha256": sha256(encoded_indices).hexdigest(),
        "selected_source_indices_preview": selected_indices[:20],
        "selected_source_indices": selected_indices,
    }
    return metadata


def _select_source_examples(
    source,
    *,
    max_samples: int | None,
    stratify_by: str | None = None,
    seed: int = 0,
    canonical_labels: List[str] | None = None,
    selection_strategy: str = "prefix",
    return_metadata: bool = False,
):
    """Select materialization rows, optionally preserving label coverage."""
    strategy = str(selection_strategy or "prefix").strip().lower()
    if stratify_by:
        if strategy not in {"prefix", "stratified"}:
            raise ValueError(
                "selection_strategy must be 'stratified' when "
                "stratify_max_samples=true"
            )
        strategy = "stratified"
    elif strategy not in {"prefix", "seeded_sample"}:
        raise ValueError(
            "selection_strategy must be one of: prefix, seeded_sample"
        )

    if max_samples is None or max_samples >= len(source):
        selected_indices = list(range(len(source)))
        metadata = _selection_metadata(
            strategy="all",
            seed=seed,
            source_records=len(source),
            selected_indices=selected_indices,
        )
        return (source, metadata) if return_metadata else source

    if not stratify_by:
        if strategy == "seeded_sample":
            index_column = _temporary_source_index_column(source)
            indexed_source = source.add_column(index_column, list(range(len(source))))
            selected = indexed_source.shuffle(seed=seed).select(range(max_samples))
            selected_indices = [int(index) for index in selected[index_column]]
            selected = _drop_temporary_source_index(selected, index_column)
        else:
            selected_indices = list(range(max_samples))
            selected = source.select(selected_indices)
        metadata = _selection_metadata(
            strategy=strategy,
            seed=seed,
            source_records=len(source),
            selected_indices=selected_indices,
        )
        return (selected, metadata) if return_metadata else selected

    if stratify_by not in source.column_names:
        available = ", ".join(sorted(source.column_names))
        raise ValueError(
            f"Cannot stratify max_samples by '{stratify_by}'. "
            f"Available columns: {available}"
        )

    labels = list(canonical_labels or _class_label_names(source, stratify_by))
    if not labels:
        raise ValueError(
            f"Cannot stratify max_samples by '{stratify_by}' because it is not "
            "a ClassLabel-style field and no canonical_labels were configured"
        )
    if max_samples < len(labels):
        raise ValueError(
            f"max_samples={max_samples} is too small to cover "
            f"{len(labels)} labels when stratify_max_samples=true"
        )
    if canonical_labels:
        canonical_set = {_normalize_canonical_label(label) for label in labels}
        buckets: Dict[str, List[int]] = {label: [] for label in canonical_set}
        for index, row in enumerate(source):
            label = _normalize_canonical_label(row[stratify_by])
            if label in buckets:
                buckets[label].append(index)
        missing = sorted(label for label, indexes in buckets.items() if not indexes)
        if missing:
            raise ValueError(
                f"Cannot cover canonical labels from '{stratify_by}': missing={missing}"
            )

        selected: list[int] = []
        for label in sorted(buckets):
            selected.append(buckets[label][0])
        for index in range(len(source)):
            if len(selected) >= max_samples:
                break
            if index not in selected:
                selected.append(index)
        selected_indices = selected[:max_samples]
        selected_source = source.select(selected_indices)
        metadata = _selection_metadata(
            strategy=strategy,
            seed=seed,
            source_records=len(source),
            selected_indices=selected_indices,
        )
        return (selected_source, metadata) if return_metadata else selected_source

    index_column = _temporary_source_index_column(source)
    indexed_source = source.add_column(index_column, list(range(len(source))))
    selected_source = indexed_source.train_test_split(
        train_size=max_samples,
        stratify_by_column=stratify_by,
        seed=seed,
    )["train"]
    selected_indices = [int(index) for index in selected_source[index_column]]
    selected_source = _drop_temporary_source_index(selected_source, index_column)
    metadata = _selection_metadata(
        strategy=strategy,
        seed=seed,
        source_records=len(source),
        selected_indices=selected_indices,
    )
    return (selected_source, metadata) if return_metadata else selected_source


def _format_template(template: str, values: Dict[str, Any]) -> str:
    """Format a user template and fail with a useful missing-key error."""
    try:
        env = create_prompt_environment()
        env.undefined = StrictUndefined
        return env.from_string(template).render(**values)
    except Exception as exc:
        available = ", ".join(sorted(values.keys()))
        raise ValueError(
            f"Failed to render template: {exc}. Available fields: {available}"
        ) from exc


def _response_text(response: Any) -> str:
    """Extract assistant text from a LiteLLM response."""
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return ""

    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message", {})
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return str(content or "").strip()


def _usage_from_response(response: Any) -> Dict[str, int]:
    """Extract token usage from a LiteLLM response."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if usage is None:
        usage = {}

    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    provider_total_tokens = int(usage.get("total_tokens", 0) or 0)
    if provider_total_tokens and provider_total_tokens < input_tokens + output_tokens:
        raise ValueError(
            "Provider total_tokens is smaller than prompt_tokens + completion_tokens"
        )
    total_tokens = provider_total_tokens or (input_tokens + output_tokens)
    return {
        "teacher_input_tokens": input_tokens,
        "teacher_output_tokens": output_tokens,
        "teacher_total_tokens": total_tokens,
        "provider_total_tokens": provider_total_tokens,
    }


def _estimate_input_tokens(model: str, messages: List[Dict[str, str]]) -> int:
    """Estimate prompt tokens for preflight masking."""
    if _HAS_TOKEN_COUNTER and _token_counter is not None:
        try:
            return int(_token_counter(model=model, messages=messages))
        except Exception as exc:
            logger.debug("Token counter failed, falling back to char estimate: %s", exc)
    chars = sum(len(message.get("content", "")) for message in messages)
    return max(1, chars // 3)


async def materialize_sft_records(
    *,
    config: ExperimentConfig,
    output_path: Path,
    split: str = "train",
    mode: str = "gold",
    max_samples: int | None = None,
    student_prompt_template: str | None = None,
    teacher_prompt_template: str | None = None,
    prompt_operator: str = "coverage",
    teacher_tier: str = "cheap",
    overwrite: bool = False,
    allow_estimated_usage: bool = False,
    allow_partial: bool = False,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """Write SFT JSONL records for one dataset split."""
    if isinstance(config.teacher, list):
        raise ValueError("materialize-sft requires a single teacher value")
    if isinstance(config.dataset, list):
        raise ValueError("materialize-sft requires a single dataset value")
    if isinstance(config.seed, list):
        raise ValueError("materialize-sft requires a single seed value")
    if isinstance(config.token_budget, list):
        raise ValueError("materialize-sft requires a single token_budget value")
    if config.paper_mode and allow_estimated_usage:
        raise ValueError("paper_mode=true rejects allow_estimated_usage")
    if mode not in {"gold", "teacher"}:
        raise ValueError("mode must be 'gold' or 'teacher'")
    if mode == "teacher" and config.teacher_max_output_tokens is None:
        raise ValueError(
            "teacher mode requires teacher_max_output_tokens for preflight accounting"
        )
    manifest_path = output_path.with_name(f"{output_path.name}.manifest.json")
    if (output_path.exists() or manifest_path.exists()) and not overwrite:
        raise ValueError(f"Output path already exists: {output_path}")

    materialize_config = config.trainer_config.get("materialize_sft", {})
    student_prompt_template = (
        student_prompt_template
        or materialize_config.get("student_prompt_template")
        or DEFAULT_STUDENT_PROMPT_TEMPLATE
    )
    teacher_prompt_template = (
        teacher_prompt_template
        or materialize_config.get("prompt_template")
        or config.prompt
        or DEFAULT_TEACHER_PROMPT_TEMPLATE
    )
    gold_answer_field = materialize_config.get("gold_answer_field")
    stratify_field = materialize_config.get("stratify_field") or (
        gold_answer_field or config.label_field
    )
    canonical_labels_field = materialize_config.get("canonical_labels_field")
    configured_canonical_labels = _configured_canonical_labels(config)
    stratify_max_samples = bool(materialize_config.get("stratify_max_samples", False))
    rejection_buffer_samples = int(
        materialize_config.get("rejection_buffer_samples", 0) or 0
    )
    configured_selection_strategy = materialize_config.get("selection_strategy")
    if configured_selection_strategy is None:
        selection_strategy = "stratified" if stratify_max_samples else "prefix"
    else:
        selection_strategy = str(configured_selection_strategy).strip().lower()
    if stratify_max_samples and selection_strategy != "stratified":
        raise ValueError(
            "materialize_sft.selection_strategy must be 'stratified' when "
            "stratify_max_samples=true"
        )
    if not stratify_max_samples and selection_strategy == "stratified":
        raise ValueError(
            "materialize_sft.selection_strategy='stratified' requires "
            "stratify_max_samples=true"
        )

    dataset = load_materialization_dataset(config)
    if split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Split '{split}' not found. Available splits: {available}")

    source = dataset[split]
    target_records = max_samples
    source_max_samples = max_samples
    if mode == "teacher" and max_samples is not None:
        if rejection_buffer_samples <= 0:
            rejection_buffer_samples = max(10, int(max_samples * 0.1))
        source_max_samples = max_samples + rejection_buffer_samples
    source, selection_info = _select_source_examples(
        source,
        max_samples=source_max_samples,
        stratify_by=stratify_field if stratify_max_samples else None,
        seed=int(config.seed),
        canonical_labels=configured_canonical_labels,
        selection_strategy=selection_strategy,
        return_metadata=True,
    )
    template_canonical_labels = list(configured_canonical_labels)
    if not template_canonical_labels and canonical_labels_field:
        template_canonical_labels = _field_label_values(source, canonical_labels_field)
    if not template_canonical_labels:
        template_canonical_labels = _class_label_names(source, stratify_field)
    normalized_canonical_labels = {
        _normalize_canonical_label(label) for label in template_canonical_labels
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    temp_manifest_path = output_path.with_name(f".{output_path.name}.manifest.tmp")
    if temp_path.exists():
        temp_path.unlink()
    if temp_manifest_path.exists():
        temp_manifest_path.unlink()

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    written = 0
    attempted = 0
    usage_estimated_records = 0
    teacher_gold_agreement_records = 0
    teacher_gold_disagreement_records = 0
    record_hashes: list[str] = []
    stop_reason = "completed"
    materialized_at = datetime.now(timezone.utc).isoformat()
    canonical_labels_path = None
    progress = _materialization_progress(show_progress)
    progress_task: TaskID | None = None

    def update_progress(status: str, advance: int = 0) -> None:
        if progress is None or progress_task is None:
            return
        progress.update(
            progress_task,
            advance=advance,
            attempted=attempted,
            rejected=max(0, attempted - written),
            tokens=total_tokens,
            status=status,
        )

    try:
        progress_context = progress if progress is not None else nullcontext()
        with progress_context:
            if progress is not None:
                progress_total = (
                    min(target_records, len(source))
                    if target_records is not None
                    else len(source)
                )
                progress_task = progress.add_task(
                    f"{split} {mode}",
                    total=progress_total,
                    attempted=0,
                    rejected=0,
                    tokens=0,
                    status="starting",
                )
            with temp_path.open("w", encoding="utf-8") as f:
                for index, row in enumerate(source):
                    if target_records is not None and written >= target_records:
                        update_progress("requested records reached")
                        break
                    attempted += 1
                    source_original_index = selection_info["selected_source_indices"][
                        index
                    ]
                    predicted_total = 0
                    row_values = dict(row)
                    gold_answer, resolved_gold_field = _label_text(
                        row_values, config, source, gold_answer_field
                    )
                    if not gold_answer:
                        raise ValueError(
                            f"Gold answer is empty for row {split}/{index} "
                            f"from field '{resolved_gold_field}'"
                        )
                    row_values["gold_answer"] = gold_answer

                    if config.text_field in row_values:
                        row_values.setdefault("text", row_values[config.text_field])
                    if config.label_field in row_values:
                        row_values.setdefault("label", row_values[config.label_field])
                    if template_canonical_labels:
                        row_values["canonical_labels"] = template_canonical_labels

                    student_prompt = _format_template(
                        student_prompt_template, row_values
                    )
                    row_values["student_prompt"] = student_prompt
                    source_id = str(row_values.get("id", f"{split}/{index}"))
                    update_progress(f"preparing row {attempted}")

                    usage = {
                        "teacher_input_tokens": 0,
                        "teacher_output_tokens": 0,
                        "teacher_total_tokens": 0,
                        "provider_total_tokens": 0,
                    }
                    teacher_prompt = ""
                    teacher_response = gold_answer
                    usage_estimated = False

                    if mode == "teacher":
                        teacher_prompt = _format_template(
                            teacher_prompt_template, row_values
                        )
                        messages = [{"role": "user", "content": teacher_prompt}]
                        predicted_total = _estimate_input_tokens(
                            config.teacher, messages
                        ) + int(config.teacher_max_output_tokens or 0)
                        if (
                            config.token_budget is not None
                            and total_tokens + predicted_total > config.token_budget
                        ):
                            stop_reason = "predicted_budget_exhausted"
                            message = (
                                "Stopping before row "
                                f"{source_id}: predicted total "
                                f"{total_tokens + predicted_total} would exceed "
                                f"token_budget={config.token_budget}"
                            )
                            update_progress("predicted budget exhausted")
                            if not allow_partial:
                                raise ValueError(
                                    f"{message}. Rerun with --allow-partial to keep a "
                                    "budget-truncated dataset."
                                )
                            logger.info(message)
                            break

                        update_progress(f"calling teacher row {attempted}")
                        response = await acompletion(
                            model=config.teacher,
                            messages=messages,
                            max_tokens=config.teacher_max_output_tokens,
                            temperature=0,
                        )
                        teacher_response = _response_text(response)
                        if not teacher_response:
                            raise ValueError(
                                f"Teacher returned an empty response for {source_id}"
                            )
                        usage = _usage_from_response(response)
                        if (
                            usage["teacher_total_tokens"] <= 0
                            and not allow_estimated_usage
                        ):
                            raise ValueError(
                                "Teacher response did not include token usage. "
                                "Rerun with allow_estimated_usage only for non-paper dry runs."
                            )
                        if usage["teacher_total_tokens"] <= 0:
                            if config.paper_mode:
                                raise ValueError(
                                    "paper_mode=true requires provider-reported token usage"
                                )
                            usage["teacher_input_tokens"] = _estimate_input_tokens(
                                config.teacher, messages
                            )
                            usage["teacher_output_tokens"] = max(
                                1, len(teacher_response) // 3
                            )
                            usage["teacher_total_tokens"] = (
                                usage["teacher_input_tokens"]
                                + usage["teacher_output_tokens"]
                            )
                            usage_estimated = True

                        if usage["teacher_total_tokens"] > predicted_total:
                            raise ValueError(
                                "Realized teacher tokens exceeded preflight estimate for "
                                f"{source_id}: realized={usage['teacher_total_tokens']} "
                                f"predicted={predicted_total}"
                            )
                        if (
                            config.token_budget is not None
                            and total_tokens + usage["teacher_total_tokens"]
                            > config.token_budget
                        ):
                            raise ValueError(
                                "Realized teacher tokens would exceed token_budget for "
                                f"{source_id}: realized_total="
                                f"{total_tokens + usage['teacher_total_tokens']} "
                                f"budget={config.token_budget}"
                            )

                    teacher_label = _normalize_canonical_label(teacher_response)
                    gold_label = _normalize_canonical_label(gold_answer)
                    input_tokens += usage["teacher_input_tokens"]
                    output_tokens += usage["teacher_output_tokens"]
                    total_tokens += usage["teacher_total_tokens"]
                    usage_estimated_records += int(usage_estimated)
                    if (
                        mode == "teacher"
                        and normalized_canonical_labels
                        and teacher_label not in normalized_canonical_labels
                    ):
                        logger.warning(
                            "Rejecting non-canonical teacher response for %s: %r",
                            source_id,
                            teacher_response,
                        )
                        update_progress(f"rejected row {attempted}")
                        continue
                    if teacher_label and gold_label and teacher_label == gold_label:
                        teacher_gold_agreement_records += 1
                    elif teacher_label and gold_label:
                        teacher_gold_disagreement_records += 1
                    record = {
                        "id": f"{config.name}/{split}/{index}",
                        "task": config.name,
                        "source_example_id": source_id,
                        "source_split": split,
                        "source_index": index,
                        "source_original_index": source_original_index,
                        "prompt_operator": prompt_operator,
                        "teacher_tier": teacher_tier if mode == "teacher" else "gold",
                        "teacher_model": (
                            config.teacher if mode == "teacher" else "gold"
                        ),
                        "student_prompt": student_prompt,
                        "teacher_prompt": teacher_prompt,
                        "teacher_response": teacher_response,
                        "gold_answer": gold_answer,
                        "gold_answer_field": resolved_gold_field,
                        "cycle": 0,
                        "seed": config.seed,
                        "materialized_at": materialized_at,
                        "materialization_mode": mode,
                        "usage_estimated": usage_estimated,
                        "predicted_teacher_total_tokens": predicted_total,
                        **usage,
                    }
                    if record["teacher_total_tokens"] < (
                        record["teacher_input_tokens"]
                        + record["teacher_output_tokens"]
                    ):
                        raise ValueError(
                            "Invalid token accounting: teacher_total_tokens must be at "
                            "least teacher_input_tokens + teacher_output_tokens"
                        )
                    line = json.dumps(record, sort_keys=True)
                    record_hashes.append(sha256(line.encode("utf-8")).hexdigest())
                    f.write(line + "\n")
                    written += 1
                    update_progress(f"accepted row {attempted}", advance=1)

        if written == 0:
            raise ValueError("No SFT records were materialized")
        if (
            mode == "teacher"
            and target_records is not None
            and len(source) >= target_records
            and written < target_records
            and not allow_partial
        ):
            raise ValueError(
                "Only materialized "
                f"{written} accepted records out of requested {target_records}; "
                "increase materialize_sft.rejection_buffer_samples or inspect "
                "non-canonical teacher responses."
            )

        canonical_labels_path = _write_canonical_labels_artifact(
            config=config,
            output_path=output_path,
            source=source,
            label_field=gold_answer_field or config.label_field,
            canonical_labels=configured_canonical_labels,
            canonical_labels_field=canonical_labels_field,
        )

        manifest = {
            "schema_version": 2,
            "output_path": str(output_path),
            "manifest_path": str(manifest_path),
            "config_name": config.name,
            "config_hash": _stable_hash(config.model_dump(mode="json")),
            "student_prompt_template_hash": _stable_hash(student_prompt_template),
            "teacher_prompt_template_hash": _stable_hash(teacher_prompt_template),
            "student_prompt_template_sha256": sha256(
                student_prompt_template.encode("utf-8")
            ).hexdigest(),
            "teacher_prompt_template_sha256": sha256(
                teacher_prompt_template.encode("utf-8")
            ).hexdigest(),
            "dataset": config.dataset,
            "dataset_subset": config.dataset_subset,
            "dataset_revision": config.dataset_revision,
            "student_model": config.student,
            "student_revision": config.student_revision,
            "tokenizer_revision": config.tokenizer_revision,
            "teacher_model": config.teacher,
            "teacher_revision": config.teacher_revision,
            "split": split,
            "materialization_request": {
                "mode": mode,
                "split": split,
                "max_samples": max_samples,
                "selection_strategy": selection_info["selection_strategy"],
                "selection_seed": selection_info["selection_seed"],
                "prompt_operator": prompt_operator,
                "teacher_tier": teacher_tier,
                "allow_partial": allow_partial,
                "allow_estimated_usage": allow_estimated_usage,
                "rejection_buffer_samples": rejection_buffer_samples,
            },
            "source_records_before_selection": selection_info[
                "source_records_before_selection"
            ],
            "source_records": len(source),
            "attempted_records": attempted,
            "accepted_records": written,
            "rejected_records": max(0, attempted - written),
            "records": written,
            "records_sha256": record_hashes,
            "stop_reason": stop_reason,
            "allow_partial": allow_partial,
            "allow_estimated_usage": allow_estimated_usage,
            "mode": mode,
            "max_samples": max_samples,
            "stratify_max_samples": stratify_max_samples,
            "selection_strategy": selection_info["selection_strategy"],
            "selection_seed": selection_info["selection_seed"],
            "selected_source_indices_count": selection_info[
                "selected_source_indices_count"
            ],
            "selected_source_indices_sha256": selection_info[
                "selected_source_indices_sha256"
            ],
            "selected_source_indices_preview": selection_info[
                "selected_source_indices_preview"
            ],
            "prompt_operator": prompt_operator,
            "teacher_tier": teacher_tier,
            "token_budget": config.token_budget,
            "teacher_input_tokens": input_tokens,
            "teacher_output_tokens": output_tokens,
            "teacher_total_tokens": total_tokens,
            "teacher_gold_agreement_records": teacher_gold_agreement_records,
            "teacher_gold_disagreement_records": teacher_gold_disagreement_records,
            "teacher_gold_disagreement_rate": (
                teacher_gold_disagreement_records
                / max(
                    1,
                    teacher_gold_agreement_records
                    + teacher_gold_disagreement_records,
                )
            ),
            "usage_estimated_records": usage_estimated_records,
            "materialized_at": materialized_at,
            "reproducibility": build_reproducibility_manifest(
                config=config,
                artifact_dir=output_path.parent,
            ),
        }
        if (
            selection_info["selected_source_indices_count"]
            <= SOURCE_INDEX_LIST_LIMIT
        ):
            manifest["selected_source_indices"] = selection_info[
                "selected_source_indices"
            ]
        if canonical_labels_path:
            manifest["canonical_labels_path"] = canonical_labels_path
        artifact_hash = sha256()
        with temp_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                artifact_hash.update(chunk)
        manifest["artifact_sha256"] = artifact_hash.hexdigest()
        with temp_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")

        temp_path.replace(output_path)
        temp_manifest_path.replace(manifest_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        if temp_manifest_path.exists():
            temp_manifest_path.unlink()
        raise

    return {
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "split": split,
        "records": written,
        "attempted_records": attempted,
        "stop_reason": stop_reason,
        "teacher_input_tokens": input_tokens,
        "teacher_output_tokens": output_tokens,
        "teacher_total_tokens": total_tokens,
    }
