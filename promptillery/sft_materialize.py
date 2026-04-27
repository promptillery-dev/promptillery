"""Materialize audited SFT JSONL records from a dataset split."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import DatasetDict, load_dataset
from jinja2 import StrictUndefined
from litellm import acompletion

from .config import ExperimentConfig
from .engine import ensure_class_label, ensure_validation_split, prepare_dataset
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


def _stable_hash(value: Any) -> str:
    """Return a stable short hash for configs and templates."""
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return sha256(payload).hexdigest()[:16]


def load_materialization_dataset(config: ExperimentConfig) -> DatasetDict:
    """Load and prepare a dataset for SFT materialization."""
    dataset_kwargs = config.dataset_kwargs or {}
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

    dataset = load_materialization_dataset(config)
    if split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Split '{split}' not found. Available splits: {available}")

    source = dataset[split]
    if max_samples is not None:
        source = source.select(range(min(max_samples, len(source))))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    temp_manifest_path = output_path.with_name(f".{output_path.name}.manifest.tmp")
    if temp_path.exists():
        temp_path.unlink()
    if temp_manifest_path.exists():
        temp_manifest_path.unlink()

    total_tokens = 0
    written = 0
    attempted = 0
    usage_estimated_records = 0
    record_hashes: list[str] = []
    stop_reason = "completed"
    materialized_at = datetime.now(timezone.utc).isoformat()

    try:
        with temp_path.open("w", encoding="utf-8") as f:
            for index, row in enumerate(source):
                attempted += 1
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

                student_prompt = _format_template(student_prompt_template, row_values)
                row_values["student_prompt"] = student_prompt
                source_id = str(row_values.get("id", f"{split}/{index}"))

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
                    teacher_prompt = _format_template(teacher_prompt_template, row_values)
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
                        if not allow_partial:
                            raise ValueError(
                                f"{message}. Rerun with --allow-partial to keep a "
                                "budget-truncated dataset."
                            )
                        logger.info(message)
                        break

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
                    if usage["teacher_total_tokens"] <= 0 and not allow_estimated_usage:
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

                total_tokens += usage["teacher_total_tokens"]
                usage_estimated_records += int(usage_estimated)
                record = {
                    "id": f"{config.name}/{split}/{index}",
                    "task": config.name,
                    "source_example_id": source_id,
                    "source_split": split,
                    "source_index": index,
                    "prompt_operator": prompt_operator,
                    "teacher_tier": teacher_tier if mode == "teacher" else "gold",
                    "teacher_model": config.teacher if mode == "teacher" else "gold",
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
                    record["teacher_input_tokens"] + record["teacher_output_tokens"]
                ):
                    raise ValueError(
                        "Invalid token accounting: teacher_total_tokens must be at "
                        "least teacher_input_tokens + teacher_output_tokens"
                    )
                line = json.dumps(record, sort_keys=True)
                record_hashes.append(sha256(line.encode("utf-8")).hexdigest())
                f.write(line + "\n")
                written += 1

        if written == 0:
            raise ValueError("No SFT records were materialized")

        manifest = {
            "schema_version": 1,
            "output_path": str(output_path),
            "manifest_path": str(manifest_path),
            "config_name": config.name,
            "config_hash": _stable_hash(config.model_dump(mode="json")),
            "student_prompt_template_hash": _stable_hash(student_prompt_template),
            "teacher_prompt_template_hash": _stable_hash(teacher_prompt_template),
            "dataset": config.dataset,
            "dataset_subset": config.dataset_subset,
            "split": split,
            "source_records": len(source),
            "attempted_records": attempted,
            "records": written,
            "records_sha256": record_hashes,
            "stop_reason": stop_reason,
            "allow_partial": allow_partial,
            "allow_estimated_usage": allow_estimated_usage,
            "mode": mode,
            "max_samples": max_samples,
            "prompt_operator": prompt_operator,
            "teacher_tier": teacher_tier,
            "token_budget": config.token_budget,
            "teacher_total_tokens": total_tokens,
            "usage_estimated_records": usage_estimated_records,
            "materialized_at": materialized_at,
        }
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
        "teacher_total_tokens": total_tokens,
    }
