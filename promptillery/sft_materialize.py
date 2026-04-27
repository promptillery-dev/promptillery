"""Materialize audited SFT JSONL records from a dataset split."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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
) -> str:
    """Return a string label/gold answer for a source row."""
    label_field = field or config.label_field
    if label_field not in row:
        return ""

    label = row[label_field]
    feature = dataset.features.get(label_field)
    if hasattr(feature, "int2str") and isinstance(label, int):
        try:
            return str(feature.int2str(label))
        except ValueError:
            return str(label)
    return str(label)


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
    total_tokens = input_tokens + output_tokens
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
) -> Dict[str, Any]:
    """Write SFT JSONL records for one dataset split."""
    if isinstance(config.teacher, list):
        raise ValueError("materialize-sft requires a single teacher value")
    if isinstance(config.seed, list):
        raise ValueError("materialize-sft requires a single seed value")
    if mode not in {"gold", "teacher"}:
        raise ValueError("mode must be 'gold' or 'teacher'")
    if mode == "teacher" and config.teacher_max_output_tokens is None:
        raise ValueError(
            "teacher mode requires teacher_max_output_tokens for preflight accounting"
        )
    if output_path.exists() and not overwrite:
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
    total_tokens = 0
    written = 0
    materialized_at = datetime.now(timezone.utc).isoformat()

    with output_path.open("w", encoding="utf-8") as f:
        for index, row in enumerate(source):
            row_values = dict(row)
            gold_answer = _label_text(row_values, config, source, gold_answer_field)
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
                predicted_total = _estimate_input_tokens(config.teacher, messages) + int(
                    config.teacher_max_output_tokens or 0
                )
                if (
                    config.token_budget is not None
                    and total_tokens + predicted_total > config.token_budget
                ):
                    logger.info(
                        "Stopping before row %s: predicted total %d would exceed token_budget=%d",
                        source_id,
                        total_tokens + predicted_total,
                        config.token_budget,
                    )
                    break

                response = await acompletion(
                    model=config.teacher,
                    messages=messages,
                    max_tokens=config.teacher_max_output_tokens,
                    temperature=0,
                )
                teacher_response = _response_text(response)
                usage = _usage_from_response(response)
                if usage["teacher_total_tokens"] <= 0 and not allow_estimated_usage:
                    raise ValueError(
                        "Teacher response did not include token usage. "
                        "Rerun with allow_estimated_usage only for non-paper dry runs."
                    )
                if usage["teacher_total_tokens"] <= 0:
                    usage["teacher_input_tokens"] = _estimate_input_tokens(
                        config.teacher, messages
                    )
                    usage["teacher_output_tokens"] = max(1, len(teacher_response) // 3)
                    usage["teacher_total_tokens"] = (
                        usage["teacher_input_tokens"] + usage["teacher_output_tokens"]
                    )
                    usage_estimated = True

            total_tokens += usage["teacher_total_tokens"]
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
                "cycle": 0,
                "seed": config.seed,
                "materialized_at": materialized_at,
                "materialization_mode": mode,
                "usage_estimated": usage_estimated,
                **usage,
            }
            if record["teacher_total_tokens"] != (
                record["teacher_input_tokens"] + record["teacher_output_tokens"]
            ):
                raise ValueError(
                    "Invalid token accounting: teacher_total_tokens must equal "
                    "teacher_input_tokens + teacher_output_tokens"
                )
            f.write(json.dumps(record, sort_keys=True) + "\n")
            written += 1

    return {
        "output_path": str(output_path),
        "split": split,
        "records": written,
        "teacher_total_tokens": total_tokens,
    }
