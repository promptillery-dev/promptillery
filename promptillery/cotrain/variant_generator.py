"""Seed-anchored variant generation (design §3.5).

The student plays *generator*: given a seed (x_seed, y_seed) plus operator
hint and optional confused_with metadata, emit K variants in JSON. The
generator never produces inputs from scratch; every variant is anchored to
a seed in the proposer's bootstrap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


_OPERATOR_HINTS = {
    "coverage": (
        "Broaden the seed via paraphrase, synonym substitution, or alternative "
        "phrasings; preserve the true label."
    ),
    "boundary": (
        "Produce variants near plausible decision boundaries: phrasings that "
        "could plausibly fit a different label but in fact preserve the true label."
    ),
    "repair": (
        "Target the specific failure pattern from the fault metadata. Generate "
        "variants that the failing student is likely to misclassify; preserve "
        "the true label."
    ),
}


@dataclass(frozen=True)
class GenerationRequest:
    seed_id: str
    seed_text: str
    seed_label: str
    operator: str
    confused_with: Optional[str]
    k: int
    temperature: float


@dataclass(frozen=True)
class Variant:
    text: str
    proposer_label: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class GenerationOutput:
    variants: List[Variant]
    parse_errors: int = 0
    raw_responses: List[str] = field(default_factory=list)


class VariantGenerator:
    """Wraps a student-model callable behind a prompt + JSON-parser contract."""

    def __init__(
        self,
        *,
        task_kind: str,
        text_field: str,
        label_field: str,
        generate_fn: Callable[..., List[str]],
    ) -> None:
        if task_kind not in {"classification", "generation"}:
            raise ValueError(task_kind)
        self.task_kind = task_kind
        self.text_field = text_field
        self.label_field = label_field
        self.generate_fn = generate_fn

    def _build_prompt(self, req: GenerationRequest) -> str:
        hint = _OPERATOR_HINTS.get(req.operator, "")
        confused_line = (
            f"The receiving student tends to confuse this label with: {req.confused_with}.\n"
            if req.confused_with
            else ""
        )
        if self.task_kind == "classification":
            schema = (
                '{"variants": [{"text": "<variant text>"}, ...]}\n'
                f"Each variant must have the label: {req.seed_label}.\n"
            )
        else:
            schema = (
                '{"variants": [{"problem": "<variant problem>", "answer": "<answer>"}, ...]}\n'
                "Numeric or cover-story perturbations only; preserve a checkable answer.\n"
            )
        return (
            "You are generating training-data variants for a peer student.\n\n"
            f"Operator: {req.operator}\n{hint}\n{confused_line}\n"
            f"Seed input: {req.seed_text}\n"
            f"Seed label: {req.seed_label}\n\n"
            f"Produce {req.k} variants in this JSON shape and nothing else:\n{schema}"
        )

    def generate(self, req: GenerationRequest) -> GenerationOutput:
        prompt = self._build_prompt(req)
        raw = self.generate_fn(prompt, n=req.k, temperature=req.temperature)
        if not raw:
            return GenerationOutput(variants=[], parse_errors=1, raw_responses=[])
        text = raw[0]
        try:
            payload = json.loads(text)
            items = payload["variants"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return GenerationOutput(variants=[], parse_errors=1, raw_responses=[text])

        variants: List[Variant] = []
        for item in items[: req.k]:
            if self.task_kind == "classification":
                v_text = item.get("text")
                if not isinstance(v_text, str) or not v_text.strip():
                    continue
                proposer_label = req.seed_label
            else:
                v_text = item.get("problem")
                proposer_label = item.get("answer", req.seed_label)
                if not isinstance(v_text, str) or not v_text.strip():
                    continue
            variants.append(Variant(
                text=v_text,
                proposer_label=str(proposer_label),
                metadata={
                    "seed_id": req.seed_id,
                    "operator": req.operator,
                    "confused_with": req.confused_with,
                    "task_kind": self.task_kind,
                    "temperature": req.temperature,
                },
            ))
        return GenerationOutput(variants=variants, parse_errors=0, raw_responses=[text])
