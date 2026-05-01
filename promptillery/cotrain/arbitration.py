"""Strong-teacher arbitration with optional self-consistency (design §3.7)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence


class ChatClient(Protocol):
    async def complete(
        self, *, model: str, messages: List[Dict[str, str]],
        temperature: float, max_tokens: int,
    ) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class ArbitrationResult:
    label: Optional[str]
    is_well_posed: bool
    samples: List[str]
    usage: Dict[str, int] = field(default_factory=dict)


class StrongTeacherArbiter:
    def __init__(
        self,
        *,
        model: str,
        client: ChatClient,
        task_kind: str,
        self_consistency_n: int,
        self_consistency_temperature: float,
        max_tokens: int = 256,
    ) -> None:
        if task_kind not in {"classification", "generation"}:
            raise ValueError(task_kind)
        self.model = model
        self.client = client
        self.task_kind = task_kind
        self.n = self_consistency_n
        self.temperature = self_consistency_temperature
        self.max_tokens = max_tokens

    def _build_messages(
        self, text: str, label_options: Optional[Sequence[str]]
    ) -> List[Dict[str, str]]:
        if self.task_kind == "classification":
            opts = ", ".join(label_options or [])
            user = (
                f"Classify the following input. Reply with exactly one of: {opts}.\n\n"
                f"Input: {text}\nLabel:"
            )
        else:
            user = (
                "Solve the problem. Reply with the final answer only, no working.\n\n"
                f"Problem: {text}\nAnswer:"
            )
        return [{"role": "user", "content": user}]

    async def label(
        self, *, text: str, label_options: Optional[Sequence[str]],
    ) -> ArbitrationResult:
        messages = self._build_messages(text, label_options)
        n = 1 if self.task_kind == "classification" else self.n
        temperature = 0.0 if n == 1 else self.temperature
        samples: List[str] = []
        usage_total = {"input": 0, "output": 0}
        for _ in range(n):
            resp = await self.client.complete(
                model=self.model, messages=messages,
                temperature=temperature, max_tokens=self.max_tokens,
            )
            samples.append(resp["text"].strip())
            for k, v in (resp.get("usage") or {}).items():
                usage_total[k] = usage_total.get(k, 0) + int(v)

        if self.task_kind == "classification":
            return ArbitrationResult(
                label=samples[0], is_well_posed=True, samples=samples, usage=usage_total
            )
        counts = Counter(samples)
        top_label, top_count = counts.most_common(1)[0]
        well_posed = top_count == n
        return ArbitrationResult(
            label=top_label if well_posed else None,
            is_well_posed=well_posed,
            samples=samples,
            usage=usage_total,
        )
