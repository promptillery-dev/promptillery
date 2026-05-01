"""Policy decision logging for budget-aware distillation experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PolicyDecision(BaseModel):
    """Single cycle-level policy decision.

    This schema is intentionally flexible while policy experiments are still
    evolving. Stable top-level fields make logs easy to analyze, while nested
    dictionaries allow policies to record richer state/action details without
    changing the file format.
    """

    schema_version: int = 1
    run_id: str = ""
    decision_id: str = ""
    cycle: int
    policy_name: str
    action_name: str
    state: Dict[str, Any] = Field(default_factory=dict)
    action: Dict[str, Any] = Field(default_factory=dict)
    action_scores: Dict[str, float] = Field(default_factory=dict)
    predicted_cost: Dict[str, Any] = Field(default_factory=dict)
    realized_cost: Dict[str, Any] = Field(default_factory=dict)
    reward: Optional[float] = None
    budget_before: Dict[str, Any] = Field(default_factory=dict)
    budget_after: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Free-form per-decision metadata. For cotrain runs, this records "
            "ProvenanceClass values and per-variant provenance counts; see "
            "promptillery.cotrain.provenance."
        ),
    )


class PolicyDecisionLogger:
    """Append policy decisions to a JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, decision: PolicyDecision) -> None:
        """Append one policy decision as compact JSON."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(decision.model_dump(), sort_keys=True) + "\n")
