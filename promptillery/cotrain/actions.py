"""Co-training action space: operator × volume × τ + STOP (design §3.8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class CoTrainAction:
    operator: str = ""
    volume: int = 0
    tau: float = 0.0
    is_stop: bool = False

    @property
    def name(self) -> str:
        if self.is_stop:
            return "STOP"
        tau_str = f"{self.tau:g}"
        return f"{self.operator}:v{self.volume}:t{tau_str}"

    def model_dump(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "operator": self.operator,
            "volume": self.volume,
            "tau": self.tau,
            "is_stop": self.is_stop,
        }


def enumerate_cotrain_actions(
    operators: Sequence[str] = ("coverage", "boundary", "repair"),
    volumes: Sequence[int] = (8, 16, 32),
    taus: Sequence[float] = (0.5, 0.7, 0.9),
    include_stop: bool = True,
) -> List[CoTrainAction]:
    actions = [
        CoTrainAction(operator=op, volume=v, tau=t)
        for op in operators
        for v in volumes
        for t in taus
    ]
    if include_stop:
        actions.append(CoTrainAction(is_stop=True))
    return actions
