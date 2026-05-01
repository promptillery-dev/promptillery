"""Co-training audit ledger schema (design §3.7, §5 amendment)."""

from __future__ import annotations

import enum
import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


class ProvenanceClass(str, enum.Enum):
    PEER_CONSENSUS = "peer_consensus"
    STRONG_TEACHER_ARBITRATION_MATCH = "strong_teacher_arbitration_match"
    STRONG_TEACHER_RELABEL = "strong_teacher_relabel"
    ILL_POSED_REJECT = "ill_posed_reject"


@dataclass
class AcceptedRecord:
    cycle: int
    seed_id: str
    variant_id: str
    proposer: str  # "a" or "b"
    receiver: str  # "a" or "b"
    proposer_label: str
    peer_label: Optional[str]
    teacher_label: Optional[str]
    accepted_label: Optional[str]
    provenance: ProvenanceClass
    proposer_confidence: float
    peer_confidence: float
    tau: float
    text: str
    operator: str
    task_kind: str
    extra: Dict[str, Any] = field(default_factory=dict)


class AuditLedger:
    def __init__(self, *, path: Path, run_id: str, schema_version: int = 1) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.schema_version = schema_version

    def append(self, record: AcceptedRecord) -> None:
        row = asdict(record)
        row["provenance"] = record.provenance.value
        row["run_id"] = self.run_id
        row["schema_version"] = self.schema_version
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def summary_counts(self) -> Dict[str, int]:
        if not self.path.exists():
            return {}
        counts: Counter[str] = Counter()
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                counts[json.loads(line)["provenance"]] += 1
        return dict(counts)
