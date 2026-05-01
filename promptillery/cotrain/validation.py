"""Per-variant validation pipeline (design §3.7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

from .actions import CoTrainAction
from .arbitration import ArbitrationResult
from .provenance import AcceptedRecord, AuditLedger, ProvenanceClass
from .variant_generator import Variant


def task_equal(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    return a.strip().lower() == b.strip().lower()


class PeerLabeller(Protocol):
    async def label(self, text: str) -> tuple[str, float]: ...


class Arbiter(Protocol):
    async def label(
        self, *, text: str, label_options: Optional[Sequence[str]]
    ) -> ArbitrationResult: ...


@dataclass
class VariantUnderReview:
    cycle: int
    variant: Variant
    proposer: str
    receiver: str
    action: CoTrainAction
    label_options: Optional[Sequence[str]]


@dataclass
class ValidationOutcome:
    accepted_label: Optional[str]
    provenance: ProvenanceClass
    proposer_confidence: float
    peer_confidence: float
    api_called: bool
    record: AcceptedRecord


class ValidationPipeline:
    def __init__(
        self,
        *,
        peer_labeller: PeerLabeller,
        arbiter: Arbiter,
        proposer_confidence: float,
        ledger: AuditLedger,
    ) -> None:
        self.peer_labeller = peer_labeller
        self.arbiter = arbiter
        self.proposer_confidence = proposer_confidence
        self.ledger = ledger

    async def validate(self, review: VariantUnderReview) -> ValidationOutcome:
        v = review.variant
        peer_label, peer_conf = await self.peer_labeller.label(v.text)
        tau = review.action.tau

        if (
            self.proposer_confidence >= tau
            and peer_conf >= tau
            and task_equal(v.proposer_label, peer_label)
        ):
            outcome = self._accept(review, peer_label, peer_conf, None,
                                   v.proposer_label, ProvenanceClass.PEER_CONSENSUS, False)
            return outcome

        arb = await self.arbiter.label(
            text=v.text, label_options=review.label_options
        )
        if not arb.is_well_posed:
            return self._accept(
                review, peer_label, peer_conf, None, None,
                ProvenanceClass.ILL_POSED_REJECT, True
            )
        if task_equal(arb.label, v.proposer_label) or task_equal(arb.label, peer_label):
            return self._accept(
                review, peer_label, peer_conf, arb.label, arb.label,
                ProvenanceClass.STRONG_TEACHER_ARBITRATION_MATCH, True
            )
        return self._accept(
            review, peer_label, peer_conf, arb.label, arb.label,
            ProvenanceClass.STRONG_TEACHER_RELABEL, True
        )

    def _accept(
        self,
        review: VariantUnderReview,
        peer_label: Optional[str],
        peer_conf: float,
        teacher_label: Optional[str],
        accepted_label: Optional[str],
        provenance: ProvenanceClass,
        api_called: bool,
    ) -> ValidationOutcome:
        v = review.variant
        record = AcceptedRecord(
            cycle=review.cycle,
            seed_id=v.metadata["seed_id"],
            variant_id=f"{review.proposer}{v.metadata['seed_id']}-{id(v)}",
            proposer=review.proposer,
            receiver=review.receiver,
            proposer_label=v.proposer_label,
            peer_label=peer_label,
            teacher_label=teacher_label,
            accepted_label=accepted_label,
            provenance=provenance,
            proposer_confidence=self.proposer_confidence,
            peer_confidence=peer_conf,
            tau=review.action.tau,
            text=v.text,
            operator=review.action.operator,
            task_kind=v.metadata["task_kind"],
            extra={"action": review.action.name},
        )
        self.ledger.append(record)
        return ValidationOutcome(
            accepted_label=accepted_label,
            provenance=provenance,
            proposer_confidence=self.proposer_confidence,
            peer_confidence=peer_conf,
            api_called=api_called,
            record=record,
        )
