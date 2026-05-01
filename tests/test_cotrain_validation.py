import asyncio
import pytest
from pathlib import Path

from promptillery.cotrain.actions import CoTrainAction
from promptillery.cotrain.arbitration import ArbitrationResult
from promptillery.cotrain.provenance import ProvenanceClass, AuditLedger
from promptillery.cotrain.validation import (
    ValidationPipeline,
    VariantUnderReview,
    ValidationOutcome,
)
from promptillery.cotrain.variant_generator import Variant


def _variant(seed_id="s0", text="x", label="alpha"):
    return Variant(
        text=text, proposer_label=label,
        metadata={"seed_id": seed_id, "operator": "boundary",
                  "confused_with": None, "task_kind": "classification",
                  "temperature": 0.8},
    )


class FakePeerLabeller:
    def __init__(self, label_seq, conf_seq):
        self.label_seq = list(label_seq)
        self.conf_seq = list(conf_seq)

    async def label(self, text):
        return self.label_seq.pop(0), self.conf_seq.pop(0)


class FakeArbiter:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    async def label(self, *, text, label_options):
        self.calls += 1
        return self.results.pop(0)


def test_peer_consensus_accepts_without_arbitration(tmp_path):
    pipe = ValidationPipeline(
        peer_labeller=FakePeerLabeller(["alpha"], [0.9]),
        arbiter=FakeArbiter([]),
        proposer_confidence=0.95,
        ledger=AuditLedger(path=tmp_path / "l.jsonl", run_id="r"),
    )
    review = VariantUnderReview(
        cycle=1, variant=_variant(),
        proposer="a", receiver="b", action=CoTrainAction("boundary", 8, 0.7),
        label_options=["alpha", "beta"],
    )
    out: ValidationOutcome = asyncio.run(pipe.validate(review))
    assert out.provenance == ProvenanceClass.PEER_CONSENSUS
    assert out.accepted_label == "alpha"
    assert out.api_called is False


def test_disagreement_triggers_arbitration_match(tmp_path):
    pipe = ValidationPipeline(
        peer_labeller=FakePeerLabeller(["beta"], [0.95]),
        arbiter=FakeArbiter([
            ArbitrationResult(label="alpha", is_well_posed=True, samples=["alpha"], usage={})
        ]),
        proposer_confidence=0.95,
        ledger=AuditLedger(path=tmp_path / "l.jsonl", run_id="r"),
    )
    review = VariantUnderReview(
        cycle=1, variant=_variant(),
        proposer="a", receiver="b", action=CoTrainAction("boundary", 8, 0.7),
        label_options=["alpha", "beta"],
    )
    out = asyncio.run(pipe.validate(review))
    assert out.provenance == ProvenanceClass.STRONG_TEACHER_ARBITRATION_MATCH
    assert out.accepted_label == "alpha"
    assert out.api_called is True


def test_low_confidence_triggers_arbitration(tmp_path):
    pipe = ValidationPipeline(
        peer_labeller=FakePeerLabeller(["alpha"], [0.5]),
        arbiter=FakeArbiter([
            ArbitrationResult(label="gamma", is_well_posed=True, samples=["gamma"], usage={})
        ]),
        proposer_confidence=0.95,
        ledger=AuditLedger(path=tmp_path / "l.jsonl", run_id="r"),
    )
    review = VariantUnderReview(
        cycle=1, variant=_variant(),
        proposer="a", receiver="b", action=CoTrainAction("boundary", 8, 0.7),
        label_options=["alpha", "beta", "gamma"],
    )
    out = asyncio.run(pipe.validate(review))
    assert out.provenance == ProvenanceClass.STRONG_TEACHER_RELABEL
    assert out.accepted_label == "gamma"


def test_ill_posed_self_consistency_rejects(tmp_path):
    pipe = ValidationPipeline(
        peer_labeller=FakePeerLabeller(["x"], [0.5]),
        arbiter=FakeArbiter([
            ArbitrationResult(label=None, is_well_posed=False, samples=["1","2","3"], usage={})
        ]),
        proposer_confidence=0.5,
        ledger=AuditLedger(path=tmp_path / "l.jsonl", run_id="r"),
    )
    review = VariantUnderReview(
        cycle=1,
        variant=Variant(text="ambiguous", proposer_label="x",
                        metadata={"seed_id": "g", "operator": "coverage",
                                  "confused_with": None, "task_kind": "generation",
                                  "temperature": 0.7}),
        proposer="a", receiver="b", action=CoTrainAction("coverage", 8, 0.7),
        label_options=None,
    )
    out = asyncio.run(pipe.validate(review))
    assert out.provenance == ProvenanceClass.ILL_POSED_REJECT
    assert out.accepted_label is None
