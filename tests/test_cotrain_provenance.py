import json
from pathlib import Path

import pytest

from promptillery.cotrain.provenance import (
    ProvenanceClass,
    AcceptedRecord,
    AuditLedger,
)


def test_provenance_class_values():
    assert ProvenanceClass.PEER_CONSENSUS.value == "peer_consensus"
    assert ProvenanceClass.STRONG_TEACHER_ARBITRATION_MATCH.value == "strong_teacher_arbitration_match"
    assert ProvenanceClass.STRONG_TEACHER_RELABEL.value == "strong_teacher_relabel"
    assert ProvenanceClass.ILL_POSED_REJECT.value == "ill_posed_reject"


def test_ledger_writes_jsonl_with_required_fields(tmp_path):
    path = tmp_path / "cotrain_ledger.jsonl"
    ledger = AuditLedger(path=path, run_id="r0")
    ledger.append(AcceptedRecord(
        cycle=1, seed_id="s0", variant_id="v0",
        proposer="a", receiver="b",
        proposer_label="card_arrival", peer_label="card_arrival",
        teacher_label=None, accepted_label="card_arrival",
        provenance=ProvenanceClass.PEER_CONSENSUS,
        proposer_confidence=0.92, peer_confidence=0.88, tau=0.7,
        text="how do i reset my pin",
        operator="boundary", task_kind="classification",
    ))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["provenance"] == "peer_consensus"
    assert rows[0]["accepted_label"] == "card_arrival"
    assert rows[0]["run_id"] == "r0"
    assert rows[0]["cycle"] == 1


def test_ledger_summarize_counts_per_class(tmp_path):
    ledger = AuditLedger(path=tmp_path / "l.jsonl", run_id="r0")
    base = dict(
        seed_id="s", variant_id="v", proposer="a", receiver="b",
        proposer_label="x", peer_label="x", teacher_label=None, accepted_label="x",
        proposer_confidence=0.9, peer_confidence=0.9, tau=0.7,
        text="t", operator="coverage", task_kind="classification",
    )
    for prov in [
        ProvenanceClass.PEER_CONSENSUS, ProvenanceClass.PEER_CONSENSUS,
        ProvenanceClass.STRONG_TEACHER_RELABEL, ProvenanceClass.ILL_POSED_REJECT,
    ]:
        ledger.append(AcceptedRecord(cycle=1, provenance=prov, **base))
    counts = ledger.summary_counts()
    assert counts["peer_consensus"] == 2
    assert counts["strong_teacher_relabel"] == 1
    assert counts["ill_posed_reject"] == 1
