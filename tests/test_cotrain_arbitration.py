import asyncio
import pytest

from promptillery.cotrain.arbitration import (
    StrongTeacherArbiter,
    ArbitrationResult,
)


class FakeChatClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def complete(self, *, model, messages, temperature, max_tokens):
        self.calls += 1
        return {"text": self.responses.pop(0), "usage": {"input": 10, "output": 5}}


def test_classification_arbitration_returns_label_in_one_call():
    client = FakeChatClient(["card_lost"])
    arbiter = StrongTeacherArbiter(
        model="openai/gpt-4o", client=client, task_kind="classification",
        self_consistency_n=3, self_consistency_temperature=0.7,
    )
    result = asyncio.run(arbiter.label(text="i lost my card", label_options=["card_lost", "card_arrival"]))
    assert isinstance(result, ArbitrationResult)
    assert result.label == "card_lost"
    assert result.is_well_posed is True
    assert result.samples == ["card_lost"]
    assert client.calls == 1


def test_generation_arbitration_runs_self_consistency_n_times():
    client = FakeChatClient(["42", "42", "42"])
    arbiter = StrongTeacherArbiter(
        model="openai/gpt-4o", client=client, task_kind="generation",
        self_consistency_n=3, self_consistency_temperature=0.7,
    )
    result = asyncio.run(arbiter.label(text="If Tim has...", label_options=None))
    assert result.label == "42"
    assert result.is_well_posed is True
    assert client.calls == 3


def test_generation_arbitration_rejects_when_samples_disagree():
    client = FakeChatClient(["42", "43", "44"])
    arbiter = StrongTeacherArbiter(
        model="openai/gpt-4o", client=client, task_kind="generation",
        self_consistency_n=3, self_consistency_temperature=0.7,
    )
    result = asyncio.run(arbiter.label(text="ambiguous", label_options=None))
    assert result.is_well_posed is False
    assert result.label is None
