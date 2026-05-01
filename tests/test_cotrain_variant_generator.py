import json
import pytest

from promptillery.cotrain.variant_generator import (
    VariantGenerator,
    GenerationRequest,
    GenerationOutput,
)


def fake_generate(prompt: str, *, n: int, temperature: float):
    payload = {"variants": [{"text": f"variant {i} of prompt#{hash(prompt) & 0xff}"} for i in range(n)]}
    return [json.dumps(payload)]


def test_generates_k_variants_for_classification_seed():
    gen = VariantGenerator(
        task_kind="classification",
        text_field="text",
        label_field="label",
        generate_fn=fake_generate,
    )
    req = GenerationRequest(
        seed_id="s0",
        seed_text="how do I reset my pin",
        seed_label="card_arrival",
        operator="boundary",
        confused_with="card_lost",
        k=3,
        temperature=0.8,
    )
    out = gen.generate(req)
    assert isinstance(out, GenerationOutput)
    assert len(out.variants) == 3
    for v in out.variants:
        assert v.proposer_label == "card_arrival"
        assert v.text.startswith("variant ")
        assert v.metadata["operator"] == "boundary"
        assert v.metadata["seed_id"] == "s0"


def test_generation_includes_operator_hint_in_prompt():
    captured = {}

    def capture_fn(prompt, *, n, temperature):
        captured["prompt"] = prompt
        return [json.dumps({"variants": [{"text": "x"}]})]

    gen = VariantGenerator(
        task_kind="classification", text_field="text", label_field="label",
        generate_fn=capture_fn,
    )
    gen.generate(GenerationRequest(
        seed_id="s", seed_text="t", seed_label="L", operator="repair",
        confused_with=None, k=1, temperature=0.7,
    ))
    assert "repair" in captured["prompt"].lower()
    assert "preserve the true label" in captured["prompt"].lower()


def test_generation_for_gsm8k_includes_answer_field():
    gen = VariantGenerator(
        task_kind="generation", text_field="problem", label_field="answer",
        generate_fn=lambda p, n, temperature: [json.dumps({"variants": [{"problem": "p", "answer": "42"}] * n})],
    )
    out = gen.generate(GenerationRequest(
        seed_id="g", seed_text="If Tim has 5 apples...", seed_label="3",
        operator="coverage", confused_with=None, k=2, temperature=0.7,
    ))
    assert all(v.metadata["task_kind"] == "generation" for v in out.variants)
    assert all(v.proposer_label == "42" for v in out.variants)


def test_malformed_response_yields_empty_output():
    gen = VariantGenerator(
        task_kind="classification", text_field="text", label_field="label",
        generate_fn=lambda p, n, temperature: ["not json"],
    )
    out = gen.generate(GenerationRequest(
        seed_id="s", seed_text="t", seed_label="L", operator="coverage",
        confused_with=None, k=2, temperature=0.7,
    ))
    assert out.variants == []
    assert out.parse_errors == 1
