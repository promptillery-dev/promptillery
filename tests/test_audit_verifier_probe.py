"""Tests for the gold-anchored label-consistency components."""

from pathlib import Path

import pytest

from promptillery.audit import (
    make_hf_predict_fn,
    require_probe_credentials,
    teacher_gold_probe,
    verifier_agreement,
)
from promptillery.config import ExperimentConfig
from promptillery.token_tracker import TokenTracker

ROWS = [
    {"text": "a wonderful heartfelt film", "label": "positive"},
    {"text": "a dreadful boring slog", "label": "negative"},
    {"text": "sharp writing and warm humor", "label": "positive"},
]


class TestVerifierAgreement:
    def test_agreement_fraction(self):
        predict_fn = lambda texts: ["positive"] * len(texts)  # noqa: E731
        assert verifier_agreement(predict_fn, ROWS) == pytest.approx(2 / 3)

    def test_agreement_normalizes_labels(self):
        predict_fn = lambda texts: ["Positive", " NEGATIVE. ", "positive"]  # noqa: E731
        assert verifier_agreement(predict_fn, ROWS) == 1.0

    def test_empty_rows_raise(self):
        with pytest.raises(ValueError):
            verifier_agreement(lambda texts: [], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            verifier_agreement(lambda texts: ["positive"], ROWS)


M5_MODEL = Path(
    "out/m5/m5_sst2_same_n_ettin_encoder_transformers_20260710_173950_"
    "275851_s13_a8775392_1cd0e0/model"
)


@pytest.mark.slow
@pytest.mark.skipif(not M5_MODEL.exists(), reason="m5 gold checkpoint not on disk")
class TestRealCheckpoint:
    def test_predict_fn_returns_label_names(self):
        predict_fn = make_hf_predict_fn(
            str(M5_MODEL), label_names={0: "negative", 1: "positive"}
        )
        predictions = predict_fn(
            ["a wonderful heartfelt film", "a dreadful boring slog"]
        )
        assert len(predictions) == 2
        assert set(predictions) <= {"negative", "positive"}

    def test_default_label_0_names_are_rejected(self):
        # The m5 checkpoint config has no id2label, so omitting label_names
        # must raise instead of silently comparing LABEL_0 strings.
        with pytest.raises(ValueError):
            make_hf_predict_fn(str(M5_MODEL))


def _probe_config():
    return ExperimentConfig(
        name="audit_probe_fixture",
        teacher="openrouter/openai/gpt-4.1",
        dataset="fixture/tiny",
        dataset_config={
            "name": "default",
            "num_classes": 2,
            "text_field": "text",
            "label_field": "label",
        },
    )


GOLD_ROWS = [
    {"text": f"gold example number {i}", "label": "positive" if i % 2 else "negative"}
    for i in range(8)
]


class TestTeacherGoldProbe:
    def _fake_completion(self, calls):
        def fake(**kwargs):
            calls.append(kwargs)
            text = kwargs["messages"][0]["content"]
            # Echo back the gold label hidden in our fixture text parity.
            number = int(text.rsplit("number", 1)[1].split()[0])
            label = "positive" if number % 2 else "negative"
            return {
                "choices": [{"message": {"content": label}}],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 3,
                    "total_tokens": 53,
                },
            }

        return fake

    def test_probe_agreement_temperature_and_ledger(self, monkeypatch):
        calls = []
        monkeypatch.setattr("litellm.completion", self._fake_completion(calls))
        tracker = TokenTracker(
            experiment_name="audit", teacher_model="x", quiet=True
        )
        with tracker.cycle(0):
            result = teacher_gold_probe(
                _probe_config(), GOLD_ROWS, k=4, seed=13, tracker=tracker
            )
        assert result.k == 4
        assert result.agreement == 1.0
        assert len(calls) == 4
        assert all(call["temperature"] == 0.0 for call in calls)
        assert all(
            call["model"] == "openrouter/openai/gpt-4.1" for call in calls
        )
        # Prompt names the label space.
        assert "negative" in calls[0]["messages"][0]["content"]
        assert "positive" in calls[0]["messages"][0]["content"]
        # Usage recorded: 4 calls x 53 tokens.
        assert result.usage["total_tokens"] == 4 * 53

    def test_probe_sampling_is_seeded(self, monkeypatch):
        calls_a, calls_b = [], []
        monkeypatch.setattr("litellm.completion", self._fake_completion(calls_a))
        first = teacher_gold_probe(_probe_config(), GOLD_ROWS, k=4, seed=13)
        monkeypatch.setattr("litellm.completion", self._fake_completion(calls_b))
        second = teacher_gold_probe(_probe_config(), GOLD_ROWS, k=4, seed=13)
        assert [row["text"] for row in first.rows] == [
            row["text"] for row in second.rows
        ]

    def test_probe_empty_gold_raises(self):
        with pytest.raises(ValueError):
            teacher_gold_probe(_probe_config(), [], k=4, seed=13)


class TestRequireProbeCredentials:
    def test_missing_provider_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            require_probe_credentials("openrouter/openai/gpt-4.1")

    def test_present_provider_key_passes(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        require_probe_credentials("openrouter/openai/gpt-4.1")
