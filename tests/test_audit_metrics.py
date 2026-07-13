"""Unit tests for the pure audit metric functions."""

import json

import pytest

from promptillery.audit import (
    FailureSummary,
    distinct_n,
    exact_duplicate_flags,
    failure_summary,
    label_drift,
    near_duplicate_flags,
    parse_teacher_records,
)
from promptillery.config import ExperimentConfig


class TestExactDuplicateFlags:
    def test_flags_duplicates_of_reference_and_earlier_texts(self):
        reference = ["The movie was great.", "a dull mess"]
        texts = [
            "the movie  was GREAT.",  # dup of reference (case/whitespace)
            "a fresh new sentence",   # novel
            "A Fresh New Sentence",   # dup of earlier text in the same batch
        ]
        assert exact_duplicate_flags(texts, reference) == [True, False, True]

    def test_first_occurrence_is_not_a_duplicate(self):
        flags = exact_duplicate_flags(["one", "one"], ["something else"])
        assert flags == [False, True]

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError):
            exact_duplicate_flags(["a"], [])


class TestDistinctN:
    def test_distinct_2_counts_unique_bigrams(self):
        # "a b c" -> (a,b),(b,c); "a b d" -> (a,b),(b,d): 3 unique / 4 total
        assert distinct_n(["a b c", "a b d"], 2) == pytest.approx(3 / 4)

    def test_distinct_1_all_unique(self):
        assert distinct_n(["alpha beta", "gamma delta"], 1) == 1.0

    def test_empty_texts_raise(self):
        with pytest.raises(ValueError):
            distinct_n([], 2)

    def test_no_ngrams_raises(self):
        with pytest.raises(ValueError):
            distinct_n(["single"], 2)


class TestLabelDrift:
    def test_identical_distributions_have_zero_drift(self):
        assert label_drift(["a", "b"], ["b", "a"]) == 0.0

    def test_disjoint_distributions_have_drift_one(self):
        assert label_drift(["a", "a"], ["b", "b"]) == 1.0

    def test_half_shift(self):
        # aug: 100% a; seed: 50% a / 50% b -> TV = 0.5
        assert label_drift(["a", "a"], ["a", "b"]) == pytest.approx(0.5)

    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError):
            label_drift([], ["a"])
        with pytest.raises(ValueError):
            label_drift(["a"], [])


# A 30-word sentence: one appended word keeps word-3-gram Jaccard ~0.9,
# comfortably above the 0.8 MinHash threshold.
LONG_BASE = (
    "the film opens with a slow deliberate sequence that patiently introduces "
    "each character while the camera lingers on small details of the town "
    "its people and their daily rituals"
)


class TestNearDuplicateFlags:
    def test_paraphrase_above_threshold_is_flagged(self):
        near_dup = LONG_BASE + " lovingly"
        flags = near_duplicate_flags([near_dup], [LONG_BASE])
        assert flags == [True]

    def test_distinct_text_below_threshold_is_not_flagged(self):
        distinct = (
            "quarterly earnings for the semiconductor sector beat analyst "
            "expectations on strong datacenter demand this year"
        )
        flags = near_duplicate_flags([distinct], [LONG_BASE])
        assert flags == [False]

    def test_earlier_batch_texts_count_as_reference(self):
        flags = near_duplicate_flags(
            [LONG_BASE, LONG_BASE + " lovingly"], ["unrelated reference text here"]
        )
        assert flags == [False, True]

    def test_short_texts_fall_back_to_unigrams(self):
        # Fewer than 3 words: identical short texts must still match.
        flags = near_duplicate_flags(["hello world"], ["hello world"])
        assert flags == [True]

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError):
            near_duplicate_flags(["a"], [])


def _classifier_config():
    return ExperimentConfig(
        name="audit_fixture",
        student_type="transformers",
        dataset="fixture/tiny",
        dataset_config={
            "name": "default",
            "num_classes": 2,
            "text_field": "text",
            "label_field": "label",
        },
    )


def _sft_config():
    return ExperimentConfig(
        name="audit_sft_fixture",
        student_type="slm",
        dataset="fixture/tiny-sft",
        dataset_config={
            "name": "default",
            "text_field": "student_prompt",
            "label_field": "gold_answer",
        },
    )


class TestParseTeacherRecords:
    def test_parses_classification_articles(self):
        content = json.dumps(
            {"articles": [{"text": "great film", "label": 1}]}
        )
        records = parse_teacher_records(_classifier_config(), content)
        assert records == [{"text": "great film", "label": 1}]

    def test_garbage_content_raises(self):
        import pytest

        with pytest.raises(Exception):
            parse_teacher_records(_classifier_config(), "not json at all {{{")

    def test_sft_label_prefers_gold_answer(self):
        # _build_augmented_sft_rows (engine.py) trains on
        # gold_answer or teacher_response; reconstruction must match, or
        # e.g. B77 SLM runs ("Label: <class>" responses, "<class>" gold)
        # report label_drift = 1.0 on every cycle.
        content = json.dumps(
            {
                "records": [
                    {
                        "student_prompt": "I still have not received my card",
                        "teacher_response": "Label: card_arrival",
                        "gold_answer": "card_arrival",
                    }
                ]
            }
        )
        records = parse_teacher_records(_sft_config(), content)
        assert records == [
            {"text": "I still have not received my card", "label": "card_arrival"}
        ]

    def test_sft_label_falls_back_to_teacher_response(self):
        # Absent or empty gold_answer -> stripped teacher_response,
        # mirroring `record.get("gold_answer") or teacher_response`.
        content = json.dumps(
            {
                "records": [
                    {
                        "student_prompt": "How do I activate my card?",
                        "teacher_response": "  activate_my_card  ",
                    },
                    {
                        "student_prompt": "Card payment was declined",
                        "teacher_response": "declined_card_payment",
                        "gold_answer": "",
                    },
                ]
            }
        )
        records = parse_teacher_records(_sft_config(), content)
        assert [r["label"] for r in records] == [
            "activate_my_card",
            "declined_card_payment",
        ]


def _attempt(cycle, attempt_id, status="success", failure_type=None, **meta):
    return {
        "cycle": cycle,
        "attempt_id": attempt_id,
        "status": status,
        "failure_type": failure_type,
        "metadata": meta,
    }


class TestFailureSummary:
    def test_counts_requested_minus_accepted_plus_failed_attempts(self):
        attempts = [
            _attempt(1, "r:c1:a1", records_requested=4, records_parsed=3,
                     records_accepted=3),
            _attempt(1, "r:c1:a2", status="failed", failure_type="Timeout",
                     records_requested=8),
        ]
        summary = failure_summary(attempts, 1)
        assert isinstance(summary, FailureSummary)
        assert summary.records_requested == 12
        assert summary.records_accepted == 3
        assert summary.n_failed_attempts == 1
        # (12 - 3) + 1 failed attempt
        assert summary.n_fail == 10

    def test_other_cycles_are_ignored(self):
        attempts = [
            _attempt(1, "r:c1:a1", records_requested=4, records_accepted=4),
            _attempt(2, "r:c2:a1", records_requested=2, records_accepted=1),
        ]
        assert failure_summary(attempts, 2).n_fail == 1

    def test_rejected_record_exemplars_recovered_by_diffing(self):
        attempts = [
            _attempt(1, "r:c1:a1", records_requested=2, records_parsed=2,
                     records_accepted=1),
        ]
        raw = {
            "choices": [{"message": {"content": json.dumps({"articles": [
                {"text": "kept row", "label": 1},
                {"text": "rejected row", "label": 0},
            ]})}}]
        }
        summary = failure_summary(
            attempts,
            1,
            raw_responses={1: raw},
            accepted_texts={"kept row"},
            config=_classifier_config(),
        )
        rejected = [e for e in summary.exemplars if e.kind == "rejected_record"]
        assert len(rejected) == 1
        assert rejected[0].text == "rejected row"
        assert rejected[0].label == "0"
        assert rejected[0].attempt_id == "r:c1:a1"

    def test_attempt_failure_exemplar_carries_raw_snippet_when_unparseable(self):
        attempts = [
            _attempt(1, "r:c1:a1", status="failed",
                     failure_type="no_records_parsed", records_requested=4),
        ]
        raw = {"choices": [{"message": {"content": "garbled {{{ output"}}]}
        summary = failure_summary(
            attempts, 1, raw_responses={1: raw},
            accepted_texts=set(), config=_classifier_config(),
        )
        failures = [e for e in summary.exemplars if e.kind == "attempt_failure"]
        assert len(failures) == 1
        assert "garbled" in failures[0].text
        assert failures[0].failure_type == "no_records_parsed"

    def test_exemplars_capped_at_max(self):
        articles = [{"text": f"row {i}", "label": 0} for i in range(15)]
        attempts = [
            _attempt(1, "r:c1:a1", records_requested=15, records_parsed=15,
                     records_accepted=0),
        ]
        raw = {"choices": [{"message": {"content": json.dumps(
            {"articles": articles})}}]}
        summary = failure_summary(
            attempts, 1, raw_responses={1: raw},
            accepted_texts=set(), config=_classifier_config(),
        )
        assert len(summary.exemplars) == 10
