"""Self-consistency screen for synthetic GSM8K-style augmentation records."""
import asyncio
import json
from types import SimpleNamespace

from datasets import Dataset, DatasetDict

from promptillery.engine import DistillationEngine, screen_votes
from promptillery.token_tracker import OperationType, TokenTracker


def test_screen_votes_majority_keeps():
    assert screen_votes("72", ["72", None], agreement=2) is True


def test_screen_votes_rejects_disagreement():
    assert screen_votes("72", ["71", "70"], agreement=2) is False


def test_screen_votes_rejects_missing_record_answer():
    assert screen_votes(None, ["72", "72"], agreement=2) is False


def test_screening_operation_type_exists():
    assert OperationType("screening") is OperationType.SCREENING


def _make_engine(trainer_config, recorded_ops):
    engine = DistillationEngine.__new__(DistillationEngine)
    engine.cfg = SimpleNamespace(
        student_type="slm",
        trainer_config=trainer_config,
        teacher_max_output_tokens=1024,
    )
    engine.token_tracker = SimpleNamespace(
        record_manual_usage=lambda usage, op: recorded_ops.append(op)
    )
    return engine


def _record(prompt, final_answer):
    solution = f"steps...\n#### {final_answer}"
    return {"student_prompt": prompt, "teacher_response": solution, "gold_answer": solution}


def test_disabled_screening_is_a_noop(monkeypatch):
    ops = []
    engine = _make_engine({}, ops)

    async def forbidden(**kwargs):
        raise AssertionError("no teacher call when screening is disabled")

    monkeypatch.setattr("promptillery.engine.acompletion", forbidden)
    records = [_record("p", 72)]
    kept, meta = asyncio.run(
        engine._screen_augmented_records(records, cycle=1, teacher_model="t")
    )
    assert kept == records and meta == {} and ops == []


SCREENING = {
    "augmentation_screening": {"enabled": True, "k": 3, "agreement": 2, "temperature": 0.7}
}


def test_screening_keeps_agreeing_rejects_disagreeing(monkeypatch):
    ops = []
    engine = _make_engine(SCREENING, ops)
    answers = iter(["#### 72", "#### 72", "#### 9", "#### 8"])

    async def fake(**kwargs):
        return {"choices": [{"message": {"content": next(answers)}}]}

    monkeypatch.setattr("promptillery.engine.acompletion", fake)
    records = [_record("good problem", 72), _record("bad problem", 7)]
    kept, meta = asyncio.run(
        engine._screen_augmented_records(records, cycle=1, teacher_model="t")
    )
    assert [r["student_prompt"] for r in kept] == ["good problem"]
    assert meta["records_kept"] == 1 and meta["records_rejected"] == 1
    assert len(ops) == 4 and all(op is OperationType.SCREENING for op in ops)


def test_screening_fails_closed_on_teacher_error(monkeypatch):
    ops = []
    engine = _make_engine(SCREENING, ops)

    async def broken(**kwargs):
        raise RuntimeError("teacher down")

    monkeypatch.setattr("promptillery.engine.acompletion", broken)
    kept, meta = asyncio.run(
        engine._screen_augmented_records([_record("p", 72)], cycle=1, teacher_model="t")
    )
    assert kept == [] and meta["records_failed"] == 1


def test_screening_uses_teacher_response_when_gold_answer_missing(monkeypatch):
    """The screen must judge the same answer the row builder trains on:
    gold_answer or teacher_response (mirrors _build_augmented_sft_rows)."""
    ops = []
    engine = _make_engine(SCREENING, ops)

    async def fake(**kwargs):
        return {"choices": [{"message": {"content": "#### 72"}}]}

    monkeypatch.setattr("promptillery.engine.acompletion", fake)
    record = {
        "student_prompt": "p",
        "teacher_response": "steps...\n#### 72",
        "gold_answer": None,
    }
    kept, meta = asyncio.run(
        engine._screen_augmented_records([record], cycle=1, teacher_model="t")
    )
    assert kept == [record]
    assert meta["records_kept"] == 1 and meta["records_rejected"] == 0


def test_augment_returns_screened_out_when_all_records_rejected(monkeypatch, tmp_path):
    """Integration: drive the real _augment and hit the total-rejection branch."""
    calls = {"augmentation": 0, "screening": 0}

    async def fake_acompletion(**kwargs):
        if "response_format" in kwargs:
            calls["augmentation"] += 1
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "records": [
                                        {
                                            "student_prompt": "novel problem",
                                            "teacher_response": "steps...\n#### 72",
                                            "gold_answer": "steps...\n#### 72",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                },
            }
        calls["screening"] += 1
        return {
            "choices": [{"message": {"content": "#### 999"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }

    monkeypatch.setattr("promptillery.engine.acompletion", fake_acompletion)

    engine = DistillationEngine.__new__(DistillationEngine)
    engine.cfg = SimpleNamespace(
        name="screening-test",
        teacher="teacher/mock",
        student_type="causal_lm_sft",
        teacher_max_output_tokens=32,
        augmentation_batch_size=1,
        trainer_config={
            "prompt_field": "student_prompt",
            "response_field": "teacher_response",
            "gold_answer_field": "gold_answer",
            "augmentation_screening": {
                "enabled": True,
                "k": 3,
                "agreement": 2,
                "temperature": 0.7,
            },
        },
        seed=13,
        token_budget=1000,
        budget_warning=None,
        budget_stop=True,
        policy_teacher_tiers={},
    )
    engine.dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "id": ["base/0"],
                    "student_prompt": ["base prompt"],
                    "teacher_response": ["base response"],
                    "gold_answer": ["base gold"],
                    "source_split": ["train"],
                    "source_idx": [0],
                    "origin_cycle": [0],
                    "teacher_input_tokens": [1],
                    "teacher_output_tokens": [1],
                    "teacher_total_tokens": [2],
                }
            )
        }
    )
    engine.out_dir = tmp_path
    engine.run_id = "screening-test-run"
    engine.augmentation_enabled = True
    engine.prompt_template = object()
    engine.prompt_vars = {}
    engine.cfg_vars = {}
    engine.token_tracker = TokenTracker(
        experiment_name="screening-test",
        teacher_model="teacher/mock",
        quiet=True,
        token_budget=1000,
        budget_stop=True,
    )
    engine.token_tracker.start_cycle(1)
    engine._attempt_counter = 0
    engine._render_augmentation_prompt = lambda sample_context, action: (
        "rendered prompt",
        {},
    )
    engine._estimate_teacher_call_tokens = lambda messages, budget, teacher_model: {
        "input_tokens": 10,
        "max_output_tokens": 32,
        "total_tokens": 42,
        "tokens_remaining": 1000,
        "token_budget": 1000,
        "allowed": True,
        "preflight_enforced": True,
        "estimator": "test",
        "teacher_model": teacher_model,
        "reason": None,
    }

    attempt_calls = []
    original_record_attempt = engine._record_teacher_attempt

    def capture_attempt(**kwargs):
        attempt_calls.append(kwargs)
        original_record_attempt(**kwargs)

    engine._record_teacher_attempt = capture_attempt

    result = asyncio.run(
        engine._augment(
            model=None,
            cycle=1,
            sample_context={"classification_report": "test"},
            budget_before={
                "token_budget": 1000,
                "tokens_remaining": 1000,
                "spent_usd": None,
            },
            decision_id="decision-1",
        )
    )

    assert result["action_name"] == "augment_screened_out"
    assert result["metadata"]["records_added"] == 0
    assert result["metadata"]["records_screened"] == 1
    assert result["metadata"]["records_rejected"] == 1
    # One augmentation call, then k-1 = 2 re-solves for the single record.
    assert calls == {"augmentation": 1, "screening": 2}
    # Nothing was appended to the training data.
    assert len(engine.dataset["train"]) == 1

    assert len(attempt_calls) == 1
    assert attempt_calls[0]["status"] == "screened_out"
    assert attempt_calls[0]["failure_type"] == "all_records_screened_out"
    assert attempt_calls[0]["metadata"]["records_added"] == 0
    assert attempt_calls[0]["metadata"]["records_rejected"] == 1

    # The real _record_teacher_attempt ran too: the audit row is on disk.
    attempts = [
        json.loads(line)
        for line in (tmp_path / "teacher_attempts.jsonl").read_text().splitlines()
    ]
    assert len(attempts) == 1
    assert attempts[0]["status"] == "screened_out"
    assert attempts[0]["failure_type"] == "all_records_screened_out"
