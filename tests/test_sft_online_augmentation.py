import asyncio
import json
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict
from jinja2 import Template

from promptillery.engine import DistillationEngine
from promptillery.policy_controller import PolicyAction
from promptillery.token_tracker import TokenTracker
from promptillery.trainers.causal_lm_sft_trainer import CausalLMSFTTrainer


def test_sft_text_builder_uses_response_when_text_field_is_prompt():
    trainer = CausalLMSFTTrainer.__new__(CausalLMSFTTrainer)
    trainer.trainer_config = {}
    trainer.explicit_text_field = False
    trainer.text_field = "student_prompt"
    trainer.prompt_field = "student_prompt"
    trainer.response_field = "teacher_response"
    trainer.add_eos_token = False

    texts = trainer._build_texts(
        {
            "student_prompt": ["Classify: hello"],
            "teacher_response": ["greeting"],
        }
    )

    assert texts == [
        "### Instruction:\nClassify: hello\n\n### Response:\ngreeting",
    ]


def test_sft_text_builder_preserves_explicit_preformatted_text():
    trainer = CausalLMSFTTrainer.__new__(CausalLMSFTTrainer)
    trainer.trainer_config = {"text_field": "text"}
    trainer.explicit_text_field = True
    trainer.text_field = "text"
    trainer.prompt_field = "student_prompt"
    trainer.response_field = "teacher_response"
    trainer.add_eos_token = False

    texts = trainer._build_texts(
        {
            "text": ["already formatted"],
            "student_prompt": ["Classify: hello"],
            "teacher_response": ["greeting"],
        }
    )

    assert texts == ["already formatted"]


def test_augmented_sft_rows_preserve_schema_fields():
    engine = DistillationEngine.__new__(DistillationEngine)
    engine.cfg = SimpleNamespace(
        trainer_config={
            "prompt_field": "student_prompt",
            "response_field": "teacher_response",
            "gold_answer_field": "gold_answer",
        },
        seed=13,
    )
    engine.run_id = "run-test"
    ds = Dataset.from_dict(
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

    rows = engine._build_augmented_sft_rows(
        [
            {
                "student_prompt": "new prompt",
                "teacher_response": "new response",
                "gold_answer": "new gold",
            }
        ],
        ds,
        cycle=1,
        teacher_model="teacher/mock",
        teacher_tier="cheap",
        prompt_operator="coverage",
    )

    assert rows == [
        {
            "id": "run-test/augmented/1/0",
            "student_prompt": "new prompt",
            "teacher_response": "new response",
            "gold_answer": "new gold",
            "source_split": "augmented",
            "source_idx": -1,
            "origin_cycle": 1,
            "teacher_input_tokens": 0,
            "teacher_output_tokens": 0,
            "teacher_total_tokens": 0,
        }
    ]


def test_augmented_sft_rows_require_sft_schema():
    engine = DistillationEngine.__new__(DistillationEngine)
    engine.cfg = SimpleNamespace(
        trainer_config={
            "prompt_field": "student_prompt",
            "response_field": "teacher_response",
        },
        seed=13,
    )
    engine.run_id = "run-test"
    ds = Dataset.from_dict({"text": ["hello"], "label": [0]})

    with pytest.raises(ValueError, match="student_prompt.*teacher_response"):
        engine._build_augmented_sft_rows(
            [{"student_prompt": "p", "teacher_response": "r"}],
            ds,
            cycle=1,
            teacher_model="teacher/mock",
            teacher_tier="cheap",
            prompt_operator="coverage",
        )


def test_same_count_cap_limits_rendered_policy_batch_size():
    engine = DistillationEngine.__new__(DistillationEngine)
    engine.cfg = SimpleNamespace(
        teacher="teacher/mock",
        augmentation_batch_size=8,
        synthetic_record_budget=2,
        policy_teacher_tiers={},
    )
    engine.dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": ["base", "already synthetic"],
                    "label": [0, 1],
                    "source_split": ["train", "augmented"],
                }
            )
        }
    )
    engine.prompt_template = Template("make {{ augmentation_batch_size }} records")
    engine.cfg_vars = {}
    engine.prompt_vars = {}

    action = PolicyAction(
        prompt_operator="coverage",
        teacher_tier="cheap",
        batch_size=4,
    )
    prompt, env = engine._render_augmentation_prompt({}, action)

    assert env["augmentation_batch_size"] == 1
    assert prompt == "make 1 records"

    seen_prompts = []
    engine._estimate_teacher_call_tokens = lambda messages, budget, teacher_model: (
        seen_prompts.append(messages[0]["content"])
        or {"total_tokens": 10, "allowed": True}
    )

    costs = engine._policy_predicted_costs([action], {}, {"token_budget": 100})

    assert seen_prompts == ["make 1 records"]
    assert costs[action.name]["total_tokens"] == 10


def test_same_count_cap_exhaustion_skips_teacher_call(monkeypatch):
    called = False

    async def fake_acompletion(**kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr("promptillery.engine.acompletion", fake_acompletion)

    engine = DistillationEngine.__new__(DistillationEngine)
    engine.cfg = SimpleNamespace(
        teacher="teacher/mock",
        augmentation_batch_size=4,
        synthetic_record_budget=1,
        policy_teacher_tiers={},
    )
    engine.dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": ["already synthetic"],
                    "label": [0],
                    "source_split": ["augmented"],
                }
            )
        }
    )
    engine.augmentation_enabled = True
    engine.prompt_template = object()

    result = asyncio.run(
        engine._augment(
            model=None,
            cycle=1,
            sample_context={"classification_report": "test"},
            budget_before={"token_budget": 100, "tokens_remaining": 100},
            decision_id="decision-1",
        )
    )

    assert result["action_name"] == "augment_empty"
    assert result["metadata"]["skip_reason"] == "synthetic_record_budget_exhausted"
    assert result["metadata"]["records_requested"] == 0
    assert not called


def test_online_sft_augmentation_appends_teacher_records(monkeypatch, tmp_path):
    async def fake_acompletion(**kwargs):
        assert kwargs["response_format"].__name__ == "AugmentedSFTResponse"
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "records": [
                                    {
                                        "student_prompt": "new banking request",
                                        "teacher_response": "cash_withdrawal",
                                        "gold_answer": "cash_withdrawal",
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

    monkeypatch.setattr("promptillery.engine.acompletion", fake_acompletion)

    engine = DistillationEngine.__new__(DistillationEngine)
    engine.cfg = SimpleNamespace(
        name="online-sft-test",
        teacher="teacher/mock",
        student_type="causal_lm_sft",
        teacher_max_output_tokens=32,
        augmentation_batch_size=1,
        trainer_config={
            "prompt_field": "student_prompt",
            "response_field": "teacher_response",
            "gold_answer_field": "gold_answer",
        },
        seed=13,
        token_budget=100,
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
    engine.run_id = "online-sft-test-run"
    engine.augmentation_enabled = True
    engine.prompt_template = object()
    engine.prompt_vars = {}
    engine.cfg_vars = {}
    engine.token_tracker = TokenTracker(
        experiment_name="online-sft-test",
        teacher_model="teacher/mock",
        quiet=True,
        token_budget=100,
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
        "tokens_remaining": 100,
        "token_budget": 100,
        "allowed": True,
        "preflight_enforced": True,
        "estimator": "test",
        "teacher_model": teacher_model,
        "reason": None,
    }

    result = asyncio.run(
        engine._augment(
            model=None,
            cycle=1,
            sample_context={"classification_report": "test"},
            budget_before={
                "token_budget": 100,
                "tokens_remaining": 100,
                "spent_usd": None,
            },
            decision_id="decision-1",
        )
    )

    assert result["action_name"] == "augment"
    assert result["metadata"]["records_added"] == 1
    assert len(engine.dataset["train"]) == 2
    added = engine.dataset["train"][-1]
    assert added["student_prompt"] == "new banking request"
    assert added["teacher_response"] == "cash_withdrawal"
    assert added["source_split"] == "augmented"

    attempts = [
        json.loads(line)
        for line in (tmp_path / "teacher_attempts.jsonl").read_text().splitlines()
    ]
    assert attempts[0]["status"] == "success"
    assert attempts[0]["decision_id"] == "decision-1"
    assert attempts[0]["metadata"]["records_parsed"] == 1
    assert attempts[0]["metadata"]["records_accepted"] == 1
