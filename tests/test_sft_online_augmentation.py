from types import SimpleNamespace

import pytest
from datasets import Dataset

from promptillery.engine import DistillationEngine
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
