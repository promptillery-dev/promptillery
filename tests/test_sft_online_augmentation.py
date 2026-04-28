import asyncio
import json
from types import SimpleNamespace

import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from jinja2 import Template

from promptillery.engine import DistillationEngine, ensure_origin_columns
from promptillery.policy_controller import PolicyAction, PolicyController
from promptillery.sft_materialize import (
    _select_source_examples,
    _write_canonical_labels_artifact,
)
from promptillery.token_tracker import TokenTracker
from promptillery.trainers.causal_lm_sft_trainer import CausalLMSFTTrainer
from promptillery.utils import (
    extract_hard_negatives,
    extract_high_entropy_samples,
    format_classification_report,
)


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


def test_sft_generation_summary_uses_canonical_labels(tmp_path):
    trainer = CausalLMSFTTrainer.__new__(CausalLMSFTTrainer)
    trainer.trainer_config = {
        "answer_extraction": "canonical_label",
        "task_metric": "macro_f1",
        "canonical_labels": [
            "cash_withdrawal",
            "cash_deposit",
            "card_arrival",
        ],
    }
    trainer.gold_answer_field = "gold_answer"
    trainer.dataset = DatasetDict(
        {
            "validation": Dataset.from_dict(
                {
                    "student_prompt": ["a", "b"],
                    "gold_answer": ["cash_withdrawal", "cash_deposit"],
                }
            )
        }
    )
    trainer.cfg = SimpleNamespace(num_classes=3, batch_size=1)
    trainer.out_dir = tmp_path

    records = [
        {
            "split": "validation",
            "index": 0,
            "normalized_prediction": "cash_withdrawal",
            "normalized_gold": "cash_withdrawal",
            "exact_match": True,
            "generation_confidence": 0.9,
            "generation_entropy": 0.1,
        },
        {
            "split": "validation",
            "index": 1,
            "normalized_prediction": "please_call_support",
            "normalized_gold": "cash_deposit",
            "exact_match": False,
            "generation_confidence": 0.8,
            "generation_entropy": 0.7,
        },
    ]

    summary = trainer._generation_summary(records)

    assert summary["canonical_label_count"] == 3
    assert summary["observed_gold_label_count"] == 2
    assert summary["invalid_label_rate"] == 0.5
    assert summary["macro_f1"] == pytest.approx(0.5)
    assert summary["macro_f1_full_canonical"] == pytest.approx(1 / 3)
    assert records[1]["prediction_is_valid_label"] is False


def test_generation_eval_cache_invalidates_after_training_version():
    trainer = CausalLMSFTTrainer.__new__(CausalLMSFTTrainer)
    trainer._generation_eval_cache = {}
    trainer._generation_eval_version = 0
    calls = []

    def fake_generate(_trainer, split, sample_limit):
        calls.append((split, sample_limit))
        return [{"normalized_prediction": str(len(calls))}]

    trainer._generate_eval_records = fake_generate
    trainer._generation_summary = lambda records: {"exact_match": len(records)}
    model = object()
    hf_trainer = SimpleNamespace(model=model)

    trainer._run_generation_eval(
        hf_trainer,
        "validation",
        4,
        write_artifacts=False,
    )
    trainer._run_generation_eval(
        hf_trainer,
        "validation",
        4,
        write_artifacts=False,
    )
    trainer._generation_eval_version += 1
    trainer._run_generation_eval(
        hf_trainer,
        "validation",
        4,
        write_artifacts=False,
    )

    assert calls == [("validation", 4), ("validation", 4)]


def test_sft_detailed_predictions_feed_policy_prompt_context(tmp_path):
    trainer = CausalLMSFTTrainer.__new__(CausalLMSFTTrainer)
    trainer.trainer_config = {
        "answer_extraction": "canonical_label",
        "task_metric": "macro_f1",
        "canonical_labels": ["cash_withdrawal", "cash_deposit"],
    }
    trainer.gold_answer_field = "gold_answer"
    trainer.dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "student_prompt": ["withdraw cash", "deposit cash"],
                    "gold_answer": ["cash_withdrawal", "cash_deposit"],
                }
            )
        }
    )
    trainer.cfg = SimpleNamespace(num_classes=2, batch_size=1)
    trainer.out_dir = tmp_path
    records = [
        {
            "split": "train",
            "index": 0,
            "normalized_prediction": "cash_withdrawal",
            "normalized_gold": "cash_withdrawal",
            "exact_match": True,
            "generation_confidence": 0.95,
            "generation_entropy": 0.1,
        },
        {
            "split": "train",
            "index": 1,
            "normalized_prediction": "please_call_support",
            "normalized_gold": "cash_deposit",
            "exact_match": False,
            "generation_confidence": 0.9,
            "generation_entropy": 1.2,
        },
    ]
    summary = trainer._generation_summary(records)

    predictions = trainer._prediction_result_from_generation(records, summary)
    hard_negatives = extract_hard_negatives(
        trainer.dataset["train"],
        predictions,
        top_k=1,
        text_column="student_prompt",
        label_column="gold_answer",
    )
    high_entropy = extract_high_entropy_samples(
        trainer.dataset["train"],
        predictions,
        top_k=1,
        text_column="student_prompt",
        label_column="gold_answer",
    )
    report = format_classification_report(predictions)

    assert hard_negatives[0]["predicted_label"] == "please_call_support"
    assert hard_negatives[0]["label"] == "cash_deposit"
    assert high_entropy[0]["text"] == "deposit cash"
    assert "invalid_label_rate: 0.5000" in report
    assert "canonical_label_count: 2" in report


def test_policy_controller_manifest_exposes_fixed_linear_weights():
    engine = DistillationEngine.__new__(DistillationEngine)
    engine.cfg = SimpleNamespace(policy_name="frugalkd_p")
    engine.policy_controller = PolicyController(
        "frugalkd_p",
        lambda_cost=0.25,
        exploration_bonus=0.5,
        seed=13,
    )

    manifest = engine._policy_controller_manifest()

    assert manifest["controller_type"] == "fixed_linear_acquisition_scorer"
    assert manifest["lambda_cost"] == 0.25
    assert manifest["exploration_bonus"] == 0.5
    assert "eval_error_rate" in manifest["linear_weights"]


def test_student_only_policy_alias_selects_stop_action():
    controller = PolicyController("student_only", seed=13)
    actions = [
        PolicyAction(prompt_operator="coverage", teacher_tier="cheap", batch_size=8),
        PolicyAction(is_stop=True),
    ]

    choice = controller.select(
        {"tokens_remaining": 10_000},
        actions=actions,
        predicted_costs={"coverage:cheap:b8": {"total_tokens": 64}},
    )

    assert choice.action.is_stop
    assert choice.action_scores == {"STOP": 0.0}


def test_fixed_mixed_teacher_alternates_teacher_tiers():
    controller = PolicyController("fixed_mixed_teacher", seed=13)
    actions = [
        PolicyAction(prompt_operator="coverage", teacher_tier="cheap", batch_size=8),
        PolicyAction(prompt_operator="coverage", teacher_tier="strong", batch_size=8),
        PolicyAction(is_stop=True),
    ]
    predicted_costs = {
        "coverage:cheap:b8": {"total_tokens": 64},
        "coverage:strong:b8": {"total_tokens": 128},
    }

    first = controller.select(
        {"cycle": 0, "tokens_remaining": 10_000},
        actions=actions,
        predicted_costs=predicted_costs,
    )
    second = controller.select(
        {"cycle": 1, "tokens_remaining": 10_000},
        actions=actions,
        predicted_costs=predicted_costs,
    )

    assert first.action.teacher_tier == "cheap"
    assert second.action.teacher_tier == "strong"


def test_materialize_writes_canonical_label_artifact(tmp_path):
    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(names=["cash withdrawal", "card-arrival"]),
        }
    )
    source = Dataset.from_dict(
        {"text": ["a", "b"], "label": [0, 1]},
        features=features,
    )
    config = SimpleNamespace(
        dataset="mteb/banking77",
        dataset_subset=None,
    )

    artifact_path = _write_canonical_labels_artifact(
        config=config,
        output_path=tmp_path / "train.jsonl",
        source=source,
        label_field="label",
    )

    payload = json.loads((tmp_path / "canonical_labels.json").read_text())
    assert artifact_path == str(tmp_path / "canonical_labels.json")
    assert payload["canonical_label_count"] == 2
    assert payload["canonical_labels"] == ["cash withdrawal", "card-arrival"]
    assert payload["normalized_canonical_labels"] == [
        "cash_withdrawal",
        "card_arrival",
    ]


def test_select_source_examples_can_stratify_materialization_cap():
    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(names=["alpha", "beta", "gamma"]),
        }
    )
    dataset = Dataset.from_dict(
        {
            "text": [f"example-{index}" for index in range(30)],
            "label": [0] * 10 + [1] * 10 + [2] * 10,
        },
        features=features,
    )

    subset = _select_source_examples(
        dataset,
        max_samples=6,
        stratify_by="label",
        seed=13,
    )

    assert len(subset) == 6
    assert set(subset["label"]) == {0, 1, 2}


def test_select_source_examples_rejects_too_small_stratified_cap():
    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(names=["alpha", "beta", "gamma"]),
        }
    )
    dataset = Dataset.from_dict(
        {
            "text": [f"example-{index}" for index in range(9)],
            "label": [0] * 3 + [1] * 3 + [2] * 3,
        },
        features=features,
    )

    with pytest.raises(ValueError, match="too small to cover"):
        _select_source_examples(
            dataset,
            max_samples=2,
            stratify_by="label",
            seed=13,
        )


def test_origin_columns_preserve_materialized_sft_metadata():
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "student_prompt": ["prompt"],
                    "teacher_response": ["answer"],
                    "source_split": ["train"],
                    "source_index": [7],
                    "cycle": [2],
                }
            )
        }
    )

    prepared = ensure_origin_columns(dataset)

    assert prepared["train"]["source_split"] == ["train"]
    assert prepared["train"]["source_idx"] == [7]
    assert prepared["train"]["origin_cycle"] == [2]


def test_paper_mode_validates_seed_usage_even_when_not_debited():
    engine = DistillationEngine.__new__(DistillationEngine)
    engine.cfg = SimpleNamespace(
        name="seed-validation-test",
        student_type="causal_lm_sft",
        paper_mode=True,
        trainer_config={
            "budget_splits": [],
            "require_teacher_token_fields": True,
        },
        token_budget=100,
    )
    engine.dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "student_prompt": ["prompt"],
                    "teacher_response": ["answer"],
                    "teacher_input_tokens": [3],
                    "teacher_output_tokens": [2],
                    "teacher_total_tokens": [5],
                    "usage_estimated": [True],
                }
            )
        }
    )
    engine.token_tracker = TokenTracker(
        experiment_name="seed-validation-test",
        teacher_model="teacher/mock",
        quiet=True,
        token_budget=100,
    )
    engine.token_tracker.start_cycle(0)
    engine._external_sft_usage_recorded = False

    with pytest.raises(ValueError, match="usage_estimated=true"):
        engine._record_external_sft_token_usage()


def test_seed_usage_validation_can_skip_online_budget_debit():
    engine = DistillationEngine.__new__(DistillationEngine)
    engine.cfg = SimpleNamespace(
        name="seed-validation-test",
        student_type="causal_lm_sft",
        paper_mode=True,
        trainer_config={
            "budget_splits": [],
            "require_teacher_token_fields": True,
        },
        token_budget=100,
    )
    engine.dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "student_prompt": ["prompt"],
                    "teacher_response": ["answer"],
                    "teacher_input_tokens": [3],
                    "teacher_output_tokens": [2],
                    "teacher_total_tokens": [5],
                    "usage_estimated": [False],
                }
            )
        }
    )
    engine.token_tracker = TokenTracker(
        experiment_name="seed-validation-test",
        teacher_model="teacher/mock",
        quiet=True,
        token_budget=100,
    )
    engine.token_tracker.start_cycle(0)
    engine._external_sft_usage_recorded = False

    engine._record_external_sft_token_usage()

    assert engine._external_sft_usage_recorded is True
    assert engine.token_tracker.current_cycle_usage().total_tokens == 0


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
    assert attempts[0]["provider_reported_cost"]["total_tokens"] == 13
    assert attempts[0]["ledger_debit_cost"]["total_tokens"] == 13
    assert attempts[0]["ledger_debit_source"] == "provider_reported"
    assert attempts[0]["metadata"]["records_parsed"] == 1
    assert attempts[0]["metadata"]["records_accepted"] == 1
