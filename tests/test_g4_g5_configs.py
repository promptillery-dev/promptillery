"""G4/G5 GSM8K configs parse and carry the generative-protocol knobs.

G4 decoders: label-free json SFT data, answer_extraction: number, 512-token
CoT generation, screening ON, per-cycle cap OFF (full-test heldout eval).
G5 pair: identical single-cycle student-only fine-tunes; only diff = kd block.
"""
import asyncio
import json

import yaml
import pytest
from datasets import Dataset, DatasetDict

from promptillery.config import ExperimentConfig
from promptillery.sft_materialize import materialize_sft_records
from promptillery.trainers.factory import TrainerFactory

G4_TRAINING = [
    ("examples/paper/G4_gsm8k_qwen3_4b.yaml", "Qwen/Qwen3-4B-Instruct-2507", True, 4),
    ("examples/paper/G4_gsm8k_smollm3.yaml", "HuggingFaceTB/SmolLM3-3B", True, 4),
    ("examples/paper/G4_gsm8k_ettin_decoder.yaml", "jhu-clsp/ettin-decoder-150m", False, 8),
    ("examples/paper/G4_gsm8k_gemma3_270m.yaml", "google/gemma-3-270m-it", False, 8),
]


@pytest.mark.parametrize("path,student,use_lora,expected_batch", G4_TRAINING)
def test_g4_training_configs(path, student, use_lora, expected_batch):
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.student == student
    assert cfg.student_type == "slm"
    assert cfg.cycles == [1, 5, 10]
    assert cfg.seed == 13
    assert "macro_f1" not in cfg.metrics and "exact_match" in cfg.metrics
    assert cfg.batch_size == expected_batch
    tc = cfg.trainer_config
    assert tc["answer_extraction"] == "number"
    assert tc["report_held_out_test"] is True
    assert tc["generation_max_new_tokens"] == 512
    assert tc["max_detailed_prediction_samples"] == 256
    assert "max_eval_generation_samples" not in tc
    assert "canonical_labels_path" not in tc
    assert tc["augmentation_screening"]["enabled"] is True
    assert tc["augmentation_screening"]["k"] == 3
    assert tc["generation_batch_size"] == expected_batch
    assert bool(tc.get("use_lora", False)) is use_lora


@pytest.mark.parametrize("path,_student,_lora,_batch", G4_TRAINING)
def test_g4_prompts_slice_few_shot(path, _student, _lora, _batch):
    # Unsliced few_shot_samples on near-unique gold answers = ~800 full
    # solutions in every augmentation prompt (utils.py:243 groups by label).
    raw = yaml.safe_load(open(path))
    assert "few_shot_samples[:6]" in raw["prompt"]


@pytest.mark.parametrize("path,_student,_lora,_batch", G4_TRAINING)
def test_g4_augmentation_wrap_matches_materialize_scaffold(path, _student, _lora, _batch):
    # Engine-side wrap for augmented rows (trainer_config.
    # augmentation_student_prompt_template) must stay byte-exact with the
    # seed/eval scaffold (materialize_sft.student_prompt_template) — this is
    # what makes augmented and seed/eval prompts train/eval-distribution
    # identical.
    materialize = yaml.safe_load(open("examples/paper/G4_gsm8k_materialize.yaml"))
    expected = materialize["trainer_config"]["materialize_sft"][
        "student_prompt_template"
    ]
    raw = yaml.safe_load(open(path))
    assert (
        raw["trainer_config"]["augmentation_student_prompt_template"] == expected
    )


def test_g4_materialize_config():
    cfg = ExperimentConfig.from_yaml("examples/paper/G4_gsm8k_materialize.yaml")
    ms = cfg.trainer_config["materialize_sft"]
    assert ms["gold_answer_field"] == "answer"
    assert "canonical_labels_field" not in ms
    assert cfg.sampling.enabled is False  # prep script did the 1K carve
    assert cfg.dataset_config.num_classes is None


def test_g5_pair_differs_only_by_kd_block():
    sft = yaml.safe_load(open("examples/paper/G5_gsm8k_kd_sft.yaml"))
    kd = yaml.safe_load(open("examples/paper/G5_gsm8k_kd_logit.yaml"))
    kd_block = kd["trainer_config"].pop("kd")
    assert kd_block == {
        "enabled": True,
        "teacher_model": "Qwen/Qwen3-32B",
        "teacher_dtype": "bfloat16",
        "temperature": 1.0,
        "ce_weight": 0.0,
        "teacher_device_map": "cuda:1",
    }
    sft.pop("name"), kd.pop("name")
    assert sft == kd  # arms identical apart from name + kd block


@pytest.mark.parametrize(
    "path", ["examples/paper/G5_gsm8k_kd_sft.yaml", "examples/paper/G5_gsm8k_kd_logit.yaml"]
)
def test_g5_configs_are_single_cycle_student_only(path):
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.cycles == 1
    assert cfg.policy_name == "student_only"
    assert cfg.trainer_config["answer_extraction"] == "number"
    assert cfg.trainer_config["report_held_out_test"] is True


ALL_CONFIGS = (
    [p for p, _, _, _ in G4_TRAINING]
    + ["examples/paper/G4_gsm8k_materialize.yaml", "examples/paper/G5_gsm8k_kd_sft.yaml",
       "examples/paper/G5_gsm8k_kd_logit.yaml"]
)


@pytest.mark.parametrize("path", ALL_CONFIGS)
def test_all_g4_g5_configs_resolve_a_trainer(path):
    # Same guard + concretize idiom as tests/test_g3_main_results_configs.py:
    # cycles: [1,5,10] is a list-valued ablation field, so concretize one member
    # (as AblationStudyRunner does) before asserting runnability.
    cfg = ExperimentConfig.from_yaml(path)
    if cfg.student_type not in TrainerFactory.get_available_types():
        pytest.skip(f"student_type {cfg.student_type!r} not installed")
    concrete = (
        cfg.generate_ablation_configs()[0] if cfg.has_list_parameters() else cfg
    )
    assert concrete.has_list_parameters() is False
    assert isinstance(concrete.cycles, int)
    assert concrete.student_type in TrainerFactory.get_available_types()


def test_gold_materialize_handles_free_text(tmp_path, monkeypatch):
    """--mode gold on label-free free-text data: records written, no
    canonical_labels.json. Adapted from the materialize_sft_records harness
    in tests/test_sft_online_augmentation.py (same monkeypatch + call shape)."""
    source = Dataset.from_dict(
        {
            "question": [
                "What is 2 + 2?",
                "What is 3 + 5?",
                "What is 10 - 4?",
            ],
            "answer": [
                "2 + 2 = 4\n#### 4",
                "3 + 5 = 8\n#### 8",
                "10 - 4 = 6\n#### 6",
            ],
            "answer_number": ["4", "8", "6"],
        }
    )
    dataset = DatasetDict({"train": source})
    monkeypatch.setattr(
        "promptillery.sft_materialize.load_materialization_dataset",
        lambda config: dataset,
    )

    config = ExperimentConfig(
        name="g4_gsm8k_materialize_gold_test",
        dataset="json",
        teacher="openrouter/openai/gpt-4.1",
        student="Qwen/Qwen3-4B-Instruct-2507",
        student_type="slm",
        seed=13,
        auto_modify_name=False,
        dataset_config={
            "name": "default",
            "text_field": "question",
            "label_field": "answer",
            # num_classes omitted: label-free generative task (Task 2).
        },
        trainer_config={
            "materialize_sft": {
                "stratify_max_samples": False,
                "gold_answer_field": "answer",
                "student_prompt_template": (
                    "Solve the grade-school math problem. Think step by step, "
                    "then give the final answer on its own line in the form "
                    '"#### <answer>".\nProblem: {{ question }}\nSolution:\n'
                ),
            }
        },
    )

    output_path = tmp_path / "train.jsonl"
    result = asyncio.run(
        materialize_sft_records(
            config=config,
            output_path=output_path,
            split="train",
            mode="gold",
        )
    )

    assert result["records"] == 3
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 3
    for record, row in zip(records, source):
        assert record["student_prompt"]
        assert record["gold_answer"] == row["answer"]
        # Pins the Jinja context columns: the raw dataset column ("question")
        # must be available to student_prompt_template, not just "text".
        assert row["question"] in record["student_prompt"]

    assert (tmp_path / "canonical_labels.json").exists() is False
