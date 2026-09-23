"""Tests for load_run_data: primary dataset_cycle_* path (issue #7)."""

import json

import pytest
import yaml
from datasets import Dataset, DatasetDict

from promptillery.audit import ReconstructionError, load_run_data, parse_teacher_records
from promptillery.config import ExperimentConfig

from conftest import ATTEMPTS, AUG_CYCLE_1, AUG_CYCLE_2, GOLD_TEST, REJECTED_CYCLE_1, SEED_ROWS


class TestLoadRunDataPrimary:
    def test_groups_augmented_rows_by_origin_cycle(self, audit_run_dir):
        data = load_run_data(audit_run_dir())
        assert data.reconstructed is False
        assert sorted(data.augmented) == [1, 2]
        assert len(data.augmented[1]) == len(AUG_CYCLE_1)
        assert len(data.augmented[2]) == len(AUG_CYCLE_2)
        assert data.augmented[2][0]["text"] == AUG_CYCLE_2[0][0]

    def test_labels_are_mapped_to_names(self, audit_run_dir):
        data = load_run_data(audit_run_dir())
        assert data.label_names == {0: "negative", 1: "positive"}
        assert data.augmented[1][0]["label"] == "positive"
        assert {row["label"] for row in data.seed_rows} == {
            "negative",
            "positive",
        }

    def test_seed_rows_exclude_augmented(self, audit_run_dir):
        data = load_run_data(audit_run_dir())
        assert len(data.seed_rows) == len(SEED_ROWS)

    def test_gold_splits_loaded(self, audit_run_dir):
        data = load_run_data(audit_run_dir())
        assert len(data.gold_test) == len(GOLD_TEST)
        assert data.gold_test[0]["label"] == "positive"

    def test_attempts_and_raw_responses_loaded(self, audit_run_dir):
        data = load_run_data(audit_run_dir())
        assert len(data.attempts) == 3
        assert sorted(data.raw_responses) == [1, 2]

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_run_data(tmp_path)

    def test_no_artifacts_raises(self, audit_run_dir):
        run_dir = audit_run_dir(with_datasets=False, with_raw=False)
        with pytest.raises(FileNotFoundError):
            load_run_data(run_dir)


class TestLoadRunDataReconstruction:
    def test_reconstructs_when_datasets_missing(
        self, audit_run_dir, audit_seed_loader
    ):
        run_dir = audit_run_dir(with_datasets=False)
        data = load_run_data(run_dir, dataset_loader=audit_seed_loader)
        assert data.reconstructed is True
        assert sorted(data.augmented) == [1, 2]
        assert len(data.augmented[1]) == 3  # 4 parsed, prefix-truncated to 3
        assert len(data.augmented[2]) == 2
        # The truncated 4th record must not appear.
        texts = [row["text"] for row in data.augmented[1]]
        assert REJECTED_CYCLE_1[0] not in texts

    def test_reconstruction_matches_primary_path_exactly(
        self, audit_run_dir, audit_seed_loader
    ):
        run_dir = audit_run_dir()  # has BOTH dataset_cycle_* and raw responses
        primary = load_run_data(run_dir)
        recon = load_run_data(
            run_dir, dataset_loader=audit_seed_loader, force_reconstruction=True
        )
        assert recon.reconstructed is True
        assert primary.augmented == recon.augmented
        assert primary.seed_rows == recon.seed_rows

    def test_count_mismatch_hard_fails(self, audit_run_dir, audit_seed_loader):
        run_dir = audit_run_dir(with_datasets=False)
        # Claim more accepted rows than the raw response can yield.
        broken = [dict(a) for a in ATTEMPTS]
        broken[0] = dict(
            broken[0],
            metadata=dict(broken[0]["metadata"], records_accepted=99),
        )
        with (run_dir / "teacher_attempts.jsonl").open("w") as f:
            for row in broken:
                f.write(json.dumps(row) + "\n")
        with pytest.raises(ReconstructionError):
            load_run_data(run_dir, dataset_loader=audit_seed_loader)

    def test_screened_runs_refuse_reconstruction(
        self, audit_run_dir, audit_seed_loader
    ):
        run_dir = audit_run_dir(with_datasets=False)
        screened = [dict(a) for a in ATTEMPTS]
        screened[0] = dict(
            screened[0],
            metadata=dict(
                screened[0]["metadata"],
                screening_mode="self_consistency",
                records_rejected=1,
            ),
        )
        with (run_dir / "teacher_attempts.jsonl").open("w") as f:
            for row in screened:
                f.write(json.dumps(row) + "\n")
        with pytest.raises(ReconstructionError):
            load_run_data(run_dir, dataset_loader=audit_seed_loader)

    def test_unparseable_response_with_zero_accepted_is_skipped(
        self, audit_run_dir, audit_seed_loader
    ):
        run_dir = audit_run_dir(with_datasets=False)
        (run_dir / "teacher_response_cycle_3.json").write_text(
            json.dumps({"choices": [{"message": {"content": "garbled {{{"}}]}),
            encoding="utf-8",
        )
        # No cycle-3 attempts exist -> expected accepted = 0 -> skip quietly.
        data = load_run_data(run_dir, dataset_loader=audit_seed_loader)
        assert 3 not in data.augmented

    def test_screening_enabled_in_config_refuses_reconstruction(
        self, audit_run_dir, audit_seed_loader
    ):
        # Screening can reject zero records: attempts metadata alone is not
        # a sufficient signal, the run config itself must hard-fail.
        run_dir = audit_run_dir(with_datasets=False)
        config_path = run_dir / "experiment_config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["trainer_config"] = {
            "augmentation_screening": {"enabled": True}
        }
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(ReconstructionError):
            load_run_data(run_dir, dataset_loader=audit_seed_loader)

    def test_screening_mode_metadata_alone_refuses_reconstruction(
        self, audit_run_dir, audit_seed_loader
    ):
        # screening_mode present with zero rejected/failed must still fail.
        run_dir = audit_run_dir(with_datasets=False)
        screened = [dict(a) for a in ATTEMPTS]
        screened[0] = dict(
            screened[0],
            metadata=dict(
                screened[0]["metadata"], screening_mode="self_consistency"
            ),
        )
        with (run_dir / "teacher_attempts.jsonl").open("w") as f:
            for row in screened:
                f.write(json.dumps(row) + "\n")
        with pytest.raises(ReconstructionError):
            load_run_data(run_dir, dataset_loader=audit_seed_loader)

    def test_missing_raw_response_for_accepted_cycle_hard_fails(
        self, audit_run_dir, audit_seed_loader
    ):
        run_dir = audit_run_dir(with_datasets=False)
        # Cycle 2 has records_accepted > 0 but its raw response is gone:
        # the cycle must not silently vanish from the reconstruction.
        (run_dir / "teacher_response_cycle_2.json").unlink()
        with pytest.raises(ReconstructionError):
            load_run_data(run_dir, dataset_loader=audit_seed_loader)


# --- SFT reconstruction acceptance-filter parity with the engine (final-review) ---

SFT_FIXTURE_CONFIG = {
    "name": "audit_sft_recon_fixture",
    "teacher": "openrouter/openai/gpt-4.1",
    "student": "some-org/tiny-slm",
    "student_type": "slm",
    "dataset": "fixture/tiny-sft",
    "dataset_config": {
        "name": "default",
        "text_field": "student_prompt",
        "label_field": "gold_answer",
    },
    "auto_modify_name": False,
    "cycles": 1,
    "seed": 13,
    "sampling": {"enabled": False},
}


def _sft_seed_dataset(config):
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "student_prompt": ["seed prompt"],
                    "gold_answer": ["seed gold"],
                    "source_split": ["train"],
                    "source_idx": [-1],
                    "origin_cycle": [0],
                }
            )
        }
    )


def _build_sft_run_dir(tmp_path, records, records_accepted):
    """A minimal reconstruction-only SFT run directory (no dataset_cycle_*)."""
    run_dir = tmp_path / SFT_FIXTURE_CONFIG["name"]
    run_dir.mkdir()
    (run_dir / "experiment_config.yaml").write_text(
        yaml.safe_dump(SFT_FIXTURE_CONFIG), encoding="utf-8"
    )
    attempts = [
        {
            "cycle": 1,
            "attempt_id": f"{SFT_FIXTURE_CONFIG['name']}:c1:a1",
            "status": "success",
            "failure_type": None,
            "metadata": {
                "records_requested": len(records),
                "records_parsed": len(records),
                "records_accepted": records_accepted,
            },
        }
    ]
    with (run_dir / "teacher_attempts.jsonl").open("w") as f:
        for row in attempts:
            f.write(json.dumps(row) + "\n")
    content = json.dumps({"records": records})
    (run_dir / "teacher_response_cycle_1.json").write_text(
        json.dumps(
            {
                "choices": [{"message": {"content": content}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir


class TestSFTReconstructionDropParity:
    """The engine drops SFT records when student_prompt or teacher_response
    is empty (engine.py:1397-1400, _build_augmented_sft_rows) — not when the
    trained-on label (gold_answer or teacher_response) is empty. A record
    with a non-empty gold_answer but a whitespace-only teacher_response is
    dropped by the engine yet was kept as a phantom row by the audit's old
    text/label-based filter."""

    def test_drops_whitespace_only_teacher_response_like_the_engine(
        self, tmp_path
    ):
        records = [
            {"student_prompt": "Q1", "teacher_response": "R1", "gold_answer": "G1"},
            {
                # Non-empty gold_answer, whitespace-only teacher_response:
                # the engine drops this record; the old audit filter kept it.
                "student_prompt": "Q2",
                "teacher_response": "   ",
                "gold_answer": "G2",
            },
            {"student_prompt": "Q3", "teacher_response": "R3", "gold_answer": "G3"},
        ]
        run_dir = _build_sft_run_dir(tmp_path, records, records_accepted=2)
        data = load_run_data(run_dir, dataset_loader=_sft_seed_dataset)
        assert data.augmented[1] == [
            {"text": "Q1", "label": "G1"},
            {"text": "Q3", "label": "G3"},
        ]


class TestParseTeacherRecordsWrapTemplate:
    """augmentation_student_prompt_template must wrap parse_teacher_records'
    text output identically to _build_augmented_sft_rows (engine.py), so
    reconstructed rows stay byte-exact with what the engine trained on."""

    def test_sft_wrap_template_renders_question_variable(self):
        config = ExperimentConfig(
            name="audit_sft_wrap_fixture",
            student_type="slm",
            dataset="fixture/tiny-sft",
            dataset_config={
                "name": "default",
                "text_field": "student_prompt",
                "label_field": "gold_answer",
            },
            trainer_config={
                "augmentation_student_prompt_template": (
                    "Solve it.\nProblem: {{ question }}\nSolution:"
                )
            },
        )
        content = json.dumps(
            {
                "records": [
                    {
                        "student_prompt": "Janet has 3 apples.",
                        "teacher_response": "She has 3.",
                        "gold_answer": "3",
                    }
                ]
            }
        )
        records = parse_teacher_records(config, content)
        assert records == [
            {
                "text": "Solve it.\nProblem: Janet has 3 apples.\nSolution:",
                "label": "3",
            }
        ]
