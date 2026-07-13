import pytest
import yaml

from promptillery.config import ExperimentConfig
from promptillery.reproducibility import (
    build_reproducibility_manifest,
    dataset_load_kwargs,
    model_revision_kwargs,
    tokenizer_revision_kwargs,
)


def test_dataset_revision_is_added_to_load_kwargs():
    config = ExperimentConfig(
        name="revision-test",
        dataset="mteb/banking77",
        dataset_revision="abc123",
        dataset_kwargs={"trust_remote_code": True},
    )

    assert dataset_load_kwargs(config) == {
        "revision": "abc123",
        "trust_remote_code": True,
    }


def test_dataset_revision_conflict_fails_fast():
    config = ExperimentConfig(
        name="revision-conflict",
        dataset_revision="abc123",
        dataset_kwargs={"revision": "def456"},
    )

    with pytest.raises(ValueError, match="dataset_revision conflicts"):
        dataset_load_kwargs(config)


def test_student_revision_kwargs_split_model_and_tokenizer():
    config = ExperimentConfig(
        name="student-revision",
        student_revision="model-sha",
        tokenizer_revision="tokenizer-sha",
    )

    assert model_revision_kwargs(config) == {"revision": "model-sha"}
    assert tokenizer_revision_kwargs(config) == {"revision": "tokenizer-sha"}


def test_tokenizer_revision_defaults_to_student_revision():
    config = ExperimentConfig(
        name="tokenizer-fallback",
        student_revision="model-sha",
    )

    assert tokenizer_revision_kwargs(config) == {"revision": "model-sha"}


def test_reproducibility_manifest_records_config_provenance(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "provenance-test",
                "dataset_revision": "dataset-sha",
                "student_revision": "student-sha",
                "teacher_revision": "teacher-date",
            }
        ),
        encoding="utf-8",
    )
    config = ExperimentConfig.from_yaml(str(config_path))
    run_copy = tmp_path / "experiment_config.yaml"
    run_copy.write_text(yaml.safe_dump(config.model_dump()), encoding="utf-8")

    manifest = build_reproducibility_manifest(config=config, artifact_dir=tmp_path)

    assert manifest["schema_version"] == 1
    assert manifest["dataset_source"]["revision"] == "dataset-sha"
    assert manifest["models"]["student"]["revision"] == "student-sha"
    assert manifest["models"]["teacher"]["revision"] == "teacher-date"
    assert manifest["config_provenance"]["source_path"] == str(config_path)
    assert manifest["config_provenance"]["source_sha256"]
    assert manifest["config_provenance"]["resolved_sha256"]
    assert manifest["config_provenance"]["run_copy_path"] == str(run_copy)
    assert manifest["config_provenance"]["run_copy_sha256"]
    assert "python" in manifest["runtime"]
    assert "torch" in manifest["hardware"]
