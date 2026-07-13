"""G3 main-results configs parse and carry the 1K-subset protocol knobs.

Encoders read the dataset (raw HF or normalized JSON) with a 1000/80-20 sampling
block and cycles:[1,5,10] (ablation → runs at 1/5/10 cycles); decoders read
materialized gold SFT and classify via canonical-label scoring with the per-cycle
cap OFF (max_eval_generation_samples unset = full-test eval).
"""
import yaml
import pytest
from promptillery.config import ExperimentConfig
from promptillery.trainers.factory import TrainerFactory

G3_CONFIGS = {
    "examples/paper/G3_agnews_ettin_encoder.yaml": {
        "student_type": "transformers", "student": "jhu-clsp/ettin-encoder-150m",
        "num_classes": 4, "is_decoder": False, "hf_config": "default", "training": True},
    "examples/paper/G3_agnews_materialize.yaml": {
        "student_type": "slm", "student": "jhu-clsp/ettin-decoder-150m",
        "num_classes": 4, "is_decoder": False, "hf_config": "default", "materialize": True,
        "gold_answer_field": "label_text", "canonical_labels_field": "label_text"},
    "examples/paper/G3_agnews_ettin_decoder.yaml": {
        "student_type": "slm", "student": "jhu-clsp/ettin-decoder-150m",
        "num_classes": 4, "is_decoder": True, "hf_config": "default", "training": True},
    "examples/paper/G3_agnews_gemma3_270m.yaml": {
        "student_type": "slm", "student": "google/gemma-3-270m-it",
        "num_classes": 4, "is_decoder": True, "hf_config": "default", "training": True},
    "examples/paper/G3_sst2_ettin_encoder.yaml": {
        "student_type": "transformers", "student": "jhu-clsp/ettin-encoder-150m",
        "num_classes": 2, "is_decoder": False, "hf_config": "default", "training": True},
    "examples/paper/G3_sst2_materialize.yaml": {
        "student_type": "slm", "student": "jhu-clsp/ettin-decoder-150m",
        "num_classes": 2, "is_decoder": False, "hf_config": "default", "materialize": True,
        "gold_answer_field": "label_text", "canonical_labels_field": "label_text"},
    "examples/paper/G3_sst2_ettin_decoder.yaml": {
        "student_type": "slm", "student": "jhu-clsp/ettin-decoder-150m",
        "num_classes": 2, "is_decoder": True, "hf_config": "default", "training": True},
    "examples/paper/G3_sst2_gemma3_270m.yaml": {
        "student_type": "slm", "student": "google/gemma-3-270m-it",
        "num_classes": 2, "is_decoder": True, "hf_config": "default", "training": True},
    "examples/paper/G3_imdb_ettin_encoder.yaml": {
        "student_type": "transformers", "student": "jhu-clsp/ettin-encoder-150m",
        "num_classes": 2, "is_decoder": False, "hf_config": "plain_text", "training": True},
    "examples/paper/G3_imdb_materialize.yaml": {
        "student_type": "slm", "student": "jhu-clsp/ettin-decoder-150m",
        "num_classes": 2, "is_decoder": False, "hf_config": "plain_text", "materialize": True,
        "gold_answer_field": "label", "canonical_labels_field": None},
    "examples/paper/G3_imdb_ettin_decoder.yaml": {
        "student_type": "slm", "student": "jhu-clsp/ettin-decoder-150m",
        "num_classes": 2, "is_decoder": True, "hf_config": "default", "training": True},
    "examples/paper/G3_imdb_gemma3_270m.yaml": {
        "student_type": "slm", "student": "google/gemma-3-270m-it",
        "num_classes": 2, "is_decoder": True, "hf_config": "default", "training": True},
    "examples/paper/G3_yahoo_ettin_encoder.yaml": {
        "student_type": "transformers", "student": "jhu-clsp/ettin-encoder-150m",
        "num_classes": 10, "is_decoder": False, "hf_config": "default", "training": True},
    "examples/paper/G3_yahoo_materialize.yaml": {
        "student_type": "slm", "student": "jhu-clsp/ettin-decoder-150m",
        "num_classes": 10, "is_decoder": False, "hf_config": "default", "materialize": True,
        "gold_answer_field": "label_text", "canonical_labels_field": "label_text"},
    "examples/paper/G3_yahoo_ettin_decoder.yaml": {
        "student_type": "slm", "student": "jhu-clsp/ettin-decoder-150m",
        "num_classes": 10, "is_decoder": True, "hf_config": "default", "training": True},
    "examples/paper/G3_yahoo_gemma3_270m.yaml": {
        "student_type": "slm", "student": "google/gemma-3-270m-it",
        "num_classes": 10, "is_decoder": True, "hf_config": "default", "training": True},
    "examples/paper/G3_huffpost_ettin_encoder.yaml": {
        "student_type": "transformers", "student": "jhu-clsp/ettin-encoder-150m",
        "num_classes": 41, "is_decoder": False, "hf_config": "default", "training": True},
    "examples/paper/G3_huffpost_materialize.yaml": {
        "student_type": "slm", "student": "jhu-clsp/ettin-decoder-150m",
        "num_classes": 41, "is_decoder": False, "hf_config": "default", "materialize": True,
        "gold_answer_field": "label_text", "canonical_labels_field": "label_text"},
    "examples/paper/G3_huffpost_ettin_decoder.yaml": {
        "student_type": "slm", "student": "jhu-clsp/ettin-decoder-150m",
        "num_classes": 41, "is_decoder": True, "hf_config": "default", "training": True},
    "examples/paper/G3_huffpost_gemma3_270m.yaml": {
        "student_type": "slm", "student": "google/gemma-3-270m-it",
        "num_classes": 41, "is_decoder": True, "hf_config": "default", "training": True},
}


def _load(path):
    with open(path) as f:
        return ExperimentConfig(**yaml.safe_load(f))


@pytest.mark.parametrize("path,expected", list(G3_CONFIGS.items()))
def test_config_parses_and_carries_protocol(path, expected):
    cfg = _load(path)
    assert cfg.teacher == "openrouter/openai/gpt-4.1"
    assert cfg.seed == 13
    assert cfg.student == expected["student"]
    assert cfg.student_type == expected["student_type"]
    assert cfg.num_classes == expected["num_classes"]
    assert cfg.base_output_dir == "out/g3"
    if expected["is_decoder"]:
        # Decoders read pre-materialized (already-subsetted) gold SFT; no sampling.
        assert cfg.sampling.enabled is False
    else:
        # Encoders / materialize read the raw dataset and carve the 1K/80-20 split.
        assert cfg.sampling.enabled is True and cfg.sampling.sample_size == 1000
        assert cfg.sampling.stratify_column == "label"
    assert cfg.dataset_config.name == expected["hf_config"]
    if expected.get("training"):
        assert cfg.cycles == [1, 5, 10]
        assert cfg.get_list_parameters() == ["cycles"]  # only cycles sweeps
    if expected.get("materialize"):
        assert cfg.has_list_parameters() is False       # companion is a single run
        msft = cfg.trainer_config["materialize_sft"]
        assert msft["gold_answer_field"] == expected["gold_answer_field"]
        if expected["canonical_labels_field"] is None:
            assert "canonical_labels_field" not in msft
        else:
            assert msft["canonical_labels_field"] == expected["canonical_labels_field"]
    if expected["is_decoder"]:
        assert cfg.trainer_config["canonical_label_decoding"] == "score"
        assert cfg.trainer_config["report_held_out_test"] is True
        assert "canonical_labels_path" in cfg.trainer_config
        assert "max_eval_generation_samples" not in cfg.trainer_config  # full-test eval


@pytest.mark.parametrize("path", list(G3_CONFIGS))
def test_trainer_instantiates(path):
    cfg = _load(path)
    assert cfg.student_type in {"transformers", "slm"}

    # Copy of tests/test_banking77_configs.py's TrainerFactory guard: confirm the
    # configured student_type actually resolves to a registered trainer class
    # before treating the config as runnable (mirrors its
    # `if expected["student_type"] not in TrainerFactory.get_available_types()`
    # skip idiom rather than inventing a new TrainerFactory call).
    if cfg.student_type not in TrainerFactory.get_available_types():
        pytest.skip(f"student_type {cfg.student_type!r} not installed")

    # Training configs carry cycles: [1, 5, 10] -- a list-valued ablation field,
    # so the raw config is not directly runnable (DistillationEngine rejects any
    # list-valued parameter). Concretize a single member first, exactly like
    # AblationStudyRunner does before handing a config to a trainer. The
    # materialize companion has no list-valued field at all, so it is already a
    # single concrete run and is used as-is.
    concrete = cfg.generate_ablation_configs()[0] if cfg.has_list_parameters() else cfg

    assert concrete.has_list_parameters() is False
    assert isinstance(concrete.cycles, int)
    assert concrete.student_type in TrainerFactory.get_available_types()
