"""Banking77 cross-architecture hub configs parse and carry the G2 knobs.

Each config drives one student architecture through the same protocol: teacher
GPT-4.1 via OpenRouter, a single monotonic 10-cycle run (read off at cycles
{1,5,10}), Banking77's 77 intents. Every student points at a shared teacher
test-label file so evaluate() reports fidelity.

Materialization is NOT inline on the training configs (full alignment with the
the shared companion config
examples/paper/G2_banking77_materialize.yaml is the single source that produces the
decoders' train/validation SFT files and the shared teacher test-label file.
See test_materialize_companion_config below.
"""

import pytest
import yaml

from promptillery.config import ExperimentConfig
from promptillery.trainers.factory import TrainerFactory

BANKING77_CONFIGS = {
    "examples/paper/G2_banking77_modernbert_base.yaml": {
        "student_type": "transformers",
        "student": "answerdotai/ModernBERT-base",
        "fidelity": True,
        "has_test_data_file": False,
        "loads_script_dataset": True,
    },
    "examples/paper/G2_banking77_ettin_encoder.yaml": {
        "student_type": "transformers",
        "student": "jhu-clsp/ettin-encoder-150m",
        "fidelity": True,
        "trust_remote_code": True,
        "has_test_data_file": False,
        "loads_script_dataset": True,
    },
    "examples/paper/G2_banking77_roberta_base.yaml": {
        "student_type": "transformers",
        "student": "FacebookAI/roberta-base",
        "fidelity": True,
        "has_test_data_file": False,
        "loads_script_dataset": True,
    },
    "examples/paper/G2_banking77_ettin_decoder.yaml": {
        "student_type": "slm",
        "student": "jhu-clsp/ettin-decoder-150m",
        "fidelity": True,
        "trust_remote_code": True,
        "has_test_data_file": True,
        "loads_script_dataset": False,
        "canonical_labels_path": "out/g2/canonical_labels.json",
    },
    "examples/paper/G2_banking77_qwen3_4b_lora.yaml": {
        "student_type": "slm",
        "student": "Qwen/Qwen3-4B-Instruct-2507",
        "fidelity": True,
        "has_test_data_file": True,
        "loads_script_dataset": False,
        "canonical_labels_path": "out/g2/canonical_labels.json",
    },
    "examples/paper/G2_banking77_smollm3.yaml": {
        "student_type": "slm",
        "student": "HuggingFaceTB/SmolLM3-3B",
        "fidelity": True,
        "has_test_data_file": True,
        "loads_script_dataset": False,
        "canonical_labels_path": "out/g2/canonical_labels.json",
    },
    "examples/paper/G2_banking77_fasttext.yaml": {
        # FastText fills its reference row's fidelity cell too: it shares the
        # classifier fidelity seam with the encoders.
        "student_type": "fasttext",
        "student": "fasttext",
        "fidelity": True,
        "has_test_data_file": False,
        "loads_script_dataset": True,
    },
}


@pytest.mark.parametrize(
    "path,expected",
    list(BANKING77_CONFIGS.items()),
    ids=list(BANKING77_CONFIGS),
)
def test_banking77_config_is_valid(path, expected):
    if expected["student_type"] not in TrainerFactory.get_available_types():
        # student_type gated behind an optional dependency (e.g. fasttext) that
        # is not installed here; ExperimentConfig validation would reject it.
        # Still guard the config's content via a raw parse so its knobs can't
        # silently drift, then skip the full round-trip.
        raw = yaml.safe_load(open(path))
        assert raw["student"] == expected["student"]
        assert raw["teacher"] == "openrouter/openai/gpt-4.1"
        assert raw["cycles"] == 10
        assert raw["dataset_config"]["num_classes"] == 77
        assert raw["prompt"]
        assert raw["require_validation_split"] is True
        assert raw["teacher_max_output_tokens"]
        if expected["loads_script_dataset"]:
            assert raw["dataset_kwargs"]["trust_remote_code"] is True
        tc = raw.get("trainer_config", {})
        assert tc.get("report_held_out_test") is True
        # Materialization moved to the companion config; training configs no
        # longer carry an inline materialize_sft block.
        assert "materialize_sft" not in tc
        assert ("fidelity" in tc) == expected["fidelity"]
        pytest.skip(f"student_type {expected['student_type']!r} not installed")

    cfg = ExperimentConfig.from_yaml(path)

    assert cfg.student == expected["student"]
    assert cfg.student_type == expected["student_type"]
    # Teacher is GPT-4.1 routed through OpenRouter (config + env only).
    assert cfg.teacher == "openrouter/openai/gpt-4.1"
    # One monotonic run to 10 cycles; the paper reads off {1,5,10}.
    assert cfg.cycles == 10
    assert cfg.num_classes == 77
    # Online active-learning loop (teacher adapts to each student's errors).
    assert cfg.prompt

    # Leakage-free selection: carve a validation split for cycle selection and
    # report finals on a clean held-out test pass.
    assert cfg.require_validation_split is True
    assert cfg.trainer_config["report_held_out_test"] is True

    # Materialization settings live in the companion config
    # (examples/paper/G2_banking77_materialize.yaml), not inline on every training config.
    # teacher_max_output_tokens stays because it is the active-learning-loop
    # teacher cap; dropping it would make the augmentation preflight mask every
    # teacher call (token_budget is set -> engine.py:959-961), silently killing
    # online augmentation.
    assert "materialize_sft" not in (cfg.trainer_config or {})
    assert cfg.teacher_max_output_tokens

    if expected["fidelity"]:
        fidelity = cfg.trainer_config["fidelity"]
        assert fidelity["teacher_labels_path"]
        assert fidelity["split"] == "test"
    else:
        assert "fidelity" not in (cfg.trainer_config or {})

    # Decoder students need a materialized test split so fidelity/held-out fire;
    # encoders read PolyAI/banking77's test split directly.
    if expected["has_test_data_file"]:
        assert cfg.dataset_kwargs["data_files"]["test"]

    # Decoders load the canonical Banking77 label schema the materializer writes
    # next to the SFT JSONL as <parent>/canonical_labels.json; the old
    # banking77_sft_train.canonical_labels.json name never existed and raised
    # FileNotFoundError.
    if "canonical_labels_path" in expected:
        assert (
            cfg.trainer_config["canonical_labels_path"]
            == expected["canonical_labels_path"]
        )

    # PolyAI/banking77 is script-based: without trust_remote_code, load_dataset
    # refuses to run its loader and a fresh run can't even load the data.
    if expected["loads_script_dataset"]:
        assert cfg.dataset_kwargs["trust_remote_code"] is True

    if expected.get("trust_remote_code"):
        assert cfg.trainer_config["trust_remote_code"] is True


@pytest.mark.parametrize(
    "path",
    [p for p in BANKING77_CONFIGS],
    ids=list(BANKING77_CONFIGS),
)
def test_banking77_json_dataset_uses_valid_hf_subset_name(path):
    """A ``dataset: json`` config's subset name must be a valid HF builder name.

    ``dataset_config.name`` is returned by ``config.dataset_subset`` and passed
    positionally to ``load_dataset(builder, name, ...)`` as the builder config
    name. For file-based builders (json) HF rejects a name containing any of its
    blacklisted characters ``<>:/\\|?*`` with InvalidConfigName -- so a decoder
    config that (mis)used the source slug ``PolyAI/banking77`` as the name dies
    on load before a single row is read. Every working json config uses
    ``default`` (examples/causal_lm_sft_tiny.yaml, the offline CI proxy).
    """
    raw = yaml.safe_load(open(path))
    if raw.get("dataset") != "json":
        pytest.skip("not a file-based json dataset config")
    name = (raw.get("dataset_config") or {}).get("name")
    hf_blacklist = set('<>:/\\|?*')
    assert name is not None, f"{path}: json config needs a dataset_config.name"
    assert not (set(name) & hf_blacklist), (
        f"{path}: dataset_config.name={name!r} contains a HF-illegal character; "
        "json datasets must use a valid builder config name such as 'default'"
    )


@pytest.mark.parametrize(
    "path,expected",
    list(BANKING77_CONFIGS.items()),
    ids=list(BANKING77_CONFIGS),
)
def test_banking77_context_fields_match_train_columns(path, expected):
    """text_field/label_field must name columns present in the AL train split.

    Every cycle the engine builds the teacher augmentation context from the
    train split via extract_few_shot/high_entropy/hard_negative, reading
    ``text_field`` and ``label_field`` (engine._prepare_sample_context). The
    encoders load PolyAI/banking77 directly, whose rows have ``text``/``label``.
    The decoders train on the materialized SFT JSONL, whose rows have
    ``student_prompt``/``gold_answer`` (no ``text``/``label``) -- so a decoder
    that kept the encoder's ``text``/``label`` dies with KeyError('label') the
    moment it prepares the cycle-1 augmentation prompt. The offline proxy hides
    this because it runs a single cycle and never augments.
    """
    raw = yaml.safe_load(open(path))
    cfg = raw.get("dataset_config") or {}
    if raw.get("dataset") == "json":
        # Decoder: reads materialized SFT records.
        assert cfg.get("text_field") == "student_prompt", (
            f"{path}: decoder AL context reads the SFT train split; text_field "
            "must be an SFT column (student_prompt), not the encoder's 'text'"
        )
        assert cfg.get("label_field") == "gold_answer", (
            f"{path}: decoder AL context reads the SFT train split; label_field "
            "must be an SFT column (gold_answer), not the encoder's 'label'"
        )
    else:
        # Encoder/FastText: reads PolyAI/banking77 directly.
        assert cfg.get("text_field") == "text"
        assert cfg.get("label_field") == "label"


def test_materialize_companion_config():
    """The shared materialization source config.

    The six training configs cannot drive materialize-sft themselves: the
    decoders are ``dataset: json`` pointing at the JSONL the step creates, and
    every config's cycle-level prompt references active-learning-only Jinja
    variables that raise under StrictUndefined during a per-row labeling pass.
    This companion loads PolyAI/banking77 directly and is the single source for
    the decoders' train/validation SFT files AND the shared teacher test-label
    file every student scores fidelity against.
    """
    path = "examples/paper/G2_banking77_materialize.yaml"
    raw = yaml.safe_load(open(path))

    # Loads the real source (not the JSONL the step writes), with the loader
    # script opt-in Banking77 requires.
    assert raw["dataset"] == "PolyAI/banking77"
    assert raw["dataset_kwargs"]["trust_remote_code"] is True
    # Carve a validation split so --split validation works (Banking77 ships
    # only train+test) and matches the training configs' carve.
    assert raw["require_validation_split"] is True
    # Single-label teacher output; the training configs keep their own larger
    # AL-loop cap for augmentation batch generation.
    assert raw["teacher_max_output_tokens"] == 16
    # Materialization-only budget, larger than the 200k AL-loop training budget.
    assert raw["token_budget"] > 200000

    materialize = raw["trainer_config"]["materialize_sft"]
    # --max-samples selection is stratified per label (e.g. 20/label subsets).
    assert materialize["stratify_max_samples"] is True
    # Per-row classification prompt lists all 77 intents (cuts non-canonical
    # teacher responses); distinct from any AL-loop augmentation prompt.
    assert "canonical_labels" in materialize["prompt_template"]
    # Short student prompt: canonical_label_decoding scores each label, so the
    # 77-label list must not be inlined into the student prompt.
    assert materialize["student_prompt_template"]

    # Full round-trip when the slm trainer's deps are installed.
    if "slm" in TrainerFactory.get_available_types():
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.dataset == "PolyAI/banking77"
        assert cfg.teacher_max_output_tokens == 16
        assert cfg.require_validation_split is True
        assert (
            cfg.trainer_config["materialize_sft"]["stratify_max_samples"] is True
        )
