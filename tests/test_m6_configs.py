"""Offline validation of the M6 baseline configs (EACL 2027 resubmission, E1/E2)."""
import pytest
import yaml

from promptillery.config import ExperimentConfig
from _paths import PAPER_EXAMPLES as EXAMPLES, REPO_ROOT

DATASETS = ["agnews", "sst2", "imdb", "yahoo", "huffpost"]
STUDENTS = {"roberta_base": "FacebookAI/roberta-base",
            "ettin_encoder": "jhu-clsp/ettin-encoder-150m"}
VARIANTS = ["ft1", "same_n", "same_n_cm", "same_n_cm_w10", "seed_x10"]
RECIPE_KEYS = ["learning_rate", "num_train_epochs", "batch_size", "seed"]


def _cells():
    for d in DATASETS:
        for s in STUDENTS:
            for v in VARIANTS:
                yield d, v, s


def _load(path):
    return ExperimentConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("d,v,s", list(_cells()), ids=[f"{d}-{v}-{s}" for d, v, s in _cells()])
def test_m6_config_invariants(d, v, s):
    cfg = _load(EXAMPLES / f"M6_{d}_{v}_{s}.yaml")
    assert cfg.name == f"m6_{d}_{v}_{s}"
    assert cfg.student == STUDENTS[s]
    assert cfg.student_type == "transformers"
    assert cfg.seed == 13
    assert cfg.base_output_dir == "out/m6"
    assert cfg.prompt is None, "M6 runs never call the teacher"
    assert cfg.has_list_parameters() is False
    assert cfg.trainer_config["report_held_out_test"] is True
    assert cfg.persist_datasets is False
    if v in ("ft1", "same_n"):
        assert cfg.cycles == 1
        assert cfg.warmup_steps == 10
    elif v == "same_n_cm_w10":
        assert cfg.cycles == 10
        assert cfg.warmup_steps == 10, "twins the corrected loop re-runs (#42, #46)"
    else:
        assert cfg.cycles == 10
        assert cfg.warmup_steps == 500, "controls mirror the promptillery arm's schedule"
    if v in ("ft1", "seed_x10"):
        g3 = _load(EXAMPLES / f"G3_{d}_ettin_encoder.yaml")
        assert cfg.dataset == g3.dataset
        assert cfg.sampling.enabled is True and cfg.sampling.sample_size == 1000
    else:
        m5 = _load(EXAMPLES / f"M5_{d}_same_n_{s}.yaml")
        assert cfg.dataset == "json"
        assert cfg.dataset_kwargs == m5.dataset_kwargs
        assert cfg.sampling.enabled is False


@pytest.mark.parametrize("d", DATASETS)
def test_m6_recipe_matches_g3(d):
    g3 = _load(EXAMPLES / f"G3_{d}_ettin_encoder.yaml")
    for s in STUDENTS:
        for v in VARIANTS:
            cfg = _load(EXAMPLES / f"M6_{d}_{v}_{s}.yaml")
            for key in RECIPE_KEYS:
                assert getattr(cfg, key) == getattr(g3, key), (d, v, s, key)


@pytest.mark.parametrize("d,s", [(d, s) for d in DATASETS for s in STUDENTS],
                          ids=[f"{d}-{s}" for d in DATASETS for s in STUDENTS])
def test_m6_same_n_cm_w10_twins_same_n_cm(d, s):
    cfg = _load(EXAMPLES / f"M6_{d}_same_n_cm_w10_{s}.yaml")
    twin = _load(EXAMPLES / f"M6_{d}_same_n_cm_{s}.yaml")
    assert cfg.warmup_steps == 10
    assert cfg.cycles == 10
    assert cfg.prompt is None
    assert cfg.name == f"m6_{d}_same_n_cm_w10_{s}"
    assert cfg.auto_modify_name is False
    assert cfg.dataset_kwargs == twin.dataset_kwargs


def test_generator_batch_override_changes_name_and_batch(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gen_m6_configs", REPO_ROOT / "scripts" / "gen_m6_configs.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    cfg = gen.build("ft1", "sst2", "roberta_base", EXAMPLES, batch_size=8)
    assert cfg["name"] == "m6_sst2_ft1_roberta_base_bs8"
    assert cfg["batch_size"] == 8
    assert gen.parse_name("m6_sst2_same_n_cm_ettin_encoder") == ("sst2", "same_n_cm", "ettin_encoder")
    assert gen.parse_name("m6_sst2_same_n_cm_w10_ettin_encoder") == ("sst2", "same_n_cm_w10", "ettin_encoder")
    assert gen.build("ft1", "sst2", "roberta_base", EXAMPLES, seed=7)["name"] == "m6_sst2_ft1_roberta_base_s7"
    assert gen.build("ft1", "sst2", "roberta_base", EXAMPLES, batch_size=8, seed=7)["name"] == \
        "m6_sst2_ft1_roberta_base_bs8_s7"
    assert gen.parse_name("m6_sst2_ft1_roberta_base_bs8_s7") == ("sst2", "ft1", "roberta_base")


@pytest.mark.parametrize("d", DATASETS)
def test_g3_roberta_rerun_config(d):
    cfg = _load(EXAMPLES / f"G3_{d}_roberta_base.yaml")
    g3 = _load(EXAMPLES / f"G3_{d}_ettin_encoder.yaml")
    assert cfg.student == "FacebookAI/roberta-base"
    assert cfg.warmup_steps == 10
    assert cfg.cycles == [1, 5, 10]
    assert cfg.prompt == g3.prompt


@pytest.mark.parametrize("d", DATASETS)
def test_g3_ettin_w10_rerun_config(d):
    cfg = _load(EXAMPLES / f"G3_{d}_ettin_encoder_w10.yaml")
    g3 = _load(EXAMPLES / f"G3_{d}_ettin_encoder.yaml")
    assert cfg.student == "jhu-clsp/ettin-encoder-150m"
    assert cfg.warmup_steps == 10
    assert cfg.cycles == [1, 5, 10]
    assert cfg.prompt == g3.prompt
    assert cfg.dataset == g3.dataset
    assert cfg.dataset_kwargs == g3.dataset_kwargs
    assert cfg.name == f"g3_{d}_ettin_encoder_w10"


def test_verifier_config_is_gold_only():
    cfg = _load(EXAMPLES / "G2_banking77_verifier_roberta.yaml")
    assert cfg.cycles == 1 and cfg.prompt is None and cfg.warmup_steps == 10
