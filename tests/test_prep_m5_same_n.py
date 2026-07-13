"""Tests for scripts/prep_m5_same_n.py (same-N gold FT baseline)."""
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest
from datasets import ClassLabel, Dataset, DatasetDict

spec = importlib.util.spec_from_file_location(
    "prep_m5", Path(__file__).resolve().parents[1] / "scripts" / "prep_m5_same_n.py"
)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def _fixture_run_dir(tmp_path, **overrides):
    manifest = {
        "run_id": "g3_sst2_ettin_encoder_transformers_cycles-10_test",
        "status": "completed",
        "cycles_completed": 10,
        "expected_cycles": 10,
        "student_model": "jhu-clsp/ettin-encoder-150m",
        "final_synthetic_count": 12,
    }
    manifest.update(overrides)
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def test_read_matched_count_returns_count_and_provenance(tmp_path):
    got = prep.read_matched_count(_fixture_run_dir(tmp_path), "sst2", "ettin_encoder")
    assert got["final_synthetic_count"] == 12
    assert got["run_id"].startswith("g3_sst2_ettin_encoder")
    assert got["run_dir"].endswith("run")


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"status": "running"}, "status"),
        ({"cycles_completed": 9}, "cycles_completed"),
        ({"expected_cycles": 5, "cycles_completed": 5}, "expected_cycles"),
        ({"student_model": "other/model"}, "student_model"),
        ({"run_id": "g3_agnews_ettin_encoder_cycles-10_x"}, "run_id"),
        ({"final_synthetic_count": 0}, "final_synthetic_count"),
    ],
)
def test_read_matched_count_rejects_unusable_runs(tmp_path, overrides, match):
    run_dir = _fixture_run_dir(tmp_path, **overrides)
    with pytest.raises(ValueError, match=match):
        prep.read_matched_count(run_dir, "sst2", "ettin_encoder")


def test_read_matched_count_requires_manifest(tmp_path):
    with pytest.raises(ValueError, match="run_manifest.json"):
        prep.read_matched_count(tmp_path, "sst2", "ettin_encoder")


def _toy_dataset(n_per_class=40, classes=("neg", "pos", "neu")):
    texts, labels, label_texts = [], [], []
    i = 0
    for ci, cname in enumerate(classes):
        for _ in range(n_per_class):
            texts.append(f"sample text {i}")
            labels.append(ci)
            label_texts.append(cname)
            i += 1
    train = Dataset.from_dict(
        {"text": texts, "label": labels, "label_text": label_texts}
    ).cast_column("label", ClassLabel(names=list(classes)))
    test = train.select(range(0, len(train), 4))
    return DatasetDict({"train": train, "test": test})


def test_reconstruction_matches_engine_sampler_and_yields_complement():
    ds = _toy_dataset()
    seed_train, seed_val, complement = prep.reconstruct_seed_split(
        ds, seed=13, sample_size=30, train_ratio=0.8
    )
    assert len(seed_train) == 24
    assert len(seed_val) == 6
    assert len(complement) == len(ds["train"]) - 30
    seed_texts = set(seed_train["text"]) | set(seed_val["text"])
    assert seed_texts.isdisjoint(set(complement["text"]))


def test_reconstruction_rejects_too_small_train():
    ds = _toy_dataset(n_per_class=5)  # 15 rows < sample_size
    with pytest.raises(ValueError, match="unused remainder"):
        prep.reconstruct_seed_split(ds, seed=13, sample_size=30, train_ratio=0.8)


def test_topup_is_deterministic_and_stratified():
    ds = _toy_dataset()
    _, _, complement = prep.reconstruct_seed_split(ds, 13, sample_size=30, train_ratio=0.8)
    a, strat_a = prep.draw_topup(complement, 9, seed=13)
    b, strat_b = prep.draw_topup(complement, 9, seed=13)
    assert a["text"] == b["text"]
    assert strat_a is True and strat_b is True
    assert set(Counter(a["label"]).values()) == {3}  # 9 rows / 3 balanced classes


def test_topup_falls_back_below_class_count():
    ds = _toy_dataset()
    _, _, complement = prep.reconstruct_seed_split(ds, 13, sample_size=30, train_ratio=0.8)
    drawn, stratified = prep.draw_topup(complement, 2, seed=13)
    assert len(drawn) == 2
    assert stratified is False


def test_topup_rejects_overdraw_and_takes_all_at_pool_size():
    ds = _toy_dataset()
    _, _, complement = prep.reconstruct_seed_split(ds, 13, sample_size=30, train_ratio=0.8)
    with pytest.raises(ValueError, match="exceeds unused pool"):
        prep.draw_topup(complement, len(complement) + 1, seed=13)
    everything, _ = prep.draw_topup(complement, len(complement), seed=13)
    assert len(everything) == len(complement)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_build_cell_writes_superset_manifest_and_is_deterministic(tmp_path, monkeypatch):
    ds = _toy_dataset()
    monkeypatch.setattr(prep, "load_source", lambda dataset, g3_root: ds)
    run_dir = _fixture_run_dir(tmp_path, final_synthetic_count=12)
    out_root = tmp_path / "m5"

    manifest = prep.build_cell(
        "sst2", "ettin_encoder", run_dir, seed=13,
        out_root=out_root, g3_root=tmp_path,
        sample_size=30, train_ratio=0.8,
    )

    train = _read_jsonl(out_root / "sst2" / "ettin_encoder" / "train.jsonl")
    val = _read_jsonl(out_root / "sst2" / "validation.jsonl")
    test = _read_jsonl(out_root / "sst2" / "test.jsonl")
    seed_train, seed_val, _ = prep.reconstruct_seed_split(
        ds, 13, sample_size=30, train_ratio=0.8
    )

    # matched N = seed train + top-up; train.jsonl is seed rows then top-up rows
    assert manifest["matched_n"] == 24 + 12 == len(train)
    assert manifest["topup_rows"] == 12
    assert manifest["seed_train_rows"] == 24
    assert manifest["source_run_id"].startswith("g3_sst2_ettin_encoder")
    assert [r["text"] for r in train[:24]] == seed_train["text"]  # strict superset
    topup_texts = {r["text"] for r in train[24:]}
    assert topup_texts.isdisjoint(set(seed_train["text"]) | set(seed_val["text"]))
    assert [r["text"] for r in val] == seed_val["text"]
    assert len(test) == len(ds["test"])
    assert set(train[0]) == {"text", "label", "label_text"}
    assert isinstance(train[0]["label"], int)

    on_disk = json.loads(
        (out_root / "sst2" / "ettin_encoder" / "matched_n_manifest.json").read_text()
    )
    assert on_disk == manifest

    before = (out_root / "sst2" / "ettin_encoder" / "train.jsonl").read_bytes()
    prep.build_cell(
        "sst2", "ettin_encoder", run_dir, seed=13,
        out_root=out_root, g3_root=tmp_path,
        sample_size=30, train_ratio=0.8,
    )
    assert (out_root / "sst2" / "ettin_encoder" / "train.jsonl").read_bytes() == before


def test_to_records_uses_classlabel_names_when_no_label_text_column():
    ds = _toy_dataset()["train"].remove_columns("label_text")
    records = prep.to_records(ds.select(range(3)))
    assert [r["label_text"] for r in records] == ["neg", "neg", "neg"]


def test_load_source_errors_when_bootstrap_cannot_produce_pools(tmp_path, monkeypatch):
    # bootstrap is attempted first (off-the-shelf mode); if the pools still
    # are not there the loud guard remains
    monkeypatch.setattr(prep, "bootstrap_prepped_pools", lambda dataset, g3_root: None)
    with pytest.raises(ValueError, match="prep_g3_datasets"):
        prep.load_source("yahoo", tmp_path)


def test_build_materialize_config_retargets_g3_config(tmp_path):
    cfg = prep.build_materialize_config(
        "sst2", tmp_path / "train.jsonl", tmp_path / "m5"
    )
    import yaml as _yaml
    g3_raw = _yaml.safe_load(
        (prep.EXAMPLES_DIR / "G3_sst2_materialize.yaml").read_text(encoding="utf-8")
    )
    mat = cfg.trainer_config["materialize_sft"]
    g3_mat = g3_raw["trainer_config"]["materialize_sft"]
    # prompt templates carried over verbatim (identity by construction)
    assert mat["prompt_template"] == g3_mat["prompt_template"]
    assert mat["student_prompt_template"] == g3_mat["student_prompt_template"]
    # retargeted at the cell file, gold fields forced to label_text
    assert cfg.dataset == "json"
    assert cfg.dataset_kwargs["data_files"] == {"train": str(tmp_path / "train.jsonl")}
    assert cfg.sampling.enabled is False
    assert cfg.require_validation_split is False
    assert mat["gold_answer_field"] == "label_text"
    assert mat["canonical_labels_field"] == "label_text"


def test_build_materialize_config_forces_gold_fields_for_imdb(tmp_path):
    # G3_imdb_materialize.yaml natively uses gold_answer_field="label" and
    # OMITS canonical_labels_field (raw IMDB has no label_text column); the
    # retarget must force both to label_text because the prep JSONL always
    # carries it. sst2 cannot catch a regression here -- its G3 file already
    # says label_text.
    cfg = prep.build_materialize_config(
        "imdb", tmp_path / "train.jsonl", tmp_path / "m5"
    )
    mat = cfg.trainer_config["materialize_sft"]
    assert mat["gold_answer_field"] == "label_text"
    assert mat["canonical_labels_field"] == "label_text"


def test_build_cell_materializes_decoder_train_sft(tmp_path, monkeypatch):
    ds = _toy_dataset()
    monkeypatch.setattr(prep, "load_source", lambda dataset, g3_root: ds)
    run_dir = _fixture_run_dir(
        tmp_path,
        run_id="g3_sst2_ettin_decoder_slm_cycles-10_test",
        student_model="jhu-clsp/ettin-decoder-150m",
    )
    out_root = tmp_path / "m5"

    manifest = prep.build_cell(
        "sst2", "ettin_decoder", run_dir, seed=13,
        out_root=out_root, g3_root=tmp_path,
        sample_size=30, train_ratio=0.8,
    )

    sft = _read_jsonl(out_root / "sst2" / "ettin_decoder" / "train_sft.jsonl")
    assert manifest["train_sft_rows"] == len(sft) == manifest["matched_n"] == 36
    assert manifest["materialize_source_config"].endswith("G3_sst2_materialize.yaml")
    record = sft[0]
    assert record["gold_answer"] in {"neg", "pos", "neu"}
    assert "student_prompt" in record and record["student_prompt"].strip()


def test_build_cell_skips_materialize_for_encoders(tmp_path, monkeypatch):
    ds = _toy_dataset()
    monkeypatch.setattr(prep, "load_source", lambda dataset, g3_root: ds)
    run_dir = _fixture_run_dir(tmp_path)
    out_root = tmp_path / "m5"
    manifest = prep.build_cell(
        "sst2", "ettin_encoder", run_dir, seed=13,
        out_root=out_root, g3_root=tmp_path,
        sample_size=30, train_ratio=0.8,
    )
    assert manifest["train_sft_rows"] is None
    assert manifest["materialize_source_config"] is None
    assert not (out_root / "sst2" / "ettin_encoder" / "train_sft.jsonl").exists()


def test_cli_runs_build_cell_and_prints_manifest(tmp_path, monkeypatch, capsys):
    ds = _toy_dataset()
    monkeypatch.setattr(prep, "load_source", lambda dataset, g3_root: ds)
    monkeypatch.setattr(prep, "SAMPLE_SIZE", 30)
    run_dir = _fixture_run_dir(tmp_path)
    prep.main([
        "--dataset", "sst2",
        "--student", "ettin_encoder",
        "--run-dir", str(run_dir),
        "--out-root", str(tmp_path / "m5"),
        "--g3-root", str(tmp_path),
    ])
    printed = json.loads(capsys.readouterr().out)
    assert printed["matched_n"] == 24 + 12
    assert (tmp_path / "m5" / "sst2" / "ettin_encoder" / "train.jsonl").exists()


def test_nominal_matched_count_is_batch_times_augmenting_cycles():
    got = prep.nominal_matched_count("sst2", "ettin_encoder")
    # G3_sst2_ettin_encoder.yaml: augmentation_batch_size 32, cycles [1, 5, 10]
    # -> 32 x (10 - 1) = 288 (the last cycle never augments)
    assert got["final_synthetic_count"] == 288
    assert got["run_id"] is None
    assert got["run_dir"] is None
    assert got["matched_n_source"] == "nominal"


def _nominal_stub(dataset, student):
    return {
        "run_id": None,
        "run_dir": None,
        "final_synthetic_count": 12,
        "matched_n_source": "nominal",
    }


def test_build_cell_nominal_mode_needs_no_run_dir(tmp_path, monkeypatch):
    ds = _toy_dataset()
    monkeypatch.setattr(prep, "load_source", lambda dataset, g3_root: ds)
    monkeypatch.setattr(prep, "nominal_matched_count", _nominal_stub)
    manifest = prep.build_cell(
        "sst2", "ettin_encoder", None, seed=13,
        out_root=tmp_path / "m5", g3_root=tmp_path,
        sample_size=30, train_ratio=0.8,
    )
    assert manifest["matched_n"] == 24 + 12
    assert manifest["matched_n_source"] == "nominal"
    assert manifest["source_run_id"] is None
    assert manifest["source_run_dir"] is None


def test_run_dir_mode_records_run_manifest_source(tmp_path, monkeypatch):
    ds = _toy_dataset()
    monkeypatch.setattr(prep, "load_source", lambda dataset, g3_root: ds)
    manifest = prep.build_cell(
        "sst2", "ettin_encoder", _fixture_run_dir(tmp_path), seed=13,
        out_root=tmp_path / "m5", g3_root=tmp_path,
        sample_size=30, train_ratio=0.8,
    )
    assert manifest["matched_n_source"] == "run_manifest"


def test_cli_requires_exactly_one_matched_n_source(tmp_path, monkeypatch):
    ds = _toy_dataset()
    monkeypatch.setattr(prep, "load_source", lambda dataset, g3_root: ds)
    base = [
        "--dataset", "sst2", "--student", "ettin_encoder",
        "--out-root", str(tmp_path / "m5"), "--g3-root", str(tmp_path),
    ]
    with pytest.raises(SystemExit):
        prep.main(base)  # neither --run-dir nor --nominal
    with pytest.raises(SystemExit):
        prep.main(base + ["--nominal", "--run-dir", str(tmp_path)])  # both


def test_cli_nominal_happy_path(tmp_path, monkeypatch, capsys):
    ds = _toy_dataset()
    monkeypatch.setattr(prep, "load_source", lambda dataset, g3_root: ds)
    monkeypatch.setattr(prep, "SAMPLE_SIZE", 30)
    monkeypatch.setattr(prep, "nominal_matched_count", _nominal_stub)
    prep.main([
        "--dataset", "sst2", "--student", "ettin_encoder", "--nominal",
        "--out-root", str(tmp_path / "m5"), "--g3-root", str(tmp_path),
    ])
    printed = json.loads(capsys.readouterr().out)
    assert printed["matched_n"] == 24 + 12
    assert printed["matched_n_source"] == "nominal"


def test_load_source_bootstraps_prepped_pools(tmp_path, monkeypatch):
    calls = []

    def fake_bootstrap(dataset, g3_root):
        calls.append((dataset, str(g3_root)))
        d = Path(g3_root) / dataset
        d.mkdir(parents=True, exist_ok=True)
        rows = [
            {"text": f"t{i} {c}", "label": ci, "label_text": c}
            for i in range(3)
            for ci, c in enumerate(["a", "b"])
        ]
        for name in ("train.jsonl", "test.jsonl"):
            with (d / name).open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

    monkeypatch.setattr(prep, "bootstrap_prepped_pools", fake_bootstrap)
    ds = prep.load_source("yahoo", tmp_path)
    assert calls == [("yahoo", str(tmp_path))]
    assert set(ds.keys()) == {"train", "test"}


def test_ensure_g3_eval_artifacts_fills_only_missing(tmp_path, monkeypatch):
    materialized = []

    async def fake_materialize(**kwargs):
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}\n", encoding="utf-8")
        (out.parent / "canonical_labels.json").write_text("{}\n", encoding="utf-8")
        materialized.append((kwargs["split"], kwargs["mode"]))
        return {}

    monkeypatch.setattr(prep, "materialize_sft_records", fake_materialize)
    created = prep.ensure_g3_eval_artifacts("sst2", tmp_path)
    assert materialized == [("validation", "gold"), ("test", "gold")]
    assert len(created) == 2
    materialized.clear()
    assert prep.ensure_g3_eval_artifacts("sst2", tmp_path) == []
    assert materialized == []


def test_cli_decoder_invokes_eval_bootstrap(tmp_path, monkeypatch, capsys):
    ds = _toy_dataset()
    monkeypatch.setattr(prep, "load_source", lambda dataset, g3_root: ds)
    monkeypatch.setattr(prep, "SAMPLE_SIZE", 30)
    monkeypatch.setattr(prep, "nominal_matched_count", _nominal_stub)
    bootstrap_calls = []
    monkeypatch.setattr(
        prep, "ensure_g3_eval_artifacts",
        lambda dataset, g3_root: bootstrap_calls.append((dataset, str(g3_root))) or [],
    )
    prep.main([
        "--dataset", "sst2", "--student", "ettin_decoder", "--nominal",
        "--out-root", str(tmp_path / "m5"), "--g3-root", str(tmp_path),
    ])
    assert bootstrap_calls == [("sst2", str(tmp_path))]
    printed = json.loads(capsys.readouterr().out)
    assert printed["matched_n"] == 24 + 12
    assert printed["train_sft_rows"] == 36
