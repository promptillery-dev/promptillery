import importlib.util
import json
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m6_results.py"
spec = importlib.util.spec_from_file_location("m6_results", SCRIPT)
m6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m6)


def _run(root, name, batch_size, acc, stamp="20260922_100000_000000"):
    d = root / f"{name}_transformers_{stamp}_s13_aa_bb"
    d.mkdir(parents=True)
    (d / "experiment_config.yaml").write_text(yaml.safe_dump({"name": name, "batch_size": batch_size}))
    (d / "metrics.json").write_text(json.dumps({"0": {"accuracy": 0.5, "f1": 0.5},
        "heldout_test": {"accuracy": acc, "f1": acc - 0.01, "_selected_cycle": 0}}))
    (d / "run_manifest.json").write_text(json.dumps({"status": "completed", "run_id": d.name}))


def test_collect_and_render(tmp_path):
    out_root = tmp_path / "out"
    _run(out_root, "m6_sst2_ft1_roberta_base", 32, 0.9312)
    _run(out_root, "m6_agnews_ft1_roberta_base_bs8", 8, 0.8801)
    _run(out_root, "m6_sst2_ft1_roberta_base_s7", 32, 0.9200)

    rows = m6.collect(out_root)
    assert {(r["dataset"], r["variant"], r["student"], r["seed"]) for r in rows} == {
        ("sst2", "ft1", "roberta_base", 13), ("agnews", "ft1", "roberta_base", 13),
        ("sst2", "ft1", "roberta_base", 7)}
    seed7 = next(r for r in rows if r["seed"] == 7)
    assert seed7["dataset"] == "sst2" and seed7["accuracy"] == 0.92

    tex = m6.table2_rows(rows)
    assert "93.12$_{-4.3}$" in tex            # sst2: 93.12 - 97.42 few-shot teacher, seed 13 unchanged
    assert "88.01$_{-0.9}^{\\dagger}$" in tex  # agnews at batch 8 gets the dagger
    assert "--" in tex                        # missing cells

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    m6.main(["--out-root", str(out_root), "--docs", str(docs_dir)])
    comment = (docs_dir / "M6_ISSUE_COMMENT.md").read_text()
    assert "| dataset | variant | student |" in comment
    assert "```latex" in comment


def test_latest_run_wins_by_timestamp_not_by_dir_name(tmp_path):
    # A `_bsN` name inserts "_bs16" before the stamp, so it always sorts after
    # the plain name lexicographically -- even when the plain run happened
    # later. "Latest wins" must go by the parsed stamp, not the raw name.
    out_root = tmp_path / "out_bs_first"
    _run(out_root, "m6_sst2_ft1_roberta_base_bs16", 16, 0.10, stamp="20260101_000000_000000")
    _run(out_root, "m6_sst2_ft1_roberta_base", 32, 0.90, stamp="20260922_120000_000000")
    rows = m6.collect(out_root)
    assert len(rows) == 1
    assert rows[0]["accuracy"] == pytest.approx(0.90)
    assert rows[0]["batch_size"] == 32

    # Reverse: the `_bs16` run happened later and should win.
    out_root2 = tmp_path / "out_plain_first"
    _run(out_root2, "m6_sst2_ft1_roberta_base", 32, 0.10, stamp="20260101_000000_000000")
    _run(out_root2, "m6_sst2_ft1_roberta_base_bs16", 16, 0.90, stamp="20260922_120000_000000")
    rows2 = m6.collect(out_root2)
    assert len(rows2) == 1
    assert rows2[0]["accuracy"] == pytest.approx(0.90)
    assert rows2[0]["batch_size"] == 16


def test_same_n_cm_w10_variant_parses(tmp_path):
    out_root = tmp_path / "out"
    _run(out_root, "m6_yahoo_same_n_cm_w10_roberta_base", 32, 0.75)
    rows = m6.collect(out_root)
    assert len(rows) == 1
    assert rows[0]["dataset"] == "yahoo"
    assert rows[0]["variant"] == "same_n_cm_w10"
    assert rows[0]["student"] == "roberta_base"


def test_table2_rows_skips_variant_with_no_cells_for_a_student(tmp_path):
    out_root = tmp_path / "out"
    _run(out_root, "m6_sst2_ft1_roberta_base", 32, 0.90)

    rows = m6.collect(out_root)
    tex = m6.table2_rows(rows)

    # roberta_base: the ft1 row is present, with "--" for the four missing datasets.
    assert tex.count("fine-tuned} (1 cycle)") == 1
    assert "--" in tex

    # ettin_encoder has no runs at all, so every one of its variant rows (all
    # five cells missing) is skipped entirely -- no all-"--" row for it.
    ettin_idx = tex.index("Ettin-encoder-150M (encoder)")
    assert "\\quad" not in tex[ettin_idx:]


def test_issue_comment_not_overwritten_unless_forced(tmp_path):
    out_root = tmp_path / "out"
    _run(out_root, "m6_sst2_ft1_roberta_base", 32, 0.9312)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    comment_path = docs_dir / "M6_ISSUE_COMMENT.md"
    comment_path.write_text("hand-written content\n\n## Deviations (E5, seeds 7/21)\nkept by hand\n")

    m6.main(["--out-root", str(out_root), "--docs", str(docs_dir)])
    assert comment_path.read_text() == "hand-written content\n\n## Deviations (E5, seeds 7/21)\nkept by hand\n"

    m6.main(["--out-root", str(out_root), "--docs", str(docs_dir), "--force-comment"])
    assert "hand-written content" not in comment_path.read_text()
    assert "| dataset | variant | student |" in comment_path.read_text()
