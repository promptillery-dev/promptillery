import importlib.util
import json
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m6_results.py"
spec = importlib.util.spec_from_file_location("m6_results", SCRIPT)
m6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m6)


def _run(root, name, batch_size, acc):
    d = root / f"{name}_transformers_20260922_100000_000000_s13_aa_bb"
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
