import importlib.util
from pathlib import Path

from _paths import PAPER_EXAMPLES

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("experiment_issues", ROOT / "scripts" / "experiment_issues.py")
ei = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ei)


def test_every_block_renders_a_self_contained_issue(tmp_path):
    paths = ei.write_all(tmp_path)
    assert len(paths) == len(ei.BLOCKS) >= 8
    for block, path in zip(ei.BLOCKS, paths):
        body = path.read_text()
        assert "git checkout resubmit/eacl2027-demo" in body
        assert "uv sync --extra fasttext" in body
        assert "Never edit" in body
        assert "gh issue comment" in body and "gh pr create" in body
        assert block["cut_line"] in body
        for cmd in block["runs"]:
            assert cmd in body
        assert set(block["labels"]) >= {"experiment"}
        assert any(label.startswith("gpu:") for label in block["labels"])


def test_laptop_blocks_are_tagged_laptop():
    laptop = [b for b in ei.BLOCKS if b["tier"] == "laptop"]
    assert {"e1-laptop-roberta", "e1-laptop-ettin", "e6-laptop-verifier"} <= {b["id"] for b in laptop}
    for b in laptop:
        assert "gpu:laptop" in b["labels"]
        assert "gpu:24gb" not in b["labels"]


def test_e4b_ettin_rerun_block_exists_and_is_not_stretch():
    by_id = {b["id"]: b for b in ei.BLOCKS}
    assert "e4b-24gb-ettin-rerun" in by_id
    assert by_id["e4b-24gb-ettin-rerun"]["stretch"] is False


def test_e2c_compute_matched_w10_block_exists_with_its_ten_configs():
    by_id = {b["id"]: b for b in ei.BLOCKS}
    assert "e2c-24gb-compute-matched-w10" in by_id
    block = by_id["e2c-24gb-compute-matched-w10"]
    assert block["stretch"] is False
    assert "gpu:24gb" in block["labels"]
    expected = {
        f"examples/M6_{d}_same_n_cm_w10_{s}.yaml"
        for d in ["agnews", "sst2", "imdb", "yahoo", "huffpost"]
        for s in ["roberta_base", "ettin_encoder"]
    }
    assert set(block["configs"]) == expected
    assert len(block["configs"]) == 10


def test_every_config_named_in_a_block_exists():
    for b in ei.BLOCKS:
        for cfg in b["configs"]:
            assert (ROOT / cfg).exists() or (PAPER_EXAMPLES / Path(cfg).name).exists(), \
                (b["id"], cfg)
