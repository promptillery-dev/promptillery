"""The shipped demo replays the recommender from committed artifacts, no GPU, no key."""
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from promptillery.cli import app

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo" / "banking77"


def test_demo_replay_recommends_from_committed_artifacts(tmp_path):
    result = CliRunner().invoke(app, [
        "recommend",
        "--main-results", str(DEMO / "paper_main_results.csv"),
        "--profile-dir", str(DEMO / "profiles"),
        "--prices", str(ROOT / "prices.yaml"),
        "--targets", str(ROOT / "targets.yaml"),
        "-o", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    rec = json.loads((tmp_path / "recommendation.json").read_text())["banking77"]
    assert rec["cell"] == "FacebookAI/roberta-base"
    assert rec["reason"] == "cheapest_feasible"
    receipt = rec["receipt"]
    assert receipt["total_usd"] == pytest.approx(0.7139, rel=1e-3)
    assert receipt["teacher_total_usd"] == pytest.approx(1100.0, rel=1e-6)
    assert receipt["break_even_volume"] == pytest.approx(237.02, rel=1e-3)
    assert receipt["training_usd"] == pytest.approx(0.0888, rel=1e-2)
    assert receipt["teacher_labelling_usd"] == pytest.approx(0.1718, rel=1e-3)
