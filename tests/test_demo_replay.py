"""The shipped demo replays the recommender from committed artifacts, no GPU, no key."""
import json
from pathlib import Path

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
    assert rec["cell"] is not None or rec["reason"] == "teacher_cheaper_at_volume"
    assert rec["receipt"]["break_even_volume"] is None or rec["receipt"]["break_even_volume"] > 0
