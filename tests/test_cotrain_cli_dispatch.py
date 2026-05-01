import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from promptillery.cli import app


def test_cli_train_dispatches_to_cotrain_engine(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "name": "x",
        "acquisition_mode": "cotrain",
        "student_type": "causal_lm_sft",
        "cycles": 1,
        "cotrain": {
            "student_a": {"model": "fake-a", "operator": "coverage"},
            "student_b": {"model": "fake-b", "operator": "boundary"},
            "strong_teacher": "openai/gpt-4o",
            "bootstrap_size": 4,
        },
    }))
    called = {}

    async def fake_run(self):
        called["ran"] = True

    class _FakeEngine:
        async def run(self):
            await fake_run(self)

    runner = CliRunner()
    with patch("promptillery.cotrain.engine.CoTrainEngine.from_config",
               classmethod(lambda cls, cfg, **kw: _FakeEngine())):
        result = runner.invoke(app, ["train", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert called.get("ran") is True
