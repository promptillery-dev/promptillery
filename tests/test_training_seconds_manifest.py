"""The engine stamps student-training wall-clock into run_manifest.json."""
import asyncio
import json
from pathlib import Path

import pytest

from promptillery.config import ExperimentConfig
from promptillery.engine import DistillationEngine

TINY = Path(__file__).resolve().parents[1] / "examples" / "causal_lm_sft_tiny.yaml"


def test_manifest_records_training_seconds(tmp_path):
    cfg = ExperimentConfig.from_yaml(str(TINY))
    cfg.base_output_dir = str(tmp_path)

    asyncio.run(DistillationEngine(cfg).run())

    manifest_path = next(Path(tmp_path).rglob("run_manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["training_seconds"] > 0
    assert len(manifest["training_seconds_by_cycle"]) == manifest["cycles_completed"]
    assert sum(manifest["training_seconds_by_cycle"]) == pytest.approx(manifest["training_seconds"], abs=1e-3)


def test_manifest_records_training_seconds_per_cycle_for_multi_cycle_run(tmp_path):
    cfg = ExperimentConfig.from_yaml(str(TINY))
    cfg.base_output_dir = str(tmp_path)
    cfg.cycles = 2

    asyncio.run(DistillationEngine(cfg).run())

    manifest_path = next(Path(tmp_path).rglob("run_manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["training_seconds_by_cycle"]) == 2
    assert all(seconds > 0 for seconds in manifest["training_seconds_by_cycle"])
    assert sum(manifest["training_seconds_by_cycle"]) == pytest.approx(manifest["training_seconds"], abs=1e-3)
