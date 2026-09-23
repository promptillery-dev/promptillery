import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "derive_targets.py"
spec = importlib.util.spec_from_file_location("derive_targets", SCRIPT)
dt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dt)


def test_quartile_floors_are_rounded_quartiles():
    # inclusive quartiles of six points: 0.9025, 0.92, 0.9375 -> no half-way rounding
    accs = [0.80, 0.90, 0.91, 0.93, 0.94, 0.95]
    assert dt.quartile_floors(accs) == [0.90, 0.92, 0.94]


def test_teacher_usd_per_call_from_logged_tokens():
    manifest = {"teacher_input_tokens": 1_624_664, "teacher_output_tokens": 15_315}
    usd = dt.teacher_usd_per_call(manifest, 3080, input_usd_per_m=2.0, output_usd_per_m=8.0)
    assert usd == pytest.approx(0.0010948, rel=1e-3)
