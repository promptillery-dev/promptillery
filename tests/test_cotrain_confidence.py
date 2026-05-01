import math
import pytest

from promptillery.cotrain.confidence import (
    fit_temperature,
    apply_temperature,
    ConfidenceCalibrator,
)


def test_fit_temperature_finds_sane_value_for_overconfident_logits():
    logits_correct = [4.0] * 50
    logits_incorrect = [4.0] * 50
    is_correct = [True] * 50 + [False] * 50
    logits = logits_correct + logits_incorrect
    T = fit_temperature(logits=logits, correctness=is_correct, num_classes=10)
    assert T > 1.0


def test_apply_temperature_monotone_decreasing_with_T():
    p_low = apply_temperature(logits=[4.0], temperature=1.0, num_classes=10)[0]
    p_high = apply_temperature(logits=[4.0], temperature=3.0, num_classes=10)[0]
    assert p_high < p_low
    assert 0.0 < p_high < 1.0


def test_calibrator_round_trip():
    cal = ConfidenceCalibrator(temperature=2.0, num_classes=10)
    raw = [0.95, 0.6, 0.99]
    calibrated = cal.calibrate(raw)
    assert all(0.0 <= c <= 1.0 for c in calibrated)
    assert calibrated[0] < raw[0]


def test_calibrator_passes_through_when_temperature_one():
    cal = ConfidenceCalibrator(temperature=1.0, num_classes=10)
    assert cal.calibrate([0.7, 0.3]) == pytest.approx([0.7, 0.3])
