"""Tests for pareto.py -- the volume-free deployment frontier.

A ``Cell`` is one row of ``paper_main_results.csv`` joined to its student
profile: two accuracies (selection = what a selector sees, held-out = the
truth), the one-time distillation spend, and the profiled latency/throughput
on the stamped hardware. Dominance runs over four volume-free axes
(accuracy up; p95 latency, distillation $, serving $/call down) so a dominated
cell loses at *every* volume -- the frontier is computed once and reused across
the whole volume sweep.
"""

from promptillery.pareto import (
    Cell,
    HardwarePrices,
    UnknownHardwareError,
    dominates,
    pareto_frontier,
    serving_cost_per_call,
)


def _cell(**overrides):
    """A candidate config cell with 4090/CUDA defaults; override per test."""
    base = dict(
        dataset="banking77",
        dataset_subset="",
        student_model="roberta-base",
        student_type="transformers",
        policy_name="uncertainty",
        control_name="",
        token_budget=20_000_000,
        selection_accuracy=0.93,
        heldout_accuracy=0.92,
        distillation_usd=1.0,
        p95_latency_ms=5.0,
        throughput_calls_per_sec=200.0,
        device="cuda:0",
        gpu_name="NVIDIA GeForce RTX 4090",
    )
    base.update(overrides)
    return Cell(**base)


def _prices():
    return HardwarePrices({"NVIDIA GeForce RTX 4090": 0.40, "cpu": 0.05})


def test_pareto_frontier_drops_a_fully_dominated_cell():
    # `weak` is worse on all four axes than `strong`: lower accuracy, higher
    # latency, higher distillation spend, and (via lower throughput on the same
    # GPU) higher serving cost. It loses at every volume, so the frontier drops it.
    strong = _cell(student_model="roberta-base", selection_accuracy=0.93,
                   p95_latency_ms=5.0, distillation_usd=1.0,
                   throughput_calls_per_sec=200.0)
    weak = _cell(student_model="ettin-decoder-150m", selection_accuracy=0.90,
                 p95_latency_ms=95.0, distillation_usd=5.0,
                 throughput_calls_per_sec=10.0)

    frontier = pareto_frontier([strong, weak], _prices(),
                               accuracy_key="selection_accuracy")

    assert strong in frontier
    assert weak not in frontier


def test_pareto_frontier_keeps_an_accuracy_latency_tradeoff():
    # `accurate` wins on accuracy, `fast` wins on latency -- neither dominates,
    # so both stay. A naive "keep the most accurate" filter would wrongly drop
    # `fast`; the frontier must not.
    accurate = _cell(student_model="roberta-base", selection_accuracy=0.95,
                     p95_latency_ms=90.0)
    fast = _cell(student_model="fasttext", selection_accuracy=0.82,
                 p95_latency_ms=1.0)

    frontier = pareto_frontier([accurate, fast], _prices(),
                               accuracy_key="selection_accuracy")

    assert accurate in frontier
    assert fast in frontier


def test_serving_cost_uses_the_devices_rate_fasttext_on_cpu():
    # fasttext is CPU-only (profiler pins it); it must be billed the CPU rate,
    # not the 4090 rate, or the one student anchoring the cheap end is destroyed.
    fasttext = _cell(student_model="fasttext", student_type="fasttext",
                     device="cpu", gpu_name=None, throughput_calls_per_sec=100.0)
    gpu = _cell(throughput_calls_per_sec=100.0)  # same throughput, 4090

    ft_cost = serving_cost_per_call(fasttext, _prices())
    gpu_cost = serving_cost_per_call(gpu, _prices())

    assert ft_cost == 0.05 / 3600.0 / 100.0        # cpu rate, not gpu
    assert gpu_cost == 0.40 / 3600.0 / 100.0       # 4090 rate
    assert ft_cost < gpu_cost                       # cheaper on CPU at equal throughput


def test_unknown_hardware_raises_rather_than_mispricing():
    # An A100 profile with no price-table entry (and no device fallback) must
    # hard-fail, not silently borrow another device's rate.
    a100 = _cell(gpu_name="NVIDIA A100-SXM4-80GB", device="cuda:0")

    import pytest

    with pytest.raises(UnknownHardwareError):
        serving_cost_per_call(a100, _prices())


def test_dominates_is_false_for_the_dominated_direction():
    # Guard the asymmetry: strong dominates weak, so weak must NOT dominate strong.
    strong = _cell(selection_accuracy=0.93, p95_latency_ms=5.0,
                   distillation_usd=1.0, throughput_calls_per_sec=200.0)
    weak = _cell(selection_accuracy=0.90, p95_latency_ms=95.0,
                 distillation_usd=5.0, throughput_calls_per_sec=10.0)

    assert dominates(strong, weak, _prices(), accuracy_key="selection_accuracy")
    assert not dominates(weak, strong, _prices(), accuracy_key="selection_accuracy")
