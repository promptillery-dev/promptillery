"""Tests for selectors.py -- the oracle, baselines, and recommender arms.

All selectors share ``select(cells, target, prices, teacher) -> Selection``.
The oracle chooses on held-out truth (and trains everything); the baselines are
fair strawmen a practitioner actually uses; the two recommender arms differ only
in whether they may escalate the teacher budget.
"""


from promptillery.pareto import Cell, HardwarePrices
from promptillery.recommender import Target, Teacher
from promptillery.selectors import (
    expert_default,
    oracle,
    random_selector,
    recommender_budget_search,
    recommender_rules,
    standard_selectors,
    volume_threshold,
)

PRICES = HardwarePrices({"NVIDIA GeForce RTX 4090": 0.40, "cpu": 0.05})
TEACHER = Teacher(usd_per_call=0.002)


def _cell(**overrides):
    base = dict(
        dataset="agnews", dataset_subset="", student_model="student",
        student_type="transformers", policy_name="uncertainty", control_name="",
        token_budget=20_000_000, selection_accuracy=0.93, heldout_accuracy=0.92,
        distillation_usd=1.0, p95_latency_ms=5.0, throughput_calls_per_sec=200.0,
        device="cuda:0", gpu_name="NVIDIA GeForce RTX 4090", expected_cycles=1,
    )
    base.update(overrides)
    return Cell(**base)


def test_oracle_chooses_cheapest_feasible_on_heldout_and_trains_all():
    # `mirage` looks best on selection but misses the floor on held-out; the
    # oracle sees the truth and skips it, taking the honest-but-dearer `real`.
    mirage = _cell(student_model="mirage", selection_accuracy=0.99,
                   heldout_accuracy=0.80, distillation_usd=0.5)
    real = _cell(student_model="real", selection_accuracy=0.92,
                 heldout_accuracy=0.93, distillation_usd=2.0)
    # volume well above real's ~1000-call break-even, so the teacher-below-
    # break-even rule (A.2) doesn't intervene here -- this test is about the
    # selection-vs-heldout accuracy gap, not the teacher trade-off.
    target = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=10_000)

    sel = oracle([mirage, real], target, PRICES, TEACHER)

    assert sel.cell == real
    assert set(sel.cells_evaluated) == {mirage, real}  # oracle trains everything


def test_expert_default_picks_the_best_encoder_ignoring_cost_and_floor():
    slm = _cell(student_model="slm", student_type="slm", selection_accuracy=0.97)
    enc_lo = _cell(student_model="ettin-enc", student_type="transformers",
                   selection_accuracy=0.91)
    enc_hi = _cell(student_model="roberta", student_type="transformers",
                   selection_accuracy=0.93)
    target = Target(accuracy_floor=0.99, latency_budget_ms=None, volume=1000)

    sel = expert_default([slm, enc_lo, enc_hi], target, PRICES, TEACHER)

    assert sel.cell == enc_hi          # best *encoder*, though the slm scores higher
    assert len(sel.cells_evaluated) == 1


def test_volume_threshold_below_threshold_calls_the_teacher():
    select = volume_threshold(threshold=1_000_000)
    target = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=1000)

    sel = select([_cell()], target, PRICES, TEACHER)

    assert sel.cell is None


def test_volume_threshold_ignores_floor_and_deploys_cheapest_to_serve():
    # Above threshold it deploys the cheapest-to-serve cell consulting neither
    # floor nor latency -- happily shipping FastText into a task it cannot do.
    fasttext = _cell(student_model="fasttext", student_type="fasttext",
                     device="cpu", gpu_name=None, selection_accuracy=0.82,
                     throughput_calls_per_sec=100_000.0)
    roberta = _cell(student_model="roberta", selection_accuracy=0.93,
                    throughput_calls_per_sec=200.0)
    target = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=1_000_000)
    select = volume_threshold(threshold=1000)

    sel = select([roberta, fasttext], target, PRICES, TEACHER)

    assert sel.cell == fasttext  # picked despite being below the floor


def test_random_selector_is_deterministic_under_a_seed():
    cells = [_cell(student_model=name) for name in ("a", "b", "c", "d")]
    target = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=1000)

    first = random_selector(seed=7)(cells, target, PRICES, TEACHER)
    again = random_selector(seed=7)(cells, target, PRICES, TEACHER)

    assert first.cell == again.cell
    assert len(first.cells_evaluated) == 1


def test_budget_search_escalates_cycles_where_rules_falls_back():
    # One student, two arms: the base 1-cycle arm misses the floor; the 10-cycle
    # arm clears it. Rules is pinned to the base arm -> teacher fallback. Budget
    # search may escalate -> deploys the 10-cycle arm.
    base = _cell(student_model="s", expected_cycles=1, distillation_usd=1.0,
                 selection_accuracy=0.85)
    escalated = _cell(student_model="s", expected_cycles=10, distillation_usd=5.0,
                      selection_accuracy=0.93)
    # volume well above escalated's ~2500-call break-even, so the teacher-
    # below-break-even rule (A.2) doesn't intervene here -- this test is about
    # budget escalation, not the teacher trade-off.
    target = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=10_000)

    rules_sel = recommender_rules([base, escalated], target, PRICES, TEACHER)
    search_sel = recommender_budget_search([base, escalated], target, PRICES, TEACHER)

    assert rules_sel.cell is None       # rules cannot escalate -> teacher fallback
    assert search_sel.cell == escalated  # budget search buys the accuracy


def test_standard_selectors_names_are_stable_and_exclude_the_oracle():
    selectors = standard_selectors(random_seed=0, volume_threshold_value=1000)

    assert "oracle" not in selectors  # oracle is the ground truth, not a row
    assert set(selectors) == {
        "expert_default", "random", "volume_threshold",
        "recommender_rules", "recommender_budget_search",
    }


def test_oracle_prefers_the_teacher_below_break_even():
    from promptillery.selectors import oracle
    student = _cell(distillation_usd=10.0, heldout_accuracy=0.95)
    target = Target(accuracy_floor=0.90, latency_budget_ms=None, volume=100)

    sel = oracle([student], target, PRICES, TEACHER)

    assert sel.cell is None
    assert sel.reason == "teacher_cheaper_at_volume"
