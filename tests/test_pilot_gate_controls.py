import json
from pathlib import Path

import yaml

from promptillery.analyze import (
    plan_same_count_control_configs,
    summarize_run,
    validate_pilot_gate,
)


def _write_run(
    root: Path,
    *,
    name: str,
    policy_name: str,
    control_name: str | None = None,
    seed: int = 13,
    token_budget: int = 25000,
    synthetic_record_budget: int | None = None,
    final_synthetic_count: int = 0,
    manifest_final_synthetic_count: int | None = None,
    cycle0_metric: float = 0.1,
    final_metric: float = 0.2,
    heldout_metric: float | None = None,
    canonical_label_count: int | None = None,
    observed_gold_label_count: int | None = None,
    heldout_observed_gold_label_count: int | None = None,
) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    run_manifest = {
        "run_id": name,
        "status": "completed",
        "selection_split": "validation",
        "policy_name": policy_name,
        "control_name": control_name,
        "seed": seed,
        "token_budget": token_budget,
        "synthetic_record_budget": synthetic_record_budget,
        "final_synthetic_count": (
            final_synthetic_count
            if manifest_final_synthetic_count is None
            else manifest_final_synthetic_count
        ),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest))
    (run_dir / "experiment_config.yaml").write_text(
        "\n".join(
            [
                f"name: {name}",
                f"policy_name: {policy_name}",
                f"seed: {seed}",
                f"token_budget: {token_budget}",
                f"control_name: {control_name or ''}",
                f"synthetic_record_budget: {synthetic_record_budget or ''}",
            ]
        )
    )
    final_metrics = {"macro_f1": final_metric}
    if canonical_label_count is not None:
        final_metrics["canonical_label_count"] = canonical_label_count
    if observed_gold_label_count is not None:
        final_metrics["observed_gold_label_count"] = observed_gold_label_count
    metrics = {"0": {"macro_f1": cycle0_metric}, "1": final_metrics}
    if heldout_metric is not None:
        metrics["heldout_test"] = {
            "macro_f1": heldout_metric,
            "_heldout_split": "test",
            "_selection_split": "validation",
            "_selected_cycle": 1,
        }
        if canonical_label_count is not None:
            metrics["heldout_test"]["canonical_label_count"] = canonical_label_count
        if heldout_observed_gold_label_count is not None:
            metrics["heldout_test"][
                "observed_gold_label_count"
            ] = heldout_observed_gold_label_count
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    (run_dir / "token_usage.json").write_text(
        json.dumps(
            {
                "cycles_completed": 2,
                "grand_total": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
                "per_cycle": [
                    {"cycle": 0, "cycle_total": {"total_tokens": 5}},
                    {"cycle": 1, "cycle_total": {"total_tokens": 10}},
                ],
            }
        )
    )
    (run_dir / "policy_decisions.jsonl").write_text(
        json.dumps(
            {
                "cycle": 1,
                "decision_id": f"{name}:d1",
                "policy_name": policy_name,
                "action_name": "final_cycle",
                "state": {"synthetic_count": final_synthetic_count},
                "metadata": {},
                "budget_before": {},
                "budget_after": {},
                "realized_cost": {},
                "predicted_cost": {},
            }
        )
        + "\n"
    )


def test_pilot_gate_requires_same_count_control(tmp_path):
    _write_run(
        tmp_path,
        name="main",
        policy_name="frugalkd_p",
        final_synthetic_count=3,
    )

    missing = validate_pilot_gate(
        tmp_path,
        metric="macro_f1",
        expected_policies=["frugalkd_p"],
        expected_seeds=["13"],
        expected_budgets=["25000"],
        require_teacher_attempts=False,
        require_frontier=False,
        require_same_count_control=True,
    )

    assert not missing["passed"]
    assert any(
        check["name"] == "same_count_controls_present"
        and not check["passed"]
        for check in missing["checks"]
    )


def test_pilot_gate_rejects_malformed_same_count_control(tmp_path):
    _write_run(
        tmp_path,
        name="main",
        policy_name="frugalkd_p",
        final_synthetic_count=3,
    )
    _write_run(
        tmp_path,
        name="bad_same_count",
        policy_name="cost_heuristic",
        control_name="same_count",
        synthetic_record_budget=3,
        final_synthetic_count=2,
    )

    report = validate_pilot_gate(
        tmp_path,
        metric="macro_f1",
        expected_policies=["frugalkd_p"],
        expected_seeds=["13"],
        expected_budgets=["25000"],
        require_teacher_attempts=False,
        require_frontier=False,
        require_same_count_control=True,
    )

    check = next(
        check
        for check in report["checks"]
        if check["name"] == "same_count_controls_present"
    )
    assert not report["passed"]
    assert check["malformed_controls"] == [
        {
            "run_id": "bad_same_count",
            "seed": "13",
            "token_budget": "25000",
            "final_synthetic_count": 2,
            "synthetic_record_budget": 3,
            "source_policy": "frugalkd_p",
            "source_final_synthetic_count": 3,
        }
    ]


def test_pilot_gate_accepts_valid_same_count_control(tmp_path):
    _write_run(
        tmp_path,
        name="main",
        policy_name="frugalkd_p",
        final_synthetic_count=3,
    )
    _write_run(
        tmp_path,
        name="same_count",
        policy_name="cost_heuristic",
        control_name="same_count",
        synthetic_record_budget=3,
        final_synthetic_count=3,
    )

    report = validate_pilot_gate(
        tmp_path,
        metric="macro_f1",
        expected_policies=["frugalkd_p"],
        expected_seeds=["13"],
        expected_budgets=["25000"],
        require_teacher_attempts=False,
        require_frontier=False,
        require_same_count_control=True,
    )

    assert report["passed"]


def test_summarize_run_prefers_manifest_final_synthetic_count(tmp_path):
    _write_run(
        tmp_path,
        name="manifest_count",
        policy_name="cost_heuristic",
        final_synthetic_count=1,
        manifest_final_synthetic_count=3,
    )

    summary = summarize_run(tmp_path / "manifest_count", metric="macro_f1")

    assert summary["final_synthetic_count"] == 3


def test_summarize_run_exposes_heldout_test_metric(tmp_path):
    _write_run(
        tmp_path,
        name="with_heldout",
        policy_name="cost_heuristic",
        heldout_metric=0.17,
    )

    summary = summarize_run(tmp_path / "with_heldout", metric="macro_f1")

    assert summary["final_metric"] == 0.2
    assert summary["heldout_split"] == "test"
    assert summary["heldout_metric_name"] == "macro_f1"
    assert summary["heldout_metric"] == 0.17


def test_summarize_run_uses_eval_time_tokens_for_auc(tmp_path):
    _write_run(tmp_path, name="eval_tokens", policy_name="cost_heuristic")
    run_dir = tmp_path / "eval_tokens"
    metrics = {
        "0": {"macro_f1": 0.1, "_teacher_tokens_at_eval": 0},
        "1": {"macro_f1": 0.2, "_teacher_tokens_at_eval": 5},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))

    summary = summarize_run(run_dir, metric="macro_f1")

    assert round(summary["cycle_quality_cost_auc"], 3) == 0.150


def test_pilot_gate_requires_heldout_test_metric(tmp_path):
    _write_run(tmp_path, name="missing", policy_name="frugalkd_p")

    missing = validate_pilot_gate(
        tmp_path,
        metric="macro_f1",
        expected_policies=["frugalkd_p"],
        expected_seeds=["13"],
        expected_budgets=["25000"],
        require_teacher_attempts=False,
        require_frontier=False,
        require_heldout=True,
    )

    assert not missing["passed"]
    assert any(
        check["name"] == "heldout_test_metrics_present"
        and not check["passed"]
        for check in missing["checks"]
    )


def test_pilot_gate_accepts_heldout_test_metric(tmp_path):
    _write_run(
        tmp_path,
        name="with_heldout",
        policy_name="frugalkd_p",
        heldout_metric=0.17,
    )

    report = validate_pilot_gate(
        tmp_path,
        metric="macro_f1",
        expected_policies=["frugalkd_p"],
        expected_seeds=["13"],
        expected_budgets=["25000"],
        require_teacher_attempts=False,
        require_frontier=False,
        require_heldout=True,
    )

    assert report["passed"]


def test_pilot_gate_requires_full_canonical_label_coverage(tmp_path):
    _write_run(
        tmp_path,
        name="partial_labels",
        policy_name="frugalkd_p",
        heldout_metric=0.17,
        canonical_label_count=77,
        observed_gold_label_count=76,
        heldout_observed_gold_label_count=77,
    )

    report = validate_pilot_gate(
        tmp_path,
        metric="macro_f1",
        expected_policies=["frugalkd_p"],
        expected_seeds=["13"],
        expected_budgets=["25000"],
        require_teacher_attempts=False,
        require_frontier=False,
        require_heldout=True,
        require_full_label_coverage=True,
    )

    check = next(
        check
        for check in report["checks"]
        if check["name"] == "full_canonical_label_coverage"
    )
    assert not report["passed"]
    assert check["failures"] == [
        {
            "run_id": "partial_labels",
            "split": "validation",
            "observed_gold_label_count": 76,
            "canonical_label_count": 77,
        }
    ]


def test_pilot_gate_rejects_missing_label_coverage_counts(tmp_path):
    _write_run(
        tmp_path,
        name="missing_label_counts",
        policy_name="frugalkd_p",
        heldout_metric=0.17,
    )

    report = validate_pilot_gate(
        tmp_path,
        metric="macro_f1",
        expected_policies=["frugalkd_p"],
        expected_seeds=["13"],
        expected_budgets=["25000"],
        require_teacher_attempts=False,
        require_frontier=False,
        require_heldout=True,
        require_full_label_coverage=True,
    )

    check = next(
        check
        for check in report["checks"]
        if check["name"] == "full_canonical_label_coverage"
    )
    assert not report["passed"]
    assert check["failures"] == [
        {
            "run_id": "missing_label_counts",
            "split": "validation",
            "observed_gold_label_count": None,
            "canonical_label_count": None,
        },
        {
            "run_id": "missing_label_counts",
            "split": "test",
            "observed_gold_label_count": None,
            "canonical_label_count": None,
        },
    ]


def test_pilot_gate_accepts_full_canonical_label_coverage(tmp_path):
    _write_run(
        tmp_path,
        name="full_labels",
        policy_name="frugalkd_p",
        heldout_metric=0.17,
        canonical_label_count=77,
        observed_gold_label_count=77,
        heldout_observed_gold_label_count=77,
    )

    report = validate_pilot_gate(
        tmp_path,
        metric="macro_f1",
        expected_policies=["frugalkd_p"],
        expected_seeds=["13"],
        expected_budgets=["25000"],
        require_teacher_attempts=False,
        require_frontier=False,
        require_heldout=True,
        require_full_label_coverage=True,
    )

    assert report["passed"]


def test_pilot_gate_requires_success_policy_auc_win(tmp_path):
    _write_run(
        tmp_path,
        name="frugal",
        policy_name="frugalkd_p",
        cycle0_metric=0.1,
        final_metric=0.2,
    )
    _write_run(
        tmp_path,
        name="heuristic",
        policy_name="cost_heuristic",
        cycle0_metric=0.2,
        final_metric=0.3,
    )

    report = validate_pilot_gate(
        tmp_path,
        metric="macro_f1",
        expected_policies=["frugalkd_p", "cost_heuristic"],
        expected_seeds=["13"],
        expected_budgets=["25000"],
        require_teacher_attempts=False,
        require_frontier=False,
        success_policy="frugalkd_p",
        success_baselines=["cost_heuristic"],
        min_auc_win_rate=1.0,
    )

    check = next(
        check
        for check in report["checks"]
        if check["name"] == "success_policy_beats_baselines_auc"
    )
    assert not report["passed"]
    assert check["win_rate"] == 0.0


def test_pilot_gate_accepts_success_policy_heldout_win(tmp_path):
    _write_run(
        tmp_path,
        name="frugal",
        policy_name="frugalkd_p",
        heldout_metric=0.4,
    )
    _write_run(
        tmp_path,
        name="heuristic",
        policy_name="cost_heuristic",
        heldout_metric=0.3,
    )

    report = validate_pilot_gate(
        tmp_path,
        metric="macro_f1",
        expected_policies=["frugalkd_p", "cost_heuristic"],
        expected_seeds=["13"],
        expected_budgets=["25000"],
        require_teacher_attempts=False,
        require_frontier=False,
        require_heldout=True,
        success_policy="frugalkd_p",
        success_baselines=["cost_heuristic"],
        min_heldout_win_rate=1.0,
    )

    assert report["passed"]


def test_pilot_gate_requires_same_count_success(tmp_path):
    _write_run(
        tmp_path,
        name="frugal",
        policy_name="frugalkd_p",
        final_synthetic_count=3,
        heldout_metric=0.3,
    )
    _write_run(
        tmp_path,
        name="same_count",
        policy_name="cost_heuristic",
        control_name="same_count",
        synthetic_record_budget=3,
        final_synthetic_count=3,
        heldout_metric=0.35,
    )

    report = validate_pilot_gate(
        tmp_path,
        metric="macro_f1",
        expected_policies=["frugalkd_p"],
        expected_seeds=["13"],
        expected_budgets=["25000"],
        require_teacher_attempts=False,
        require_frontier=False,
        require_heldout=True,
        require_same_count_control=True,
        success_policy="frugalkd_p",
        min_heldout_win_rate=1.0,
        require_same_count_success=True,
    )

    check = next(
        check
        for check in report["checks"]
        if check["name"] == "success_policy_beats_same_count_heldout"
    )
    assert not report["passed"]
    assert check["win_rate"] == 0.0


def test_pilot_gate_requires_acquisition_decision_attempt_join(tmp_path):
    _write_run(tmp_path, name="missing_attempt", policy_name="frugalkd_p")
    run_dir = tmp_path / "missing_attempt"
    (run_dir / "policy_decisions.jsonl").write_text(
        json.dumps(
            {
                "cycle": 0,
                "decision_id": "missing_attempt:d0",
                "policy_name": "frugalkd_p",
                "action_name": "coverage:cheap:b8",
                "state": {},
                "metadata": {"acquisition_outcome": "augment"},
                "budget_before": {},
                "budget_after": {},
                "realized_cost": {},
                "predicted_cost": {},
            }
        )
        + "\n"
    )
    (run_dir / "teacher_attempts.jsonl").write_text("")

    report = validate_pilot_gate(
        tmp_path,
        metric="macro_f1",
        expected_policies=["frugalkd_p"],
        expected_seeds=["13"],
        expected_budgets=["25000"],
        require_teacher_attempts=False,
        require_frontier=False,
    )

    check = next(
        check
        for check in report["checks"]
        if check["name"] == "acquisition_decisions_have_teacher_attempts"
    )
    assert not report["passed"]
    assert check["missing_decision_ids"] == ["missing_attempt:d0"]


def test_plan_same_count_control_configs_from_source_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    _write_run(
        runs_dir,
        name="source_25k",
        policy_name="frugalkd_p",
        seed=13,
        token_budget=25000,
        final_synthetic_count=7,
    )
    _write_run(
        runs_dir,
        name="source_100k",
        policy_name="frugalkd_p",
        seed=13,
        token_budget=100000,
        final_synthetic_count=11,
    )
    _write_run(
        runs_dir,
        name="heuristic_25k",
        policy_name="cost_heuristic",
        seed=13,
        token_budget=25000,
        final_synthetic_count=3,
    )

    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        "\n".join(
            [
                "name: banking_pilot",
                "policy_name:",
                "  - cost_heuristic",
                "  - frugalkd_p",
                "seed:",
                "  - 13",
                "token_budget:",
                "  - 25000",
                "  - 100000",
                "base_output_dir: ../experiments/policy_pilot",
            ]
        ),
        encoding="utf-8",
    )

    plan = plan_same_count_control_configs(
        runs_dir,
        base_config,
        tmp_path / "same_count_configs",
        metric="macro_f1",
        control_base_output_dir="../experiments/policy_pilot/same_count",
    )

    assert [row["synthetic_record_budget"] for row in plan] == [7, 11]
    configs = sorted((tmp_path / "same_count_configs").glob("*.yaml"))
    assert len(configs) == 2

    first_config = yaml.safe_load(configs[0].read_text())
    assert first_config["policy_name"] == "cost_heuristic"
    assert first_config["control_name"] == "same_count"
    assert first_config["seed"] == 13
    assert first_config["token_budget"] in {25000, 100000}
    assert first_config["synthetic_record_budget"] in {7, 11}
    assert first_config["base_output_dir"] == "../experiments/policy_pilot/same_count"

    manifest = json.loads(
        (tmp_path / "same_count_configs" / "same_count_plan.json").read_text()
    )
    assert len(manifest) == 2
