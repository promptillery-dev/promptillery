import csv
import json
from pathlib import Path

import yaml

from promptillery.analyze import (
    analyze_runs,
    plan_same_count_control_configs,
    summarize_run,
    validate_pilot_gate,
    write_paper_report,
)


def _write_run(
    root: Path,
    *,
    name: str,
    policy_name: str,
    control_name: str | None = None,
    status: str = "completed",
    paper_mode: bool = True,
    seed: int = 13,
    token_budget: int = 25000,
    synthetic_record_budget: int | None = None,
    final_synthetic_count: int = 0,
    manifest_final_synthetic_count: int | None = None,
    expected_cycles: int | None = 2,
    cycles_completed: int = 2,
    action_space_id: str | None = "action-space-a",
    config_extra: dict | None = None,
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
        "status": status,
        "selection_split": "validation",
        "paper_mode": paper_mode,
        "policy_name": policy_name,
        "control_name": control_name,
        "seed": seed,
        "token_budget": token_budget,
        "cycles_completed": cycles_completed,
        "action_space": {"action_space_id": action_space_id or ""},
        "synthetic_record_budget": synthetic_record_budget,
        "final_synthetic_count": (
            final_synthetic_count
            if manifest_final_synthetic_count is None
            else manifest_final_synthetic_count
        ),
    }
    if expected_cycles is not None:
        run_manifest["expected_cycles"] = expected_cycles
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest))
    config_data = {
        "name": name,
        "policy_name": policy_name,
        "seed": seed,
        "token_budget": token_budget,
        "control_name": control_name or "",
        "synthetic_record_budget": synthetic_record_budget or "",
    }
    config_data.update(config_extra or {})
    (run_dir / "experiment_config.yaml").write_text(
        yaml.safe_dump(config_data, sort_keys=False),
        encoding="utf-8",
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
            metrics["heldout_test"]["observed_gold_label_count"] = (
                heldout_observed_gold_label_count
            )
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    (run_dir / "token_usage.json").write_text(
        json.dumps(
            {
                "cycles_completed": cycles_completed,
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
        check["name"] == "same_count_controls_present" and not check["passed"]
        for check in missing["checks"]
    )


def test_paper_report_writes_reviewer_tables(tmp_path):
    _write_run(
        tmp_path,
        name="frugal",
        policy_name="frugalkd_p",
        cycle0_metric=0.4,
        final_metric=0.8,
        heldout_metric=0.75,
        final_synthetic_count=4,
    )
    _write_run(
        tmp_path,
        name="heuristic",
        policy_name="cost_heuristic",
        cycle0_metric=0.4,
        final_metric=0.6,
        heldout_metric=0.55,
        final_synthetic_count=4,
    )

    rows = analyze_runs(tmp_path, metric="macro_f1")
    paths = write_paper_report(
        tmp_path,
        rows,
        tmp_path / "paper_report",
        baseline_policies=["cost_heuristic"],
    )

    assert paths["paper_readiness_report"].exists()
    assert paths["paper_main_results"].exists()
    assert paths["paper_budget_audit"].exists()
    assert paths["paper_pairwise_summary"].exists()
    assert paths["paper_quality_cost_points"].exists()
    assert paths["paper_action_cycle_frequencies"].exists()
    with paths["paper_pairwise_deltas"].open(newline="", encoding="utf-8") as f:
        delta_rows = list(csv.DictReader(f))
    with paths["paper_pairwise_summary"].open(newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    with paths["paper_quality_cost_points"].open(newline="", encoding="utf-8") as f:
        point_rows = list(csv.DictReader(f))
    with paths["paper_action_cycle_frequencies"].open(
        newline="", encoding="utf-8"
    ) as f:
        action_cycle_rows = list(csv.DictReader(f))

    assert len(delta_rows) == 1
    assert delta_rows[0]["success_policy"] == "frugalkd_p"
    assert delta_rows[0]["baseline_policy"] == "cost_heuristic"
    assert float(delta_rows[0]["delta_final_metric"]) > 0
    assert delta_rows[0]["final_win"] == "True"
    assert len(summary_rows) == 2
    budget_summary = next(
        row for row in summary_rows if row["summary_scope"] == "budget"
    )
    all_budget_summary = next(
        row for row in summary_rows if row["summary_scope"] == "all_budgets"
    )
    assert budget_summary["final_wins"] == "1"
    assert budget_summary["final_losses"] == "0"
    assert budget_summary["final_win_rate"] == "1.0"
    assert budget_summary["final_sign_test_p"] == "1.0"
    assert float(budget_summary["final_mean_delta_ci_low"]) > 0
    assert (
        budget_summary["final_mean_delta_ci_low"]
        == budget_summary["final_mean_delta_ci_high"]
    )
    assert all_budget_summary["token_budget"] == "ALL"
    assert all_budget_summary["final_n"] == "1"
    frugal_points = [row for row in point_rows if row["run_id"] == "frugal"]
    assert [row["split"] for row in frugal_points] == [
        "validation",
        "validation",
        "test",
    ]
    assert frugal_points[-1]["metric_value"] == "0.75"
    assert frugal_points[-1]["cumulative_online_teacher_total_tokens"] == "15"
    assert any(
        row["cycle"] == "1" and row["action_name"] == "final_cycle"
        for row in action_cycle_rows
    )


def test_paper_report_combines_control_roots(tmp_path):
    core_root = tmp_path / "core"
    control_root = tmp_path / "active_control"
    _write_run(
        core_root,
        name="frugal",
        policy_name="frugalkd_p",
        final_metric=0.8,
        heldout_metric=0.75,
    )
    _write_run(
        control_root,
        name="active",
        policy_name="cheap_only",
        control_name="active_kd_uncertainty",
        final_metric=0.6,
        heldout_metric=0.55,
    )

    rows = analyze_runs(core_root, metric="macro_f1") + analyze_runs(
        control_root, metric="macro_f1"
    )
    paths = write_paper_report(
        [core_root, control_root],
        rows,
        tmp_path / "paper_report",
        baseline_policies=["cheap_only"],
    )

    with paths["paper_pairwise_deltas"].open(newline="", encoding="utf-8") as f:
        delta_rows = list(csv.DictReader(f))
    with paths["paper_main_results"].open(newline="", encoding="utf-8") as f:
        main_rows = list(csv.DictReader(f))
    report = paths["paper_readiness_report"].read_text(encoding="utf-8")

    assert {row["policy_name"] for row in main_rows} == {"frugalkd_p", "cheap_only"}
    assert len(delta_rows) == 1
    assert delta_rows[0]["baseline_control_name"] == "active_kd_uncertainty"
    assert str(core_root) in report
    assert str(control_root) in report


def test_pilot_gate_requires_paper_mode_when_requested(tmp_path):
    _write_run(
        tmp_path,
        name="nonpaper",
        policy_name="frugalkd_p",
        paper_mode=False,
    )

    report = validate_pilot_gate(
        tmp_path,
        metric="macro_f1",
        expected_policies=["frugalkd_p"],
        expected_seeds=["13"],
        expected_budgets=["25000"],
        require_teacher_attempts=False,
        require_frontier=False,
        require_paper_mode=True,
    )

    check = next(
        check for check in report["checks"] if check["name"] == "paper_mode_enabled"
    )
    assert not report["passed"]
    assert check["violating_run_ids"] == ["nonpaper"]


def test_pilot_gate_rejects_bad_paper_cycle_metadata(tmp_path):
    _write_run(
        tmp_path,
        name="bad_cycles",
        policy_name="frugalkd_p",
        cycles_completed=3,
        expected_cycles=2,
    )

    report = validate_pilot_gate(
        tmp_path,
        metric="macro_f1",
        expected_policies=["frugalkd_p"],
        expected_seeds=["13"],
        expected_budgets=["25000"],
        require_teacher_attempts=False,
        require_frontier=False,
        require_paper_mode=True,
    )

    check = next(
        check for check in report["checks"] if check["name"] == "paper_cycle_bounds"
    )
    assert not report["passed"]
    assert check["failures"] == [
        {
            "run_id": "bad_cycles",
            "cycles_completed": 3,
            "expected_cycles": 2,
        }
    ]


def test_pilot_gate_rejects_non_provider_teacher_debits(tmp_path):
    _write_run(tmp_path, name="reserved", policy_name="frugalkd_p")
    run_dir = tmp_path / "reserved"
    (run_dir / "teacher_attempts.jsonl").write_text(
        json.dumps(
            {
                "cycle": 0,
                "attempt_id": "reserved:a0",
                "decision_id": "reserved:d1",
                "run_id": "reserved",
                "status": "failed",
                "failure_type": "ValueError",
                "predicted_cost": {"total_tokens": 10},
                "provider_reported_cost": {},
                "ledger_debit_cost": {"total_tokens": 10},
                "realized_cost": {"total_tokens": 10},
                "ledger_debit_source": "reserved_bound",
                "budget_before": {"tokens_remaining": 100},
                "budget_after": {"tokens_remaining": 90},
                "metadata": {},
            }
        )
        + "\n"
    )

    report = validate_pilot_gate(
        tmp_path,
        metric="macro_f1",
        expected_policies=["frugalkd_p"],
        expected_seeds=["13"],
        expected_budgets=["25000"],
        require_teacher_attempts=True,
        require_frontier=False,
        require_provider_reported_usage=True,
    )

    teacher_check = next(
        check
        for check in report["checks"]
        if check["name"] == "teacher_attempts_successful"
    )
    provider_check = next(
        check
        for check in report["checks"]
        if check["name"] == "provider_reported_usage_present"
    )
    assert not report["passed"]
    assert teacher_check["violating_run_ids"] == ["reserved"]
    assert provider_check["missing_provider_attempt_ids"] == ["reserved:a0"]
    assert provider_check["non_provider_debit_attempt_ids"] == ["reserved:a0"]


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


def test_pilot_gate_rejects_same_count_action_space_mismatch(tmp_path):
    _write_run(
        tmp_path,
        name="main",
        policy_name="frugalkd_p",
        final_synthetic_count=3,
        action_space_id="source-actions",
    )
    _write_run(
        tmp_path,
        name="same_count",
        policy_name="cost_heuristic",
        control_name="same_count",
        synthetic_record_budget=3,
        final_synthetic_count=3,
        action_space_id="control-actions",
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
    assert check["action_space_mismatches"] == [
        {
            "run_id": "same_count",
            "seed": "13",
            "token_budget": "25000",
            "action_space_id": "control-actions",
            "source_policy": "frugalkd_p",
            "source_action_space_id": "source-actions",
        }
    ]


def test_pilot_gate_rejects_duplicate_same_count_controls(tmp_path):
    _write_run(
        tmp_path,
        name="main",
        policy_name="frugalkd_p",
        final_synthetic_count=3,
    )
    _write_run(
        tmp_path,
        name="same_count_a",
        policy_name="cost_heuristic",
        control_name="same_count",
        synthetic_record_budget=3,
        final_synthetic_count=3,
    )
    _write_run(
        tmp_path,
        name="same_count_b",
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

    check = next(
        check
        for check in report["checks"]
        if check["name"] == "same_count_controls_present"
    )
    assert not report["passed"]
    assert check["duplicate_controls"] == [
        {
            "seed": "13",
            "token_budget": "25000",
            "run_ids": ["same_count_a", "same_count_b"],
        }
    ]


def test_pilot_gate_rejects_same_count_config_mismatch(tmp_path):
    _write_run(
        tmp_path,
        name="main",
        policy_name="frugalkd_p",
        final_synthetic_count=3,
        config_extra={"trainer_config": {"budget_splits": []}},
    )
    _write_run(
        tmp_path,
        name="same_count",
        policy_name="cost_heuristic",
        control_name="same_count",
        synthetic_record_budget=3,
        final_synthetic_count=3,
        config_extra={"trainer_config": {"budget_splits": ["train"]}},
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
    assert check["config_mismatches"][0]["run_id"] == "same_count"
    assert (
        check["config_mismatches"][0]["config_hash"]
        != check["config_mismatches"][0]["source_config_hash"]
    )


def test_pilot_gate_rejects_main_action_space_mismatch(tmp_path):
    _write_run(
        tmp_path,
        name="frugal",
        policy_name="frugalkd_p",
        action_space_id="frugal-actions",
    )
    _write_run(
        tmp_path,
        name="heuristic",
        policy_name="cost_heuristic",
        action_space_id="heuristic-actions",
    )

    report = validate_pilot_gate(
        tmp_path,
        metric="macro_f1",
        expected_policies=["frugalkd_p", "cost_heuristic"],
        expected_seeds=["13"],
        expected_budgets=["25000"],
        require_teacher_attempts=False,
        require_frontier=False,
    )

    check = next(
        check for check in report["checks"] if check["name"] == "action_space_parity"
    )
    assert not report["passed"]
    assert check["failures"] == [
        {
            "key": ["", "", "", "", "13", "25000", "validation", "macro_f1", "max"],
            "action_space_ids": ["frugal-actions", "heuristic-actions"],
            "run_ids": ["frugal", "heuristic"],
        }
    ]


def test_pilot_gate_rejects_duplicate_policy_seed_budget_runs(tmp_path):
    _write_run(tmp_path, name="run_a", policy_name="frugalkd_p")
    _write_run(tmp_path, name="run_b", policy_name="frugalkd_p")

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
        if check["name"] == "expected_policy_seed_budget_grid"
    )
    assert not report["passed"]
    assert check["duplicates"] == [
        {
            "combo": ["frugalkd_p", "13", "25000"],
            "run_ids": ["run_a", "run_b"],
        }
    ]


def test_pilot_gate_rejects_unexpected_fixed_policy_operator(tmp_path):
    _write_run(tmp_path, name="fixed", policy_name="fixed_coverage")
    (tmp_path / "fixed" / "policy_decisions.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "cycle": 0,
                        "decision_id": "fixed:d0",
                        "policy_name": "fixed_coverage",
                        "action_name": "coverage:cheap:b8",
                        "action": {
                            "prompt_operator": "coverage",
                            "teacher_tier": "cheap",
                            "batch_size": 8,
                        },
                        "state": {},
                        "metadata": {},
                        "budget_before": {},
                        "budget_after": {},
                        "realized_cost": {},
                        "predicted_cost": {},
                    }
                ),
                json.dumps(
                    {
                        "cycle": 1,
                        "decision_id": "fixed:d1",
                        "policy_name": "fixed_coverage",
                        "action_name": "repair:cheap:b8",
                        "action": {
                            "prompt_operator": "repair",
                            "teacher_tier": "cheap",
                            "batch_size": 8,
                        },
                        "state": {},
                        "metadata": {},
                        "budget_before": {},
                        "budget_after": {},
                        "realized_cost": {},
                        "predicted_cost": {},
                    }
                ),
            ]
        )
        + "\n"
    )

    report = validate_pilot_gate(
        tmp_path,
        metric="macro_f1",
        expected_policies=["fixed_coverage"],
        expected_seeds=["13"],
        expected_budgets=["25000"],
        require_teacher_attempts=False,
        require_frontier=False,
    )

    check = next(
        check
        for check in report["checks"]
        if check["name"] == "fixed_policies_use_distinct_prompt_operators"
    )
    assert not report["passed"]
    assert check["failures"] == [
        {
            "policy_name": "fixed_coverage",
            "expected_operator": "coverage",
            "found_operators": ["coverage", "repair"],
            "unexpected_operators": ["repair"],
        }
    ]


def test_pilot_gate_rejects_wrong_same_count_control_policy(tmp_path):
    _write_run(
        tmp_path,
        name="main",
        policy_name="frugalkd_p",
        final_synthetic_count=3,
    )
    _write_run(
        tmp_path,
        name="same_count",
        policy_name="random_feasible",
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
        same_count_control_policy="cost_heuristic",
    )

    check = next(
        check
        for check in report["checks"]
        if check["name"] == "same_count_controls_present"
    )
    assert not report["passed"]
    assert check["wrong_policy_controls"] == [
        {
            "run_id": "same_count",
            "policy_name": "random_feasible",
            "expected_policy_name": "cost_heuristic",
        }
    ]


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
    _write_run(
        tmp_path,
        name="eval_tokens",
        policy_name="cost_heuristic",
        token_budget=5,
    )
    run_dir = tmp_path / "eval_tokens"
    metrics = {
        "0": {"macro_f1": 0.1, "_teacher_tokens_at_eval": 0},
        "1": {"macro_f1": 0.2, "_teacher_tokens_at_eval": 5},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))

    summary = summarize_run(run_dir, metric="macro_f1")

    assert round(summary["cycle_quality_cost_auc"], 3) == 0.150


def test_summarize_run_extends_auc_to_token_budget(tmp_path):
    _write_run(
        tmp_path,
        name="budget_horizon",
        policy_name="cost_heuristic",
        token_budget=10,
    )
    run_dir = tmp_path / "budget_horizon"
    metrics = {
        "0": {"macro_f1": 0.1, "_teacher_tokens_at_eval": 0},
        "1": {"macro_f1": 0.2, "_teacher_tokens_at_eval": 5},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))

    summary = summarize_run(run_dir, metric="macro_f1")

    assert round(summary["cycle_quality_cost_auc"], 3) == 0.175


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
        check["name"] == "heldout_test_metrics_present" and not check["passed"]
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


def test_pilot_gate_does_not_count_ties_as_wins(tmp_path):
    _write_run(
        tmp_path,
        name="frugal",
        policy_name="frugalkd_p",
        final_metric=0.2,
    )
    _write_run(
        tmp_path,
        name="heuristic",
        policy_name="cost_heuristic",
        final_metric=0.2,
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
        min_final_win_rate=1.0,
    )

    check = next(
        check
        for check in report["checks"]
        if check["name"] == "success_policy_beats_baselines_final"
    )
    assert not report["passed"]
    assert check["failures"][0]["delta"] == 0.0
    assert check["failures"][0]["won"] is False


def test_pilot_gate_requires_each_baseline_to_pass(tmp_path):
    _write_run(
        tmp_path,
        name="frugal",
        policy_name="frugalkd_p",
        final_metric=0.3,
    )
    _write_run(
        tmp_path,
        name="heuristic",
        policy_name="cost_heuristic",
        final_metric=0.4,
    )
    _write_run(
        tmp_path,
        name="random",
        policy_name="random_feasible",
        final_metric=0.1,
    )

    report = validate_pilot_gate(
        tmp_path,
        metric="macro_f1",
        expected_policies=["frugalkd_p", "cost_heuristic", "random_feasible"],
        expected_seeds=["13"],
        expected_budgets=["25000"],
        require_teacher_attempts=False,
        require_frontier=False,
        success_policy="frugalkd_p",
        success_baselines=["cost_heuristic", "random_feasible"],
        min_final_win_rate=0.5,
    )

    check = next(
        check
        for check in report["checks"]
        if check["name"] == "success_policy_beats_baselines_final"
    )
    assert not report["passed"]
    assert [
        (row["baseline"], row["passed"], row["win_rate"])
        for row in check["per_baseline"]
    ] == [
        ("cost_heuristic", False, 0.0),
        ("random_feasible", True, 1.0),
    ]


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
        config_extra={"trainer_config": {"budget_splits": []}},
    )
    _write_run(
        runs_dir,
        name="source_100k",
        policy_name="frugalkd_p",
        seed=13,
        token_budget=100000,
        final_synthetic_count=11,
        config_extra={"trainer_config": {"budget_splits": []}},
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
                "trainer_config:",
                "  budget_splits:",
                "    - train",
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
    assert first_config["trainer_config"] == {"budget_splits": []}
    assert first_config["base_output_dir"] == "../experiments/policy_pilot/same_count"

    manifest = json.loads(
        (tmp_path / "same_count_configs" / "same_count_plan.json").read_text()
    )
    assert len(manifest) == 2
