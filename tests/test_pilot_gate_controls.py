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
    (run_dir / "metrics.json").write_text(
        json.dumps({"0": {"macro_f1": 0.1}, "1": {"macro_f1": 0.2}})
    )
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
