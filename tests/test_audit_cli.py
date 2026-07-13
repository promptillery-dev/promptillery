"""audit_run orchestrator and `promptillery audit` CLI tests."""

import csv
import json

import pytest
from typer.testing import CliRunner

from promptillery.audit import audit_run, latex_rows
from promptillery.cli import app

from conftest import AUG_CYCLE_1, AUG_CYCLE_2

runner = CliRunner()


def _fake_predict(texts):
    # A "verifier" that always answers positive.
    return ["positive"] * len(texts)


class TestAuditRun:
    def test_cycle_and_cumulative_metrics(self, audit_run_dir):
        result = audit_run(audit_run_dir(), predict_fn=_fake_predict)
        assert [c.cycle for c in result.cycles] == [1, 2]
        cycle1 = result.cycles[0]
        # 1 exact dup of seed among 3 rows.
        assert cycle1.n_rows == len(AUG_CYCLE_1)
        assert cycle1.dup_pct == pytest.approx(1 / 3)
        # 1 near-dup (LONG_BASE + " lovingly"), excluding the exact dup.
        assert cycle1.ndup_pct == pytest.approx(1 / 3)
        # (4 requested - 3 accepted) + 0 failed attempts.
        assert cycle1.n_fail == 1
        cycle2 = result.cycles[1]
        assert cycle2.n_rows == len(AUG_CYCLE_2)
        assert cycle2.dup_pct == 0.0
        # (10 requested - 2 accepted) + 1 failed attempt.
        assert cycle2.n_fail == 9
        # Cumulative rolls everything up.
        assert result.cumulative.cycle == "all"
        assert result.cumulative.n_rows == len(AUG_CYCLE_1) + len(AUG_CYCLE_2)
        assert result.cumulative.n_fail == 10

    def test_verifier_columns(self, audit_run_dir):
        result = audit_run(audit_run_dir(), predict_fn=_fake_predict)
        # Cycle 1 has 2 positive / 1 negative rows; verifier says positive.
        assert result.cycles[0].lblc_verifier == pytest.approx(2 / 3)
        # Calibration on the gold test split (1 positive / 1 negative).
        assert result.verifier_test_accuracy == pytest.approx(1 / 2)

    def test_writes_json_and_csv(self, audit_run_dir):
        run_dir = audit_run_dir()
        audit_run(run_dir, predict_fn=_fake_predict)
        audit_dir = run_dir / "audit"
        payload = json.loads((audit_dir / "audit.json").read_text())
        assert payload["experiment"] == "audit_fixture_run"
        assert len(payload["cycles"]) == 2
        assert payload["cycles"][0]["failures"]["exemplars"]
        with (audit_dir / "audit.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert [row["cycle"] for row in rows] == ["1", "2", "all"]

    def test_output_dir_override_uses_run_subdir(self, audit_run_dir, tmp_path):
        run_dir = audit_run_dir()
        out = tmp_path / "audits"
        audit_run(run_dir, output_dir=str(out))
        assert (out / run_dir.name / "audit.json").exists()

    def test_no_verifier_leaves_column_none(self, audit_run_dir):
        result = audit_run(audit_run_dir())
        assert result.cycles[0].lblc_verifier is None
        assert result.verifier_test_accuracy is None

    def test_probe_writes_usage_ledger(self, audit_run_dir, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        def fake(**kwargs):
            return {
                "choices": [{"message": {"content": "positive"}}],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 2,
                    "total_tokens": 42,
                },
            }

        monkeypatch.setattr("litellm.completion", fake)
        run_dir = audit_run_dir()
        result = audit_run(run_dir, probe=True, probe_k=3, seed=13)
        assert result.probe is not None
        assert result.probe.k == 3
        usage = json.loads((run_dir / "audit" / "audit_usage.json").read_text())
        assert usage["grand_total"]["total_tokens"] == 3 * 42
        # Probe cost surfaces on the CSV's cumulative row.
        with (run_dir / "audit" / "audit.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert rows[-1]["probe_total_tokens"] == str(3 * 42)

    def test_probe_without_key_refuses(self, audit_run_dir, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            audit_run(audit_run_dir(), probe=True, probe_k=2)

    def test_probe_failure_still_saves_partial_usage_ledger(
        self, audit_run_dir, monkeypatch
    ):
        # A mid-loop litellm error (call 3 of 4) must not lose the ledger for
        # the two calls that already spent tokens (final-review finding 2).
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        calls = []

        def flaky(**kwargs):
            calls.append(kwargs)
            if len(calls) > 2:
                raise RuntimeError("simulated API failure")
            return {
                "choices": [{"message": {"content": "positive"}}],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 2,
                    "total_tokens": 42,
                },
            }

        monkeypatch.setattr("litellm.completion", flaky)
        run_dir = audit_run_dir()
        with pytest.raises(RuntimeError, match="simulated API failure"):
            audit_run(run_dir, probe=True, probe_k=5, seed=13)
        assert len(calls) == 3
        usage = json.loads((run_dir / "audit" / "audit_usage.json").read_text())
        assert usage["grand_total"]["total_tokens"] == 2 * 42

    def test_rejects_invalid_probe_split_regardless_of_probe_flag(
        self, audit_run_dir
    ):
        # A typo'd probe_split ("vlaidation") must not fail-open to the
        # untouchable test split (final-review finding 3).
        with pytest.raises(ValueError):
            audit_run(audit_run_dir(), probe=False, probe_split="vlaidation")


class TestLatexRows:
    def test_latex_rows_shape(self, audit_run_dir):
        result = audit_run(audit_run_dir(), predict_fn=_fake_predict)
        lines = latex_rows(result).splitlines()
        assert len(lines) == 2
        # Data & Cyc & Dup% & NDup% & Lbl-c(v) & Lbl-c(t) & Div. & #Fail
        first_cells = [cell.strip() for cell in lines[0].split("&")]
        assert first_cells[0] == "sst2-tiny"
        assert first_cells[1] == "1"
        assert lines[0].rstrip().endswith("\\\\")
        # No probe ran: Lbl-c(t) is "--".
        assert first_cells[5] == "--"
        # Second row leaves the dataset cell empty.
        assert lines[1].split("&")[0].strip() == ""


class TestAuditCli:
    def test_end_to_end_writes_outputs(self, audit_run_dir):
        run_dir = audit_run_dir()
        result = runner.invoke(app, ["audit", str(run_dir)])
        assert result.exit_code == 0, result.output
        assert (run_dir / "audit" / "audit.json").exists()
        assert (run_dir / "audit" / "audit.csv").exists()
        assert "Audit" in result.output

    def test_latex_flag_prints_rows(self, audit_run_dir):
        result = runner.invoke(app, ["audit", str(audit_run_dir()), "--latex"])
        assert result.exit_code == 0, result.output
        assert "\\\\" in result.output
        assert "sst2-tiny" in result.output

    def test_multiple_run_dirs_write_combined_csv(
        self, audit_run_dir, tmp_path
    ):
        run_dir = audit_run_dir()
        out = tmp_path / "combined_out"
        result = runner.invoke(
            app,
            [
                "audit",
                str(run_dir),
                str(run_dir),
                "--output-dir",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (out / "audit_combined.csv").exists()

    def test_bad_probe_split_errors(self, audit_run_dir):
        result = runner.invoke(
            app, ["audit", str(audit_run_dir()), "--probe-split", "train"]
        )
        assert result.exit_code == 1

    def test_probe_without_key_errors(self, audit_run_dir, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        result = runner.invoke(app, ["audit", str(audit_run_dir()), "--probe"])
        assert result.exit_code == 1
        assert "OPENROUTER_API_KEY" in result.output

    def test_not_a_run_dir_errors(self, tmp_path):
        result = runner.invoke(app, ["audit", str(tmp_path)])
        assert result.exit_code == 1
        assert "experiment_config.yaml" in result.output
