"""Test for the regret-curve figure (issue #6, M4). Needs the `paper` extra."""

import csv

import pytest

pytest.importorskip("matplotlib")

from promptillery.figures import _plot_regret_curve, write_paper_figures  # noqa: E402


def _regret_rows():
    rows = []
    for dataset in ("agnews", "imdb"):
        for selector in ("recommender_budget_search", "volume_threshold"):
            for volume, regret in ((1000, 0.0), (1_000_000, 3.5), (100_000_000, 40.0)):
                rows.append({
                    "dataset": dataset, "selector": selector,
                    "volume": str(volume), "regret_usd": str(regret),
                })
    return rows


def test_plot_regret_curve_writes_one_panel_per_dataset(tmp_path):
    paths = _plot_regret_curve(_regret_rows(), tmp_path, "png")

    stems = {p.stem for p in paths}
    assert stems == {"regret_curve_agnews", "regret_curve_imdb"}
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_write_paper_figures_picks_up_regret_curve_csv(tmp_path):
    report = tmp_path / "report"
    report.mkdir()
    with (report / "regret_curve.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "selector", "volume", "regret_usd"])
        w.writeheader()
        w.writerows(_regret_rows())

    manifest = write_paper_figures(report, fmt="png")

    created = " ".join(manifest["created"])
    assert "regret_curve_agnews" in created
    assert "regret_curve_imdb" in created
