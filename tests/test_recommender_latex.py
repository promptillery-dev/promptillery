import csv
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "recommender_latex", REPO_ROOT / "scripts" / "recommender_latex.py")
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)


def test_render_recommender_table_has_one_row_per_selector(tmp_path):
    table = tmp_path / "recommender_table.csv"
    with table.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "selector", "agreement_with_gt",
                                          "pct_exhaustive_cost", "n_targets"])
        w.writeheader()
        w.writerow({"dataset": "banking77", "selector": "recommender_rules",
                    "agreement_with_gt": "0.9", "pct_exhaustive_cost": "0.5", "n_targets": "54"})
        w.writerow({"dataset": "banking77", "selector": "volume_threshold",
                    "agreement_with_gt": "0.4", "pct_exhaustive_cost": "0.1", "n_targets": "54"})
    rec = tmp_path / "recommendation.json"
    rec.write_text(json.dumps({"banking77": {"cell": "FacebookAI/roberta-base",
        "reason": "cheapest_feasible", "receipt": {"total_usd": 1.23, "teacher_total_usd": 1100.0,
        "break_even_volume": 1234.6, "training_usd": 0.1, "teacher_labelling_usd": 0.17}}}))
    regret = tmp_path / "regret_curve.csv"
    with regret.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "selector", "volume", "regret_usd"])
        w.writeheader()
        w.writerow({"dataset": "banking77", "selector": "recommender_rules", "volume": "1000000", "regret_usd": "0.0"})
        w.writerow({"dataset": "banking77", "selector": "volume_threshold", "volume": "1000000", "regret_usd": "12.5"})

    tex = rl.render_recommender_table(table, regret, rec)

    assert "recommender\\_rules" not in tex          # labels are humanised
    assert "Volume threshold" in tex and "Recommender" in tex
    assert "90.0" in tex and "12.50" in tex
    assert "1{,}235" in tex                          # break-even volume, thousands sep
