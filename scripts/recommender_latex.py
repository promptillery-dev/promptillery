#!/usr/bin/env python
"""Render the recommender evaluation as LaTeX rows for the paper (tab:recommender).

    uv run python scripts/recommender_latex.py out/recommender_g2 --output docs/RECOMMENDER_TABLE.tex
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LABELS = {
    "expert_default": "Best encoder (expert default)",
    "random": "Random student",
    "volume_threshold": "Volume threshold",
    "recommender_rules": "Recommender (fixed budget)",
    "recommender_budget_search": "Recommender (budget search)",
}
ORDER = list(LABELS)


def _thousands(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", "{,}")


def render_recommender_table(table_csv, regret_csv, recommendation_json,
                             primary_volume: int = 1_000_000) -> str:
    table = list(csv.DictReader(Path(table_csv).open()))
    regret = list(csv.DictReader(Path(regret_csv).open()))
    rec = json.loads(Path(recommendation_json).read_text())
    regret_at = {(r["selector"], int(r["volume"])): float(r["regret_usd"]) for r in regret}
    lines = ["\\toprule",
             "\\textbf{Policy} & \\textbf{Agree.\\ (\\%)} & \\textbf{Cost of search (\\%)} & "
             f"\\textbf{{Regret @{_thousands(primary_volume)} (\\$)}} \\\\",
             "\\midrule"]
    by_sel = {r["selector"]: r for r in table}
    for sel in ORDER:
        if sel not in by_sel:
            continue
        r = by_sel[sel]
        reg = regret_at.get((sel, primary_volume))
        lines.append(
            f"{LABELS[sel]} & {100 * float(r['agreement_with_gt']):.1f} & "
            f"{100 * float(r['pct_exhaustive_cost']):.0f} & "
            f"{'--' if reg is None else f'{reg:.2f}'} \\\\"
        )
    lines.append("\\bottomrule")
    for dataset, entry in rec.items():
        receipt = entry["receipt"]
        be = receipt.get("break_even_volume")
        lines.append(f"% {dataset}: pick={entry['cell']} reason={entry['reason']} "
                     f"total=${receipt['total_usd']:.2f} teacher=${receipt['teacher_total_usd']:.2f} "
                     f"break_even={'n/a' if be is None else _thousands(be)} "
                     f"training=${receipt.get('training_usd', 0):.3f} "
                     f"labelling=${receipt.get('teacher_labelling_usd', 0):.3f}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("report_dir")
    p.add_argument("--output", required=True)
    p.add_argument("--primary-volume", type=int, default=1_000_000)
    args = p.parse_args(argv)
    d = Path(args.report_dir)
    tex = render_recommender_table(d / "recommender_table.csv", d / "regret_curve.csv",
                                   d / "recommendation.json", args.primary_volume)
    Path(args.output).write_text(tex)
    print(tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
