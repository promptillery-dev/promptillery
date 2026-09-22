#!/usr/bin/env python
"""How much do the recommender's inputs move its pick? (spec §4 E3 step 4)

Re-runs the primary-target recommendation with the GPU rate, the CPU rate and
the teacher price each scaled by 0.5x, 1x and 2x, and reports the pick, its
reason, its cost and the break-even volume.

    uv run python scripts/recommender_sensitivity.py \
        --cells out/recommender_g2/paper_main_results.csv --profile-dir out/g2 \
        --output out/recommender_g2/sensitivity.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from promptillery.pareto import HardwarePrices
from promptillery.recommender import Teacher
from promptillery.recommender_io import load_cells, load_prices, load_targets
from promptillery.selectors import recommender_budget_search

FIELDS = ["factor", "multiplier", "pick", "reason", "total_usd",
          "teacher_total_usd", "break_even_volume"]


def _scaled(base: HardwarePrices, factor: str, m: float) -> HardwarePrices:
    rates = {}
    for key, rate in base.usd_per_hour.items():
        is_cpu = key == "cpu"
        scale = m if (factor == "gpu_rate" and not is_cpu) or (factor == "cpu_rate" and is_cpu) else 1.0
        rates[key] = rate * scale
    return HardwarePrices(rates)


def sensitivity_rows(cells_csv, profile_dir, prices_path, targets_path,
                     multipliers: Iterable[float] = (0.5, 1.0, 2.0)) -> list[dict]:
    config = load_targets(targets_path)
    base = load_prices(prices_path)
    rows = []
    for factor in ("gpu_rate", "cpu_rate", "teacher_price"):
        for m in multipliers:
            prices = _scaled(base, factor, m) if factor != "teacher_price" else base
            teacher = (Teacher(usd_per_call=config.teacher.usd_per_call * m)
                       if factor == "teacher_price" else config.teacher)
            cells = load_cells(cells_csv, profile_dir,
                               expect_gpu_name=config.expect_gpu_name, prices=prices)
            sel = recommender_budget_search(cells, config.primary_target, prices, teacher)
            be = sel.receipt.break_even_volume
            rows.append({
                "factor": factor, "multiplier": m,
                "pick": sel.cell.student_model if sel.cell else "teacher",
                "reason": sel.reason,
                "total_usd": round(sel.receipt.total_usd, 4),
                "teacher_total_usd": round(sel.receipt.teacher_total_usd, 4),
                "break_even_volume": None if be is None else round(be),
            })
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cells", required=True)
    p.add_argument("--profile-dir", required=True)
    p.add_argument("--prices", default="prices.yaml")
    p.add_argument("--targets", default="targets.yaml")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    rows = sensitivity_rows(args.cells, args.profile_dir, args.prices, args.targets)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['factor']:<14} x{r['multiplier']:<4} -> {r['pick']:<32} "
              f"${r['total_usd']:>10.2f} vs teacher ${r['teacher_total_usd']:>10.2f} "
              f"break-even={r['break_even_volume']}")
    picks = {r["pick"] for r in rows}
    print("pick is stable across all perturbations" if len(picks) == 1
          else f"pick changes across perturbations: {sorted(picks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
