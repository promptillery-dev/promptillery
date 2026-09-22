#!/usr/bin/env python
"""Derive the pre-registered recommender inputs by stated rules (spec §5, A6).

    uv run python scripts/derive_targets.py \
        --cells out/recommender_g2/paper_main_results.csv --profile-dir out/g2 \
        --teacher-manifest out/g2/banking77_teacher_test.jsonl.manifest.json --teacher-calls 3080

Rules (paper Appendix B):
  accuracy_floors  = quartiles (25/50/75 %) of the cells' selection accuracies, 2 dp
  volume_threshold = median break-even volume across cells (teacher vs. serving rate)
  teacher.usd_per_call = logged tokens of one full teacher pass / calls, at list rates
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Optional

from promptillery.recommender import Teacher, break_even_volume
from promptillery.recommender_io import load_cells, load_prices


def quartile_floors(accuracies: list[float]) -> list[float]:
    q = statistics.quantiles(sorted(accuracies), n=4, method="inclusive")
    return [round(v, 2) for v in q]


def teacher_usd_per_call(manifest: dict, calls: int, *, input_usd_per_m: float,
                         output_usd_per_m: float) -> float:
    usd = (manifest["teacher_input_tokens"] / 1e6 * input_usd_per_m
           + manifest["teacher_output_tokens"] / 1e6 * output_usd_per_m)
    return usd / calls


def median_break_even(cells, prices, teacher: Teacher) -> Optional[float]:
    volumes = [v for v in (break_even_volume(c, prices, teacher) for c in cells) if v is not None]
    return statistics.median(volumes) if volumes else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cells", required=True)
    p.add_argument("--profile-dir", required=True)
    p.add_argument("--prices", default="prices.yaml")
    p.add_argument("--teacher-manifest", required=True)
    p.add_argument("--teacher-calls", type=int, required=True)
    p.add_argument("--input-usd-per-m", type=float, default=2.0)
    p.add_argument("--output-usd-per-m", type=float, default=8.0)
    args = p.parse_args(argv)

    prices = load_prices(args.prices)
    cells = load_cells(args.cells, args.profile_dir, prices=prices)
    manifest = json.loads(Path(args.teacher_manifest).read_text())
    usd_per_call = teacher_usd_per_call(manifest, args.teacher_calls,
                                        input_usd_per_m=args.input_usd_per_m,
                                        output_usd_per_m=args.output_usd_per_m)
    teacher = Teacher(usd_per_call=usd_per_call)
    floors = quartile_floors([c.selection_accuracy for c in cells])
    threshold = median_break_even(cells, prices, teacher)

    print("# paste into targets.yaml (rules: see scripts/derive_targets.py)")
    print(f"teacher:\n  usd_per_call: {usd_per_call:.4f}")
    print(f"accuracy_floors: {floors}")
    print(f"volume_threshold: {round(threshold) if threshold else 'null'}")
    for c in cells:
        be = break_even_volume(c, prices, teacher)
        print(f"#   {c.student_model:<32} sel={c.selection_accuracy:.4f} "
              f"intercept=${c.distillation_usd:.3f} break_even={be and round(be)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
