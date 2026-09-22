#!/usr/bin/env python
"""Build the recommender's candidate-cell CSV from finished run directories.

Each run directory holding metrics.json, run_manifest.json and token_usage.json
becomes one row of a paper_main_results.csv that `promptillery recommend` reads.
Runs without a profile.json are skipped; the recommender cannot use them.
Teacher spend is priced from logged tokens at list rates because the tracker
stored estimated_cost=null for OpenRouter-routed GPT-4.1. Training wall-clock
is taken from the manifest when the engine recorded it (runs after 2026-09-22);
older runs get the run's wall-clock (run-id stamp to manifest created_at) as an
upper bound, labelled as such in training_seconds_source.

    uv run python scripts/build_cells.py out/g2 --dataset banking77 \
        --output out/recommender_g2/paper_main_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

GPT41_INPUT_USD_PER_M = 2.0
GPT41_OUTPUT_USD_PER_M = 8.0

FIELDS = [
    "dataset", "dataset_subset", "student_model", "student_type", "metric", "mode",
    "token_budget", "expected_cycles", "policy_name", "control_name", "seeds",
    "run_count", "mean_heldout_metric", "std_heldout_metric", "mean_final_metric",
    "std_final_metric", "mean_estimated_cost", "std_estimated_cost",
    "teacher_input_tokens", "teacher_output_tokens", "training_seconds",
    "training_seconds_source", "run_id",
]

_STAMP = re.compile(r"_(\d{8})_(\d{6})_")


def price_usd(input_tokens: int, output_tokens: int, *, input_usd_per_m: float,
              output_usd_per_m: float) -> float:
    """Dollars for a token count at per-million list rates."""
    return input_tokens / 1e6 * input_usd_per_m + output_tokens / 1e6 * output_usd_per_m


def metric_key(block: dict) -> str:
    """accuracy for classifiers and FastText; exact_match for decoders.

    On a label task exact_match on the canonical label *is* accuracy, which is
    why both families can share one column.
    """
    if "accuracy" in block:
        return "accuracy"
    if "exact_match" in block:
        return "exact_match"
    raise KeyError(f"no accuracy-like metric in {sorted(block)}")


def run_start(run_id: str, utc_offset_hours: float) -> Optional[datetime]:
    m = _STAMP.search(run_id)
    if not m:
        return None
    local = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return local.replace(tzinfo=timezone(timedelta(hours=utc_offset_hours)))


def wall_clock_seconds(manifest: dict, utc_offset_hours: float) -> Optional[float]:
    """Seconds from the run-id stamp (box local time) to manifest created_at (UTC)."""
    start = run_start(str(manifest.get("run_id", "")), utc_offset_hours)
    created = (manifest.get("reproducibility") or {}).get("created_at")
    if start is None or not created:
        return None
    seconds = (datetime.fromisoformat(created) - start).total_seconds()
    return seconds if seconds > 0 else None


def cell_row(run_dir: Path, *, dataset: str, utc_offset_hours: float,
             input_usd_per_m: float, output_usd_per_m: float,
             training_seconds_override: Optional[float] = None) -> dict:
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    usage = json.loads((run_dir / "token_usage.json").read_text())
    heldout = metrics["heldout_test"]
    key = metric_key(heldout)
    selected = str(heldout["_selected_cycle"])
    grand = usage["grand_total"]
    cost = price_usd(int(grand["input_tokens"]), int(grand["output_tokens"]),
                     input_usd_per_m=input_usd_per_m, output_usd_per_m=output_usd_per_m)
    if training_seconds_override is not None:
        seconds, source = float(training_seconds_override), "override"
    elif manifest.get("training_seconds") is not None:
        seconds, source = float(manifest["training_seconds"]), "manifest"
    else:
        wall = wall_clock_seconds(manifest, utc_offset_hours)
        seconds = wall or 0.0
        source = "run_wall_clock_upper_bound" if wall is not None else "unavailable"
    return {
        "dataset": dataset,
        "dataset_subset": manifest.get("dataset_subset") or "",
        "student_model": manifest["student_model"],
        "student_type": manifest["student_type"],
        "metric": key,
        "mode": "max",
        "token_budget": manifest.get("token_budget") or 0,
        "expected_cycles": manifest.get("expected_cycles") or 0,
        "policy_name": manifest.get("policy_name") or "",
        "control_name": manifest.get("control_name") or "",
        "seeds": manifest.get("seed", ""),
        "run_count": 1,
        "mean_heldout_metric": heldout[key],
        "std_heldout_metric": 0.0,
        "mean_final_metric": metrics[selected][key],
        "std_final_metric": 0.0,
        "mean_estimated_cost": round(cost, 6),
        "std_estimated_cost": 0.0,
        "teacher_input_tokens": grand["input_tokens"],
        "teacher_output_tokens": grand["output_tokens"],
        "training_seconds": round(seconds, 1),
        "training_seconds_source": source,
        "run_id": manifest["run_id"],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_root", help="directory whose children are run dirs")
    parser.add_argument("--dataset", required=True, help="dataset label for every row")
    parser.add_argument("--output", required=True, help="CSV to write")
    parser.add_argument("--utc-offset-hours", type=float, default=2.0,
                        help="timezone of the run-id stamp (campaign box: CEST = +2)")
    parser.add_argument("--input-usd-per-m", type=float, default=GPT41_INPUT_USD_PER_M)
    parser.add_argument("--output-usd-per-m", type=float, default=GPT41_OUTPUT_USD_PER_M)
    parser.add_argument("--training-seconds-json", default=None,
                        help="optional {run_id: seconds} overrides (from box logs)")
    args = parser.parse_args(argv)

    overrides = {}
    if args.training_seconds_json:
        overrides = json.loads(Path(args.training_seconds_json).read_text())

    rows = []
    for run_dir in sorted(Path(args.run_root).iterdir()):
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "completed":
            print(f"skip {run_dir.name}: status={manifest.get('status')}")
            continue
        # Check for profile.json in model/ or at run root
        profile_path = run_dir / "model" / "profile.json"
        if not profile_path.exists():
            profile_path = run_dir / "profile.json"
        if not profile_path.exists():
            print(f"skip {run_dir.name}: no profile.json", file=sys.stderr)
            continue
        rows.append(cell_row(
            run_dir, dataset=args.dataset, utc_offset_hours=args.utc_offset_hours,
            input_usd_per_m=args.input_usd_per_m, output_usd_per_m=args.output_usd_per_m,
            training_seconds_override=overrides.get(manifest["run_id"]),
        ))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['student_model']:<32} {row['metric']:<11} sel={row['mean_final_metric']:.4f} "
              f"held={row['mean_heldout_metric']:.4f} teacher=${row['mean_estimated_cost']:.3f} "
              f"train={row['training_seconds']:.0f}s ({row['training_seconds_source']})")
    print(f"wrote {len(rows)} cells to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
