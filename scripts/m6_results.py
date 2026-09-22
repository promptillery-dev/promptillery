#!/usr/bin/env python
"""Collect M6 runs into docs/M6_RESULTS.{md,json} and Table 2 LaTeX rows.

    uv run python scripts/m6_results.py            # reads out/m6, writes docs/
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

DATASETS = ["agnews", "yahoo", "huffpost", "imdb", "sst2"]     # Table 2 column order
FEW_SHOT_TEACHER = {"agnews": 88.95, "yahoo": 70.84, "huffpost": 45.00, "imdb": 96.32, "sst2": 97.42}
STUDENT_LABEL = {"roberta_base": "RoBERTa-base-125M (encoder)",
                 "ettin_encoder": "Ettin-encoder-150M (encoder)"}
VARIANT_LABEL = {
    "ft1": "\\quad \\textbf{fine-tuned} (1 cycle)",
    "same_n": "\\quad \\textbf{same-$N$ gold FT}",
    "same_n_cm": "\\quad \\textbf{same-$N$ gold FT}, compute-matched",
    "seed_x10": "\\quad \\textbf{seed-only} $\\times$10 rounds",
}
PROTOCOL_BATCH = 32
_NAME = re.compile(r"^m6_(?P<d>[a-z0-9]+)_(?P<v>ft1|same_n_cm|same_n|seed_x10)_(?P<s>roberta_base|ettin_encoder)(?:_bs(?P<bs>\d+))?(?:_s(?P<seed>\d+))?$")


def collect(out_root: Path) -> list[dict]:
    rows = []
    for run_dir in sorted(Path(out_root).iterdir()):
        cfg_path, metrics_path = run_dir / "experiment_config.yaml", run_dir / "metrics.json"
        if not (cfg_path.exists() and metrics_path.exists()):
            continue
        name = yaml.safe_load(cfg_path.read_text())["name"]
        m = _NAME.match(name)
        if not m:
            continue
        heldout = json.loads(metrics_path.read_text()).get("heldout_test")
        if not heldout:
            continue
        rows.append({
            "dataset": m.group("d"), "variant": m.group("v"), "student": m.group("s"),
            "batch_size": int(m.group("bs") or PROTOCOL_BATCH),
            "seed": int(m.group("seed") or 13),
            "accuracy": float(heldout["accuracy"]), "f1": float(heldout.get("f1", 0.0)),
            "run_id": run_dir.name,
        })
    # latest run per cell wins (sorted by run dir name = timestamp)
    latest = {}
    for r in rows:
        latest[(r["dataset"], r["variant"], r["student"], r["seed"])] = r
    return list(latest.values())


def _cell(r: dict | None, dataset: str) -> str:
    if r is None:
        return "--"
    acc = 100 * r["accuracy"]
    delta = acc - FEW_SHOT_TEACHER[dataset]
    dagger = "^{\\dagger}" if r["batch_size"] != PROTOCOL_BATCH else ""
    return f"{acc:.2f}$_{{{delta:+.1f}}}{dagger}$"


def table2_rows(rows: list[dict]) -> str:
    by = {(r["dataset"], r["variant"], r["student"]): r for r in rows if r["seed"] == 13}
    out = []
    for student, s_label in STUDENT_LABEL.items():
        out.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{s_label}}}}}\\\\")
        for variant, v_label in VARIANT_LABEL.items():
            cells = [by.get((d, variant, student)) for d in DATASETS]
            if all(c is None for c in cells):
                continue
            accs = [100 * c["accuracy"] for c in cells if c is not None]
            avg = f"{sum(accs) / len(accs):.2f}" if len(accs) == len(DATASETS) else "--"
            out.append(f"{v_label} & " + " & ".join(_cell(c, d) for c, d in zip(cells, DATASETS)) + f" & {avg} \\\\")
    return "\n".join(out) + "\n"


def markdown(rows: list[dict]) -> str:
    lines = ["# M6 results (held-out test accuracy)", "",
             "| dataset | variant | student | batch | seed | accuracy | f1 | run |",
             "|---|---|---|---:|---:|---:|---:|---|"]
    for r in sorted(rows, key=lambda r: (r["student"], r["variant"], r["dataset"])):
        lines.append(f"| {r['dataset']} | {r['variant']} | {r['student']} | {r['batch_size']} | {r['seed']} | "
                     f"{r['accuracy']:.4f} | {r['f1']:.4f} | `{r['run_id']}` |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default="out/m6")
    p.add_argument("--docs", default="docs")
    args = p.parse_args(argv)
    rows = collect(Path(args.out_root))
    docs = Path(args.docs)
    (docs / "M6_RESULTS.md").write_text(markdown(rows))
    (docs / "M6_RESULTS.json").write_text(json.dumps(rows, indent=2))
    (docs / "M6_TABLE2_ROWS.tex").write_text(table2_rows(rows))
    comment = markdown(rows) + "\n```latex\n" + table2_rows(rows) + "```\n"
    (docs / "M6_ISSUE_COMMENT.md").write_text(comment)
    print(markdown(rows))
    print(table2_rows(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
