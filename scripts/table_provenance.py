#!/usr/bin/env python
"""Generate table-provenance index for resubmission numbers.

    uv run python scripts/table_provenance.py [--output FILE] [--repo-root PATH]

Outputs table provenance markdown (default: docs/TABLE_PROVENANCE.md) and prints row count.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional


def build_rows(repo_root: Path) -> list[dict]:
    """Build list of (table, row, column, value, file) dicts for all numbers in brief order."""
    rows = []

    # Table 3: Deployment results
    rows.extend(_table3_accuracy_rows(repo_root))
    rows.extend(_table3_latency_rows(repo_root))

    # Table recommender
    rows.extend(_table_recommender_rows(repo_root))

    # Sec 5.3 receipt
    rows.extend(_receipt_rows(repo_root))

    # Appendix B sensitivity
    rows.extend(_sensitivity_rows(repo_root))

    # targets.yaml
    rows.extend(_targets_rows(repo_root))

    # Table 2 (M6 results)
    rows.extend(_table2_rows(repo_root))

    # Table 2 (promptillery loop rows re-run with warmup 10)
    rows.extend(_table2_loop_rows(repo_root))

    # Table 5 (audit)
    rows.extend(_table5_rows(repo_root))

    return rows


def _table3_accuracy_rows(repo_root: Path) -> list[dict]:
    """Accuracy rows from out/g2/*/metrics.json heldout_test blocks."""
    rows = []
    out_g2 = repo_root / "out" / "g2"

    if not out_g2.exists():
        return rows

    # Find all run directories with metrics.json
    for run_dir in sorted(out_g2.iterdir()):
        if not run_dir.is_dir():
            continue

        metrics_file = run_dir / "metrics.json"
        if not metrics_file.exists():
            continue

        metrics = json.loads(metrics_file.read_text())

        # Skip runs without heldout_test
        if "heldout_test" not in metrics:
            continue

        heldout = metrics["heldout_test"]

        # Get student name
        student = _get_student_name(run_dir)
        if not student:
            continue

        # Get accuracy or exact_match
        value = None
        if "accuracy" in heldout:
            value = heldout["accuracy"]
        elif "exact_match" in heldout:
            value = heldout["exact_match"]

        if value is not None:
            rows.append({
                "table": "Table 3",
                "row": student,
                "column": "Acc",
                "value": f"{value:.4f}",
                "file": _relative_path(metrics_file, repo_root),
            })

    return rows


def _table3_latency_rows(repo_root: Path) -> list[dict]:
    """Latency rows from out/g2/**/profile.json."""
    rows = []
    out_g2 = repo_root / "out" / "g2"

    if not out_g2.exists():
        return rows

    # Use rglob to find all profile.json files
    for profile_file in sorted(out_g2.rglob("profile.json")):
        # Skip profile-bs*.json files
        if profile_file.name.startswith("profile-bs"):
            continue

        profile = json.loads(profile_file.read_text())

        # Get model name from profile.json
        model_name = profile.get("model")
        if not model_name:
            continue

        # Get p50_latency_ms from student section
        student = profile.get("student", {})
        p50_latency = student.get("p50_latency_ms")

        if p50_latency is not None:
            rows.append({
                "table": "Table 3",
                "row": model_name,
                "column": "p50 (ms)",
                "value": f"{p50_latency:.2f}",
                "file": _relative_path(profile_file, repo_root),
            })

    return rows


def _table_recommender_rows(repo_root: Path) -> list[dict]:
    """Rows from out/recommender_g2/recommender_table.csv."""
    rows = []
    table_file = repo_root / "out" / "recommender_g2" / "recommender_table.csv"

    if not table_file.exists():
        return rows

    # Load recommender_table.csv
    recommender_data = {}
    with open(table_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            selector = row.get("selector")
            if selector:
                recommender_data[selector] = row

    # Load regret_curve.csv to find regret at volume 1000000
    regret_curve_file = repo_root / "out" / "recommender_g2" / "regret_curve.csv"
    regret_at_1m = {}
    if regret_curve_file.exists():
        with open(regret_curve_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("volume") == "1000000":
                    selector = row.get("selector")
                    regret_usd = row.get("regret_usd")
                    if selector and regret_usd:
                        regret_at_1m[selector] = float(regret_usd)

    # Create three rows per selector
    for selector in sorted(recommender_data.keys()):
        row_data = recommender_data[selector]

        # Agree. (%)
        agreement = row_data.get("agreement_with_gt")
        if agreement:
            rows.append({
                "table": "Table recommender",
                "row": selector,
                "column": "Agree. (%)",
                "value": f"{float(agreement) * 100:.1f}",
                "file": _relative_path(table_file, repo_root),
            })

        # Cost of search (%)
        pct_cost = row_data.get("pct_exhaustive_cost")
        if pct_cost:
            rows.append({
                "table": "Table recommender",
                "row": selector,
                "column": "Cost of search (%)",
                "value": f"{float(pct_cost) * 100:.0f}",
                "file": _relative_path(table_file, repo_root),
            })

        # Regret @1e6 ($)
        if selector in regret_at_1m:
            rows.append({
                "table": "Table recommender",
                "row": selector,
                "column": "Regret @1e6 ($)",
                "value": f"{regret_at_1m[selector]:.2f}",
                "file": _relative_path(regret_curve_file, repo_root),
            })

    return rows


def _receipt_rows(repo_root: Path) -> list[dict]:
    """Rows from out/recommender_g2/recommendation.json receipt."""
    rows = []
    rec_file = repo_root / "out" / "recommender_g2" / "recommendation.json"

    if not rec_file.exists():
        return rows

    recommendation = json.loads(rec_file.read_text())

    # Get banking77 section
    banking77 = recommendation.get("banking77")
    if not banking77:
        return rows

    receipt = banking77.get("receipt", {})

    # Map field names: "pick" comes from "cell"
    field_mapping = {
        "pick": banking77.get("cell"),
        "total_usd": receipt.get("total_usd"),
        "teacher_total_usd": receipt.get("teacher_total_usd"),
        "break_even_volume": receipt.get("break_even_volume"),
        "training_usd": receipt.get("training_usd"),
        "teacher_labelling_usd": receipt.get("teacher_labelling_usd"),
    }

    for field_name, value in field_mapping.items():
        if value is not None:
            # Format numeric values as %.4f, keep strings as-is
            if isinstance(value, str):
                formatted_value = value
            else:
                formatted_value = f"{float(value):.4f}"

            rows.append({
                "table": "Sec 5.3 receipt",
                "row": field_name,
                "column": "—",
                "value": formatted_value,
                "file": _relative_path(rec_file, repo_root),
            })

    return rows


def _sensitivity_rows(repo_root: Path) -> list[dict]:
    """Rows from out/recommender_g2/sensitivity.csv."""
    rows = []
    sens_file = repo_root / "out" / "recommender_g2" / "sensitivity.csv"

    if not sens_file.exists():
        return rows

    with open(sens_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            factor = row.get("factor")
            multiplier = row.get("multiplier")
            break_even = row.get("break_even_volume")

            if factor and multiplier and break_even:
                rows.append({
                    "table": "Appendix B sensitivity",
                    "row": f"{factor} x{multiplier}",
                    "column": "break_even_volume",
                    "value": str(break_even),
                    "file": _relative_path(sens_file, repo_root),
                })

    return rows


def _targets_rows(repo_root: Path) -> list[dict]:
    """Rows from docs/TARGETS_DERIVATION.md."""
    rows = []
    targets_file = repo_root / "docs" / "TARGETS_DERIVATION.md"

    if not targets_file.exists():
        raise FileNotFoundError(targets_file)

    content = targets_file.read_text()

    # Parse values from the YAML-like section
    import re

    # accuracy_floors
    match = re.search(r'accuracy_floors:\s*\[(.*?)\]', content)
    if match:
        floors_str = match.group(1)
        rows.append({
            "table": "targets.yaml",
            "row": "accuracy_floors",
            "column": "—",
            "value": f"[{floors_str}]",
            "file": _relative_path(targets_file, repo_root),
        })

    # volume_threshold
    match = re.search(r'volume_threshold:\s*(\d+)', content)
    if match:
        rows.append({
            "table": "targets.yaml",
            "row": "volume_threshold",
            "column": "—",
            "value": match.group(1),
            "file": _relative_path(targets_file, repo_root),
        })

    # teacher.usd_per_call
    match = re.search(r'usd_per_call:\s*([\d.]+)', content)
    if match:
        rows.append({
            "table": "targets.yaml",
            "row": "teacher.usd_per_call",
            "column": "—",
            "value": match.group(1),
            "file": _relative_path(targets_file, repo_root),
        })

    return rows


def _table2_rows(repo_root: Path) -> list[dict]:
    """Rows from docs/M6_RESULTS.json or pending marker."""
    rows = []
    m6_file = repo_root / "docs" / "M6_RESULTS.json"

    if not m6_file.exists():
        # Emit pending marker
        rows.append({
            "table": "Table 2",
            "row": "baseline rows",
            "column": "—",
            "value": "pending",
            "file": "docs/M6_RESULTS.json",
        })
    else:
        # Parse M6_RESULTS.json and emit rows
        results = json.loads(m6_file.read_text())

        # Assuming results is structured as {student_variant: {dataset: {...}}}
        for entry in results:
            if isinstance(entry, dict):
                student = entry.get("student")
                variant = entry.get("variant")
                dataset = entry.get("dataset")
                accuracy = entry.get("accuracy")
                batch_size = entry.get("batch_size", 32)
                seed = entry.get("seed", 13)

                if student and dataset and accuracy is not None:
                    row_name = f"{student} {variant}" if variant else student
                    if seed != 13:
                        row_name += f" (seed {seed})"
                    value = f"{accuracy * 100:.2f}"
                    if batch_size != 32:
                        value += f" (bs{batch_size})"

                    rows.append({
                        "table": "Table 2",
                        "row": row_name,
                        "column": dataset,
                        "value": value,
                        "file": _relative_path(m6_file, repo_root),
                    })

    return rows


_LOOP_STUDENTS = {"roberta_base": "roberta_base", "ettin_encoder_w10": "ettin_encoder"}


def _table2_loop_rows(repo_root: Path) -> list[dict]:
    """Table 2 promptillery (N cycles) cells from out/g3/ablation_g3_<dataset>_<student>_*/<arm>/metrics.json.

    Only the corrected-warmup re-runs are indexed (RoBERTa-base and the Ettin-encoder ``_w10`` configs).
    When a dataset has several ablation directories, the one with the latest timestamp wins
    (an out-of-memory re-run at ``_bsN`` is a later directory, and its batch size is appended
    to the value as ``(bsN)``). Arms without a held-out accuracy (failed runs) are skipped.
    """
    import re

    rows = []
    out_g3 = repo_root / "out" / "g3"
    if not out_g3.is_dir():
        return rows
    latest: dict[tuple[str, str], tuple[str, Path, str | None]] = {}
    for suffix, student in _LOOP_STUDENTS.items():
        pattern = re.compile(
            rf"^ablation_g3_(?P<dataset>[a-z0-9]+)_{re.escape(suffix)}(?:_transformers)?(?:_bs(?P<bs>\d+))?_(?P<stamp>\d{{8}}_\d{{6}})$"
        )
        for ablation_dir in out_g3.iterdir():
            match = pattern.match(ablation_dir.name)
            if not match or not ablation_dir.is_dir():
                continue
            key = (student, match.group("dataset"))
            if key not in latest or match.group("stamp") > latest[key][0]:
                latest[key] = (match.group("stamp"), ablation_dir, match.group("bs"))
    for (student, dataset), (_stamp, ablation_dir, batch_size) in sorted(latest.items()):
        for metrics_file in sorted(ablation_dir.glob("*_cycles-*/metrics.json")):
            cycles_match = re.search(r"_cycles-(\d+)_", metrics_file.parent.name)
            if not cycles_match:
                continue
            cycles = int(cycles_match.group(1))
            try:
                accuracy = json.loads(metrics_file.read_text())["heldout_test"]["accuracy"]
            except (KeyError, TypeError, ValueError):
                continue
            rows.append({
                "table": "Table 2",
                "row": f"{student} promptillery ({cycles} cycle{'s' if cycles != 1 else ''})",
                "column": dataset,
                "value": f"{accuracy * 100:.2f}" + (f" (bs{batch_size})" if batch_size else ""),
                "file": _relative_path(metrics_file, repo_root),
            })
    return rows


def _slug(text: str) -> str:
    """Filesystem-safe lowercase slug (``jhu-clsp/ettin-encoder-150m`` -> ``jhu-clsp-ettin-encoder-150m``)."""
    import re
    slug = re.sub(r"[^0-9a-zA-Z]+", "-", text.lower()).strip("-")
    return slug or "unknown"


def _table5_rows(repo_root: Path) -> list[dict]:
    """Table 5 (tab:audit): the gold-only verifier row plus audit.json fields."""
    rows = []

    # Verifier: held-out accuracy from out/g2_verifier/*/metrics.json.
    for metrics_file in sorted((repo_root / "out" / "g2_verifier").glob("*/metrics.json")):
        metrics = json.loads(metrics_file.read_text())
        accuracy = (metrics.get("heldout_test") or {}).get("accuracy")
        if accuracy is not None:
            rows.append({
                "table": "Table 5",
                "row": "verifier (gold-only RoBERTa)",
                "column": "held-out accuracy",
                "value": f"{accuracy:.4f}",
                "file": _relative_path(metrics_file, repo_root),
            })

    # Per-run audit.json: one row per top-level numeric field, plus one level
    # of flattening for numeric fields nested under an obvious key (e.g.
    # "cumulative").
    for audit_file in sorted((repo_root / "out" / "g2").glob("*/audit/audit.json")):
        audit = json.loads(audit_file.read_text())
        run_dir = audit_file.parent.parent
        student = _get_student_name(run_dir) or run_dir.name
        student_slug = _slug(student)

        for field_name, value in audit.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                rows.append({
                    "table": "Table 5",
                    "row": student_slug,
                    "column": field_name,
                    "value": str(value),
                    "file": _relative_path(audit_file, repo_root),
                })
            elif isinstance(value, dict):
                for child_name, child_value in value.items():
                    if isinstance(child_value, bool):
                        continue
                    if isinstance(child_value, (int, float)):
                        rows.append({
                            "table": "Table 5",
                            "row": student_slug,
                            "column": f"{field_name}.{child_name}",
                            "value": str(child_value),
                            "file": _relative_path(audit_file, repo_root),
                        })

    return rows


def _get_student_name(run_dir: Path) -> Optional[str]:
    """Get student name from experiment_config.yaml or run_manifest.json."""
    # Try experiment_config.yaml first
    config_file = run_dir / "experiment_config.yaml"
    if config_file.exists():
        content = config_file.read_text()
        import re
        match = re.search(r'^student:\s*(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()

    # Try run_manifest.json
    manifest_file = run_dir / "run_manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        return manifest.get("student_model")

    return None


def _relative_path(file_path: Path, repo_root: Path) -> str:
    """Return relative path from repo_root to file_path with forward slashes."""
    rel = file_path.relative_to(repo_root)
    return str(rel).replace("\\", "/")


def _escape_value(value: str) -> str:
    """Escape pipe characters in value for markdown table."""
    return value.replace("|", "\\|")


def render(rows: list[dict]) -> str:
    """Render rows as flat markdown table with exactly 5 columns."""
    lines = [
        "# Table provenance (generated by scripts/table_provenance.py)",
        "",
        "| table | row | column | value | file |",
        "|---|---|---|---|---|",
    ]

    # Add data rows in order
    for row in rows:
        escaped_value = _escape_value(row["value"])
        line = (
            f"| {row['table']} | {row['row']} | {row['column']} | "
            f"{escaped_value} | {row['file']} |"
        )
        lines.append(line)

    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate table-provenance index for resubmission numbers"
    )
    parser.add_argument(
        "--output",
        default="docs/TABLE_PROVENANCE.md",
        help="Output file path (default: docs/TABLE_PROVENANCE.md)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Repository root path (default: script's parent's parent)",
    )
    args = parser.parse_args(argv)

    # Determine repo root
    if args.repo_root:
        repo_root = args.repo_root
    else:
        repo_root = Path(__file__).resolve().parents[1]

    rows = build_rows(repo_root)
    output = render(rows)

    # Write output file
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(output)

    # Print row count
    print(f"{len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
