#!/usr/bin/env bash
# Export this private dev repo into the public package checkout.
#   scripts/export_public.sh /path/to/promptillery-public
# Paper configs move to examples/paper/; M5 ships too (superseded, see M5_README.md's
# superseded note) because the exported scripts/gen_m6_configs.py derives the same-N M6
# configs from the M5 configs. Results under out/ are force-added (out/ is gitignored in
# both repos). Review `git status` there, then commit.
set -euo pipefail
DST="${1:?public checkout path}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"

rsync -a --delete --exclude '__pycache__' "$SRC/promptillery/" "$DST/promptillery/"
rsync -a --delete --exclude '__pycache__' "$SRC/tests/" "$DST/tests/"
rsync -a --delete "$SRC/scripts/" "$DST/scripts/"
rsync -a "$SRC/images/" "$DST/images/"
for f in README.md CHANGELOG.md LICENSE pyproject.toml uv.lock prices.yaml targets.yaml; do
  cp "$SRC/$f" "$DST/$f"
done

mkdir -p "$DST/examples/paper" "$DST/examples/demo"
# non-paper examples (tiny fixtures, generic configs) stay at examples/
rsync -a --delete --exclude 'G[0-9]_*' --exclude 'M[0-9]_*' --exclude 'demo/' --exclude 'paper/' \
  "$SRC/examples/" "$DST/examples/"
# paper configs: G2..G5 and M6 (+ their READMEs); M5 (superseded; the M6 same-N configs
# derive from it) ships with its README
rsync -a --delete --delete-excluded --include 'G[0-9]_*' --include 'M6_*' --include 'M5_*' --exclude '*' "$SRC/examples/" "$DST/examples/paper/"
rsync -a --delete "$SRC/examples/demo/" "$DST/examples/demo/"

# frozen artifacts (JSON/CSV/YAML/PDF only; never weights or dataset snapshots; g3/m5 ship run records but not their jsonl data pools, which scripts/prep_g3_datasets.py and scripts/prep_m5_same_n.py regenerate)
for d in g2 g2_verifier g3 g4 g5 m5 m6 recommender_g2; do
  [ -d "$SRC/out/$d" ] || continue
  mkdir -p "$DST/out/$d"
  extra=()
  case "$d" in g3|m5) extra=(--exclude '*.jsonl');; esac
  rsync -a --delete --prune-empty-dirs \
    --exclude '*.safetensors' --exclude '*.bin' --exclude 'training/' --exclude 'dataset_cycle_*' \
    --exclude 'logs/' --exclude 'checkpoint-*' "${extra[@]}" "$SRC/out/$d/" "$DST/out/$d/"
done
mkdir -p "$DST/docs"
for f in M6_RESULTS.md M6_RESULTS.json TARGETS_DERIVATION.md TABLE_PROVENANCE.md RECOMMENDER_TABLE.tex; do
  [ -f "$SRC/docs/$f" ] && cp "$SRC/docs/$f" "$DST/docs/$f"
done

cd "$DST"
git add -A
git add -f out/
git status --short | head -40 || true
echo "Now: uv sync && uv run pytest -q && uv run ruff check . ; then commit and push."
