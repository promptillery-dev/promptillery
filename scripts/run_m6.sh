#!/usr/bin/env bash
# Run M6 configs one after another with an out-of-memory ladder (32 -> 16 -> 8).
#   scripts/run_m6.sh examples/M6_sst2_ft1_roberta_base.yaml examples/M6_sst2_same_n_roberta_base.yaml
# Logs: out/m6/logs/<config>.log. A smaller batch is recorded in the run name (_bsN).
set -uo pipefail
mkdir -p out/m6/logs
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  runname=$(echo "$name" | tr 'A-Z' 'a-z')
  for bs in "" 16 8; do
    if [ -n "$bs" ]; then
      variant=$(uv run python scripts/gen_m6_configs.py --only "$runname" --batch-size "$bs" --print-path | tail -n1)
    else
      variant="$cfg"
    fi
    if [ -z "$variant" ]; then
      echo "== $(date -Is) FAIL  could not generate a batch-$bs config for $runname"; break
    fi
    log="out/m6/logs/$(basename "$variant" .yaml).log"
    echo "== $(date -Is) start $variant"
    if uv run promptillery train "$variant" > "$log" 2>&1; then
      echo "== $(date -Is) ok    $variant"; break
    elif grep -qi "out of memory" "$log"; then
      echo "== $(date -Is) OOM   $variant -> smaller batch"; continue
    else
      echo "== $(date -Is) FAIL  $variant (see $log)"; break
    fi
  done
done
