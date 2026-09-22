# Recommender targets derivation (2026-09-22)

```
uv run python scripts/derive_targets.py --cells out/recommender_g2/paper_main_results.csv --profile-dir out/g2 --teacher-manifest out/g2/banking77_teacher_test.jsonl.manifest.json --teacher-calls 3080
```

```
# paste into targets.yaml (rules: see scripts/derive_targets.py)
teacher:
  usd_per_call: 0.0011
accuracy_floors: [0.89, 0.91, 0.92]
volume_threshold: 263
#   jhu-clsp/ettin-decoder-150m      sel=0.8840 intercept=$0.792 break_even=731
#   jhu-clsp/ettin-encoder-150m      sel=0.9180 intercept=$0.290 break_even=265
#   fasttext                         sel=0.8066 intercept=$0.183 break_even=167
#   google/gemma-3-270m-it           sel=0.9020 intercept=$1.205 break_even=1113
#   answerdotai/ModernBERT-base      sel=0.9165 intercept=$0.286 break_even=262
#   FacebookAI/roberta-base          sel=0.9330 intercept=$0.261 break_even=238
```
