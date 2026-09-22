# Changelog

## 0.2.0 — 2026-09-24 (EACL 2027 demo submission)
- Recommender: the teacher is a candidate below the break-even volume; student-training GPU time is priced into the distillation intercept; `recommend` reads `training_seconds`.
- Recommender evaluation on the frozen Banking77 cells (`examples/demo/`), price sensitivity (`scripts/recommender_sensitivity.py`).
- Encoder baselines re-run with `warmup_steps: 10` plus compute-matched controls (`examples/paper/M6_*`).
- Engine records `training_seconds` in `run_manifest.json`.
- Prices and targets fixed by stated rules with dated sources.

## 0.1.0 — 2026-07-13
- Initial public release.
