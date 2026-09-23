# Demo replay (no GPU, no API key)

The six Banking77 students from the paper, frozen: `banking77/paper_main_results.csv`
(accuracy, teacher spend, training time per student, built by `scripts/build_cells.py`
from `out/g2`) and `banking77/profiles/` (latency and throughput on one RTX 4090).

    uv run promptillery recommend \
        --main-results examples/demo/banking77/paper_main_results.csv \
        --profile-dir examples/demo/banking77/profiles \
        --prices prices.yaml --targets targets.yaml -o out/demo_report
    uv run promptillery paper-figures out/demo_report --format pdf
    cat out/demo_report/recommendation.json

`recommendation.json` is the cost receipt for the primary target in `targets.yaml`
(pick, dollars vs. calling the teacher, break-even volume). `recommender_table.csv`
scores the recommender against always-the-best-encoder, random, and a volume
threshold; `figures/regret_curve_banking77.pdf` plots dollars lost vs. volume.

Change `primary_target` in `targets.yaml` (for example `volume: 100`) and re-run:
below the break-even volume of 237 calls the answer flips to keep calling the teacher.

One real training cycle with a teacher key (costs cents): `uv run promptillery train examples/demo/sst2_encoder_live.yaml`.
The audit on that run: `uv run promptillery audit out/demo/demo_sst2_encoder_live_*`.
