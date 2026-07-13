# M5 Same-N Gold-FT Baseline — the main-results table `same-N gold FT` row

**Status:** prep runs OFF THE SHELF on any checkout (no campaign artifacts
needed in `--nominal` mode; the prep script bootstraps every free prerequisite
itself). Training is not exercised in CI and needs a GPU.

For each (student, dataset) cell this trains ONE plain supervised fine-tune on
real gold data volume-matched to the `promptillery (10 cycles)` arm:

- matched N = gold seed-train (800) + the synthetic rows a 10-cycle run
  creates. Default (`--nominal`): computed from the cell's G3 config as
  `augmentation_batch_size x (cycles - 1)` = 32 x 9 = **288** (the last cycle
  never augments) ⇒ matched N = **1088 for every cell**. Exact mode
  (`--run-dir`): read from a completed run's
  `run_manifest.json -> final_synthetic_count` (nominal minus filtered
  rejects; observed SST-2: 288/288/284),
- top-up rows are a seeded, stratified draw from the UNUSED remainder of the
  original train split — real gold labels, never teacher-relabeled; the train
  set is a strict superset of the seed rows,
- same recipe/seed as each student's G3 "fine-tuned (1 cycle)" row; `cycles: 1`
  and no `prompt:` key ⇒ **zero teacher calls** (GPU-only cost).

This row isolates **augmentation value vs. data volume**: cycles grow the
training set, and this baseline controls for that. A null (gold ≥ promptillery)
is reportable — the pitch then rests on cost/label-availability.

Students: `roberta_base`, `ettin_encoder`, `ettin_decoder`, `gemma3_270m`.
RoBERTa's promptillery rows are prior-submission reruns (no current-schema
artifacts), so its matched N reuses the Ettin-encoder cell's N per dataset:
same seed + same N => identical data, and `M5_<D>_same_n_roberta_base.yaml`
reads `out/m5/<D>/ettin_encoder/train.jsonl` directly — run the ettin_encoder
prep first; there is no `roberta_base` prep invocation. Datasets: agnews,
sst2, imdb, yahoo, huffpost (NOT Banking77).

There are NO M5 materialize configs, and no manual G3 pre-steps: for decoder
cells the prep script gold-materializes `train_sft.jsonl` by reusing
`examples/paper/G3_<D>_materialize.yaml` (prompt-format identity with the
promptillery arms by construction), auto-fills any missing
`out/g3/<D>/{validation,test}_sft.jsonl` + `canonical_labels.json` the same
way (never overwriting existing campaign files), and auto-runs
`prep_g3_datasets.py` when the normalized yahoo/huffpost pools are absent.
All of that is gold-mode/deterministic: **zero teacher calls, no API key**.

## Per-cell run order (dataset D, student S) — two commands

```bash
# 1) build the cell's matched-N gold data (bootstraps all free prerequisites;
#    for decoder cells this ALSO gold-materializes train_sft.jsonl):
uv run python scripts/prep_m5_same_n.py --dataset <D> --student <S> --nominal
# -> out/m5/<D>/<S>/train.jsonl (+ train_sft.jsonl for decoders),
#    out/m5/<D>/{validation,test}.jsonl,
#    out/m5/<D>/<S>/matched_n_manifest.json (matched N + provenance)
#
#    Exact-N alternative (needs the cell's completed cycles-10 run dir, e.g.
#    on the campaign box; matches the post-filter count instead of nominal):
#      ... --run-dir out/g3/ablation_g3_<D>_<S>_*/g3_<D>_<S>_*_cycles-10_*

# 2) single fine-tune (zero teacher calls; no OPENROUTER_API_KEY needed):
uv run promptillery train examples/paper/M5_<D>_same_n_<S>.yaml

# 3) the table cell = the run's heldout_test accuracy (encoders) /
#    exact_match (decoders), from out/m5/<run>/metrics.json:
uv run promptillery analyze out/m5 --metric accuracy
```

## Full example: SST-2, Ettin decoder

```bash
uv run python scripts/prep_m5_same_n.py --dataset sst2 --student ettin_decoder --nominal
uv run promptillery train examples/paper/M5_sst2_same_n_ettin_decoder.yaml
```

## Notes

- **Pick ONE matched-N mode per table and state it in the paper.** `--nominal`
  gives a uniform N (1088) with no artifact dependency; `--run-dir` matches
  each run's actual post-filter count (differs by at most a few rows, e.g.
  SST-2 gemma 1084). The paper's Baselines paragraph currently describes the
  artifact-read variant — reword it if the campaign fills the row nominally.
- Never reuse another cell's train.jsonl across students in `--run-dir` mode
  (N differs per run). The ONE sanctioned exception is RoBERTa, whose N is
  defined as the Ettin-encoder cell's N (see above); under `--nominal` all
  cells share N=1088 by construction but the files are still written per cell.
- Decoder validation/test SFT and `canonical_labels.json` are the **G3 files**
  (identical evaluation to the promptillery arms); only the train SFT is new.
  When they pre-exist (campaign box) they are used as-is; when missing they
  are regenerated gold-mode from the same config — same record content.
- The prep script cross-checks its seed-carve reconstruction against
  `promptillery.engine.prepare_dataset` and hard-fails on drift; in
  `--run-dir` mode it validates the source run (completed, 10/10 cycles,
  matching student/dataset); for decoders it verifies the materialized SFT
  row count equals matched N.
- **IMDB gold-answer note:** IMDB is the one dataset whose gold-answer path
  differs from its G3 materialize config (M5 forces `label_text`; G3 mapped
  ints via ClassLabel). Verified end-to-end 2026-07-10: all materialized IMDB
  gold answers are canonical (`neg`/`pos`). Re-check after any prep change:
  `python -c "import json; canon=set(json.load(open('out/g3/imdb/canonical_labels.json'))['canonical_labels']); assert all(json.loads(l)['gold_answer'] in canon for l in open('out/m5/imdb/ettin_decoder/train_sft.jsonl')); print('imdb gold answers canonical')"`

## Fresh checkout = just run it

Run `uv sync`, then the two commands above per cell. `--nominal` mode
was executed end-to-end on a machine without campaign artifacts for all 15
prep cells (5 datasets x 3 students) on 2026-07-10: the script auto-downloaded
the HF datasets, auto-normalized yahoo/huffpost, auto-materialized the missing
gold eval SFT, and produced matched N = 1088 everywhere. The only remaining
requirements are a GPU for step 2 and HF hub access (plus the accepted Gemma
license + `hf auth login` for the `gemma3_270m` cells).

`--run-dir` (exact post-filter N) additionally needs the cell's
`run_manifest.json` from the campaign box — the script reads only that one
~6 KB file from the given directory and validates its content, so
`rsync -avR 'gpubox:promptillery/./out/g3/ablation_g3_*/g3_*_cycles-10_*/run_manifest.json' .`
is sufficient.
