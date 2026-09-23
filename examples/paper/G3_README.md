# G3 Main Results — Tab:main-results Cross-Architecture Configs

**Status:** NOT CI-verified — run on the remote 2×A100 box.

This document walks through the complete run order for all five datasets (AG News, SST-2, IMDB, Yahoo, HuffPost) used to fill `tab:main-results` with a 2-encoder / 2-decoder roster.

## Roster

- **Encoders:** RoBERTa-base (reused from prior submission), Ettin-encoder-150m
- **Decoders:** Ettin-decoder-150m, Gemma-3-270M
- **Teacher:** `openrouter/openai/gpt-4.1`
- **Protocol:** 1,000-instance stratified subset, 80/20 train/val, full test set evaluation
- **Seed:** 13

**Gemma-3-270M License Note:** Requires accepting the Gemma license and running `hf auth login` before the first run.

---

## Per-Dataset Run Order

For each dataset D ∈ {agnews, sst2, imdb, yahoo, huffpost}:

### Step 0: Prep (Yahoo & HuffPost only)

Only **yahoo** and **huffpost** require the prep step. Skip this for agnews, sst2, and imdb.

```bash
uv run python scripts/prep_g3_datasets.py --dataset <D> --seed 13
```

This writes normalized JSONL under `out/g3/<D>/` (train pool and full test split).

### Step 1: Materialize Gold SFT + Canonical Labels

Run once per dataset (outputs are shared across all three students):

```bash
uv run promptillery materialize-sft examples/G3_<D>_materialize.yaml \
  --mode gold --split train      --output out/g3/<D>/train_sft.jsonl      --overwrite
uv run promptillery materialize-sft examples/G3_<D>_materialize.yaml \
  --mode gold --split validation --output out/g3/<D>/validation_sft.jsonl --overwrite
uv run promptillery materialize-sft examples/G3_<D>_materialize.yaml \
  --mode gold --split test       --output out/g3/<D>/test_sft.jsonl       --overwrite
```

Each pass also writes `out/g3/<D>/canonical_labels.json` (overwrites on final pass; they are identical).

### Step 2: Run Each Student as a Cycles Ablation

For each of the three students, run:

```bash
uv run promptillery ablation examples/G3_<D>_ettin_encoder.yaml --no-cleanup
uv run promptillery ablation examples/G3_<D>_ettin_decoder.yaml --no-cleanup
uv run promptillery ablation examples/G3_<D>_gemma3_270m.yaml  --no-cleanup
```

**Important:** `--no-cleanup` is mandatory. Each config specifies `cycles: [1, 5, 10]`, which expands into three independent training runs (one per cycle point). Without `--no-cleanup`, the ablation default would keep only the best config and discard cycles 1 and 5. Each run completes with a `report_held_out_test` pass that scores the **full** test split on that run's final model — this is what fills each table cell.

### Step 3: Read Heldout Test Metrics

After all ablations, extract the three `heldout_test` accuracy numbers per student (from each run's `metrics.json`):

```bash
uv run promptillery analyze out/g3/<D> --metric accuracy
```

---

## Full Example: AG News

For a complete walkthrough with ag_news (no prep step):

```bash
# Step 1: Materialize gold SFT + canonical labels
uv run promptillery materialize-sft examples/G3_agnews_materialize.yaml \
  --mode gold --split train      --output out/g3/agnews/train_sft.jsonl      --overwrite
uv run promptillery materialize-sft examples/G3_agnews_materialize.yaml \
  --mode gold --split validation --output out/g3/agnews/validation_sft.jsonl --overwrite
uv run promptillery materialize-sft examples/G3_agnews_materialize.yaml \
  --mode gold --split test       --output out/g3/agnews/test_sft.jsonl       --overwrite

# Step 2: Run the three students (each as a 1/5/10 ablation)
uv run promptillery ablation examples/G3_agnews_ettin_encoder.yaml --no-cleanup
uv run promptillery ablation examples/G3_agnews_ettin_decoder.yaml --no-cleanup
uv run promptillery ablation examples/G3_agnews_gemma3_270m.yaml  --no-cleanup

# Step 3: Extract metrics
uv run promptillery analyze out/g3/agnews --metric accuracy
```

---

## Full Example: Yahoo

For yahoo (with prep step):

```bash
# Step 0: Prep (yahoo only)
uv run python scripts/prep_g3_datasets.py --dataset yahoo --seed 13

# Step 1: Materialize gold SFT + canonical labels
uv run promptillery materialize-sft examples/G3_yahoo_materialize.yaml \
  --mode gold --split train      --output out/g3/yahoo/train_sft.jsonl      --overwrite
uv run promptillery materialize-sft examples/G3_yahoo_materialize.yaml \
  --mode gold --split validation --output out/g3/yahoo/validation_sft.jsonl --overwrite
uv run promptillery materialize-sft examples/G3_yahoo_materialize.yaml \
  --mode gold --split test       --output out/g3/yahoo/test_sft.jsonl       --overwrite

# Step 2: Run the three students (each as a 1/5/10 ablation)
uv run promptillery ablation examples/G3_yahoo_ettin_encoder.yaml --no-cleanup
uv run promptillery ablation examples/G3_yahoo_ettin_decoder.yaml --no-cleanup
uv run promptillery ablation examples/G3_yahoo_gemma3_270m.yaml  --no-cleanup

# Step 3: Extract metrics
uv run promptillery analyze out/g3/yahoo --metric accuracy
```

---

## Notes

- **OpenRouter API key:** Set `OPENROUTER_API_KEY` in `.env` or export it before running. The CLI auto-loads `.env`.
- **Full-test evaluation:** The decoder configs omit `max_eval_generation_samples`, so the final held-out pass scores the entire test split (not a per-cycle capped sample).
- **Cycles 1/5/10:** Each ablation run independently trains for 1, 5, or 10 cycles. The three `heldout_test` numbers become the three cycle rows in `tab:main-results`.
- **Recomputation:** The three ablation runs recompute the shared cycle prefix (cycles 1–5 overlap between the cycles-5 and cycles-10 runs), so total training cost ≈ 1.6× a single cycles-10 run. This is a modest, one-time GPU cost on these tiny models.
