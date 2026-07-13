# G4/G5 — GSM8K Generative Table + Logit-KD Probe

**Requires GPUs.** These campaigns are not exercised in CI; the GSM8K prep and
all gold materialization steps are free and CPU-only, but training the students
(and the KD arm in particular) needs an 80GB-class card.

Two campaigns share the GSM8K prep pipeline and the same 1K-subset/seed-13
protocol as the main-results table:

- **G4** (the generative table): the cross-architecture generative-KD
  table on GSM8K. Roster: GPT-4.1 teacher (`openrouter/openai/gpt-4.1`), four
  decoder students — Qwen3-4B-Instruct, SmolLM3-3B, Ettin-decoder-150m,
  Gemma-3-270M — each run as a `cycles: [1, 5, 10]` ablation, plus a same-N
  gold-FT control per student (below).
- **G5** (KD probe): a single-shot SFT-vs-logit-KD comparison —
  Qwen3-4B fine-tuned on identical Qwen3-32B-generated solutions, once with
  plain SFT and once with an added token-level forward-KL loss against the
  Qwen3-32B teacher (pinned to its own GPU via `kd.teacher_device_map`, since
  its ~66GB bf16 footprint cannot co-locate with the student on one 80GB
  card). No cycles, no teacher API calls.

---

## Run order

```bash
export OPENROUTER_API_KEY=...              # G4 augmentation + teacher rows only

# 0) prep (free)
uv run python scripts/prep_gsm8k.py --seed 13

# 1) gold SFT (free)
uv run promptillery materialize-sft examples/paper/G4_gsm8k_materialize.yaml \
  --mode gold --split train      --output out/g4/gsm8k/train_sft.jsonl      --overwrite
uv run promptillery materialize-sft examples/paper/G4_gsm8k_materialize.yaml \
  --mode gold --split validation --output out/g4/gsm8k/validation_sft.jsonl --overwrite
uv run promptillery materialize-sft examples/paper/G4_gsm8k_materialize.yaml \
  --mode gold --split test       --output out/g4/gsm8k/test_sft.jsonl       --overwrite

# 2) teacher ceiling rows (API)
uv run python -m promptillery.baseline_eval --task generative ...   # zero- and 5-shot

# 3) students as cycles:[1,5,10] ablations (API for augmentation)
uv run promptillery ablation examples/paper/G4_gsm8k_qwen3_4b.yaml     --no-cleanup
uv run promptillery ablation examples/paper/G4_gsm8k_smollm3.yaml      --no-cleanup
uv run promptillery ablation examples/paper/G4_gsm8k_ettin_decoder.yaml --no-cleanup
uv run promptillery ablation examples/paper/G4_gsm8k_gemma3_270m.yaml  --no-cleanup

# 4) same-N gold FT per student (free of teacher calls): read N -> --topup-to N
#    -> materialize gold -> run G4_gsm8k_goldft_<student>.yaml

# 5) G5 (no API key): generate teacher solutions + ceiling, then both arms
uv run python scripts/g5_generate_teacher_solutions.py
uv run promptillery train examples/paper/G5_gsm8k_kd_sft.yaml
uv run promptillery train examples/paper/G5_gsm8k_kd_logit.yaml
```

Both G5 configs read `out/g4/gsm8k/test_sft.jsonl` as their held-out test split, so
G4's prep + test-split materialization (steps 0 and 1) must run before any G5 arm —
even on a G5-only box with no G4 training planned.

`--no-cleanup` on the G4 ablations is mandatory: each config expands
`cycles: [1, 5, 10]` into three independent training runs (one per cycle
point), and without the flag the ablation runner keeps only the best config
and discards the rest. Each run ends with a `report_held_out_test` pass over
the full 1,319-row GSM8K test split — that pass fills the table cell.

---

## Smoke gates before the full campaign

Run these cheap/short checks before committing GPU time to the full 1K/5/10
sweep — a silent failure here degrades every downstream cycle row to noise:

1. **Materialize smoke:** run step 1 above (gold-mode train/val/test) and
   confirm no `canonical_labels.json` is written and the records look sane
   (non-empty `student_prompt` containing the problem text, `gold_answer`
   ending `#### <number>`). Cheap, CPU-only.
2. **AL go/no-go (standing paper `\todo`):** a 2-cycle Qwen3-4B run. Verify
   `generation_entropy` spreads across the validation predictions (not
   degenerate), screened augmentation records parse, and the augmentation
   `prompt:` renders under `StrictUndefined` with the mandatory
   `few_shot_samples[:6]` slice (unsliced, near-unique GSM8K gold answers
   make the group ≈ the whole 800-row train split — see
   `utils.py:extract_few_shot_samples`).
3. **KD smoke:** a 50-step `G5_gsm8k_kd_logit.yaml` run. Confirm the KD loss
   is finite and decreasing, and that the Qwen3-32B teacher (pinned to
   `cuda:1` via `kd.teacher_device_map`) plus the Qwen3-4B LoRA student
   (`cuda:0`) fit H100 memory at `batch_size: 4` / `max_seq_length: 1024`.

---

## Same-N gold-FT control

Each G4 student's 10-cycle ablation run accumulates augmented training data
gated by the self-consistency screen, so the final train-split size is
run-dependent. The same-N gold-FT arm isolates "more data" from "AL-selected
data" by fine-tuning the same student on an equally-sized *gold* (unaugmented)
train set:

1. Read the **post-screening** accumulated train size **N** for the student
   from its cycles-10 run artifacts (`metrics.json` / manifest record counts).
2. Top up the seed train pool to N: `uv run python scripts/prep_gsm8k.py
   --topup-to N --seed 13` — this writes `out/g4/gsm8k/train_topup_<N>.jsonl`,
   a seeded superset of the original 800-row seed train drawn from the
   ~6,673 unused GSM8K train problems (never touching the seed val or test
   splits).
3. Point `G4_gsm8k_materialize.yaml` at the top-up file for a one-off
   `--mode gold --split train` pass (free, no teacher calls).
4. Run `G4_gsm8k_goldft_<student>.yaml` — the student's training config with
   `cycles: 1`, the enlarged gold train SFT, and no augmentation. (These four
   post-run configs are not built yet; they depend on each run's N and are
   out of scope for this plan — see the plan's "out of scope" notes.)

A null result (same-N gold FT ≈ the cycles-10 AL run) is reportable: it
isolates whether promptillery's augmentation value comes from data volume or
from AL-selected content.

---

## Notes

- **OpenRouter API key:** only needed for G4 (materialization is `--mode
  gold`, so no key required for step 1; steps 2–3 need it). G5 uses a local
  co-located teacher and needs no API key at all.
- **Label-free protocol:** GSM8K has no class labels. The 1K subset (800
  train / 200 validation, seed 13) plus full 1,319-row test is carved in
  `scripts/prep_gsm8k.py`, not the engine's `sampling:` block — the engine
  has no unstratified sampling path, so `sampling.enabled: false` in every
  G4/G5 config.
- **Full-test evaluation:** the G4 training configs omit
  `max_eval_generation_samples`, so the final `report_held_out_test` pass
  scores the entire 1,319-row test split, matching the uncapped
  the main-results table decoder columns.
- **G5 never mixes teachers:** the SFT and KD arms train on the identical
  Qwen3-32B-generated, gold-rejection-filtered solutions
  (`scripts/g5_generate_teacher_solutions.py`); only the loss function
  differs (`trainer_config.kd` block). Do not compare G5 rows against G4's
  GPT-4.1-augmented rows as a KD-vs-SFT result — the teachers differ.
- **G5 GPU footprint:** the KD arm needs two 80GB GPUs — the ~66GB bf16
  Qwen3-32B teacher pinned to `cuda:1` (`kd.teacher_device_map`) alongside
  the student on `cuda:0`. The SFT arm (no `kd` block) and the teacher-
  solution generation step (`scripts/g5_generate_teacher_solutions.py`,
  `device_map="auto"`) each run on a single 80GB card.
