# Examples

Two kinds of config live here.

## Start here — runnable examples

Small configs that demonstrate one feature each. These are what the main README
walks through, and several run with no API key at all.

| Config | What it shows |
|---|---|
| `text_classification_transformers.yaml` | The basic loop: a BERT student distilled from an LLM teacher |
| `text_classification_fasttext.yaml` | The same loop with a CPU-only FastText student |
| `causal_lm_sft_tiny.yaml` | Causal-LM SFT path — no API calls, runs offline |
| `materialize_sft_tiny.yaml` | Materializing an SFT JSONL dataset at zero teacher cost |
| `student_only_tiny.yaml` | Training a student with no teacher in the loop |
| `early_stopping.yaml` | Stopping once the monitored metric plateaus |
| `budget_stopping.yaml` | Stopping once the teacher token budget is spent |
| `policy_stop_tiny.yaml` | Budget-aware acquisition policy with an explicit STOP action |
| `augmentation_without_active_learning_selection.yaml` | Augmentation with random rather than uncertainty-based selection |
| `ablation_minimal.yaml` | Sweeping one parameter via list syntax |
| `ablation_model_comparison.yaml` | Comparing student architectures |
| `ablation_teacher_comparison.yaml` | Comparing teacher models |
| `hyperparameter_sweep.yaml` | Grid search over learning rate, epochs, and cycles |

## Inspecting results without training

`analyze_demo/` is a completed two-cycle run, checked in so that the analysis
tooling has something to chew on before you have run anything yourself:

```bash
promptillery analyze examples/analyze_demo
```

That prints the quality-cost summary — best cycle, teacher token spend, and the
quality-cost AUC — as CSV. Its numbers are invented; what is real is the set of
artifacts a run leaves behind (`metrics.json`, `token_usage.json`,
`run_manifest.json`), which is what `analyze` reads.

## Supporting files

`data/` holds the tiny JSONL datasets the offline examples train on, and
`fixtures/` holds a small WordPiece tokenizer that stands in for a real student
model. Together they are what lets the `*_tiny` configs above run on a laptop
CPU in seconds, with no API key. You should not need to edit either.

## `paper/` — the campaign configs

The configs behind the paper's experiments. They are grouped by campaign, and
each group has its own README with the run order, expected costs, and hardware
requirements. Most need a GPU and a teacher API key; the materialize and prep
steps are free.

| Campaign | What it runs |
|---|---|
| `G2_banking77_*` | Banking77 across seven students, from FastText to a 4B LoRA decoder — the cost/quality frontier |
| `G3_*` (see `paper/G3_README.md`) | Five datasets × three students — the main results table |
| `G4_gsm8k_*` (see `paper/G4_G5_README.md`) | GSM8K generative distillation across four decoder students |
| `G5_gsm8k_*` (see `paper/G4_G5_README.md`) | Logit-KD probe: token-level forward-KL against a local 32B teacher |
| `M5_*` (see `paper/M5_README.md`) | Same-N gold fine-tuning baseline — the control arm |

Configs whose name ends in `_materialize` do not train anything: they write the
dataset an experiment consumes. Configs ending in `_rescue_4090` are reduced-memory
variants of their namesake, for a 24GB consumer card rather than an 80GB
datacenter one.
