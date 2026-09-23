#!/usr/bin/env python
"""Render and post one GitHub issue per experiment block (EACL 2027 sprint).

    uv run python scripts/experiment_issues.py            # render docs/experiments/*.md
    uv run python scripts/experiment_issues.py --post     # create labels + issues with gh

Each issue is self-contained: an agent on any machine clones, checks out the
sprint branch, runs the listed commands verbatim, and hands results back as a
PR against the sprint branch plus a comment with the paper-ready numbers.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = "promptillery-dev/promptillery-dev"
BRANCH = "resubmit/eacl2027-demo"
DATASETS = ["agnews", "sst2", "imdb", "yahoo", "huffpost"]
SHORT_TEXT = ["agnews", "sst2", "yahoo", "huffpost"]

LABELS = {
    "experiment": "sprint experiment block",
    "gpu:laptop": "an 8 GB laptop GPU (RTX 4060) is enough",
    "gpu:24gb": "needs a 24 GB card (RTX 3090/4090 class)",
    "gpu:4090-box": "needs the July RTX 4090 machine that holds the checkpoints",
    "stretch": "run only if time allows; the paper ships without it",
}

M6_RESULTS_GLOBS = [
    "out/m6/*/metrics.json", "out/m6/*/run_manifest.json",
    "out/m6/*/token_usage.json", "out/m6/*/experiment_config.yaml",
]
M6_DOCS = ["docs/M6_RESULTS.md", "docs/M6_RESULTS.json", "docs/M6_TABLE2_ROWS.tex", "docs/M6_ISSUE_COMMENT.md"]
PREP = "for D in agnews sst2 imdb yahoo huffpost; do uv run python scripts/prep_m5_same_n.py --dataset $D --student ettin_encoder --nominal; done"


def _m6(datasets, variants, students):
    return [f"examples/M6_{d}_{v}_{s}.yaml" for d in datasets for s in students for v in variants]


BLOCKS = [
    {
        "id": "e1-laptop-roberta", "tier": "laptop",
        "title": "E1 (laptop): RoBERTa fine-tuned 1-cycle and same-N gold, corrected warmup, 5 datasets",
        "labels": ["experiment", "gpu:laptop"], "cut_line": "2026-09-23 18:00 CEST",
        "fills": "Table 2 RoBERTa `fine-tuned (1 cycle)` and `same-N gold FT` rows",
        "extra_setup": "", "prep": PREP,
        "configs": _m6(DATASETS, ["ft1", "same_n"], ["roberta_base"]),
        "runs": ["scripts/run_m6.sh " + " ".join(_m6(DATASETS, ["ft1", "same_n"], ["roberta_base"]))],
        "results_script": "uv run python scripts/m6_results.py",
        "results_globs": M6_RESULTS_GLOBS + M6_DOCS, "stretch": False,
        "notes": "About 3-5 minutes per run on short text; IMDB may drop to batch 16 or 8 through the ladder, which the run name records.",
    },
    {
        "id": "e1-laptop-ettin", "tier": "laptop",
        "title": "E1 (laptop): Ettin-encoder fine-tuned 1-cycle and same-N gold, corrected warmup, 4 short-text datasets",
        "labels": ["experiment", "gpu:laptop"], "cut_line": "2026-09-23 18:00 CEST",
        "fills": "Table 2 Ettin-encoder baseline rows for AG News, SST-2, Yahoo, HuffPost",
        "extra_setup": "", "prep": PREP,
        "configs": _m6(SHORT_TEXT, ["ft1", "same_n"], ["ettin_encoder"]),
        "runs": ["scripts/run_m6.sh " + " ".join(_m6(SHORT_TEXT, ["ft1", "same_n"], ["ettin_encoder"]))],
        "results_script": "uv run python scripts/m6_results.py",
        "results_globs": M6_RESULTS_GLOBS + M6_DOCS, "stretch": False,
        "notes": "IMDB is excluded on purpose (8 GB is not enough for Ettin at 512+ tokens); it is issue e1d-24gb-ettin-imdb.",
    },
    {
        "id": "e1d-24gb-ettin-imdb", "tier": "24gb",
        "title": "E1d (24 GB): Ettin-encoder on IMDB, fine-tuned 1-cycle and same-N gold",
        "labels": ["experiment", "gpu:24gb"], "cut_line": "2026-09-23 18:00 CEST",
        "fills": "Table 2 Ettin-encoder IMDB baseline cells (dagger footnote if batch 8)",
        "extra_setup": "", "prep": "uv run python scripts/prep_m5_same_n.py --dataset imdb --student ettin_encoder --nominal",
        "configs": _m6(["imdb"], ["ft1", "same_n"], ["ettin_encoder"]),
        "runs": ["scripts/run_m6.sh " + " ".join(_m6(["imdb"], ["ft1", "same_n"], ["ettin_encoder"]))],
        "results_script": "uv run python scripts/m6_results.py",
        "results_globs": M6_RESULTS_GLOBS + M6_DOCS, "stretch": False,
        "notes": "In July this cell ran out of memory at batch 32 and 16 on a 24 GB card and completed at 8; the ladder handles that.",
    },
    {
        "id": "e2-24gb-compute-matched", "tier": "24gb",
        "title": "E2 (24 GB): same-N gold, compute-matched (10 rounds), both encoders, 5 datasets",
        "labels": ["experiment", "gpu:24gb"], "cut_line": "2026-09-23 22:00 CEST",
        "fills": "Table 2 new row `same-N gold FT, compute-matched` for both encoders",
        "extra_setup": "", "prep": PREP,
        "configs": _m6(DATASETS, ["same_n_cm"], ["roberta_base", "ettin_encoder"]),
        "runs": ["scripts/run_m6.sh " + " ".join(_m6(DATASETS, ["same_n_cm"], ["roberta_base"])),
                 "scripts/run_m6.sh " + " ".join(_m6(DATASETS, ["same_n_cm"], ["ettin_encoder"]))],
        "results_script": "uv run python scripts/m6_results.py",
        "results_globs": M6_RESULTS_GLOBS + M6_DOCS, "stretch": False,
        "notes": "The two run lines can go in parallel on two GPUs (CUDA_VISIBLE_DEVICES=0 / 1). About 20-40 minutes per run.",
    },
    {
        "id": "e2c-24gb-compute-matched-w10", "tier": "24gb",
        "title": "E2c (24 GB): same-N gold, compute-matched, warm-up 10 (twin of #42/#46), both encoders, 5 datasets",
        "labels": ["experiment", "gpu:24gb"], "cut_line": "2026-09-23 22:00 CEST",
        "fills": "Table 2 row `same-N gold FT, compute-matched` at warm-up 10 for both encoders (replaces the warm-up-500 row from #40 when it lands)",
        "extra_setup": "", "prep": PREP,
        "configs": _m6(DATASETS, ["same_n_cm_w10"], ["roberta_base", "ettin_encoder"]),
        "runs": ["scripts/run_m6.sh " + " ".join(_m6(DATASETS, ["same_n_cm_w10"], ["roberta_base"])),
                 "scripts/run_m6.sh " + " ".join(_m6(DATASETS, ["same_n_cm_w10"], ["ettin_encoder"]))],
        "results_script": "uv run python scripts/m6_results.py",
        "results_globs": M6_RESULTS_GLOBS + M6_DOCS, "stretch": False,
        "notes": "Same as #40 but with `warmup_steps: 10` in every one of the 10 rounds, so this control trains exactly like the corrected loop re-runs (#42, #46). No teacher calls, no API key. On a 24 GB card the imdb/ettin arm needs the batch ladder (32 -> 16 -> 8) and takes about 2.5 h at batch 8; on an 80 GB card it runs at batch 32 in well under an hour. The two run lines can go in parallel on two GPUs (CUDA_VISIBLE_DEVICES=0 / 1). The other nine runs take 7-25 minutes each.",
    },
    {
        "id": "e2b-24gb-seed-x10", "tier": "24gb",
        "title": "E2b (24 GB, stretch): gold seed only, 10 rounds, both encoders, 5 datasets",
        "labels": ["experiment", "gpu:24gb", "stretch"], "cut_line": "2026-09-23 22:00 CEST",
        "fills": "Appendix B repeated-training control",
        "extra_setup": "",
        "prep": "for D in yahoo huffpost; do uv run python scripts/prep_g3_datasets.py --dataset $D --seed 13; done",
        "configs": _m6(DATASETS, ["seed_x10"], ["roberta_base", "ettin_encoder"]),
        "runs": ["scripts/run_m6.sh " + " ".join(_m6(DATASETS, ["seed_x10"], ["roberta_base", "ettin_encoder"]))],
        "results_script": "uv run python scripts/m6_results.py",
        "results_globs": M6_RESULTS_GLOBS + M6_DOCS, "stretch": True,
        "notes": "AG News, SST-2 and IMDB need no prep (the configs carve the seed subset from the HF dataset); Yahoo and HuffPost read the normalized out/g3/<D>/{train,test}.jsonl that the prep line writes.",
    },
    {
        "id": "e4-24gb-roberta-rerun", "tier": "24gb",
        "title": "E4 (24 GB, ~$1 API): RoBERTa promptillery arms 1/5/10 cycles with corrected warmup, 5 datasets, shipped configs",
        "labels": ["experiment", "gpu:24gb"], "cut_line": "launch by 2026-09-23 12:00 CEST, results by 22:00",
        "fills": "Table 2 RoBERTa `promptillery (5/10 cycles)` rows, replacing the prior-version numbers (trained with 500 warm-up steps per cycle)",
        "extra_setup": "printf 'OPENROUTER_API_KEY=%s\\n' '<your OpenRouter key>' > .env   # the CLI loads .env; about $0.20 per dataset",
        "prep": "for D in yahoo huffpost; do uv run python scripts/prep_g3_datasets.py --dataset $D --seed 13; done",
        "configs": [f"examples/G3_{d}_roberta_base.yaml" for d in DATASETS],
        "runs": [f"uv run promptillery ablation examples/G3_{d}_roberta_base.yaml --no-cleanup" for d in DATASETS],
        "results_script": "uv run promptillery analyze out/g3 --metric accuracy --output out/g3/roberta_rerun_summary.csv",
        "results_globs": ["out/g3/ablation_g3_*_roberta_base_*/*/metrics.json", "out/g3/ablation_g3_*_roberta_base_*/*/run_manifest.json",
                          "out/g3/ablation_g3_*_roberta_base_*/*/token_usage.json", "out/g3/ablation_g3_*_roberta_base_*/*/experiment_config.yaml",
                          "out/g3/ablation_g3_*_roberta_base_*/ablation_summary.md", "out/g3/roberta_rerun_summary.csv"],
        "stretch": False,
        "notes": "Paste the heldout_test accuracy of the cycles-1, cycles-5 and cycles-10 arms per dataset into the comment. Disk: with `--no-cleanup` each ablation leaves about 15 GB of checkpoints under `training/`; after a run line finishes you may `rm -rf out/g3/ablation_g3_*/*/training` (those files are never committed).",
    },
    {
        "id": "e4b-24gb-ettin-rerun", "tier": "24gb",
        "title": "E4b (24 GB, ~$1 API): Ettin-encoder promptillery arms 1/5/10 cycles with corrected warmup, 5 datasets",
        "labels": ["experiment", "gpu:24gb"], "cut_line": "launch by 2026-09-23 12:00 CEST, results by 20:00",
        "fills": "Table 2 Ettin-encoder promptillery (5/10 cycles) rows, replacing the July runs trained with 500 warm-up steps per cycle",
        "extra_setup": "printf 'OPENROUTER_API_KEY=%s\\n' '<your OpenRouter key>' > .env   # the CLI loads .env; about $0.20 per dataset",
        "prep": "for D in yahoo huffpost; do uv run python scripts/prep_g3_datasets.py --dataset $D --seed 13; done",
        "configs": [f"examples/G3_{d}_ettin_encoder_w10.yaml" for d in DATASETS],
        "runs": [f"uv run promptillery ablation examples/G3_{d}_ettin_encoder_w10.yaml --no-cleanup" for d in DATASETS],
        "results_script": "uv run promptillery analyze out/g3 --metric accuracy --output out/g3/ettin_w10_rerun_summary.csv",
        "results_globs": ["out/g3/ablation_g3_*_ettin_encoder_w10_*/*/metrics.json", "out/g3/ablation_g3_*_ettin_encoder_w10_*/*/run_manifest.json",
                          "out/g3/ablation_g3_*_ettin_encoder_w10_*/*/token_usage.json", "out/g3/ablation_g3_*_ettin_encoder_w10_*/*/experiment_config.yaml",
                          "out/g3/ablation_g3_*_ettin_encoder_w10_*/ablation_summary.md", "out/g3/ettin_w10_rerun_summary.csv"],
        "stretch": False,
        "notes": "Paste the heldout_test accuracy of the cycles-1, cycles-5 and cycles-10 arms per dataset into the comment. IMDB is the long one; the others take 20-40 minutes each on a 24 GB card. Disk: with `--no-cleanup` each ablation leaves about 15 GB of checkpoints under `training/`; after a run line finishes you may `rm -rf out/g3/ablation_g3_*/*/training` (those files are never committed).",
    },
    {
        "id": "e5-laptop-seeds", "tier": "laptop",
        "title": "E5 (laptop, stretch): seeds 7 and 21 for the RoBERTa 1-cycle baseline on SST-2 and AG News",
        "labels": ["experiment", "gpu:laptop", "stretch"], "cut_line": "2026-09-23 22:00 CEST",
        "fills": "Appendix B variance sentence",
        "extra_setup": "", "prep": "",
        "configs": ["examples/M6_sst2_ft1_roberta_base.yaml", "examples/M6_agnews_ft1_roberta_base.yaml"],
        "runs": ["for S in 7 21; do for D in sst2 agnews; do uv run python scripts/gen_m6_configs.py --only m6_${D}_ft1_roberta_base --seed $S; done; done",
                 "scripts/run_m6.sh examples/M6_sst2_ft1_roberta_base_s7.yaml examples/M6_sst2_ft1_roberta_base_s21.yaml examples/M6_agnews_ft1_roberta_base_s7.yaml examples/M6_agnews_ft1_roberta_base_s21.yaml"],
        "results_script": "uv run python scripts/m6_results.py",
        "results_globs": M6_RESULTS_GLOBS + M6_DOCS + ["examples/M6_*_s7.yaml", "examples/M6_*_s21.yaml"], "stretch": True,
        "notes": "Seed runs appear in docs/M6_RESULTS.md with their seed; Table 2 keeps seed 13.",
    },
    {
        "id": "e6-laptop-verifier", "tier": "laptop",
        "title": "E6 (laptop, stretch): gold-only Banking77 verifier, then the audit on the encoder and decoder runs",
        "labels": ["experiment", "gpu:laptop", "stretch"], "cut_line": "2026-09-23 18:00 CEST",
        "fills": "Table 5 verifier gold accuracy and decoder audit rows",
        "extra_setup": "", "prep": "",
        "configs": ["examples/G2_banking77_verifier_roberta.yaml"],
        "runs": ["uv run promptillery train examples/G2_banking77_verifier_roberta.yaml",
                 "uv run promptillery audit out/g2/g2_banking77_ettin_encoder_transformers_* out/g2/g2_banking77_ettin_decoder_slm_* out/g2/g2_banking77_gemma3_270m_slm_* --verifier-model out/g2_verifier/g2_banking77_verifier_roberta_*/model --latex | tee out/g2_verifier/audit_rows.tex"],
        "results_script": "cat out/g2_verifier/g2_banking77_verifier_roberta_*/metrics.json | head -c 2000",
        "results_globs": ["out/g2_verifier/*/metrics.json", "out/g2_verifier/*/run_manifest.json", "out/g2_verifier/audit_rows.tex", "out/g2/*/audit/*.json", "out/g2/*/audit/*.csv"],
        "stretch": True,
        "notes": "Paste the verifier's heldout_test accuracy and the printed LaTeX audit rows into the comment.",
    },
    {
        "id": "e8-4090-batch8", "tier": "4090-box",
        "title": "E8 (4090 box only, stretch): batch-8 latency profiles of the six Banking77 students",
        "labels": ["experiment", "gpu:4090-box", "stretch"], "cut_line": "2026-09-23 12:00 CEST",
        "fills": "Table 3 footnote and Appendix B batch-1 vs batch-8 comparison",
        "extra_setup": "# run on the machine that still holds the July checkpoints under out/g2/*/model; the profile stamp guard requires the RTX 4090",
        "prep": "",
        "configs": [f"examples/G2_banking77_{s}.yaml" for s in ["roberta_base", "ettin_encoder", "modernbert_base", "ettin_decoder", "gemma3_270m", "fasttext"]],
        "runs": [f"uv run promptillery profile examples/G2_banking77_{s}.yaml --model-path $(ls -d out/g2/g2_banking77_{s}_*/model | head -n1) --batch-size 8 --iterations 50"
                 for s in ["roberta_base", "ettin_encoder", "modernbert_base", "ettin_decoder", "gemma3_270m"]]
                + ["uv run promptillery profile examples/G2_banking77_fasttext.yaml --model-path $(ls -d out/g2/g2_banking77_fasttext_* | head -n1) --batch-size 8 --iterations 50"],
        "results_script": "ls out/g2/*/model/profile-bs8.json out/g2/*/profile-bs8.json",
        "results_globs": ["out/g2/*/model/profile-bs8.json", "out/g2/*/profile-bs8.json"], "stretch": True,
        "notes": "The profiler --batch-size option is on the sprint branch (profile-bs8.json is written next to profile.json). Paste p50/p95/throughput per student into the comment.",
    },
]


def render(block: dict) -> str:
    runs = "\n".join(block["runs"])
    globs = " ".join(block["results_globs"])
    stretch = "**Stretch: the paper ships without this block if the cut line passes.**\n\n" if block["stretch"] else ""
    ablation = any("promptillery ablation" in run for run in block["runs"])
    oom_rule = (
        "If an arm dies with CUDA out of memory at batch 32, copy that config next to it with `batch_size: 16` "
        "and its `name` suffixed `_bs16` (the copy is not committed), rerun the copy, and report it. "
        "`scripts/run_m6.sh` does not apply to `promptillery ablation`."
        if ablation else
        "The only permitted change is the out-of-memory ladder built into `scripts/run_m6.sh` (batch 32 → 16 → 8), "
        "which records the batch size in the run name."
    )
    return f"""# {block['title']}

Labels: {', '.join(block['labels'])} · GPU tier: **{block['tier']}** · Cut line: **{block['cut_line']}**

{stretch}**Fills:** {block['fills']}.

## Setup (once)

```bash
git clone https://github.com/{REPO}.git && cd promptillery-dev
git checkout {BRANCH}
uv sync --extra fasttext
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
{block['extra_setup']}
```

## Data prep (no API key)

```bash
{block['prep'] or '# none'}
```

## Run (verbatim)

```bash
{runs}
```

{block['notes']}

## Ground rules

1. **Never edit a config.** {oom_rule}
2. One seed (13) unless the block says otherwise. No repeats, no hyper-parameter changes.
3. If a run fails for a reason other than memory, keep going with the next config and report it.
4. Report every deviation, even ones you fixed.

## Hand results back

```bash
{block['results_script']}
git checkout -b results/{block['id']}
git add -f {globs}
git commit -m "results({block['id']}): <one line on what completed>"
git push -u origin results/{block['id']}
gh pr create --repo {REPO} --base {BRANCH} --title "results: {block['id']}" --body-file docs/M6_ISSUE_COMMENT.md
gh issue comment <THIS ISSUE NUMBER> --repo {REPO} --body-file docs/M6_ISSUE_COMMENT.md
```

If this block does not produce `docs/M6_ISSUE_COMMENT.md`, write the comment by hand: one markdown table with the numbers named under **Fills**, then a `Deviations` list. Never commit model weights, `training/`, `dataset_cycle_*`, or logs.

If the first hand-back line fails because a run is not `completed` (a crashed or failed arm), drop the summary CSV from the `git add` line, commit the rest, and list that arm under Deviations; the `metrics.json` files carry every number the paper needs.
"""


def write_all(out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for block in BLOCKS:
        path = out_dir / f"{block['id']}.md"
        path.write_text(render(block), encoding="utf-8")
        paths.append(path)
    return paths


def post(out_dir: Path, only: str | None = None) -> None:
    for name, desc in LABELS.items():
        subprocess.run(["gh", "label", "create", name, "--repo", REPO, "--description", desc, "--force"], check=True)
    for block in BLOCKS:
        if only and block["id"] != only:
            continue
        body = out_dir / f"{block['id']}.md"
        try:
            result = subprocess.run(
                ["gh", "issue", "create", "--repo", REPO, "--title", block["title"],
                 "--body-file", str(body), "--label", ",".join(block["labels"])],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            print(e.stderr, file=sys.stderr)
            raise
        print(block["id"], result.stdout.strip())


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="docs/experiments")
    p.add_argument("--post", action="store_true")
    p.add_argument("--only", default=None, help="restrict --post to one block id")
    args = p.parse_args(argv)
    paths = write_all(Path(args.out_dir))
    for path in paths:
        print("wrote", path)
    if args.post:
        post(Path(args.out_dir), args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
