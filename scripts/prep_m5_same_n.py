"""Build M5 same-N gold-FT baseline data (issue #8): seed + matched gold top-up.

For one (dataset, student) cell of tab:main-results:
1. read the completed cycles-10 run's run_manifest.json ->
   final_synthetic_count (the post-filtering synthetic rows the run actually
   accumulated; nominally augmentation_batch_size x 9 minus filtered rejects),
2. reconstruct the exact 1K-seed train/val carve of the G3 protocol
   (sample_size 1000 / train_ratio 0.8 / stratify label / seed 13),
   cross-checked against promptillery.engine.prepare_dataset,
3. draw final_synthetic_count EXTRA rows (seeded, stratified) from the UNUSED
   remainder of the original train split -- real gold labels, never
   teacher-relabeled,
4. write out/m5/<D>/<student>/train.jsonl (seed-train + top-up: a strict
   superset of the seed-train rows), out/m5/<D>/validation.jsonl,
   out/m5/<D>/test.jsonl, and out/m5/<D>/<student>/matched_n_manifest.json,
5. decoder cells only: gold-materialize out/m5/<D>/<student>/train_sft.jsonl
   by reusing examples/G3_<D>_materialize.yaml (prompt-format identity with
   the promptillery arms by construction; gold mode = zero teacher calls).

RoBERTa's cell has NO prep invocation: its matched N reuses the Ettin-encoder
run's N (decision 2026-07-10), and same seed + same N => identical data, so
M5_<D>_same_n_roberta_base.yaml reads the ettin_encoder train file.

Deterministic given --seed. Requires the cell's completed G3 cycles-10 run dir
(plus out/g3/<D>/{train,test}.jsonl for yahoo/huffpost). Zero teacher calls.
See docs/superpowers/plans/2026-07-10-m5-same-n-gold-ft.md and issue #8.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path

from datasets import DatasetDict, load_dataset

from promptillery.config import ExperimentConfig, SamplingConfig
from promptillery.engine import ensure_class_label, prepare_dataset
from promptillery.sft_materialize import materialize_sft_records

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"

STUDENTS = {
    "ettin_encoder": "jhu-clsp/ettin-encoder-150m",
    "ettin_decoder": "jhu-clsp/ettin-decoder-150m",
    "gemma3_270m": "google/gemma-3-270m-it",
}
DECODER_STUDENTS = ("ettin_decoder", "gemma3_270m")

# Mirrors the G3 config dataset sources (examples/G3_<D>_*.yaml).
HF_SOURCES = {
    "agnews": ("SetFit/ag_news", "default"),
    "sst2": ("SetFit/sst2", "default"),
    "imdb": ("stanfordnlp/imdb", "plain_text"),
}
PREPPED = ("yahoo", "huffpost")
DATASETS = tuple(HF_SOURCES) + PREPPED

# The G3 1K-subset protocol (must match the G3 configs' sampling blocks).
SAMPLE_SIZE = 1000
TRAIN_RATIO = 0.8
STRATIFY = "label"


def read_matched_count(run_dir: Path, dataset: str, student: str) -> dict:
    """Validate the cell's cycles-10 source run and return its matched count."""
    manifest_path = Path(run_dir) / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"no run_manifest.json under {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = str(manifest.get("run_id") or "")
    problems = []
    if manifest.get("status") != "completed":
        problems.append(f"status={manifest.get('status')!r} (want 'completed')")
    if manifest.get("cycles_completed") != manifest.get("expected_cycles"):
        problems.append(
            f"cycles_completed={manifest.get('cycles_completed')} != "
            f"expected_cycles={manifest.get('expected_cycles')}"
        )
    if manifest.get("expected_cycles") != 10:
        problems.append(
            f"expected_cycles={manifest.get('expected_cycles')} "
            "(want 10; same-N matches the 10-cycle arm)"
        )
    if manifest.get("student_model") != STUDENTS[student]:
        problems.append(
            f"student_model={manifest.get('student_model')!r} "
            f"(want {STUDENTS[student]!r})"
        )
    if not run_id.startswith(f"g3_{dataset}_{student}"):
        problems.append(f"run_id={run_id!r} does not start with g3_{dataset}_{student}")
    count = manifest.get("final_synthetic_count")
    if not isinstance(count, int) or count <= 0:
        problems.append(f"final_synthetic_count={count!r} (want a positive int)")
    if problems:
        raise ValueError(
            f"{run_dir} is not a usable cycles-10 source run: " + "; ".join(problems)
        )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "final_synthetic_count": count,
        "matched_n_source": "run_manifest",
    }


def nominal_matched_count(dataset: str, student: str) -> dict:
    """Approximate matched N from config, no campaign artifacts needed.

    A 10-cycle run augments after every cycle except the last (engine.py:2038
    gates on cycle < cycles - 1), so the cycle-10 training set nominally gains
    augmentation_batch_size x (cycles - 1) synthetic rows. Per-run filtering
    can only shave a few rows off (observed SST-2: 288/288/284 kept of 288
    nominal). Values are read from the cell's G3 training config rather than
    hardcoded, so a protocol change there propagates here.
    """
    source_path = EXAMPLES_DIR / f"G3_{dataset}_{student}.yaml"
    if not source_path.exists():
        raise ValueError(f"no G3 training config at {source_path}")
    cfg = ExperimentConfig.from_yaml(str(source_path))
    cycles = max(cfg.cycles) if isinstance(cfg.cycles, list) else int(cfg.cycles)
    count = int(cfg.augmentation_batch_size) * (cycles - 1)
    return {
        "run_id": None,
        "run_dir": None,
        "final_synthetic_count": count,
        "matched_n_source": "nominal",
    }


def reconstruct_seed_split(
    ds: DatasetDict,
    seed: int,
    sample_size: int = SAMPLE_SIZE,
    train_ratio: float = TRAIN_RATIO,
):
    """Replay the G3 sampling carve; return (seed_train, seed_val, complement).

    The complement is the UNUSED remainder of the original train split -- the
    pool the same-N top-up is drawn from. The replay is cross-checked against
    promptillery.engine.prepare_dataset so any sampler change fails loudly
    instead of silently building the wrong baseline.
    """
    train_ds = ds["train"]
    if len(train_ds) <= sample_size:
        raise ValueError(
            f"train split has {len(train_ds)} rows <= sample_size {sample_size}; "
            "the same-N protocol needs an unused remainder to top up from"
        )
    first = train_ds.train_test_split(
        train_size=sample_size, stratify_by_column=STRATIFY, seed=seed
    )
    seed_pool, complement = first["train"], first["test"]
    second = seed_pool.train_test_split(
        test_size=1 - train_ratio, stratify_by_column=STRATIFY, seed=seed
    )
    seed_train, seed_val = second["train"], second["test"]

    check = prepare_dataset(
        DatasetDict({"train": train_ds}),
        SamplingConfig(
            enabled=True,
            sample_size=sample_size,
            train_ratio=train_ratio,
            stratify_column=STRATIFY,
            seed=seed,
        ),
    )
    if (
        check["train"].to_dict() != seed_train.to_dict()
        or check["validation"].to_dict() != seed_val.to_dict()
    ):
        raise RuntimeError(
            "seed-split reconstruction drifted from promptillery.engine.prepare_dataset"
        )
    return seed_train, seed_val, complement


def draw_topup(complement, n_extra: int, seed: int):
    """Seeded draw of n_extra gold rows from the unused pool; stratified when possible."""
    if n_extra > len(complement):
        raise ValueError(
            f"top-up of {n_extra} exceeds unused pool of {len(complement)} rows"
        )
    if n_extra == len(complement):
        return complement, True
    if n_extra < len(complement.unique(STRATIFY)):
        return complement.shuffle(seed=seed).select(range(n_extra)), False
    try:
        drawn = complement.train_test_split(
            train_size=n_extra, stratify_by_column=STRATIFY, seed=seed
        )["train"]
        return drawn, True
    except ValueError:
        # HF stratification can fail on sparse tail classes; keep it deterministic.
        return complement.shuffle(seed=seed).select(range(n_extra)), False


def bootstrap_prepped_pools(dataset: str, g3_root: Path) -> None:
    """Regenerate the normalized yahoo/huffpost pools (G3 step 0; free).

    Delegates to scripts/prep_g3_datasets.py at the protocol seed (its
    default, 13), which is deterministic: same script + seed produce pools
    byte-identical to the campaign's.
    """
    spec = importlib.util.spec_from_file_location(
        "prep_g3_datasets", REPO_ROOT / "scripts" / "prep_g3_datasets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(["--dataset", dataset, "--out-dir", str(g3_root)])


def ensure_g3_eval_artifacts(dataset: str, g3_root: Path) -> list[str]:
    """Fill missing out/g3/<D> gold eval files the decoder configs read.

    Replays the G3 README step-1 gold passes (validation/test) from the
    dataset's G3 materialize config verbatim -- gold mode makes zero teacher
    calls and record content is deterministic. Existing campaign files are
    never overwritten; only gaps are filled, so every dataset is runnable
    off the shelf without waiting on prior runs or copied artifacts.
    """
    created: list[str] = []
    dataset_dir = Path(g3_root) / dataset
    for split in ("validation", "test"):
        out_path = dataset_dir / f"{split}_sft.jsonl"
        if out_path.exists():
            continue
        cfg = ExperimentConfig.from_yaml(
            str(EXAMPLES_DIR / f"G3_{dataset}_materialize.yaml")
        )
        asyncio.run(
            materialize_sft_records(
                config=cfg,
                output_path=out_path,
                split=split,
                mode="gold",
                overwrite=False,
            )
        )
        created.append(str(out_path))
    if not (dataset_dir / "canonical_labels.json").exists():
        raise ValueError(
            f"{dataset_dir}/canonical_labels.json missing despite gold SFT files; "
            "rerun the materialize passes from examples/G3_README.md step 1"
        )
    return created


def load_source(dataset: str, g3_root: Path) -> DatasetDict:
    """Load the cell's original dataset exactly as the G3 configs read it."""
    if dataset in HF_SOURCES:
        slug, config_name = HF_SOURCES[dataset]
        ds = load_dataset(slug, config_name)
    elif dataset in PREPPED:
        files = {
            "train": str(Path(g3_root) / dataset / "train.jsonl"),
            "test": str(Path(g3_root) / dataset / "test.jsonl"),
        }
        if any(not Path(p).exists() for p in files.values()):
            bootstrap_prepped_pools(dataset, Path(g3_root))
        missing = [p for p in files.values() if not Path(p).exists()]
        if missing:
            raise ValueError(
                f"{missing} missing -- run scripts/prep_g3_datasets.py "
                f"--dataset {dataset} --seed 13 first"
            )
        ds = load_dataset("json", data_files=files)
    else:
        raise ValueError(f"unknown dataset {dataset!r} (want one of {DATASETS})")
    return ensure_class_label(ds, STRATIFY)


def to_records(ds) -> list[dict]:
    """Normalize rows to {text, label, label_text} (IMDB: names via ClassLabel)."""
    names = getattr(ds.features[STRATIFY], "names", None)
    has_label_text = "label_text" in ds.column_names
    records = []
    for row in ds:
        label = int(row[STRATIFY])
        if has_label_text:
            label_text = row["label_text"]
        elif names is not None:
            label_text = names[label]
        else:
            raise ValueError(
                "cannot derive label_text: no label_text column and no ClassLabel names"
            )
        records.append({"text": row["text"], "label": label, "label_text": str(label_text)})
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def build_materialize_config(
    dataset: str, cell_train_file: Path, out_root: Path
) -> ExperimentConfig:
    """Retarget the dataset's G3 materialize config at the cell's train file.

    Reusing examples/G3_<D>_materialize.yaml keeps the decoder prompt format
    identical to the promptillery arms BY CONSTRUCTION. Gold fields are forced
    to label_text because the prep JSONL always carries it (IMDB's G3 config
    maps ints via ClassLabel names, which a json-loaded file lacks).
    """
    source_path = EXAMPLES_DIR / f"G3_{dataset}_materialize.yaml"
    cfg = ExperimentConfig.from_yaml(str(source_path))
    trainer_config = dict(cfg.trainer_config)
    mat = dict(trainer_config.get("materialize_sft") or {})
    mat["gold_answer_field"] = "label_text"
    mat["canonical_labels_field"] = "label_text"
    trainer_config["materialize_sft"] = mat
    return cfg.model_copy(
        update={
            "name": f"m5_{dataset}_same_n_materialize",
            "dataset": "json",
            "dataset_kwargs": {"data_files": {"train": str(cell_train_file)}},
            "sampling": SamplingConfig(enabled=False),
            "require_validation_split": False,
            "base_output_dir": str(out_root),
            "trainer_config": trainer_config,
        }
    )


def build_cell(
    dataset: str,
    student: str,
    run_dir: Path,
    seed: int,
    out_root: Path,
    g3_root: Path,
    sample_size: int = SAMPLE_SIZE,
    train_ratio: float = TRAIN_RATIO,
) -> dict:
    """Build one (dataset, student) same-N cell; return the written manifest.

    run_dir=None selects nominal mode: matched N is computed from the cell's
    G3 training config (augmentation_batch_size x (cycles - 1)) instead of a
    completed run's manifest.
    """
    if run_dir is None:
        source = nominal_matched_count(dataset, student)
    else:
        source = read_matched_count(Path(run_dir), dataset, student)
    ds = load_source(dataset, Path(g3_root))
    seed_train, seed_val, complement = reconstruct_seed_split(
        ds, seed, sample_size=sample_size, train_ratio=train_ratio
    )
    n_extra = source["final_synthetic_count"]
    topup, stratified = draw_topup(complement, n_extra, seed)
    topup_records = to_records(topup)

    out_root = Path(out_root)
    cell_dir = out_root / dataset / student
    write_jsonl(cell_dir / "train.jsonl", to_records(seed_train) + topup_records)
    write_jsonl(out_root / dataset / "validation.jsonl", to_records(seed_val))
    write_jsonl(out_root / dataset / "test.jsonl", to_records(ds["test"]))

    train_sft_rows = None
    materialize_source = None
    if student in DECODER_STUDENTS:
        materialize_source = str(EXAMPLES_DIR / f"G3_{dataset}_materialize.yaml")
        mat_cfg = build_materialize_config(
            dataset, cell_dir / "train.jsonl", out_root
        )
        asyncio.run(
            materialize_sft_records(
                config=mat_cfg,
                output_path=cell_dir / "train_sft.jsonl",
                split="train",
                mode="gold",
                overwrite=True,
            )
        )
        with (cell_dir / "train_sft.jsonl").open(encoding="utf-8") as f:
            train_sft_rows = sum(1 for _ in f)
        expected = len(seed_train) + len(topup)
        if train_sft_rows != expected:
            raise RuntimeError(
                f"gold materialization wrote {train_sft_rows} records, "
                f"expected matched N = {expected}"
            )

    histogram: dict[str, int] = {}
    for r in topup_records:
        histogram[str(r["label"])] = histogram.get(str(r["label"]), 0) + 1
    manifest = {
        "dataset": dataset,
        "student": student,
        "student_model": STUDENTS[student],
        "source_run_id": source["run_id"],
        "source_run_dir": source["run_dir"],
        "matched_n_source": source["matched_n_source"],
        "final_synthetic_count": n_extra,
        "seed_train_rows": len(seed_train),
        "validation_rows": len(seed_val),
        "topup_rows": len(topup),
        "matched_n": len(seed_train) + len(topup),
        "stratified_topup": stratified,
        "topup_label_histogram": dict(sorted(histogram.items())),
        "seed": seed,
        "sample_size": sample_size,
        "train_ratio": train_ratio,
        "train_sft_rows": train_sft_rows,
        "materialize_source_config": materialize_source,
    }
    (cell_dir / "matched_n_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build one M5 same-N gold-FT cell (issue #8)."
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--student", required=True, choices=sorted(STUDENTS))
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--run-dir",
        help="the cell's completed g3 cycles-10 run directory "
        "(exact post-filter matched N from run_manifest.json)",
    )
    source_group.add_argument(
        "--nominal", action="store_true",
        help="compute matched N from the cell's G3 config instead: "
        "augmentation_batch_size x (cycles - 1); needs no campaign artifacts",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out-root", default="out/m5")
    parser.add_argument(
        "--g3-root", default="out/g3",
        help="root of the G3 artifacts (yahoo/huffpost normalized JSONL)",
    )
    args = parser.parse_args(argv)
    manifest = build_cell(
        args.dataset,
        args.student,
        Path(args.run_dir) if args.run_dir else None,
        args.seed,
        Path(args.out_root),
        Path(args.g3_root),
        sample_size=SAMPLE_SIZE,
        train_ratio=TRAIN_RATIO,
    )
    if args.student in DECODER_STUDENTS:
        # Leave the machine fully ready to train: the decoder configs read
        # out/g3/<D> gold validation/test SFT, regenerable for free.
        manifest = dict(manifest)
        manifest["bootstrapped_g3_eval_artifacts"] = ensure_g3_eval_artifacts(
            args.dataset, Path(args.g3_root)
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
