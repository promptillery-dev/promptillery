"""Normalize GSM8K into the G4 1K-subset protocol JSONL (+ same-N gold top-ups).

Deterministic and seeded (default 13). Writes {question, answer, answer_number}
JSONL under out/g4/gsm8k/. GSM8K has no class labels and the engine sampling
block has no unstratified path (engine.py:187-192 skips silently), so the
1000-row carve happens here. Calculator annotations <<...>> are stripped (free
parameter, recorded here). See
docs/superpowers/specs/2026-07-10-g4-g5-gsm8k-generative-kd-design.md §4.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from promptillery.utils import extract_final_number

CALC_ANNOTATION = re.compile(r"<<[^>]*>>")


def strip_calculator_annotations(answer: str) -> str:
    return CALC_ANNOTATION.sub("", answer)


def normalize_gsm8k(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        answer = strip_calculator_annotations(row["answer"]).strip()
        number = extract_final_number(answer)
        if number is None:
            raise ValueError(f"GSM8K row without numeric final answer: {row['answer']!r}")
        out.append(
            {"question": row["question"].strip(), "answer": answer, "answer_number": number}
        )
    return out


def sample_and_split(
    records: list[dict], sample_size: int, train_ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Seeded UNstratified sample + split. Sorts by question first so the
    carve is independent of upstream row order."""
    pool = sorted(records, key=lambda r: r["question"])
    rng = random.Random(seed)
    rng.shuffle(pool)
    sample = pool[:sample_size]
    cut = int(round(train_ratio * len(sample)))
    return sample[:cut], sample[cut:]


def topup_pool(
    all_train: list[dict],
    seed_train: list[dict],
    target_n: int,
    seed: int,
    *,
    exclude: list[dict] | None = None,
) -> list[dict]:
    """Seeded SUPERSET of seed_train topped up to target_n from problems unused
    by BOTH the seed train and exclude (e.g. the validation carve)."""
    if target_n < len(seed_train):
        raise ValueError(f"target_n={target_n} < seed train size {len(seed_train)}")
    seed_questions = {r["question"] for r in seed_train}
    barred = seed_questions | {r["question"] for r in (exclude or [])}
    unused = sorted(
        (r for r in all_train if r["question"] not in barred),
        key=lambda r: r["question"],
    )
    rng = random.Random(seed)
    rng.shuffle(unused)
    need = target_n - len(seed_train)
    if need > len(unused):
        raise ValueError(f"Not enough unused problems: need {need}, have {len(unused)}")
    return list(seed_train) + unused[:need]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--out-dir", type=Path, default=Path("out/g4/gsm8k"))
    parser.add_argument(
        "--topup-to",
        type=int,
        default=None,
        help="Emit train_topup_<N>.jsonl (superset of the seed train) and exit",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    raw = load_dataset("openai/gsm8k", "main")
    train_records = normalize_gsm8k(list(raw["train"]))
    test_records = normalize_gsm8k(list(raw["test"]))
    seed_train, seed_val = sample_and_split(
        train_records, args.sample_size, args.train_ratio, args.seed
    )

    if args.topup_to is not None:
        topped = topup_pool(train_records, seed_train, args.topup_to, args.seed, exclude=seed_val)
        out = args.out_dir / f"train_topup_{args.topup_to}.jsonl"
        write_jsonl(out, topped)
        print(f"Wrote {len(topped)} records to {out}")
        return

    write_jsonl(args.out_dir / "train.jsonl", seed_train)
    write_jsonl(args.out_dir / "validation.jsonl", seed_val)
    write_jsonl(args.out_dir / "test.jsonl", test_records)
    print(
        f"Wrote {len(seed_train)}/{len(seed_val)}/{len(test_records)} "
        f"train/validation/test records to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
