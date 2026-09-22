"""Normalize Yahoo and HuffPost into {text,label,label_text} JSONL for the G3 hub.

Deterministic and seeded (default 13). Only Yahoo and HuffPost need prep; AG News,
SST-2 and IMDB are read raw by the configs. See
docs/superpowers/specs/2026-07-09-g3-main-results-configs-design.md.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

# Single-pair merge reducing heegyu/News-Category (42 raw) to 41, per dipalo2024pgkd.
# normalize_huffpost asserts exactly 41 after merge, so a different raw count fails loudly.
HUFFPOST_MERGE_MAP = {"THE WORLDPOST": "WORLDPOST"}


def merge_huffpost_category(category: str) -> str:
    return HUFFPOST_MERGE_MAP.get(category.strip(), category.strip())


def _compose_dot(*parts: str) -> str:
    return ". ".join(p.strip() for p in parts if p and p.strip())


def normalize_huffpost(rows: list[dict]) -> list[dict]:
    merged = [(merge_huffpost_category(r["category"]),
               _compose_dot(r.get("headline", ""), r.get("short_description", ""))) for r in rows]
    label_texts = sorted({m[0] for m in merged})
    if len(label_texts) != 41:
        raise ValueError(f"Expected 41 HuffPost classes after merge, got {len(label_texts)}")
    label_of = {name: i for i, name in enumerate(label_texts)}
    return [{"text": text, "label": label_of[cat], "label_text": cat} for cat, text in merged]


def stratified_split(records: list[dict], ratios: dict[str, float], seed: int) -> dict[str, list[dict]]:
    if len(ratios) != 2:
        raise ValueError(
            f"stratified_split supports exactly two splits (got {sorted(ratios)}); "
            f"use a 2-way ratio like {{'train': 0.9, 'test': 0.1}}"
        )
    by_label: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)
    rng = random.Random(seed)
    names = list(ratios)
    out: dict[str, list[dict]] = {name: [] for name in names}
    for label in sorted(by_label):
        items = list(by_label[label])
        rng.shuffle(items)
        cut = int(round(ratios[names[0]] * len(items)))
        out[names[0]].extend(items[:cut])
        rest = items[cut:]
        for name in names[1:]:
            out[name].extend(rest)
    for name in out:
        out[name].sort(key=lambda r: r["text"])
    return out


def write_jsonl(path: str, records: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def _compose_lines(*parts: str) -> str:
    return "\n".join(p.strip() for p in parts if p and p.strip())


def normalize_yahoo(rows: list[dict], topic_names: list[str]) -> list[dict]:
    out = []
    for r in rows:
        label = int(r["topic"])
        text = _compose_lines(r.get("question_title", ""), r.get("question_content", ""))
        out.append({"text": text, "label": label, "label_text": topic_names[label]})
    return out


def stratified_pool(records: list[dict], cap: int, seed: int) -> list[dict]:
    by_label: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)
    rng = random.Random(seed)
    per = max(1, cap // max(1, len(by_label)))
    pooled: list[dict] = []
    for label in sorted(by_label):
        items = list(by_label[label])
        rng.shuffle(items)
        pooled.extend(items[:per])
    pooled.sort(key=lambda r: r["text"])
    return pooled[:cap]


def _prep_yahoo(out_dir: Path, seed: int, pool: int) -> None:
    from datasets import load_dataset
    ds = load_dataset("community-datasets/yahoo_answers_topics")
    names = ds["train"].features["topic"].names
    train = normalize_yahoo([dict(r) for r in ds["train"]], names)
    test = normalize_yahoo([dict(r) for r in ds["test"]], names)
    write_jsonl(str(out_dir / "yahoo" / "train.jsonl"), stratified_pool(train, pool, seed))
    write_jsonl(str(out_dir / "yahoo" / "test.jsonl"), test)


def _prep_huffpost(out_dir: Path, seed: int) -> None:
    from datasets import load_dataset
    ds = load_dataset("heegyu/news-category-dataset")
    records = normalize_huffpost([dict(r) for r in ds["train"]])
    parts = stratified_split(records, {"train": 0.9, "test": 0.1}, seed=seed)
    write_jsonl(str(out_dir / "huffpost" / "train.jsonl"), parts["train"])
    write_jsonl(str(out_dir / "huffpost" / "test.jsonl"), parts["test"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Prep Yahoo/HuffPost for the G3 hub")
    ap.add_argument("--dataset", required=True, choices=["yahoo", "huffpost"])
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-dir", default="out/g3")
    ap.add_argument("--yahoo-pool", type=int, default=20000)
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.dataset == "yahoo":
        _prep_yahoo(out_dir, args.seed, args.yahoo_pool)
    else:
        _prep_huffpost(out_dir, args.seed)


if __name__ == "__main__":
    main()
