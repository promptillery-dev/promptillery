"""Generate local-teacher (Qwen3-32B) GSM8K solutions for the G5 KD probe.

Greedy decode over the G4 seed train/validation problems (out/g4/gsm8k/,
written by scripts/prep_gsm8k.py); rejection-filter the TRAIN records against
the dataset gold final answers so the SFT and KD arms train on the identical
surviving set; optionally score teacher zero-shot EM on the full test split
(the KD block's ceiling row). GPU work — not CI-verified; the pure pieces are
unit-tested in tests/test_g5_teacher_solutions.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from promptillery.utils import extract_final_number

# MUST stay in sync with examples/paper/G4_gsm8k_materialize.yaml student_prompt_template.
SOLVE_PROMPT = (
    "Solve the grade-school math problem. Think step by step, then give the\n"
    'final answer on its own line in the form "#### <answer>".\n'
    "Problem: {question}\n"
    "Solution:"
)


def build_solve_prompt(question: str) -> str:
    return SOLVE_PROMPT.format(question=question)


def filter_against_gold(records: list[dict]) -> tuple[list[dict], float]:
    """Keep records whose teacher solution ends in the gold final answer."""
    kept = [
        record
        for record in records
        if extract_final_number(record["teacher_response"]) == record["answer_number"]
    ]
    rate = len(kept) / len(records) if records else 0.0
    return kept, rate


def build_gold_test_records(rows: list[dict]) -> list[dict]:
    """Normalize held-out gold rows to the teacher-SFT JSON schema."""
    return [
        {
            "student_prompt": build_solve_prompt(row["question"]),
            "teacher_response": row["answer"],
            "gold_answer": row["answer"],
            "answer_number": row["answer_number"],
        }
        for row in rows
    ]


def add_teacher_token_counts(records: list[dict], tokenizer) -> list[dict]:
    """Add auditable local-teacher token counts to pre-materialized rows."""
    enriched = []
    for record in records:
        record = dict(record)
        input_tokens = len(tokenizer(record["student_prompt"])["input_ids"])
        output_tokens = len(tokenizer(record["teacher_response"])["input_ids"])
        record["teacher_input_tokens"] = input_tokens
        record["teacher_output_tokens"] = output_tokens
        record["teacher_total_tokens"] = input_tokens + output_tokens
        enriched.append(record)
    return enriched


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_solutions(model, tokenizer, questions, batch_size, max_new_tokens):
    """Greedy batched generation; returns one solution string per question."""
    import torch

    solutions = []
    for start in range(0, len(questions), batch_size):
        prompts = [build_solve_prompt(q) for q in questions[start : start + batch_size]]
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False
            )
        prompt_len = encoded["input_ids"].shape[1]
        for row in generated:
            completion = tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
            solutions.append(completion.strip())
        print(f"  generated {min(start + batch_size, len(questions))}/{len(questions)}")
    return solutions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--data-dir", type=Path, default=Path("out/g4/gsm8k"))
    parser.add_argument("--out-dir", type=Path, default=Path("out/g5/gsm8k"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--skip-test-em", action="store_true")
    parser.add_argument("--prepare-test-only", action="store_true")
    parser.add_argument("--prepare-data-only", action="store_true")
    args = parser.parse_args()

    test_rows = read_jsonl(args.data_dir / "test.jsonl")
    if args.prepare_test_only:
        write_jsonl(
            args.out_dir / "test_teacher_sft.jsonl",
            build_gold_test_records(test_rows),
        )
        print(f"Wrote {len(test_rows)} normalized test records")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    write_jsonl(
        args.out_dir / "test_teacher_sft.jsonl",
        add_teacher_token_counts(build_gold_test_records(test_rows), tokenizer),
    )
    if args.prepare_data_only:
        for split in ("train", "validation"):
            path = args.out_dir / f"{split}_teacher_sft.jsonl"
            write_jsonl(path, add_teacher_token_counts(read_jsonl(path), tokenizer))
        print("Added teacher token counts to train/validation/test records")
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=getattr(torch, args.dtype),
        device_map="auto",
    )
    model.eval()

    meta = {
        "teacher_model": args.teacher_model,
        "decode": {"greedy": True, "max_new_tokens": args.max_new_tokens},
    }

    for split, do_filter in (("train", True), ("validation", False)):
        rows = read_jsonl(args.data_dir / f"{split}.jsonl")
        print(f"Generating {split} solutions for {len(rows)} problems ...")
        solutions = generate_solutions(
            model, tokenizer, [r["question"] for r in rows], args.batch_size, args.max_new_tokens
        )
        records = [
            {
                "student_prompt": build_solve_prompt(row["question"]),
                "teacher_response": solution,
                "gold_answer": row["answer"],
                "answer_number": row["answer_number"],
            }
            for row, solution in zip(rows, solutions)
        ]
        if do_filter:
            records, survival = filter_against_gold(records)
            meta["train_survival_rate"] = survival
            print(f"Rejection filter kept {len(records)} ({survival:.1%})")
        write_jsonl(
            args.out_dir / f"{split}_teacher_sft.jsonl",
            add_teacher_token_counts(records, tokenizer),
        )

    if not args.skip_test_em:
        rows = test_rows
        print(f"Scoring teacher zero-shot EM on {len(rows)} test problems ...")
        solutions = generate_solutions(
            model, tokenizer, [r["question"] for r in rows], args.batch_size, args.max_new_tokens
        )
        correct = sum(
            extract_final_number(sol) == row["answer_number"]
            for row, sol in zip(rows, solutions)
        )
        meta["teacher_test_exact_match"] = correct / len(rows)
        (args.out_dir / "teacher_test_em.json").write_text(
            json.dumps({"exact_match": correct / len(rows), "n": len(rows)}, indent=2)
        )

    (args.out_dir / "generation_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
