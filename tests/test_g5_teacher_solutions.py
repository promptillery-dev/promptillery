"""G5 pure pieces: solve-prompt contract + gold rejection filter."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "g5_generate_teacher_solutions", Path(__file__).resolve().parents[1] / "scripts" / "g5_generate_teacher_solutions.py"
)
g5_generate_teacher_solutions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g5_generate_teacher_solutions)

build_solve_prompt = g5_generate_teacher_solutions.build_solve_prompt
filter_against_gold = g5_generate_teacher_solutions.filter_against_gold
build_gold_test_records = g5_generate_teacher_solutions.build_gold_test_records
add_teacher_token_counts = g5_generate_teacher_solutions.add_teacher_token_counts


def test_solve_prompt_contract():
    prompt = build_solve_prompt("What is 2+2?")
    assert '#### <answer>' in prompt
    assert "What is 2+2?" in prompt
    assert prompt.rstrip().endswith("Solution:")


def _rec(response, gold_number):
    return {
        "student_prompt": "p",
        "teacher_response": response,
        "gold_answer": f"gold steps #### {gold_number}",
        "answer_number": str(gold_number),
    }


def test_filter_keeps_matching_final_answers():
    kept, rate = filter_against_gold(
        [_rec("steps #### 72", 72), _rec("steps #### 9", 7), _rec("no number at all!", 7)]
    )
    assert len(kept) == 1 and kept[0]["teacher_response"] == "steps #### 72"
    assert rate == pytest.approx(1 / 3)


def test_filter_empty_input():
    kept, rate = filter_against_gold([])
    assert kept == [] and rate == 0.0


def test_gold_test_records_match_teacher_sft_schema():
    records = build_gold_test_records(
        [{"question": "2+2?", "answer": "steps #### 4", "answer_number": "4"}]
    )
    assert set(records[0]) == {
        "student_prompt",
        "teacher_response",
        "gold_answer",
        "answer_number",
    }
    assert records[0]["teacher_response"] == records[0]["gold_answer"]


def test_teacher_token_counts_are_auditable():
    class Tokenizer:
        def __call__(self, text):
            return {"input_ids": text.split()}

    record = _rec("two output tokens", 4) | {"student_prompt": "one input"}
    enriched = add_teacher_token_counts([record], Tokenizer())[0]
    assert enriched["teacher_input_tokens"] == 2
    assert enriched["teacher_output_tokens"] == 3
    assert enriched["teacher_total_tokens"] == 5
