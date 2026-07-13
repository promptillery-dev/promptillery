"""Generative (GSM8K) task mode for teacher baseline eval: prompts + EM."""
import asyncio
import json

from datasets import Dataset, DatasetDict

from promptillery.baseline_eval import (
    GENERATIVE_SOLVE_INSTRUCTIONS,
    build_generative_prompt,
    evaluate_dataset_generative,
    run_baseline_evaluation,
)


def test_zero_shot_prompt_carries_solve_contract():
    prompt = build_generative_prompt("What is 2+2?")
    assert prompt.startswith(GENERATIVE_SOLVE_INSTRUCTIONS)
    assert '#### <answer>' in prompt
    assert prompt.rstrip().endswith("Solution:")
    assert "What is 2+2?" in prompt


def test_few_shot_prompt_includes_exemplars():
    prompt = build_generative_prompt(
        "What is 2+2?", [{"question": "1+1?", "answer": "Add.\n#### 2"}]
    )
    assert "1+1?" in prompt and "#### 2" in prompt
    assert prompt.index("1+1?") < prompt.index("What is 2+2?")


def _dataset():
    return DatasetDict(
        {
            "train": Dataset.from_list(
                [{"question": f"train {i}", "answer": f"#### {i}"} for i in range(6)]
            ),
            "test": Dataset.from_list(
                [
                    {"question": "easy", "answer": "so #### 4"},
                    {"question": "hard", "answer": "thus #### 7"},
                ]
            ),
        }
    )


def test_generative_em_scoring(monkeypatch):
    async def fake_acompletion(**kwargs):
        content = kwargs["messages"][0]["content"]
        # solve "easy" correctly, "hard" wrong
        answer = "#### 4" if "easy" in content else "#### 999"
        return {"choices": [{"message": {"content": f"steps...\n{answer}"}}]}

    monkeypatch.setattr("promptillery.baseline_eval.acompletion", fake_acompletion)
    result = asyncio.run(
        evaluate_dataset_generative(_dataset(), teacher="fake", mode="zero-shot", seed=13)
    )
    assert result.metrics == {"exact_match": 0.5}
    assert result.num_samples == 2


def test_generative_few_shot_pulls_from_train(monkeypatch):
    seen_prompts = []

    async def fake_acompletion(**kwargs):
        seen_prompts.append(kwargs["messages"][0]["content"])
        return {"choices": [{"message": {"content": "#### 4"}}]}

    monkeypatch.setattr("promptillery.baseline_eval.acompletion", fake_acompletion)
    asyncio.run(
        evaluate_dataset_generative(
            _dataset(), teacher="fake", mode="few-shot", num_shots=3, seed=13
        )
    )
    assert all("train" in p for p in seen_prompts)  # exemplars drawn from train split


def test_run_baseline_evaluation_generative_end_to_end(monkeypatch, tmp_path):
    """The --task generative CLI path must not crash in the shared summary tail.

    print_summary_table/save_summary_csv are written for classification's
    metrics={"accuracy", "f1"} shape; generative results only carry
    "exact_match", so this exercises the full run_baseline_evaluation call
    the CLI makes and checks it survives to a written summary CSV.
    """

    async def fake_acompletion(**kwargs):
        return {"choices": [{"message": {"content": "steps...\n#### 4"}}]}

    monkeypatch.setattr("promptillery.baseline_eval.acompletion", fake_acompletion)

    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    with open(train_path, "w") as f:
        for i in range(3):
            f.write(json.dumps({"question": f"train {i}", "answer": f"#### {i}"}) + "\n")
    with open(test_path, "w") as f:
        f.write(json.dumps({"question": "easy", "answer": "so #### 4"}) + "\n")

    output_dir = tmp_path / "out"
    results = asyncio.run(
        run_baseline_evaluation(
            dataset_name="gsm8k",
            task="generative",
            data_files=[f"train={train_path}", f"test={test_path}"],
            modes=["zero-shot"],
            output_dir=str(output_dir),
            seed=13,
        )
    )

    assert len(results) == 1
    assert results[0].metrics == {"exact_match": 1.0}
    assert (output_dir / "baseline_summary.csv").exists()
