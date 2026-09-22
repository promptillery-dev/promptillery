"""Tests for the student-vs-teacher fidelity metric (issue #3)."""

import json
from pathlib import Path

import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from promptillery.config import ExperimentConfig
from promptillery.fidelity import (
    agreement_by_index,
    load_teacher_eval_labels,
    teacher_agreement,
)
from promptillery.trainers.factory import TrainerFactory

FIXTURE_TOKENIZER = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "fixtures"
    / "tiny_wordpiece_tokenizer"
)

BANKING_LABELS = ["card_arrival", "card_linking"]


def _write_teacher_jsonl(path, rows):
    """Write teacher-materialized SFT records (one JSON object per line)."""
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _tiny_classifier_trainer(tmp_path, teacher_labels_path=None):
    """A tiny from-config BERT sequence classifier with train+test splits."""
    from transformers import (
        AutoTokenizer,
        BertConfig,
        BertForSequenceClassification,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(FIXTURE_TOKENIZER))
    bert_config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=len(BANKING_LABELS),
        pad_token_id=tokenizer.pad_token_id,
    )
    model_dir = tmp_path / "tiny_bert"
    BertForSequenceClassification(bert_config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    trainer_config = {}
    if teacher_labels_path is not None:
        trainer_config["fidelity"] = {
            "teacher_labels_path": str(teacher_labels_path),
            "split": "test",
        }
    config = ExperimentConfig(
        name="fidelity-clf-smoke",
        student=str(model_dir),
        student_type="transformers",
        num_labels=len(BANKING_LABELS),
        text_field="text",
        label_field="label",
        metrics=[],
        auto_modify_name=False,
        num_train_epochs=1,
        batch_size=2,
        trainer_config=trainer_config,
    )
    features = Features(
        {"text": Value("string"), "label": ClassLabel(names=BANKING_LABELS)}
    )
    rows = {
        "text": ["my card came today", "linking my card", "where is my card"],
        "label": [0, 1, 0],
    }
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(rows, features=features),
            "test": Dataset.from_dict(rows, features=features),
        }
    )
    return TrainerFactory.create_trainer(config, dataset, tmp_path)


def _tiny_sft_trainer(tmp_path):
    """A tiny from-config GPT-2 SFT (slm) trainer with train+test splits."""
    trainer_config = {
        "model_from_config": True,
        "model_type": "gpt2",
        "model_config": {
            "n_embd": 32,
            "n_head": 4,
            "n_layer": 2,
            "n_positions": 128,
            "n_ctx": 128,
        },
        "use_lora": False,
        "prompt_field": "student_prompt",
        "response_field": "teacher_response",
        "gold_answer_field": "gold_answer",
        "answer_extraction": "text",
        "generation_max_new_tokens": 4,
        "add_eos_token": False,
        "report_to": [],
    }
    config = ExperimentConfig(
        name="fidelity-slm-smoke",
        student=str(FIXTURE_TOKENIZER),
        student_type="slm",
        auto_modify_name=False,
        num_train_epochs=1,
        batch_size=2,
        trainer_config=trainer_config,
    )
    rows = {
        "student_prompt": ["Classify: nice", "Classify: awful", "Classify: okay"],
        "teacher_response": ["card_arrival", "card_linking", "card_arrival"],
        "gold_answer": ["card_arrival", "card_linking", "card_arrival"],
    }
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(rows),
            "test": Dataset.from_dict(rows),
        }
    )
    return TrainerFactory.create_trainer(config, dataset, tmp_path)


def _tiny_fasttext_trainer(tmp_path, teacher_labels_path=None):
    """A tiny FastText classifier with train+test splits (reference profile)."""
    trainer_config = {}
    if teacher_labels_path is not None:
        trainer_config["fidelity"] = {
            "teacher_labels_path": str(teacher_labels_path),
            "split": "test",
        }
    config = ExperimentConfig(
        name="fidelity-fasttext-smoke",
        student="fasttext",
        student_type="fasttext",
        num_labels=len(BANKING_LABELS),
        text_field="text",
        label_field="label",
        metrics=[],
        auto_modify_name=False,
        num_train_epochs=5,
        learning_rate=0.5,
        batch_size=2,
        trainer_config=trainer_config,
    )
    features = Features(
        {"text": Value("string"), "label": ClassLabel(names=BANKING_LABELS)}
    )
    rows = {
        "text": ["my card came today", "linking my card", "where is my card"],
        "label": [0, 1, 0],
    }
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(rows, features=features),
            "test": Dataset.from_dict(rows, features=features),
        }
    )
    return TrainerFactory.create_trainer(config, dataset, tmp_path)


def _bare_hf_trainer(trainer, tmp_path):
    """A HuggingFace Trainer wrapping the (untrained) student for predict()."""
    from transformers import Trainer, TrainingArguments

    return Trainer(
        model=trainer.model,
        args=TrainingArguments(
            output_dir=str(tmp_path / "hf"), report_to=[]
        ),
        tokenizer=trainer.tokenizer,
    )


def test_teacher_agreement_counts_normalized_top1_matches():
    # Row 0 matches exactly; row 1 matches after label normalization
    # (case + spacing); row 2 is a different label -> disagreement.
    student = ["card_arrival", "Card Arrival", "wrong_label"]
    teacher = ["card_arrival", "card_arrival", "card_arrival"]

    assert teacher_agreement(student, teacher) == pytest.approx(2 / 3)


def test_teacher_agreement_raises_on_empty_input():
    # With no teacher reference labels the metric is undefined. Raise rather
    # than silently reporting 0.0 -- a 0.0 fidelity is indistinguishable from a
    # student that agreed with the teacher on nothing, which would launder a
    # missing/empty teacher-label file into a plausible-looking table cell.
    with pytest.raises(ValueError):
        teacher_agreement([], [])


def test_teacher_agreement_raises_on_length_mismatch():
    # Student and teacher labels must align 1:1 by row; unequal lengths are a
    # data/programming error, not a partial agreement. The legitimate
    # "student emitted fewer predictions than teacher rows" case is handled one
    # level up by agreement_by_index, which pads missing rows with None so the
    # lists handed here are always equal length (see the sparse-join test).
    with pytest.raises(ValueError):
        teacher_agreement(["card_arrival"], ["card_arrival", "card_arrival"])


def test_agreement_by_index_joins_on_shared_row_index():
    # Teacher labeled rows 0 and 2 (row 1 was rejected -> absent). The student
    # agrees on row 0, disagrees on row 2, and never predicted a row the teacher
    # skipped -> 1 of the teacher's 2 rows == 0.5. Row 1's stray student
    # prediction is ignored because the teacher defines the evaluated rows.
    student_by_index = {0: "card_arrival", 1: "noise", 2: "card_linking"}
    teacher_labels = {0: "card_arrival", 2: "card_arrival"}

    assert agreement_by_index(student_by_index, teacher_labels) == pytest.approx(0.5)


def test_load_teacher_eval_labels_maps_split_index_to_teacher_response(tmp_path):
    # A teacher-materialized test file keys each teacher prediction by its
    # split row index (source_index); the label is the raw teacher_response.
    path = _write_teacher_jsonl(
        tmp_path / "test.jsonl",
        [
            {"source_index": 0, "teacher_response": "card_arrival",
             "materialization_mode": "teacher"},
            {"source_index": 1, "teacher_response": "card_linking",
             "materialization_mode": "teacher"},
        ],
    )

    labels = load_teacher_eval_labels(path)

    assert labels == {0: "card_arrival", 1: "card_linking"}


def test_load_teacher_eval_labels_keys_on_original_row_index(tmp_path):
    # source_index is the position within the (possibly subset/reordered)
    # materialization selection; source_original_index is the true row index in
    # the split. Student predictions are keyed by original row index, so the
    # join must prefer source_original_index when present -- otherwise a teacher
    # file produced with --max-samples under a non-prefix (shuffled/stratified)
    # selection strategy silently misaligns with the student's predictions.
    path = _write_teacher_jsonl(
        tmp_path / "test.jsonl",
        [
            {"source_index": 0, "source_original_index": 5,
             "teacher_response": "card_arrival", "materialization_mode": "teacher"},
            {"source_index": 1, "source_original_index": 2,
             "teacher_response": "card_linking", "materialization_mode": "teacher"},
        ],
    )

    labels = load_teacher_eval_labels(path)

    assert labels == {5: "card_arrival", 2: "card_linking"}


def test_evaluate_reports_teacher_fidelity_for_encoder(tmp_path):
    # Build the student, read off its own per-row predicted label names, then
    # hand the teacher those exact labels. A faithful student agrees with the
    # teacher on every row -> teacher_fidelity == 1.0. This exercises the full
    # wiring (class-id -> label name -> normalized index join) independently
    # of what the untrained model happens to predict.
    teacher_path = tmp_path / "teacher_test.jsonl"
    trainer = _tiny_classifier_trainer(tmp_path, teacher_labels_path=teacher_path)
    hf = _bare_hf_trainer(trainer, tmp_path)

    res = hf.predict(trainer.prepare_data("test"))
    preds = res.predictions.argmax(axis=1)
    names = trainer.dataset["test"].features["label"].names
    _write_teacher_jsonl(
        teacher_path,
        [
            {"source_index": i, "teacher_response": names[int(p)],
             "materialization_mode": "teacher"}
            for i, p in enumerate(preds)
        ],
    )

    scores = trainer.evaluate(hf, split="test")

    assert scores["teacher_fidelity"] == pytest.approx(1.0)


def test_evaluate_reports_teacher_fidelity_for_fasttext(tmp_path):
    # FastText predicts integer class ids per row just like the encoder, so it
    # reuses the same id->name->index fidelity join. Seed the teacher with the
    # student's own predicted label names -> perfect self-agreement == 1.0. This
    # fills the reference-profile row of tab:deployment (issue #3 story 28).
    fasttext = pytest.importorskip("fasttext")  # noqa: F841
    teacher_path = tmp_path / "teacher_test.jsonl"
    trainer = _tiny_fasttext_trainer(tmp_path, teacher_labels_path=teacher_path)
    model = trainer.train()

    names = trainer.dataset["test"].features["label"].names
    detailed = trainer.get_detailed_predictions(model, split="test")
    _write_teacher_jsonl(
        teacher_path,
        [
            {"source_index": i, "source_original_index": i,
             "teacher_response": names[int(p)], "materialization_mode": "teacher"}
            for i, p in enumerate(detailed.predicted_labels)
        ],
    )

    scores = trainer.evaluate(model, split="test")

    assert scores["teacher_fidelity"] == pytest.approx(1.0)


def test_fasttext_evaluate_omits_teacher_fidelity_when_unconfigured(tmp_path):
    # With no teacher label file configured, FastText evaluate() adds no
    # teacher_fidelity key -- reference runs without fidelity are unaffected.
    pytest.importorskip("fasttext")
    trainer = _tiny_fasttext_trainer(tmp_path)
    model = trainer.train()

    scores = trainer.evaluate(model, split="test")

    assert "teacher_fidelity" not in scores


def test_fasttext_handles_newlines_in_text(tmp_path):
    # Real Banking77 messages contain newlines, and fasttext.predict raises on
    # any input with a '\n'. The trainer must sanitize text before predicting
    # (and before writing training lines, since fasttext is line-oriented).
    # Guards a crash seen when smoke-running the real G2_banking77_fasttext config.
    pytest.importorskip("fasttext")
    config = ExperimentConfig(
        name="fasttext-newline",
        student="fasttext",
        student_type="fasttext",
        num_labels=len(BANKING_LABELS),
        text_field="text",
        label_field="label",
        metrics=["accuracy"],
        auto_modify_name=False,
        num_train_epochs=5,
        learning_rate=0.5,
        batch_size=2,
        trainer_config={},
    )
    features = Features(
        {"text": Value("string"), "label": ClassLabel(names=BANKING_LABELS)}
    )
    rows = {
        "text": ["my card\narrived today", "linking\nmy card", "where is my card"],
        "label": [0, 1, 0],
    }
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(rows, features=features),
            "test": Dataset.from_dict(rows, features=features),
        }
    )
    trainer = TrainerFactory.create_trainer(config, dataset, tmp_path)
    model = trainer.train()

    # Must not raise "predict processes one line at a time".
    scores = trainer.evaluate(model, split="test")
    assert "accuracy" in scores
    # The augmentation path predicts too -- also must tolerate newlines.
    trainer.predict_for_augmentation(model, split="train")


def _write_run_dir(tmp_path, metrics):
    """A minimal completed run directory summarize_run() can read."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(json.dumps({"status": "completed"}))
    (run_dir / "experiment_config.yaml").write_text("name: fidelity-run\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    (run_dir / "token_usage.json").write_text(json.dumps({"grand_total": {}}))
    return run_dir


def test_summary_surfaces_teacher_fidelity_alongside_accuracy(tmp_path):
    # The deployment table needs fidelity next to accuracy in one analysis
    # row. Selecting accuracy as the primary metric must still surface the
    # per-run teacher fidelity as its own dedicated, always-present column.
    from promptillery.analyze import summarize_run

    run_dir = _write_run_dir(
        tmp_path,
        {
            "1": {"accuracy": 0.5, "teacher_fidelity": 0.4},
            "10": {"accuracy": 0.8, "teacher_fidelity": 0.7},
            "heldout_test": {
                "accuracy": 0.82,
                "teacher_fidelity": 0.71,
                "_heldout_split": "test",
            },
        },
    )

    row = summarize_run(run_dir, metric="accuracy")

    # The chosen primary metric is unaffected...
    assert row["heldout_metric_name"] == "accuracy"
    assert row["heldout_metric"] == pytest.approx(0.82)
    # ...and fidelity rides along as its own column, distinct from accuracy.
    assert row["heldout_teacher_fidelity"] == pytest.approx(0.71)
    assert row["final_teacher_fidelity"] == pytest.approx(0.7)


def test_summary_csv_includes_teacher_fidelity_column(tmp_path):
    # The fidelity columns must be part of the CSV schema so a summarized row
    # writes without a DictWriter "field not in fieldnames" error.
    import csv

    from promptillery.analyze import summarize_run, write_summary_csv

    run_dir = _write_run_dir(
        tmp_path,
        {
            "1": {"accuracy": 0.8, "teacher_fidelity": 0.7},
            "heldout_test": {
                "accuracy": 0.82,
                "teacher_fidelity": 0.71,
                "_heldout_split": "test",
            },
        },
    )
    row = summarize_run(run_dir, metric="accuracy")

    csv_path = tmp_path / "summary.csv"
    write_summary_csv([row], csv_path)

    written = next(csv.DictReader(csv_path.open()))
    assert written["heldout_teacher_fidelity"] == "0.71"


def test_evaluate_omits_teacher_fidelity_when_unconfigured(tmp_path):
    # With no teacher label file configured, evaluate() behaves exactly as
    # before: it never adds a teacher_fidelity key. Existing runs are untouched.
    trainer = _tiny_classifier_trainer(tmp_path)
    hf = _bare_hf_trainer(trainer, tmp_path)

    scores = trainer.evaluate(hf, split="test")

    assert "teacher_fidelity" not in scores


def test_evaluate_reports_teacher_fidelity_for_decoder(tmp_path):
    # The SFT (decoder) student joins its per-row generated predictions to the
    # teacher labels by row index. Seed the teacher with the student's own
    # (deterministic, greedy) predictions -> perfect self-agreement == 1.0.
    trainer = _tiny_sft_trainer(tmp_path)
    hf = trainer.train()

    first = trainer.evaluate(hf, split="test")
    assert "teacher_fidelity" not in first

    records = [
        json.loads(line)
        for line in (tmp_path / "eval_predictions_test.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    teacher_path = tmp_path / "teacher_test.jsonl"
    _write_teacher_jsonl(
        teacher_path,
        [
            {"source_index": r["index"], "teacher_response": r["normalized_prediction"],
             "materialization_mode": "teacher"}
            for r in records
        ],
    )

    trainer.trainer_config["fidelity"] = {
        "teacher_labels_path": str(teacher_path),
        "split": "test",
    }
    scores = trainer.evaluate(hf, split="test")

    assert scores["teacher_fidelity"] == pytest.approx(1.0)
