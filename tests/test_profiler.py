"""Tests for the student inference profiler."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict

from promptillery.config import ExperimentConfig
from promptillery.profiler import (
    latency_stats,
    load_profile,
    profile_model,
    profile_student,
    save_profile,
    teacher_cost_per_1k_calls,
)
from promptillery.trainers.factory import TrainerFactory

FIXTURE_TOKENIZER = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "fixtures"
    / "tiny_wordpiece_tokenizer"
)


def _tiny_slm_trainer(tmp_path):
    """A tiny from-config GPT-2 SLM trainer (CPU, no training needed)."""
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
        name="profiler-slm-smoke",
        student=str(FIXTURE_TOKENIZER),
        student_type="slm",
        auto_modify_name=False,
        num_train_epochs=1,
        batch_size=2,
        trainer_config=trainer_config,
    )
    dataset = DatasetDict(
        {
            "validation": Dataset.from_dict(
                {
                    "student_prompt": ["Classify: nice", "Classify: awful"],
                    "teacher_response": ["positive", "negative"],
                    "gold_answer": ["positive", "negative"],
                }
            )
        }
    )
    return TrainerFactory.create_trainer(config, dataset, tmp_path)


def test_latency_stats_computes_percentiles_mean_and_throughput():
    # Five sequential single-request calls with known per-call durations (s).
    per_call_seconds = [0.010, 0.020, 0.030, 0.040, 0.050]

    stats = latency_stats(per_call_seconds)

    # p50 is the interpolated median: index 0.5*(5-1)=2 -> 0.030s -> 30ms.
    assert math.isclose(stats["p50_latency_ms"], 30.0, rel_tol=1e-9)
    # p95: index 0.95*4=3.8 -> 0.040*0.2 + 0.050*0.8 = 0.048s -> 48ms.
    assert math.isclose(stats["p95_latency_ms"], 48.0, rel_tol=1e-9)
    # mean: 0.030s -> 30ms.
    assert math.isclose(stats["mean_latency_ms"], 30.0, rel_tol=1e-9)
    # throughput: 5 calls / 0.150s total = 33.333... calls/sec.
    assert math.isclose(
        stats["throughput_calls_per_sec"], 5 / 0.150, rel_tol=1e-9
    )


def _write_token_usage(dir_path, estimated_cost):
    payload = {
        "cycles_completed": 1,
        "grand_total": {
            "estimated_cost": estimated_cost,
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        },
        "per_cycle": [],
    }
    path = dir_path / "token_usage.json"
    path.write_text(json.dumps(payload))
    return path


def test_teacher_cost_per_1k_reuses_token_usage_estimated_cost(tmp_path):
    # Teacher spent $4.50 over 3000 calls -> $1.50 per 1K calls.
    usage_path = _write_token_usage(tmp_path, estimated_cost=4.50)

    cost = teacher_cost_per_1k_calls(usage_path, n_teacher_calls=3000)

    assert math.isclose(cost, 1.50, rel_tol=1e-9)


def test_teacher_cost_per_1k_is_none_when_unpriced_or_missing(tmp_path):
    # Unpriced teacher model: litellm-derived cost is null -> null column.
    unpriced = _write_token_usage(tmp_path, estimated_cost=None)
    assert teacher_cost_per_1k_calls(unpriced, n_teacher_calls=3000) is None

    # No artifact on disk at all -> null, not an error.
    missing = tmp_path / "does_not_exist.json"
    assert teacher_cost_per_1k_calls(missing, n_teacher_calls=3000) is None


def _tiny_classifier_trainer(tmp_path):
    """A tiny from-config BERT sequence classifier saved to disk (CPU)."""
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
        num_labels=2,
        pad_token_id=tokenizer.pad_token_id,
    )
    model_dir = tmp_path / "tiny_bert"
    BertForSequenceClassification(bert_config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    config = ExperimentConfig(
        name="profiler-clf-smoke",
        student=str(model_dir),
        student_type="transformers",
        num_labels=2,
        text_field="text",
        label_field="label",
        metrics=[],
        auto_modify_name=False,
        num_train_epochs=1,
        batch_size=2,
    )
    dataset = DatasetDict(
        {
            "validation": Dataset.from_dict(
                {"text": ["nice product", "awful service"], "label": [1, 0]}
            )
        }
    )
    return TrainerFactory.create_trainer(config, dataset, tmp_path)


def test_profile_student_measures_classifier_latency_on_transformers(tmp_path):
    # Same public entry point, a non-SFT student type -> classifier path.
    trainer = _tiny_classifier_trainer(tmp_path)

    result = profile_student(
        trainer, split="validation", device="cpu", iterations=6, warmup=2
    )

    student = result["student"]
    assert result["measurement"]["n_iterations"] == 6
    assert student["p95_latency_ms"] >= student["p50_latency_ms"] > 0
    assert student["throughput_calls_per_sec"] > 0


class _StubFastText:
    """Stands in for the external fasttext model (no torch, no .to()/.eval())."""

    def predict(self, text, k=1):
        return ((f"__label__{len(text) % 2}",), [1.0])


def test_profile_student_profiles_fasttext_without_torch():
    # fasttext is latency-profiled through the same public entry point, but it
    # is non-torch: profile_student must not call .to()/.eval() on it.
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(
            student_type="fasttext",
            student="fasttext-tiny",
            text_field="text",
            dataset="ag_news",
        ),
        dataset={
            "validation": Dataset.from_dict(
                {"text": ["good", "bad", "meh"], "label": [1, 0, 1]}
            )
        },
    )

    result = profile_student(
        trainer,
        split="validation",
        device="cpu",
        iterations=6,
        warmup=2,
        model=_StubFastText(),
        tokenizer=None,
    )

    student = result["student"]
    assert result["measurement"]["n_iterations"] == 6
    assert student["p95_latency_ms"] >= student["p50_latency_ms"] >= 0
    assert student["throughput_calls_per_sec"] > 0
    assert result["student_type"] == "fasttext"
    assert result["hardware"]["device"] == "cpu"


def test_profile_student_measures_decoder_latency_on_slm(tmp_path):
    trainer = _tiny_slm_trainer(tmp_path)

    result = profile_student(
        trainer, split="validation", device="cpu", iterations=6, warmup=2
    )

    student = result["student"]
    # Exactly `iterations` calls are measured; warmup is excluded.
    assert result["measurement"]["n_iterations"] == 6
    # Real generation happened, so latencies are positive and ordered.
    assert student["p95_latency_ms"] >= student["p50_latency_ms"] > 0
    assert student["throughput_calls_per_sec"] > 0


def test_profile_student_raises_clear_error_on_empty_split(tmp_path):
    # An empty split must raise a clear, actionable error -- not the confusing
    # ZeroDivisionError from `index % len(prompts)` in the measurement loop.
    trainer = _tiny_slm_trainer(tmp_path)
    trainer.dataset["validation"] = Dataset.from_dict(
        {"student_prompt": [], "teacher_response": [], "gold_answer": []}
    )

    with pytest.raises(ValueError, match="empty"):
        profile_student(
            trainer, split="validation", device="cpu", iterations=4, warmup=1
        )


def test_inputs_for_split_materializes_at_most_limit_rows():
    # Only warmup+iterations prompts are ever replayed, so a huge test split
    # must not force the whole column into memory -- _inputs_for_split caps it.
    from promptillery.profiler import _inputs_for_split

    trainer = SimpleNamespace(
        cfg=SimpleNamespace(student_type="fasttext", text_field="text"),
        dataset={
            "validation": Dataset.from_dict(
                {"text": [f"row {i}" for i in range(100)]}
            )
        },
    )

    inputs = _inputs_for_split(trainer, "fasttext", "validation", limit=7)

    assert inputs == [f"row {i}" for i in range(7)]


def test_resolve_device_auto_selects_mps_when_cuda_absent(monkeypatch):
    # On Apple hardware (no CUDA, MPS present) the default should be MPS, not
    # CPU -- the code path already works, it just wasn't being auto-selected.
    import torch

    from promptillery.profiler import _resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert _resolve_device(None) == "mps"


def test_resolve_device_prefers_cuda_then_cpu_fallback(monkeypatch):
    import torch

    from promptillery.profiler import _resolve_device

    # CUDA wins when present, even if MPS is also reported available.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _resolve_device(None) == "cuda"

    # Neither accelerator -> CPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert _resolve_device(None) == "cpu"


def test_resolve_device_rejects_mps_when_unavailable(monkeypatch):
    import torch

    from promptillery.profiler import _resolve_device

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="mps"):
        _resolve_device("mps")


def test_synchronize_device_drains_gpu_queue_but_noops_on_cpu(monkeypatch):
    # GPU ops are async: the timed region must end with an explicit device sync
    # so latency isn't undercounted. CPU is synchronous -> no-op.
    import torch

    from promptillery.profiler import _synchronize_device

    calls = []
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda *a, **k: calls.append("cuda")
    )
    monkeypatch.setattr(
        torch.mps, "synchronize", lambda *a, **k: calls.append("mps")
    )

    _synchronize_device("cpu")
    assert calls == []

    _synchronize_device("cuda")
    _synchronize_device("cuda:0")
    _synchronize_device("mps")
    assert calls == ["cuda", "cuda", "mps"]


def test_measure_latencies_syncs_the_device_inside_the_timed_region(
    tmp_path, monkeypatch
):
    # The sync must be wired into measurement (once per timed call, on the
    # resolved device), not left implicit -- this is the regression it guards.
    import promptillery.profiler as profiler

    trainer = _tiny_classifier_trainer(tmp_path)
    calls = []
    monkeypatch.setattr(
        profiler, "_synchronize_device", lambda device: calls.append(device)
    )

    profile_student(
        trainer, split="validation", device="cpu", iterations=6, warmup=2
    )

    assert calls.count("cpu") >= 6
    assert set(calls) == {"cpu"}


def _sample_profile():
    """A saved profile as it would look for a Qwen3-4B run on the A100 box."""
    return {
        "model": "Qwen/Qwen3-4B",
        "student_type": "slm",
        "dataset": "banking77",
        "hardware": {
            "device": "cuda:0",
            "gpu_name": "NVIDIA A100-SXM4-80GB",
            "cpu": "x86_64",
            "torch_version": "2.7.1+cu126",
        },
        "measurement": {"n_iterations": 50, "warmup": 5, "latency_batch_size": 1},
        "student": {
            "p50_latency_ms": 12.0,
            "p95_latency_ms": 20.0,
            "mean_latency_ms": 13.0,
            "throughput_calls_per_sec": 80.0,
        },
    }


def test_load_profile_raises_on_model_mismatch(tmp_path):
    # Downstream runs reuse a saved 4B profile; loading the wrong model's must fail
    # loudly rather than silently paste its latency into the paper.
    from promptillery.profiler import ProfileStampError

    path = save_profile(_sample_profile(), tmp_path)

    with pytest.raises(ProfileStampError, match="model"):
        load_profile(path, expect_model="Qwen/Qwen3-0.6B")


def test_load_profile_raises_on_hardware_mismatch(tmp_path):
    # Right model, wrong GPU: the A100 profile must not be reused on an H100.
    from promptillery.profiler import ProfileStampError

    path = save_profile(_sample_profile(), tmp_path)

    with pytest.raises(ProfileStampError, match="gpu_name"):
        load_profile(
            path,
            expect_model="Qwen/Qwen3-4B",
            expect_hardware={"gpu_name": "NVIDIA H100 80GB HBM3"},
        )


def test_load_profile_returns_profile_when_stamp_matches(tmp_path):
    profile = _sample_profile()
    path = save_profile(profile, tmp_path)

    loaded = load_profile(
        path,
        expect_model="Qwen/Qwen3-4B",
        expect_hardware={"gpu_name": "NVIDIA A100-SXM4-80GB", "device": "cuda:0"},
    )

    assert loaded == profile


def test_save_profile_stamped_name_encodes_model_and_hardware(tmp_path):
    # A fixed profile.json collides when two models / two GPUs write the same
    # directory; the stamped name keeps distinct (model, hardware) profiles apart.
    profile = _sample_profile()

    path = save_profile(profile, tmp_path, stamped_name=True)

    assert path.parent == tmp_path
    assert path.suffix == ".json"
    assert "qwen3-4b" in path.name.lower()
    assert "a100" in path.name.lower()
    assert load_profile(path) == profile
    # Default remains the fixed profile.json (backward-compatible).
    assert save_profile(profile, tmp_path).name == "profile.json"


def test_profile_result_stamps_hardware_and_round_trips(tmp_path):
    trainer = _tiny_slm_trainer(tmp_path)

    result = profile_student(
        trainer, split="validation", device="cpu", iterations=4, warmup=1
    )

    # Hardware identity is stamped so a number is reproducible / reusable.
    assert result["hardware"]["device"] == "cpu"
    assert result["hardware"]["torch_version"]
    assert result["model"] == str(FIXTURE_TOKENIZER)
    assert result["student_type"] == "slm"

    # The artifact is a cacheable profile.json that loads back unchanged.
    path = save_profile(result, tmp_path)
    assert path.name == "profile.json"
    assert load_profile(path) == result


def _saved_tiny_slm_run(tmp_path):
    """A run dir with a saved tiny SLM checkpoint + a token_usage.json."""
    config = ExperimentConfig.from_yaml("examples/causal_lm_sft_tiny.yaml")
    run_dir = tmp_path / "run"
    seed_trainer = TrainerFactory.create_trainer(
        config,
        DatasetDict(
            {
                "validation": Dataset.from_dict(
                    {
                        "student_prompt": ["x"],
                        "teacher_response": ["y"],
                        "gold_answer": ["y"],
                    }
                )
            }
        ),
        run_dir,
    )
    # load_model expects the checkpoint at <run>/model.
    model_dir = run_dir / "model"
    seed_trainer.model.save_pretrained(model_dir)
    seed_trainer.tokenizer.save_pretrained(model_dir)
    # Teacher token usage produced by the run's TokenTracker.
    (run_dir / "token_usage.json").write_text(
        json.dumps(
            {
                "grand_total": {
                    "estimated_cost": 3.0,
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                }
            }
        )
    )
    return config, run_dir


def test_profile_model_writes_profile_json_with_teacher_cost(tmp_path):
    config, run_dir = _saved_tiny_slm_run(tmp_path)

    profile = profile_model(
        config,
        run_dir,
        split="validation",
        device="cpu",
        iterations=4,
        warmup=1,
        n_teacher_calls=1000,
    )

    # Student latency was measured on the loaded checkpoint.
    assert profile["student"]["p50_latency_ms"] > 0
    # Teacher cost reused from token tracking: 3.0 / 1000 * 1000 = 3.0 per 1K.
    assert math.isclose(profile["teacher"]["cost_per_1k_calls"], 3.0, rel_tol=1e-9)
    # Cacheable artifact written beside the run.
    assert (run_dir / "profile.json").exists()


def test_profile_cli_command_writes_profile(tmp_path):
    from typer.testing import CliRunner

    from promptillery.cli import app

    _, run_dir = _saved_tiny_slm_run(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "profile",
            "examples/causal_lm_sft_tiny.yaml",
            "--model-path",
            str(run_dir),
            "--split",
            "validation",
            "--device",
            "cpu",
            "--iterations",
            "4",
            "--warmup",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (run_dir / "profile.json").exists()
