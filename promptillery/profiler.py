"""Student inference profiler: p50/p95 latency + throughput (issue #2).

Measures how fast a trained student runs on a fixed device and reuses the
teacher-side cost model from ``token_tracker``. The student side reports
latency and throughput only (no derived dollars); teacher cost stays a
per-1K-calls figure. Results serialise to a cacheable profile that is *stamped
with* the (model, hardware) it was measured on; ``load_profile`` can assert
that stamp, and ``save_profile(..., stamped_name=True)`` encodes it into the
filename, so downstream experiments (#5) reuse the right numbers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from .analyze import _percentile
from .trainers.factory import SFT_STUDENT_TYPES, TrainerFactory


def latency_stats(
    per_call_seconds: List[float], *, calls_per_measurement: int = 1
) -> Dict[str, float]:
    """Summarise a list of per-call latencies (seconds) into ms + throughput.

    Returns p50/p95/mean latency in milliseconds and the achieved throughput
    in calls per second (total calls / total wall time). Percentiles reuse the
    analysis module's linearly-interpolated ``_percentile`` for consistent
    semantics with the rest of the pipeline. ``calls_per_measurement`` is the
    number of requests each timed measurement actually served (the profiling
    batch size); it only scales throughput -- latency stays per-measurement,
    since that is the wall-clock cost of one timed call regardless of how many
    requests it served.
    """
    if not per_call_seconds:
        raise ValueError("per_call_seconds must be non-empty")

    ordered = sorted(per_call_seconds)
    total_seconds = sum(per_call_seconds)
    return {
        "p50_latency_ms": _percentile(ordered, 0.5) * 1000.0,
        "p95_latency_ms": _percentile(ordered, 0.95) * 1000.0,
        "mean_latency_ms": (total_seconds / len(per_call_seconds)) * 1000.0,
        "throughput_calls_per_sec": (
            len(per_call_seconds) * calls_per_measurement / total_seconds
        ),
    }


def teacher_cost_per_1k_calls(
    token_usage: Union[str, Path, Mapping],
    n_teacher_calls: int,
) -> Optional[float]:
    """Teacher cost per 1000 calls, reused from the run's token tracking.

    ``token_usage`` is either a path to a persisted ``token_usage.json`` or an
    already-loaded mapping in the same shape (``grand_total.estimated_cost``).
    Cost is the litellm-derived ``estimated_cost`` produced by ``TokenTracker``;
    this function does not re-derive prices. Returns ``None`` when the artifact
    is missing, the cost is unpriced (``None``), or the call count is not
    positive, so callers get a null column rather than a raised error.
    """
    if isinstance(token_usage, Mapping):
        summary = token_usage
    else:
        path = Path(token_usage)
        if not path.exists():
            return None
        summary = json.loads(path.read_text())

    estimated_cost = summary.get("grand_total", {}).get("estimated_cost")
    if estimated_cost is None or not n_teacher_calls or n_teacher_calls <= 0:
        return None
    return estimated_cost / n_teacher_calls * 1000.0


def _hardware_identity(device: str) -> Dict[str, Optional[str]]:
    """Record the hardware a measurement ran on, for reproducibility/reuse."""
    import platform

    import torch

    gpu_name: Optional[str] = None
    if device.startswith("cuda") and torch.cuda.is_available():
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        gpu_name = torch.cuda.get_device_name(index)
    return {
        "device": device,
        "gpu_name": gpu_name,
        "cpu": platform.processor() or platform.machine(),
        "torch_version": torch.__version__,
    }


def _mps_available() -> bool:
    """Whether Apple's Metal (MPS) backend is present and usable."""
    import torch

    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def _resolve_device(device: Optional[str]) -> str:
    """Resolve the target device, preferring an accelerator when none is asked.

    Auto-selection order is CUDA -> MPS -> CPU, so local dev profiling on Apple
    hardware uses the GPU instead of silently falling back to CPU.
    """
    import torch

    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device '{device}' requested but no CUDA GPU is available"
        )
    if device == "mps" and not _mps_available():
        raise RuntimeError("device 'mps' requested but MPS is not available")
    return device


def _synchronize_device(device: str) -> None:
    """Block until the device has finished the timed work.

    CUDA/MPS kernels are dispatched asynchronously, so a timer could stop before
    the GPU actually finishes -- undercounting latency. An explicit sync at the
    end of the timed region makes the measurement correct and, unlike relying on
    a ``.decode()``/``.item()`` side effect, can't be silently dropped by a
    refactor. CPU execution is synchronous, so this is a no-op there.
    """
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _inputs_for_split(
    trainer: Any, student_type: str, split: str, limit: Optional[int] = None
) -> List[str]:
    """The student's own task inputs for the given split.

    Decoder/SLM students read the SFT prompt field; classifier students read
    the dataset's text field -- the same columns their inference paths use.
    ``limit`` caps how many rows are materialised: profiling only ever replays
    the first ``warmup + iterations`` prompts, so a large test split need not be
    pulled into memory in full.
    """
    dataset = trainer.dataset
    if split not in dataset:
        raise ValueError(f"split '{split}' not in dataset")
    ds = dataset[split]
    if len(ds) == 0:
        raise ValueError(f"split '{split}' is empty; nothing to profile")
    if student_type in SFT_STUDENT_TYPES:
        field = getattr(trainer, "prompt_field", "student_prompt")
    else:
        field = trainer.cfg.text_field
    if field not in ds.column_names:
        raise ValueError(f"input field '{field}' not in split '{split}'")
    # Slice before reading the column so only `limit` rows are materialised.
    values = ds[:limit][field] if limit is not None else ds[field]
    return [str(value) for value in values]


def _decoder_inference_op(
    trainer: Any, device: str, model: Any, tokenizer: Any
) -> Callable[[Union[str, List[str]]], None]:
    """A generate() call mirroring the trainer's eval path.

    ``run_one`` accepts either a single prompt (single-request, batch size 1)
    or a ``list[str]`` of prompts (batched). Batched decoder generation needs
    left padding so every sequence's last real token lines up at the same
    position for ``generate()``; that is set here, scoped to this op's own
    tokenizer, not touched anywhere else.
    """
    import torch

    max_new_tokens = int(trainer.trainer_config.get("generation_max_new_tokens", 32))
    max_length = getattr(trainer, "max_seq_length", None)
    # Left padding is required for batched generation; this deliberately
    # mutates the trainer's tokenizer for the rest of the profiling call.
    tokenizer.padding_side = "left"

    def run_one(prompt_text: Union[str, List[str]]) -> None:
        texts = (
            prompt_text if isinstance(prompt_text, (list, tuple)) else [prompt_text]
        )
        formatted = [trainer._format_generation_prompt(str(t)) for t in texts]
        encoded = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
        input_width = int(encoded["input_ids"].shape[1])
        completion_ids = generated.sequences[:, input_width:]
        # Decoding the completion is the real inference output; the timed region
        # is drained explicitly via _synchronize_device, not by this call.
        tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    return run_one


def _classifier_inference_op(
    trainer: Any, device: str, model: Any, tokenizer: Any
) -> Callable[[Union[str, List[str]]], None]:
    """A classifier forward pass (logits -> predicted labels), 1 or N requests."""
    import torch

    max_length = getattr(trainer, "max_seq_length", None)

    def run_one(texts) -> None:
        batch = [
            str(t) for t in (texts if isinstance(texts, (list, tuple)) else [texts])
        ]
        encoded = tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        # argmax -> predicted labels is the real inference output; the timed
        # region is drained explicitly via _synchronize_device, not by this call.
        logits.argmax(dim=-1).tolist()

    return run_one


def _fasttext_inference_op(model: Any) -> Callable[[Union[str, List[str]]], None]:
    """A fasttext prediction (non-torch, CPU-only), 1 or N requests.

    fasttext is latency-profiled but never cost-profiled -- student cost is
    latency/throughput only, and its speed is itself a useful deployment
    anchor. It has no device/tokenizer; inference is ``model.predict(text)``.
    """

    def run_one(texts) -> None:
        batch = texts if isinstance(texts, (list, tuple)) else [texts]
        for t in batch:
            model.predict(str(t))

    return run_one


def _build_inference_op(
    trainer: Any, student_type: str, device: str, model: Any, tokenizer: Any
) -> Callable[[Union[str, List[str]]], None]:
    """Dispatch on student_type to the matching single/batched inference call."""
    if student_type in SFT_STUDENT_TYPES:
        return _decoder_inference_op(trainer, device, model, tokenizer)
    if student_type == "transformers":
        return _classifier_inference_op(trainer, device, model, tokenizer)
    if student_type == "fasttext":
        return _fasttext_inference_op(model)
    raise NotImplementedError(
        f"profiling student_type '{student_type}' is not supported"
    )


def _measure_latencies(
    run_one: Callable[[Union[str, List[str]]], None],
    prompts: Sequence[str],
    *,
    iterations: int,
    warmup: int,
    device: str,
    batch_size: int = 1,
) -> List[float]:
    """Warm up (untimed), then time `iterations` calls of `batch_size` requests.

    ``batch_size == 1`` (the default) times single-request calls, unchanged
    from before. ``batch_size > 1`` instead passes each timed call a
    ``list[str]`` of ``batch_size`` prompts, cycling through ``prompts``.
    Each call is followed by an explicit device sync so async GPU work is
    included in the timed region (see ``_synchronize_device``); the warmup
    syncs too so its kernels don't spill into the first measured call.
    """

    def _arg(index: int):
        if batch_size == 1:
            return prompts[index % len(prompts)]
        return [
            prompts[(index * batch_size + j) % len(prompts)]
            for j in range(batch_size)
        ]

    for index in range(warmup):
        run_one(_arg(index))
        _synchronize_device(device)
    per_call: List[float] = []
    for index in range(iterations):
        arg = _arg(index)
        start = perf_counter()
        run_one(arg)
        _synchronize_device(device)
        per_call.append(perf_counter() - start)
    return per_call


def profile_student(
    trainer: Any,
    *,
    split: str = "validation",
    device: Optional[str] = None,
    iterations: int = 50,
    warmup: int = 5,
    model: Any = None,
    tokenizer: Any = None,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Profile a trained student's inference on a fixed device.

    Moves the student to ``device`` (torch students only) and replays its own
    task inputs through the architecture-appropriate inference call: generate()
    for decoder/SLM students, a forward pass for encoder classifiers, and
    predict() for non-torch fasttext students. Reports p50/p95/mean latency plus
    throughput; the student side is latency/throughput only -- no derived
    dollars.

    ``batch_size`` (default 1) is the number of requests served per timed
    call -- single-stream by default, matching every existing profile.json.
    Passing ``batch_size > 1`` times batched calls instead and stamps
    ``measurement.latency_batch_size`` accordingly so downstream readers know
    the numbers are batched throughput, not single-stream.

    ``model``/``tokenizer`` default to the trainer's own, but a loaded
    checkpoint can be passed explicitly (the trainer's ``load_model`` returns
    a wrapper without mutating ``trainer.model``).
    """
    student_type = trainer.cfg.student_type
    # fasttext is non-torch/CPU-only; torch students honour the requested device.
    target = "cpu" if student_type == "fasttext" else _resolve_device(device)

    model = model if model is not None else trainer.model
    tokenizer = (
        tokenizer if tokenizer is not None else getattr(trainer, "tokenizer", None)
    )
    # Only torch models need explicit placement + eval mode.
    if hasattr(model, "to"):
        model.to(target)
    if hasattr(model, "eval"):
        model.eval()

    inputs = _inputs_for_split(
        trainer, student_type, split, limit=warmup + iterations
    )
    run_one = _build_inference_op(trainer, student_type, target, model, tokenizer)
    per_call = _measure_latencies(
        run_one, inputs, iterations=iterations, warmup=warmup, device=target,
        batch_size=batch_size,
    )

    return {
        "model": trainer.cfg.student,
        "student_type": student_type,
        "dataset": getattr(trainer.cfg, "dataset", None),
        "hardware": _hardware_identity(target),
        "measurement": {
            "n_iterations": iterations,
            "warmup": warmup,
            "latency_batch_size": batch_size,
        },
        "student": latency_stats(per_call, calls_per_measurement=batch_size),
    }


def _slug(text: str) -> str:
    """Filesystem-safe lowercase slug (``Qwen/Qwen3-4B`` -> ``qwen-qwen3-4b``)."""
    slug = re.sub(r"[^0-9a-z]+", "-", text.lower()).strip("-")
    return slug or "unknown"


def profile_filename(result: Mapping) -> str:
    """A ``(model, hardware)``-encoded filename so distinct profiles don't collide.

    Two models -- or the same model on two GPUs -- writing beside their
    checkpoints would otherwise clobber a shared ``profile.json``. A batched
    measurement (``measurement.latency_batch_size > 1``) also gets a
    ``-bsN`` suffix, so it can never collide with -- or be mistaken for -- the
    single-stream profile the paper's tables read.
    """
    model_slug = _slug(str(result.get("model", "model")))
    hardware = result.get("hardware") or {}
    hw_slug = _slug(str(hardware.get("gpu_name") or hardware.get("device") or "cpu"))
    batch_size = (result.get("measurement") or {}).get("latency_batch_size", 1)
    bs_suffix = f"-bs{batch_size}" if batch_size and batch_size > 1 else ""
    return f"profile-{model_slug}-{hw_slug}{bs_suffix}.json"


def save_profile(
    result: Mapping,
    dir_or_path: Union[str, Path],
    *,
    stamped_name: bool = False,
) -> Path:
    """Write a profile to ``profile.json`` (cacheable, stamped with model+hardware).

    Accepts a directory (writes ``profile.json`` inside it) or an explicit
    file path. Pass ``stamped_name=True`` with a directory to write a
    ``(model, hardware)``-encoded filename instead, so distinct profiles in one
    directory don't overwrite each other. Returns the path written.
    """
    path = Path(dir_or_path)
    if path.suffix != ".json":
        filename = profile_filename(result) if stamped_name else "profile.json"
        path = path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return path


class ProfileStampError(ValueError):
    """A loaded profile doesn't match the (model, hardware) a consumer expects.

    A profile is *stamped with* the model/hardware it was measured on, not keyed
    by it, so reuse (#5) must assert the stamp before trusting the numbers --
    otherwise a filename collision or a wrong-GPU cache silently pastes bad
    latency into ``tab:deployment``/``tab:recommender``.
    """


def _assert_stamp(
    profile: Mapping,
    expect_model: Optional[str],
    expect_hardware: Optional[Mapping[str, Any]],
) -> None:
    """Raise ``ProfileStampError`` if the profile's stamp misses expectations."""
    if expect_model is not None and profile.get("model") != expect_model:
        raise ProfileStampError(
            f"profile is for model {profile.get('model')!r}, "
            f"expected {expect_model!r}"
        )
    if expect_hardware:
        hardware = profile.get("hardware") or {}
        for key, want in expect_hardware.items():
            got = hardware.get(key)
            if got != want:
                raise ProfileStampError(
                    f"profile hardware {key}={got!r}, expected {want!r}"
                )


def load_profile(
    path: Union[str, Path],
    *,
    expect_model: Optional[str] = None,
    expect_hardware: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Load a previously-saved profile (e.g. to reuse a 4B profile, #5).

    Pass ``expect_model`` and/or ``expect_hardware`` (a subset of the stamped
    ``hardware`` block, e.g. ``{"gpu_name": ...}``) to assert the cached profile
    was measured for the right model/hardware before reusing its numbers;
    a mismatch raises ``ProfileStampError``. With no expectations the load is
    unvalidated (backward-compatible).
    """
    profile = json.loads(Path(path).read_text())
    _assert_stamp(profile, expect_model, expect_hardware)
    return profile


def _find_token_usage(model_path: Path) -> Optional[Path]:
    """Locate the run's token_usage.json next to the checkpoint."""
    for candidate in (
        model_path / "token_usage.json",
        model_path.parent / "token_usage.json",
    ):
        if candidate.exists():
            return candidate
    return None


def _teacher_block(
    config: Any, model_path: Path, n_teacher_calls: Optional[int]
) -> Dict[str, Any]:
    """Teacher cost columns, reused from the run's token tracking."""
    usage_path = _find_token_usage(model_path)
    estimated_cost = None
    if usage_path is not None:
        summary = json.loads(usage_path.read_text())
        estimated_cost = summary.get("grand_total", {}).get("estimated_cost")
    cost_per_1k = (
        teacher_cost_per_1k_calls(usage_path, n_teacher_calls)
        if usage_path is not None and n_teacher_calls
        else None
    )
    return {
        "model": getattr(config, "teacher", None),
        "estimated_cost": estimated_cost,
        "n_calls": n_teacher_calls,
        "cost_per_1k_calls": cost_per_1k,
    }


def profile_model(
    config: Any,
    model_path: Union[str, Path],
    *,
    split: str = "test",
    device: Optional[str] = None,
    iterations: int = 50,
    warmup: int = 5,
    n_teacher_calls: Optional[int] = None,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Profile a trained student checkpoint and write a cacheable profile.json.

    Loads the config's dataset, builds the trainer, loads the checkpoint at
    ``model_path``, measures student latency/throughput on ``split``, merges
    the teacher cost-per-1k from the run's token tracking, and persists the
    profile beside the checkpoint. Mirrors ``evaluate_model``'s assembly.

    ``batch_size`` (default 1, single-stream) writes ``profile.json`` as
    before. ``batch_size > 1`` writes to a separate ``profile-bs{N}.json``
    instead, so the batch-1 profile the paper reads is never overwritten by a
    batched measurement.
    """
    from datasets import load_dataset

    from .reproducibility import dataset_load_kwargs

    model_path = Path(model_path)
    dataset_subset = config.dataset_subset
    dataset_kwargs = dataset_load_kwargs(config)
    if dataset_subset:
        dataset = load_dataset(config.dataset, dataset_subset, **dataset_kwargs)
    else:
        dataset = load_dataset(config.dataset, **dataset_kwargs)

    trainer = TrainerFactory.create_trainer(config, dataset, model_path)
    loaded = trainer.load_model(model_path)
    model = getattr(loaded, "model", loaded)
    tokenizer = getattr(loaded, "processing_class", None) or getattr(
        loaded, "tokenizer", None
    )

    profile = profile_student(
        trainer,
        split=split,
        device=device,
        iterations=iterations,
        warmup=warmup,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    profile["teacher"] = _teacher_block(config, model_path, n_teacher_calls)
    save_profile(
        profile,
        model_path if batch_size == 1 else model_path / f"profile-bs{batch_size}.json",
    )
    return profile
