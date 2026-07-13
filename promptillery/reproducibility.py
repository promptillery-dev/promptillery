"""Best-effort reproducibility metadata for paper artifacts."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any


PACKAGE_VERSION_NAMES = (
    "promptillery",
    "datasets",
    "transformers",
    "torch",
    "accelerate",
    "peft",
    "litellm",
    "numpy",
    "pandas",
    "scikit-learn",
    "evaluate",
    "pydantic",
    "typer",
    "PyYAML",
)

SAFE_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "TOKENIZERS_PARALLELISM",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTHONHASHSEED",
)


def dataset_load_kwargs(config: Any) -> dict[str, Any]:
    """Return dataset kwargs with an optional explicit Hugging Face revision."""
    kwargs = dict(getattr(config, "dataset_kwargs", {}) or {})
    dataset_revision = getattr(config, "dataset_revision", None)
    if dataset_revision is None:
        return kwargs

    existing_revision = kwargs.get("revision")
    if existing_revision is not None and str(existing_revision) != dataset_revision:
        raise ValueError(
            "dataset_revision conflicts with dataset_kwargs.revision: "
            f"{dataset_revision!r} != {existing_revision!r}"
        )
    kwargs["revision"] = dataset_revision
    return kwargs


def model_revision_kwargs(config: Any) -> dict[str, str]:
    """Return optional Hugging Face model kwargs for the configured student."""
    student_revision = getattr(config, "student_revision", None)
    return {"revision": student_revision} if student_revision else {}


def tokenizer_revision_kwargs(config: Any) -> dict[str, str]:
    """Return optional Hugging Face tokenizer kwargs for the configured student."""
    tokenizer_revision = getattr(config, "tokenizer_revision", None)
    if tokenizer_revision:
        return {"revision": tokenizer_revision}
    return model_revision_kwargs(config)


def config_sha256(config: Any) -> str:
    """Return the stable hash used to identify a resolved experiment config."""
    return sha256(
        json.dumps(config.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path | str) -> str:
    """Return the SHA-256 digest of a file."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_reproducibility_manifest(
    config: Any | None = None,
    artifact_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Build a reviewer-facing, best-effort reproducibility snapshot."""
    package_root = Path(__file__).resolve().parents[1]
    workspace_root = package_root.parent
    artifact_path = Path(artifact_dir) if artifact_dir is not None else None
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_control": {
            "package": _git_state(package_root),
            "workspace": _git_state(workspace_root),
        },
        "runtime": {
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "implementation": platform.python_implementation(),
            },
            "platform": {
                "platform": platform.platform(),
                "system": platform.system(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "packages": _package_versions(),
            "lockfiles": _lockfile_hashes(package_root),
        },
        "hardware": {
            "cpu_count": os.cpu_count(),
            "torch": _torch_state(),
        },
        "environment": {
            name: os.environ[name] for name in SAFE_ENV_VARS if name in os.environ
        },
    }
    if artifact_path is not None:
        manifest["artifact_dir"] = str(artifact_path)
    if config is not None:
        manifest["dataset_source"] = _dataset_source(config)
        manifest["models"] = _models(config)
        manifest["config_provenance"] = _config_provenance(config, artifact_path)
    return manifest


def _dataset_source(config: Any) -> dict[str, Any]:
    return {
        "dataset": getattr(config, "dataset", None),
        "subset": getattr(config, "dataset_subset", None),
        "revision": getattr(config, "dataset_revision", None),
        "load_kwargs": dataset_load_kwargs(config),
    }


def _models(config: Any) -> dict[str, Any]:
    return {
        "student": {
            "name": getattr(config, "student", None),
            "revision": getattr(config, "student_revision", None),
            "tokenizer_revision": getattr(config, "tokenizer_revision", None),
        },
        "teacher": {
            "name": getattr(config, "teacher", None),
            "revision": getattr(config, "teacher_revision", None),
            "policy_teacher_tiers": getattr(config, "policy_teacher_tiers", {}) or {},
        },
    }


def _config_provenance(
    config: Any,
    artifact_dir: Path | None,
) -> dict[str, Any]:
    provenance = {
        "source_path": getattr(config, "_source_path", None),
        "source_sha256": getattr(config, "_source_sha256", None),
        "resolved_sha256": config_sha256(config),
    }
    if artifact_dir is not None:
        run_copy = artifact_dir / "experiment_config.yaml"
        if run_copy.exists():
            provenance["run_copy_path"] = str(run_copy)
            provenance["run_copy_sha256"] = file_sha256(run_copy)
    return provenance


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in PACKAGE_VERSION_NAMES:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _lockfile_hashes(package_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename in ("pyproject.toml", "uv.lock"):
        path = package_root / filename
        if path.exists():
            hashes[filename] = file_sha256(path)
    return hashes


def _git_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path), "error": "path_missing"}

    def run_git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout.strip()

    try:
        root = run_git("rev-parse", "--show-toplevel")
        head_sha = run_git("rev-parse", "HEAD")
        branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
        status = run_git("status", "--short")
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }

    status_entries = status.splitlines()
    return {
        "available": True,
        "path": str(path),
        "repo_root": root,
        "head_sha": head_sha,
        "branch": branch,
        "is_dirty": bool(status_entries),
        "status_entries": status_entries[:200],
    }


def _torch_state() -> dict[str, Any]:
    state: dict[str, Any] = {"available": False}
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local environment
        state["error"] = f"{type(exc).__name__}: {exc}"
        return state

    state["available"] = True
    state["version"] = getattr(torch, "__version__", None)
    state["cuda_available"] = bool(torch.cuda.is_available())
    state["cuda_version"] = getattr(torch.version, "cuda", None)
    state["cuda_device_count"] = int(torch.cuda.device_count())
    state["cuda_devices"] = [
        torch.cuda.get_device_name(index)
        for index in range(state["cuda_device_count"])
    ]
    cudnn = getattr(torch.backends, "cudnn", None)
    state["cudnn_version"] = cudnn.version() if cudnn else None
    mps = getattr(torch.backends, "mps", None)
    state["mps_available"] = bool(mps and mps.is_available())
    return state
