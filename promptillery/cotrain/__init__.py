"""Co-training acquisition (design 2026-05-01).

Two heterogeneous students generate seed-anchored variants targeting each
other's faults; strong teacher arbitrates on disagreement. Imports are lazy
to avoid loading torch/transformers when only config validation runs.
"""

__all__ = ["CoTrainEngine"]


def __getattr__(name):
    if name == "CoTrainEngine":
        from .engine import CoTrainEngine
        return CoTrainEngine
    raise AttributeError(name)
