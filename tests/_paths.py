"""Where the paper configs live: examples/ here, examples/paper/ in the public export."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def paper_examples_dir() -> Path:
    for candidate in (REPO_ROOT / "examples" / "paper", REPO_ROOT / "examples"):
        if (candidate / "G3_agnews_ettin_encoder.yaml").exists():
            return candidate
    raise FileNotFoundError("paper configs not found under examples/ or examples/paper/")


PAPER_EXAMPLES = paper_examples_dir()
