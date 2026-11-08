from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repository root
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = (
    ROOT / "artifacts"
)  # trained models / exports (git-ignored, published to Hugging Face)
PROCESSED_PATH = DATA_DIR / "processed"
SEED = 42


def dataset_dirs(name="mnist"):
    """``(results_dir, artifacts_dir)`` of a dataset. MNIST keeps the original top-level folders."""
    if name == "mnist":
        return RESULTS_DIR, ARTIFACTS_DIR
    return RESULTS_DIR / "datasets" / name, ARTIFACTS_DIR / "datasets" / name
