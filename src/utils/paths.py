from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repository root
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = (
    ROOT / "artifacts"
)  # trained models / exports (git-ignored, published to Hugging Face)
PROCESSED_PATH = DATA_DIR / "processed"
SEED = 42
