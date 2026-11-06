from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # WriteRHand
DATA_DIR = ROOT / "data"                     # WriteRHand/data
RESULTS_DIR = ROOT / "results"               # WriteRHand/results
PROCESSED_PATH = DATA_DIR / "processed"
SEED = 42