"""Central config. Override any path with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- model ---
MODEL_ID = os.environ.get("VLM_MODEL_ID", "THChou1220/gemma-4-e2b-kinetics54K_FT")

# --- data ---
DATASET_REPO = os.environ.get("VLM_DATASET_REPO", "gnitoahc/vlm-eval-videos")
DATA_DIR = Path(os.environ.get("VLM_DATA_DIR", ROOT / "data"))
VIDEO_DIR = Path(os.environ.get("VLM_VIDEO_DIR", DATA_DIR / "videos"))
RESULTS_DIR = Path(os.environ.get("VLM_RESULTS_DIR", ROOT / "results"))

# --- inference ---
NUM_FRAMES = int(os.environ.get("VLM_NUM_FRAMES", "8"))   # "use the model's example": 8 frames
MAX_NEW_TOKENS = int(os.environ.get("VLM_MAX_NEW_TOKENS", "64"))
DTYPE = os.environ.get("VLM_DTYPE", "bfloat16")            # matches the model's bf16 weights
SEED = int(os.environ.get("VLM_SEED", "0"))

for d in (DATA_DIR, VIDEO_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
