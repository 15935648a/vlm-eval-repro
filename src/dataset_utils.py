"""Download the benchmark and iterate its clips + metadata."""
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import config


def download_dataset() -> Path:
    """Snapshot the dataset repo (videos + per-category metadata.csv) into DATA_DIR."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=config.DATASET_REPO,
        repo_type="dataset",
        local_dir=str(config.DATA_DIR / "vlm-eval-videos"),
    )
    return Path(path)


def load_metadata(root: Optional[Path] = None) -> pd.DataFrame:
    """Concatenate every metadata.csv under the dataset, tagging the source category dir."""
    root = Path(root or (config.DATA_DIR / "vlm-eval-videos"))
    frames: List[pd.DataFrame] = []
    for csv in root.rglob("metadata.csv"):
        df = pd.read_csv(csv)
        df["__category_dir"] = csv.parent.name
        df["__csv"] = str(csv)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no metadata.csv found under {root}; run download_dataset() first")
    return pd.concat(frames, ignore_index=True)


def resolve_video_path(row: Dict, root: Optional[Path] = None) -> Optional[Path]:
    """Best-effort map a metadata row to an actual .mp4 on disk."""
    root = Path(root or (config.DATA_DIR / "vlm-eval-videos"))
    fname = str(row.get("filename", "")).strip()
    if not fname:
        return None
    hits = list(root.rglob(fname))
    return hits[0] if hits else None
