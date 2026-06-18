"""Train/test leakage check: do the eval clips appear in the Kinetics-54K training set?

Both the eval benchmark (gnitoahc/vlm-eval-videos) and the training set
(bear7011/gemma-4-e4b-kinetics_54K) are sourced from Kinetics/YouTube, so the same video
can be in both. We compare at the YouTube-video-id level.

CPU-only and numpy-free (stdlib + huggingface_hub only), but run it in the same container
as everything else (lab policy = always Docker):
    bash scripts/00_docker_run.sh bash scripts/05_check_leakage.sh
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from huggingface_hub import snapshot_download

from . import config

TRAIN_REPO = "bear7011/gemma-4-e4b-kinetics_54K"

# Kinetics clip id = "<ytid>_<start:06d>_<end:06d>"; strip the trailing two numeric groups.
_SEGMENT = re.compile(r"_\d{6}_\d{6}$")


def _ytid(raw: str) -> str:
    raw = str(raw).strip()
    raw = raw.rsplit("/", 1)[-1]          # drop any "class/clip" path prefix
    return _SEGMENT.sub("", raw)          # drop _start_end if present


def eval_ytids(eval_root: Path):
    """Return {ytid: [(category, filename), ...]} from every metadata.csv."""
    out = defaultdict(list)
    csvs = list(eval_root.rglob("metadata.csv"))
    if not csvs:
        raise FileNotFoundError(f"no metadata.csv under {eval_root}; download the dataset first")
    for csvp in csvs:
        with open(csvp) as f:
            for row in csv.DictReader(f):
                vid = row.get("video_id") or row.get("youtube_id") or ""
                if not vid and row.get("filename"):     # fallback: parse from filename
                    stem = Path(row["filename"]).stem
                    vid = stem.split("_")[-1]
                if vid:
                    out[_ytid(vid)].append((csvp.parent.name, row.get("filename", "")))
    return out


def train_ytids(train_root: Path):
    """Collect ytids from the training records (json/jsonl preferred; parquet fallback)."""
    ids = set()
    data_files = [p for ext in ("*.json", "*.jsonl") for p in train_root.rglob(ext)]
    if data_files:
        for fp in data_files:
            with open(fp) as f:
                if fp.suffix == ".jsonl":
                    records = (json.loads(line) for line in f if line.strip())
                else:
                    loaded = json.load(f)
                    records = loaded if isinstance(loaded, list) else [loaded]
                for rec in records:
                    vid = rec.get("video_id")
                    if vid:
                        ids.add(_ytid(vid))
        return ids
    # fallback: parquet (needs pyarrow); fail loudly with guidance if unavailable.
    parquets = list(train_root.rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"no json/jsonl/parquet data found under {train_root}")
    import pyarrow.parquet as pq
    for fp in parquets:
        col = pq.read_table(fp, columns=["video_id"])["video_id"].to_pylist()
        ids.update(_ytid(v) for v in col if v)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "leakage.csv"))
    args = ap.parse_args()

    eval_root = config.DATA_DIR / "vlm-eval-videos"
    print(f"loading eval ids from {eval_root}")
    ev = eval_ytids(eval_root)

    print(f"downloading training metadata from {TRAIN_REPO} ...")
    train_root = Path(snapshot_download(
        repo_id=TRAIN_REPO, repo_type="dataset",
        local_dir=str(config.DATA_DIR / "kinetics_54K_train"),
        allow_patterns=["*.json", "*.jsonl", "*.parquet"],
    ))
    tr = train_ytids(train_root)
    print(f"training set: {len(tr)} unique youtube ids")

    overlap = sorted(set(ev) & set(tr))
    per_cat = defaultdict(lambda: [0, 0])  # category -> [clips, leaked]
    rows = []
    for ytid, occ in ev.items():
        leaked = ytid in tr
        for cat, fname in occ:
            per_cat[cat][0] += 1
            per_cat[cat][1] += int(leaked)
            if leaked:
                rows.append({"category": cat, "video_id": ytid, "filename": fname})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "video_id", "filename"])
        w.writeheader()
        w.writerows(rows)

    total_clips = sum(len(v) for v in ev.values())
    leaked_clips = len(rows)
    print(f"\n=== LEAKAGE: {leaked_clips}/{total_clips} eval clips share a youtube id with training "
          f"({len(overlap)} unique videos) ===")
    print(f"{'category':<22}{'clips':>8}{'leaked':>8}")
    for cat in sorted(per_cat):
        c, l = per_cat[cat]
        print(f"{cat:<22}{c:>8}{l:>8}")
    print(f"\nleaked clips written to {args.out}")


if __name__ == "__main__":
    main()
