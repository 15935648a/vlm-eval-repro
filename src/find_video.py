"""Locate the clip that produced the skating/fall description.

Runs DESCRIBE_PROMPT (greedy) over every .mp4 in the benchmark, saves all outputs, and
flags rows whose description matches the originally-observed keywords (skating / rink /
yellow shirt / getting back up). Inspect results/find_video.csv and pick the match.
"""
import argparse
import csv

from tqdm import tqdm

from . import config
from .model_runner import GemmaVideoRunner
from .prompts import DESCRIBE_PROMPT, TARGET_KEYWORDS
from .video_utils import iter_videos, sample_frames


def score(text: str) -> int:
    t = text.lower()
    return sum(kw in t for kw in TARGET_KEYWORDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", default=str(config.DATA_DIR / "vlm-eval-videos"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips")
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "find_video.csv"))
    args = ap.parse_args()

    videos = iter_videos(args.video_dir)
    if args.limit:
        videos = videos[: args.limit]
    print(f"scanning {len(videos)} clips with DESCRIBE_PROMPT (greedy)")

    runner = GemmaVideoRunner()
    rows = []
    for vp in tqdm(videos):
        try:
            frames = sample_frames(vp, config.NUM_FRAMES)
            out = runner.generate(frames, DESCRIBE_PROMPT)
        except Exception as e:  # keep scanning even if one clip fails
            out = f"<ERROR: {e}>"
        rows.append({"video": str(vp), "match_score": score(out), "output": out})

    rows.sort(key=lambda r: r["match_score"], reverse=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "match_score", "output"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {args.out}\nTop candidates:")
    for r in rows[:5]:
        print(f"  [{r['match_score']}] {r['video']}\n      {r['output']}")


if __name__ == "__main__":
    main()
