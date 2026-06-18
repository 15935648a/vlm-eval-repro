"""Mine the benchmark for the actual contradiction:
   describe-prompt reports a fall, but the yes/no prompt answers "No".

Reuses the describe outputs already in results/find_video.csv (no re-describe), filters to
clips whose description mentions a fall, then runs ONLY the cheap yes/no first-token forward
pass on those. Flags + ranks the clips where describe says fall yet P(no) wins — those are
the clean reproductions (the "falling and then getting back up" / H2 cases especially).
"""
import argparse
import csv
import re

from tqdm import tqdm

from . import config
from .model_runner import GemmaVideoRunner
from .prompts import YESNO_PROMPT
from .video_utils import sample_frames

FALL_RE = re.compile(
    r"\b(fall|falls|fell|fallen|falling|trip|trips|tripped|slip|slips|slipped|"
    r"stumbl\w*|tumbl\w*|collaps\w*|topple\w*|knocked\s+over|loses?\s+balance)\b",
    re.I,
)
# describe outputs that explicitly say a fall was transient/recovered -> strongest H2 cases.
RECOVER_RE = re.compile(r"get(s|ting)?\s+back\s+up|gets?\s+up|stands?\s+back|recover", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--find-csv", default=str(config.RESULTS_DIR / "find_video.csv"))
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "contradictions.csv"))
    args = ap.parse_args()

    with open(args.find_csv) as f:
        rows = list(csv.DictReader(f))
    fall_rows = [r for r in rows if r.get("output") and FALL_RE.search(r["output"])
                 and not r["output"].startswith("<ERROR")]
    print(f"{len(fall_rows)}/{len(rows)} clips describe a fall; checking their yes/no answers")

    runner = GemmaVideoRunner()
    out = []
    for r in tqdm(fall_rows):
        try:
            frames = sample_frames(r["video"], config.NUM_FRAMES)
            yn = runner.first_token_yes_no(frames, YESNO_PROMPT)
        except Exception as e:
            print(f"skip {r['video']}: {e}")
            continue
        out.append({
            "video": r["video"],
            "p_yes": round(yn["p_yes"], 3),
            "p_no": round(yn["p_no"], 3),
            "yesno_says_no": yn["argmax_is_no"],
            "recovered_fall": bool(RECOVER_RE.search(r["output"])),
            "describe": r["output"],
        })

    # contradictions first, then by how strongly the model said "No".
    out.sort(key=lambda d: (d["yesno_says_no"], d["p_no"]), reverse=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "p_yes", "p_no", "yesno_says_no",
                                          "recovered_fall", "describe"])
        w.writeheader()
        w.writerows(out)

    contradictions = [d for d in out if d["yesno_says_no"]]
    print(f"\n=== {len(contradictions)} contradictions (describe=fall, yes/no=No) ===")
    for d in contradictions[:15]:
        tag = " [recovered]" if d["recovered_fall"] else ""
        print(f"  P(no)={d['p_no']:.3f}{tag}  {d['video'].split('/')[-1]}")
        print(f"      {d['describe']}")
    print(f"\nfull ranking -> {args.out}")


if __name__ == "__main__":
    main()
