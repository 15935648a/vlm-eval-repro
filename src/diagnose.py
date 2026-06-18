"""Subject-referent diagnosis on the mined contradiction clips.

For the top-N contradictions (describe=fall, yes/no=No), vary ONLY the question's subject
and print a P(yes) matrix. Separates H4 (the word "person" triggers No) from H1/H3 (No
persists across all subjects, including the explicit-"man" clip).
"""
import argparse
import csv
import json

from . import config
from .model_runner import GemmaVideoRunner
from .prompts import SUBJECT_PROBES
from .video_utils import sample_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contradictions", default=str(config.RESULTS_DIR / "contradictions.csv"))
    ap.add_argument("--top", type=int, default=6, help="how many top-P(no) clips to probe")
    ap.add_argument("--videos", nargs="*", help="explicit video paths (overrides --top)")
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "diagnose.json"))
    args = ap.parse_args()

    if args.videos:
        clips = [{"video": v, "describe": ""} for v in args.videos]
    else:
        with open(args.contradictions) as f:
            rows = [r for r in csv.DictReader(f) if r.get("yesno_says_no") in ("True", "true", "1")]
        clips = rows[: args.top]

    runner = GemmaVideoRunner()
    probe_keys = list(SUBJECT_PROBES)
    record = []

    header = f"{'clip':<28}" + "".join(f"{k:>20}" for k in probe_keys)
    print(header)
    print("-" * len(header))
    for c in clips:
        frames = sample_frames(c["video"], config.NUM_FRAMES)
        cells = {k: runner.first_token_yes_no(frames, p)["p_yes"] for k, p in SUBJECT_PROBES.items()}
        record.append({"video": c["video"], "describe": c.get("describe", ""), "p_yes": cells})
        name = c["video"].split("/")[-1][:27]
        print(f"{name:<28}" + "".join(f"{cells[k]:>20.3f}" for k in probe_keys))

    with open(args.out, "w") as f:
        json.dump({"probes": SUBJECT_PROBES, "clips": record}, f, indent=2)
    print(f"\n(cells = P(yes) at first token)\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
