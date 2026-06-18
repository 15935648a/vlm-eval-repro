"""Reproduce the contradiction on a single clip.

Runs both prompts under greedy decoding, repeated N times to prove determinism, and also
reports the first-token Yes/No probabilities for the binary prompt + every wording variant
(this is the first data point separating H1 format-prior from H2 semantic-threshold).
"""
import argparse
import json

from . import config
from .model_runner import GemmaVideoRunner
from .prompts import (
    DESCRIBE_PROMPT,
    YESNO_CONTROL,
    YESNO_PROMPT,
    YESNO_VARIANTS,
)
from .video_utils import sample_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--repeats", type=int, default=3, help="greedy should give identical output each time")
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "reproduce.json"))
    args = ap.parse_args()

    frames = sample_frames(args.video, config.NUM_FRAMES)
    runner = GemmaVideoRunner()

    record = {"video": args.video, "num_frames": config.NUM_FRAMES, "model": config.MODEL_ID}

    # 1. determinism check on the two canonical prompts
    record["yesno_prompt"] = YESNO_PROMPT
    record["describe_prompt"] = DESCRIBE_PROMPT
    record["yesno_outputs"] = [runner.generate(frames, YESNO_PROMPT) for _ in range(args.repeats)]
    record["describe_outputs"] = [runner.generate(frames, DESCRIBE_PROMPT) for _ in range(args.repeats)]

    # 2. first-token Yes/No probabilities: canonical + control + variants
    record["yesno_logits"] = {
        "canonical": runner.first_token_yes_no(frames, YESNO_PROMPT),
        "control_is_there_a_person": runner.first_token_yes_no(frames, YESNO_CONTROL),
        "variants": {v: runner.first_token_yes_no(frames, v) for v in YESNO_VARIANTS},
    }

    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\n=== {args.video} ===")
    print(f"\n[yes/no prompt] {YESNO_PROMPT}")
    for o in record["yesno_outputs"]:
        print(f"   -> {o!r}")
    print(f"\n[describe prompt] {DESCRIBE_PROMPT}")
    for o in record["describe_outputs"]:
        print(f"   -> {o!r}")

    c = record["yesno_logits"]["canonical"]
    print(f"\n[first-token probs] canonical yes/no:  P(yes)={c['p_yes']:.3f}  P(no)={c['p_no']:.3f}  top={c['top_token']!r}")
    print("[variants]")
    for v, r in record["yesno_logits"]["variants"].items():
        print(f"   P(yes)={r['p_yes']:.3f} P(no)={r['p_no']:.3f}  <- {v}")

    contradicted = any("fall" in o.lower() or "fell" in o.lower() for o in record["describe_outputs"]) and all(
        r["argmax_is_no"] for r in [c]
    )
    print(f"\ncontradiction reproduced: {contradicted}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
