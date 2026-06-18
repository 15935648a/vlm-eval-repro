"""Is the 'compute Yes mid-stack, overwrite to No late' pattern general, or just one clip?

Runs the logit lens on 'Did the person fall?' for the top-N contradiction clips and reports,
per clip: the peak mid-layer P(yes), the layer it peaks at, the layer where Yes flips to No,
and the final P(no). A consistent peak~1.0 -> late flip across clips means the late-layer
override is a systematic mechanism, not a single-clip artifact.
"""
import argparse
import csv
import json

from . import config
from .analyze import logit_lens
from .model_runner import GemmaVideoRunner
from .prompts import SUBJECT_PROBES
from .video_utils import sample_frames

PERSON_FALL = SUBJECT_PROBES["person_fall"]


def summarize(lens):
    """lens: list of {layer, p_yes, p_no}. Return peak-yes + flip-to-no layer (late stack)."""
    peak = max(lens, key=lambda r: r["p_yes"])
    flip_layer = None
    for r in lens:                       # first layer after the peak where No takes over
        if r["layer"] > peak["layer"] and r["p_no"] > 0.5:
            flip_layer = r["layer"]
            break
    return {
        "peak_yes_layer": peak["layer"],
        "peak_p_yes": round(peak["p_yes"], 3),
        "flip_layer": flip_layer,
        "final_p_no": round(lens[-1]["p_no"], 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contradictions", default=str(config.RESULTS_DIR / "contradictions.csv"))
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "flip_layers.json"))
    args = ap.parse_args()

    with open(args.contradictions) as f:
        clips = [r for r in csv.DictReader(f)
                 if r.get("yesno_says_no") in ("True", "true", "1")][: args.top]

    runner = GemmaVideoRunner()
    rows = []
    print(f"{'clip':<28}{'peakYesL':>10}{'peakP(yes)':>12}{'flipL':>8}{'finalP(no)':>12}")
    print("-" * 70)
    for c in clips:
        frames = sample_frames(c["video"], config.NUM_FRAMES)
        s = summarize(logit_lens(runner, frames, PERSON_FALL))
        s["video"] = c["video"]
        rows.append(s)
        name = c["video"].split("/")[-1][:27]
        print(f"{name:<28}{s['peak_yes_layer']:>10}{s['peak_p_yes']:>12.3f}"
              f"{str(s['flip_layer']):>8}{s['final_p_no']:>12.3f}")

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
