"""Net-accuracy eval using the only real labels available: the benchmark CATEGORY.

The reference `answer` is a per-category template ("The person climbs a ladder"), not a
per-clip fall label, so it can't adjudicate the contradiction. The category can, weakly:
  - fall categories (face_planting, falling_off_bike, falling_off_chair) -> a fall is curated (y=1)
  - general_mp4 (diverse everyday actions)                               -> mostly no fall   (y=0)
  - climbing_ladder                                                      -> AMBIGUOUS (the action
        is climbing; some clips also contain a fall) -> reported separately, NOT scored.

Compares the final-layer decoder (the model's real answer) against early-exit decoding at
--layer, on 'Did the person fall?'. A general fix must raise recall on real falls WITHOUT
blowing up false-positives on the no-fall category. This is still weak ground truth — a real
claim needs human labels — but it is the best the dataset itself supports.
"""
import argparse

from .analyze import logit_lens
from . import config
from .model_runner import GemmaVideoRunner
from .prompts import SUBJECT_PROBES
from .video_utils import iter_videos, sample_frames

PERSON_FALL = SUBJECT_PROBES["person_fall"]
FALL_CATS = ["face_planting", "falling_off_bike", "falling_off_chair"]
NEG_CATS = ["general_mp4"]
AMBIG_CATS = ["climbing_ladder"]


def _clips_for(cats, cap):
    root = config.DATA_DIR / "vlm-eval-videos"
    out = []
    for cat in cats:
        vids = []
        for d in root.rglob(cat):
            if d.is_dir():
                vids += iter_videos(d)
        out.append((cat, sorted(set(vids))[:cap]))
    return out


def _yes_rates(runner, clips, layer):
    """Return (final_yes_rate, early_yes_rate) over clips, decoding 'Did the person fall?'."""
    fin = ear = 0
    for v in clips:
        lens = logit_lens(runner, sample_frames(v, config.NUM_FRAMES), PERSON_FALL)
        by_layer = {r["layer"]: r["p_yes"] for r in lens}
        fin += int(by_layer[max(by_layer)] > 0.5)   # final layer = the model's real answer
        ear += int(by_layer.get(layer, 0.0) > 0.5)   # early-exit
    n = max(1, len(clips))
    return fin / n, ear / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=28, help="early-exit layer")
    ap.add_argument("--cap", type=int, default=15, help="max clips per category")
    args = ap.parse_args()

    runner = GemmaVideoRunner()
    print(f"early-exit layer = {args.layer}   (final = the model's deployed answer)\n")
    print(f"{'group':<22}{'category':<20}{'n':>4}{'final Yes%':>12}{'early Yes%':>12}")
    print("-" * 70)

    def report(title, cats, want):  # want: "Yes" desired (positives) or "No" desired
        for cat, clips in _clips_for(cats, args.cap):
            if not clips:
                continue
            fin, ear = _yes_rates(runner, clips, args.layer)
            print(f"{title:<22}{cat:<20}{len(clips):>4}{fin:>12.0%}{ear:>12.0%}")

    report("POSITIVE (fall)", FALL_CATS, "Yes")     # Yes% = recall; higher is better
    report("NEGATIVE (no fall)", NEG_CATS, "No")     # Yes% = false-positive rate; lower is better
    report("AMBIGUOUS", AMBIG_CATS, "?")             # Yes% inflation here is the open question
    print("\nread: POSITIVE Yes% = recall (want high), NEGATIVE Yes% = false-positive (want low).")
    print("if early-exit raises NEGATIVE Yes% a lot, it is not a safe general fix.")


if __name__ == "__main__":
    main()
