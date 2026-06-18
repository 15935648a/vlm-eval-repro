"""Early-exit fix: read the fall answer BEFORE the late-layer 'No' override.

L17 additive steering recovered contradictions but blew up false-positives — it just injects
"Yes" regardless of content, because at L27 both real falls and contradictions already read
"Yes"; they differ only in the late override (L29-34). So the specific fix is not to add Yes
but to decode the answer at an earlier layer, before the override fires.

Specificity is free: a true negative never computed "Yes" at the exit layer, so it still reads
"No" — unlike additive steering, which forces Yes on everything. This evaluates logit-lens
decoding at each candidate exit layer over the same three disjoint sets.
"""
import argparse
import csv

from .analyze import logit_lens
from . import config
from .model_runner import GemmaVideoRunner
from .prompts import SUBJECT_PROBES
from .steer import _load_sets
from .video_utils import sample_frames

PERSON_FALL = SUBJECT_PROBES["person_fall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="*", default=list(range(24, 33)))
    ap.add_argument("--cap", type=int, default=12)
    ap.add_argument("--contradictions", default=str(config.RESULTS_DIR / "contradictions.csv"))
    ap.add_argument("--find-csv", default=str(config.RESULTS_DIR / "find_video.csv"))
    args = ap.parse_args()

    false_no, true_yes, negatives = _load_sets(args.contradictions, args.find_csv, args.cap)
    print(f"eval: {len(false_no)} contradictions, {len(true_yes)} true-pos, {len(negatives)} negatives")
    runner = GemmaVideoRunner()

    # one forward per clip; read p_yes at every layer from the lens.
    def lens_by_layer(clips):
        rows = []
        for v in clips:
            lens = logit_lens(runner, sample_frames(v, config.NUM_FRAMES), PERSON_FALL)
            rows.append({r["layer"]: r["p_yes"] for r in lens})
        return rows

    fn, ty, neg = lens_by_layer(false_no), lens_by_layer(true_yes), lens_by_layer(negatives)

    def frac(rows, layer, want_yes):
        if not rows:
            return 0.0
        return sum(((r.get(layer, 0.0) > 0.5) == want_yes) for r in rows) / len(rows)

    print(f"\n{'exitL':>6}{'recovery':>12}{'retention':>12}{'false-pos':>12}")
    print("-" * 42)
    for L in args.layers:
        rec = frac(fn, L, True)        # contradictions: want Yes
        ret = frac(ty, L, True)        # true falls: stay Yes
        fp = 1 - frac(neg, L, False)   # negatives wrongly Yes
        print(f"{L:>6}{rec:>12.0%}{ret:>12.0%}{fp:>12.0%}")
    print("\nwant: recovery high, retention high, false-pos low (vs final-layer baseline = 0/100/low).")


if __name__ == "__main__":
    main()
