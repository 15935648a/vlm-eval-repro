"""Activation-steering fix for the late-layer 'No' override on fall questions.

The mechanism (from analyze.py): the model reads out 'Yes' mid-stack, then layers ~33-35
overwrite it to 'No'; injecting a different answer-position residual at L17 flips it back and
survives the override. Here we build a FIXED steering vector (no donor needed) via diff-of-means
at one layer:
    v = mean(answer-resid | fall clips the model correctly answered Yes)
      - mean(answer-resid | contradiction clips it wrongly answered No)
and add alpha*v at that layer's answer position during the yes/no fall forward pass.

Evaluated on three disjoint sets so a real fix is distinguished from "make everything Yes":
    1. contradictions (held-out)  -> want flip to Yes      (recovery)
    2. true-positive falls        -> want stays Yes        (no harm)
    3. true negatives (no fall)   -> must stay No          (specificity)
"""
import argparse
import csv

import torch

from . import config
from .analyze import _language_model
from .find_contradiction import FALL_RE
from .model_runner import GemmaVideoRunner
from .prompts import SUBJECT_PROBES
from .video_utils import sample_frames

PERSON_FALL = SUBJECT_PROBES["person_fall"]


def _layers(runner):
    return _language_model(runner.model).layers


@torch.no_grad()
def _answer_resid(runner, frames, prompt, layer):
    store = {}

    def hook(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        store["v"] = t[0, -1].detach().float().clone()

    h = _layers(runner)[layer].register_forward_hook(hook)
    runner.model(**runner._build_inputs(frames, prompt))
    h.remove()
    return store["v"]


@torch.no_grad()
def _p_yes_steered(runner, frames, prompt, layer, vec, alpha):
    h = None
    if alpha:
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            t[0, -1] = t[0, -1] + (alpha * vec).to(t.dtype)
            return out
        h = _layers(runner)[layer].register_forward_hook(hook)
    logits = runner.model(**runner._build_inputs(frames, prompt)).logits[0, -1].float()
    if h:
        h.remove()
    return logits.softmax(-1)[runner._yes_ids].sum().item()


def _load_sets(contradictions_csv, find_csv, cap):
    """Return (false_no, true_yes, negatives) lists of video paths."""
    with open(contradictions_csv) as f:
        fall = list(csv.DictReader(f))
    false_no = [r["video"] for r in fall if r.get("yesno_says_no") in ("True", "true", "1")]
    true_yes = [r["video"] for r in fall if r.get("yesno_says_no") in ("False", "false", "0")]
    with open(find_csv) as f:
        neg = [r["video"] for r in csv.DictReader(f)
               if r.get("output") and not r["output"].startswith("<ERROR")
               and not FALL_RE.search(r["output"])]
    return false_no[:cap], true_yes[:cap], neg[:cap]


def _rate(runner, clips, layer, vec, alpha, want_yes):
    """Fraction answered as desired (p_yes>0.5 if want_yes else p_yes<=0.5)."""
    hit = 0
    for v in clips:
        frames = sample_frames(v, config.NUM_FRAMES)
        p = _p_yes_steered(runner, frames, PERSON_FALL, layer, vec, alpha)
        hit += int((p > 0.5) == want_yes)
    return hit / max(1, len(clips))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=17, help="build+inject layer (patch showed L17 works)")
    ap.add_argument("--alphas", type=float, nargs="*", default=[0, 2, 4, 6, 8])
    ap.add_argument("--cap", type=int, default=12, help="max clips per set")
    ap.add_argument("--contradictions", default=str(config.RESULTS_DIR / "contradictions.csv"))
    ap.add_argument("--find-csv", default=str(config.RESULTS_DIR / "find_video.csv"))
    args = ap.parse_args()

    false_no, true_yes, negatives = _load_sets(args.contradictions, args.find_csv, args.cap)
    runner = GemmaVideoRunner()

    # split build/eval so recovery isn't measured on the build set.
    half_fn, half_ty = len(false_no) // 2, len(true_yes) // 2
    build_fn, eval_fn = false_no[:half_fn], false_no[half_fn:]
    build_ty, eval_ty = true_yes[:half_ty], true_yes[half_ty:]
    print(f"build: {len(build_ty)} yes / {len(build_fn)} no | "
          f"eval: {len(eval_fn)} contradictions, {len(eval_ty)} true-pos, {len(negatives)} negatives")

    def mean_resid(clips):
        vs = [_answer_resid(runner, sample_frames(v, config.NUM_FRAMES), PERSON_FALL, args.layer)
              for v in clips]
        return torch.stack(vs).mean(0)

    vec = mean_resid(build_ty) - mean_resid(build_fn)
    print(f"steering vector ||v|| = {vec.norm().item():.2f} at layer {args.layer}\n")

    print(f"{'alpha':>6}{'recovery':>12}{'retention':>12}{'false-pos':>12}")
    print("-" * 42)
    for a in args.alphas:
        rec = _rate(runner, eval_fn, args.layer, vec, a, want_yes=True)    # No -> Yes
        ret = _rate(runner, eval_ty, args.layer, vec, a, want_yes=True)    # Yes stays Yes
        fp = 1 - _rate(runner, negatives, args.layer, vec, a, want_yes=False)  # No -> Yes (bad)
        print(f"{a:>6.1f}{rec:>12.0%}{ret:>12.0%}{fp:>12.0%}")
    print("\nwant: recovery high, retention high, false-pos low.")


if __name__ == "__main__":
    main()
