"""Which side is actually wrong? Join the mined contradictions against the benchmark's own
reference `answer` (human label), not the model's describe output.

So far "contradiction" = describe says fall, yes/no says No, and we ASSUMED the fall is real.
But describe is the model's own text and may hallucinate. If the dataset's reference answer
says NO fall, then the yes/no "No" is correct and the late override is doing useful work —
which would mean early-exit is the wrong fix. This decides the direction of the bug.

CPU-only (stdlib), runs in the container like everything else.
"""
import argparse
import csv
from pathlib import Path

from . import config
from .find_contradiction import FALL_RE


def load_reference(eval_root: Path):
    """filename -> {label, answer} from every metadata.csv."""
    ref = {}
    for csvp in eval_root.rglob("metadata.csv"):
        with open(csvp) as f:
            for row in csv.DictReader(f):
                fn = (row.get("filename") or "").strip()
                if fn:
                    ref[fn] = {"label": row.get("label", ""), "answer": row.get("answer", "")}
    if not ref:
        raise FileNotFoundError(f"no metadata.csv under {eval_root}")
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contradictions", default=str(config.RESULTS_DIR / "contradictions.csv"))
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "check_labels.csv"))
    args = ap.parse_args()

    ref = load_reference(config.DATA_DIR / "vlm-eval-videos")
    with open(args.contradictions) as f:
        rows = [r for r in csv.DictReader(f) if r.get("yesno_says_no") in ("True", "true", "1")]

    out, ref_fall, ref_nofall, missing = [], 0, 0, 0
    for r in rows:
        fn = Path(r["video"]).name
        meta = ref.get(fn)
        if meta is None:
            missing += 1
            continue
        ans = meta["answer"]
        ref_says_fall = bool(FALL_RE.search(ans))
        ref_fall += ref_says_fall
        ref_nofall += not ref_says_fall
        out.append({
            "video": fn,
            "ref_says_fall": ref_says_fall,
            "reference_answer": ans,
            "model_describe": r.get("describe", ""),
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "ref_says_fall", "reference_answer", "model_describe"])
        w.writeheader()
        w.writerows(out)

    print(f"{len(out)} contradiction clips matched to a reference answer ({missing} unmatched)\n")
    print(f"  reference SAYS fall  -> model yes/no 'No' is WRONG (genuine bug, override harmful): {ref_fall}")
    print(f"  reference NO fall    -> model yes/no 'No' is RIGHT (describe hallucinated, override OK): {ref_nofall}\n")
    for o in out:
        flag = "FALL " if o["ref_says_fall"] else "nofall"
        print(f"  [{flag}] {o['video']}")
        print(f"       ref:      {o['reference_answer']}")
        print(f"       describe: {o['model_describe']}")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
