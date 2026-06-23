# Summary — yes/no vs describe contradiction in the Kinetics-FT Gemma-4 video VLM

**Question investigated.** On `THChou1220/gemma-4-e2b-kinetics54K_FT`, the free-form
description says a fall happened ("...falls to the floor") while the binary prompt
"Did the person fall? yes/no" answers **No** — same video, same 8 frames. Can hidden states
explain it, and can it be fixed?

Run on a DGX Spark (GB10, aarch64) in Docker (NGC PyTorch base). All steps are in `scripts/`.

## What we found

**1. Not perception, not referent.** (`diagnose.py`) On every contradiction clip
`is_person_present` ≈ 0.97–0.998 (the model sees a person, even babies), yet *every* fall
phrasing — person/someone/anyone/child fall — stays ~0.005–0.11. The yes/no **format** works
(presence → Yes); the **fall judgment** specifically collapses to No.

**2. Mechanism: compute "Yes" mid-stack, overwrite to "No" late.**
- Logit lens (`analyze.py`) on the failing prompt: layers **L26–32 decode to 'Yes' (P≈1.0)**,
  then **L33–35 flip to 'No' (P=1.0)**. The answer is computed, then overwritten.
- General, not a one-clip artifact (`flip_layers.py`): 8/8 clips peak P(yes)≈1.0 at L26–28,
  flip at L29–34.
- Causally corroborated (`analyze.py` patching): injecting `describe` content does **not**
  flip it (the fall content is already present); injecting the `presence` residual flips it to
  Yes from L17 and **survives** the late override → the override is *content-gated*, not a
  blanket No.

**3. The late override is a conservative precision mechanism, not a bug.**
(`labeled_eval.py`, category as weak ground truth)

| group | final-layer Yes% | early-exit @L28 Yes% |
|---|---|---|
| real falls (face_planting / bike / chair) | 60 / 67 / 33% (recall) | 100 / 100 / 93% |
| no-fall (general_mp4) | **0% (FPR)** | **60% (FPR)** |
| climbing_ladder (ambiguous) | 7% | 87% |

Removing the late layers (early-exit) makes false-positives jump 0→60% on clean negatives —
so the late layers' *function* is to suppress spurious mid-stack "Yes". That is precision/
conservatism, measured. The same mechanism over-fires on genuine falls (recall only 33–67%).

**4. Which side is actually wrong.** (`check_labels.py`) The benchmark's reference `answer`
is a per-category template ("The person climbs a ladder"), i.e. **action-class ground truth**.
Taking it at face value:
- climbing/general "contradictions" → reference says *no fall* → the model's **"No" is correct**
  and the `describe` output is hallucinating a fall. The override is doing the right thing.
- the genuine yes/no bug is narrower: **low recall on the real fall categories** (says No to
  clips whose ground-truth answer is "falls off a bike/chair").
- Residual uncertainty: Kinetics labels are noisy; a few "climbing" clips might be mislabeled
  falls (then describe is right). Only watching the clips resolves this.

**5. A wrong fix was blocked.** Logit-lens + patching alone suggested "just early-exit before
the override." Adding a **true-negative control set** killed it: early-exit FPR 0→60%, steering
false-pos 0→67–100%, balanced accuracy 77→69% (worse). A fix validated only on the failures you
went looking for will always look good — you must test the class that should stay "No".

## What is / isn't claimable

- ✅ **Mechanism** (compute-Yes-then-conservatively-override): robust, reproducible, no labels needed.
- ✅ **The genuine defect**: low recall on real falls (the describe/yes/no disagreement on
  climbing clips is mostly `describe` hallucination, where yes/no is the *correct* one).
- ❌ **"early-exit fixes it"** — refuted by the negative control.
- ⚠️ **Any fix as "general"** requires an independently-labeled precision/recall comparison that
  *dominates* the model's current operating point. The dataset has no per-clip fall label.

## How to actually solve the contradiction (open, needs labels)

The contradiction = two opposite miscalibrations: `describe` over-claims falls (false positives),
`yes/no` is conservative (false negatives). "Consistent" is easy; "consistent **and correct**"
needs a labeled fall signal.

- **B — threshold-calibrate** `P(yes)` on a labeled dev set instead of argmax@0.5 (cheap; moves
  along the ROC with a control, unlike early-exit).
- **C — linear probe on the L27 residual** (where "Yes" is cleanest) as a deployable fall
  detector; ship it only if it dominates both raw outputs on a held-out ROC. *This is the path
  where the hidden-state analysis becomes the solution.* Guard against learning a category
  shortcut (e.g. "climbing") rather than "fall".
- **D — small balanced SFT** on yes/no fall QA (root cause: yes/no QA is OOD; training was
  captioning only). Most robust, most effort.
- **Bonus** — use the `describe` vs `yes/no` disagreement as a free hallucination/uncertainty
  flag for human review.

**Open items:** (a) get real per-clip fall labels (category is weak; climbing is ambiguous);
(b) run `check_leakage.py` — eval and training both come from Kinetics, so scores may be inflated
(`general_mp4` clips even carry Kinetics `ytid_start_end` filenames).

## Script map

`00` docker build/run · `01` download · `02` find_video · `03` reproduce · `04` analyze
(logit lens + patching) · `05` check_leakage · `06` find_contradiction · `07` diagnose ·
`08` flip_layers · `09` steer · `10` early_exit · `11` check_labels · `12` labeled_eval.
