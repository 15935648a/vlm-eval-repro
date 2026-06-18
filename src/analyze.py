"""Milestone 2 — why does yes/no say "No" while describe says a fall happened?

Two interventions on a single clip, both operating at the *answer position* (the last
token, where generation starts). Recall the architectural point: the vision tokens are
identical across both prompts, so the divergence must be in how the answer position reads
out from them — that is exactly what these probe.

  (A) logit lens : project each layer's answer-position hidden state to vocab and watch
                   when P(Yes) vs P(No) is decided across depth. A late-layer flip to "No"
                   over an earlier "Yes" lead is evidence for H1/H3 (a prior override).
  (B) activation patching : run the describe prompt, copy its residual stream at the answer
                   position into the yes/no run, layer by layer, and measure the shift in
                   P(Yes). The layer whose patch flips No->Yes is the causal locus ("決定性").

Run AFTER reproduction is confirmed. Module paths below are best-effort for
Gemma4ForConditionalGeneration; if discovery fails, the error prints the model structure so
we can pin the right attribute names.
"""
import argparse
import json

import torch

from . import config
from .model_runner import GemmaVideoRunner
from .prompts import DESCRIBE_PROMPT, YESNO_PROMPT
from .video_utils import sample_frames


# --- locate the decoder layers / final norm / lm_head inside the wrapper ----------------
def _language_model(model):
    # Gemma4ForConditionalGeneration -> .model (Gemma4Model) -> .language_model (text stack)
    for path in ("model.language_model", "language_model", "model.model.language_model", "model"):
        obj = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and hasattr(obj, "layers"):
            return obj
    raise AttributeError(
        "could not find the text decoder stack; model structure:\n"
        + "\n".join(n for n, _ in model.named_modules())[:4000]
    )


def _final_norm_and_head(model, lm):
    head = model.get_output_embeddings()           # tied lm_head
    norm = getattr(lm, "norm", None) or getattr(lm, "final_layernorm", None)
    if norm is None:
        raise AttributeError("could not find final norm on the language model")
    return norm, head


@torch.no_grad()
def logit_lens(runner: GemmaVideoRunner, frames, prompt: str):
    """P(Yes)/P(No) read from every layer's answer-position hidden state."""
    inputs, outputs = runner.forward_hidden(frames, prompt)
    lm = _language_model(runner.model)
    norm, head = _final_norm_and_head(runner.model, lm)
    yes_ids, no_ids = runner._yes_ids, runner._no_ids

    rows = []
    for layer_idx, h in enumerate(outputs.hidden_states):   # tuple: embeddings + each layer
        vec = h[0, -1]                                       # answer position
        logits = head(norm(vec.unsqueeze(0))).float().squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        rows.append({
            "layer": layer_idx,
            "p_yes": probs[yes_ids].sum().item(),
            "p_no": probs[no_ids].sum().item(),
            "top": runner.processor.tokenizer.decode([int(logits.argmax())]),
        })
    return rows


@torch.no_grad()
def activation_patch(runner: GemmaVideoRunner, frames):
    """Patch the describe-run residual (answer position) into the yes/no run, per layer.
    Returns P(yes) after patching each layer; compare to the unpatched baseline."""
    lm = _language_model(runner.model)
    layers = lm.layers

    # 1. capture describe-run residual at the answer position, per layer (block output).
    donor = {}

    def make_capture(i):
        def hook(_m, _inp, out):
            donor[i] = (out[0] if isinstance(out, tuple) else out)[0, -1].detach().clone()
        return hook

    handles = [layers[i].register_forward_hook(make_capture(i)) for i in range(len(layers))]
    runner.forward_hidden(frames, DESCRIBE_PROMPT)
    for h in handles:
        h.remove()

    # 2. baseline P(yes) on the yes/no prompt.
    base = runner.first_token_yes_no(frames, YESNO_PROMPT)
    yesno_inputs = runner._build_inputs(frames, YESNO_PROMPT)

    # 3. for each layer, inject the donor vector at the answer position and re-read P(yes).
    results = [{"layer": "baseline", "p_yes": base["p_yes"], "p_no": base["p_no"]}]
    for i in range(len(layers)):
        def patch_hook(_m, _inp, out, vec=donor[i]):
            t = out[0] if isinstance(out, tuple) else out
            t[0, -1] = vec.to(t.dtype)
            return out
        hp = layers[i].register_forward_hook(patch_hook)
        logits = runner.model(**yesno_inputs).logits[0, -1].float()
        hp.remove()
        probs = torch.softmax(logits, dim=-1)
        results.append({
            "layer": i,
            "p_yes": probs[runner._yes_ids].sum().item(),
            "p_no": probs[runner._no_ids].sum().item(),
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "analyze.json"))
    args = ap.parse_args()

    frames = sample_frames(args.video, config.NUM_FRAMES)
    runner = GemmaVideoRunner()

    record = {
        "video": args.video,
        "logit_lens_yesno": logit_lens(runner, frames, YESNO_PROMPT),
        "logit_lens_describe": logit_lens(runner, frames, DESCRIBE_PROMPT),
        "activation_patch_describe_into_yesno": activation_patch(runner, frames),
    }
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)

    print("logit lens (yes/no prompt) — P(yes)/P(no) by layer:")
    for r in record["logit_lens_yesno"]:
        print(f"  L{r['layer']:>2}  yes={r['p_yes']:.3f} no={r['p_no']:.3f}  top={r['top']!r}")
    print("\nactivation patch (describe -> yes/no), P(yes) by patched layer:")
    for r in record["activation_patch_describe_into_yesno"]:
        print(f"  {str(r['layer']):>8}  p_yes={r['p_yes']:.3f}")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
