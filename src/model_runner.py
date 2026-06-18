"""Load the Gemma 4 video VLM and run it on sampled frames.

Exposes three things the project needs:
  - generate()           : greedy text output (milestone 1, reproduction)
  - first_token_yes_no() : P(Yes)/P(No) at the first answer token (analysis: H1/H2/H3)
  - forward_hidden()     : full hidden_states tuple at the answer position (analysis: probe/lens/patch)

Model class is `Gemma4ForConditionalGeneration`; we load via AutoModelForImageTextToText,
which maps to it. If your transformers build needs the explicit class, swap the import.
"""
from typing import Dict, List, Tuple

import torch
from PIL import Image

from . import config


def _dtype(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


class GemmaVideoRunner:
    def __init__(self, model_id: str = config.MODEL_ID, device: str = "cuda"):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=_dtype(config.DTYPE),   # `torch_dtype` is deprecated in recent transformers
            device_map=device,
        )
        self.model.eval()
        self._yes_ids, self._no_ids = self._yes_no_token_ids()

    # --- prompt assembly -------------------------------------------------
    def _build_inputs(self, frames: List[Image.Image], prompt: str) -> Dict[str, torch.Tensor]:
        # 8 frames passed as image content + the text question (Gemma multimodal chat format).
        content = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    # --- milestone 1: reproduction --------------------------------------
    @torch.no_grad()
    def generate(self, frames: List[Image.Image], prompt: str,
                 max_new_tokens: int = config.MAX_NEW_TOKENS) -> str:
        inputs = self._build_inputs(frames, prompt)
        in_len = inputs["input_ids"].shape[-1]
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,      # greedy -> deterministic
            num_beams=1,
        )
        text = self.processor.decode(out[0][in_len:], skip_special_tokens=True)
        return text.strip()

    # --- analysis hooks --------------------------------------------------
    def _yes_no_token_ids(self) -> Tuple[List[int], List[int]]:
        tok = self.processor.tokenizer
        yes_words = ["Yes", " Yes", "yes", " yes", "YES"]
        no_words = ["No", " No", "no", " no", "NO"]

        def first_ids(words):
            ids = set()
            for w in words:
                enc = tok.encode(w, add_special_tokens=False)
                if enc:
                    ids.add(enc[0])
            return sorted(ids)

        return first_ids(yes_words), first_ids(no_words)

    @torch.no_grad()
    def first_token_yes_no(self, frames: List[Image.Image], prompt: str) -> Dict[str, float]:
        """Softmax mass on Yes-tokens vs No-tokens at the first generated position."""
        inputs = self._build_inputs(frames, prompt)
        logits = self.model(**inputs).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        p_yes = probs[self._yes_ids].sum().item()
        p_no = probs[self._no_ids].sum().item()
        top_id = int(logits.argmax())
        return {
            "p_yes": p_yes,
            "p_no": p_no,
            "argmax_is_yes": top_id in self._yes_ids,
            "argmax_is_no": top_id in self._no_ids,
            "top_token": self.processor.tokenizer.decode([top_id]),
        }

    @torch.no_grad()
    def forward_hidden(self, frames: List[Image.Image], prompt: str):
        """Return (inputs, outputs-with-hidden_states) for the answer position.
        Used by milestone-2 analysis (logit lens / linear probe / activation patching).
        hidden_states: tuple(num_layers+1) of (1, seq, hidden); position -1 is the readout."""
        inputs = self._build_inputs(frames, prompt)
        outputs = self.model(**inputs, output_hidden_states=True)
        return inputs, outputs
