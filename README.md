# vlm-eval-repro

Reproduce and analyze an inconsistency in a video VLM
(`THChou1220/gemma-4-e2b-kinetics54K_FT`, a `Gemma4ForConditionalGeneration` model
fine-tuned on Kinetics, evaluated on the `gnitoahc/vlm-eval-videos` benchmark).

## The bug we are chasing

Same video, same 8 sampled frames, two prompts that disagree:

| Prompt | Observed answer |
|---|---|
| `Did the person fall in the video? Answer either yes or no` | **No** |
| `Detect if any accident happens. If yes, give a very consice description, otherwise say everything is normal` | *"...skating on an outdoor rink, falling and then getting back up"* |

The binary-QA answer ("No") contradicts the free-form description (which describes a fall).
Three competing hypotheses we want to separate:

- **H1 — format/prior**: yes/no is out-of-distribution for a caption fine-tune; the model
  has a strong "No" prior for `Did X happen?` regardless of content.
- **H2 — semantic threshold**: the model sees "falling **and then getting back up**" and
  judges it does not count as a fall/accident, so "No" is content-grounded but strict.
- **H3 — perception vs readout**: the fall *is* encoded in the hidden state; the binary head
  just fails to surface it.

> Architectural note that shapes the analysis: this is a decoder-only causal model, so the
> **vision-token hidden states are identical across the two prompts** (image tokens precede the
> question and cannot attend to it). The divergence must live in the *answer/readout positions*,
> not in the image encoding. So we probe text/generation positions, not vision tokens.

## Milestones

1. **Stable reproduction** (this repo's first goal)
   - Locate the exact clip (`src/find_video.py`)
   - Greedy decoding, fixed 8-frame sampling, confirm the contradiction is deterministic
2. **Analysis** (next): yes/no logits → logit lens → linear probe → activation patching.
   `model_runner` already exposes first-token yes/no probabilities and `output_hidden_states`.

## Setup (DGX Spark — GB10 Grace-Blackwell, aarch64, Docker)

Native arm64, so **no QEMU** (build and run on the same machine). The base image is the NGC
PyTorch container, which ships a CUDA torch built for aarch64 + Blackwell (sm_121) — plain
`pip install torch` on arm64 does not. Everything runs inside the container; the repo is
bind-mounted (so `data/` and `results/` land on the host) and the HF cache is mounted too.

```bash
git clone https://github.com/15935648a/vlm-eval-repro.git
cd vlm-eval-repro

# 0. smoke-test GPU + base tag (bump tag if it reports "no kernel image ... sm_121"):
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.01-py3 \
  python -c "import torch; x=torch.randn(4,4,device='cuda'); print('ok', float((x@x).sum()))"
# if the GPU isn't visible, use CDI instead: export GPU_FLAG='--device nvidia.com/gpu=all'

# 1. build the image (NGC PyTorch base + transformers for Gemma 4)
#    override base tag if needed: docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.05-py3 -t vlm-eval-repro .
bash scripts/00_docker_build.sh

# 2. download benchmark videos (~1.15 GB) + model weights (inside the container)
bash scripts/00_docker_run.sh bash scripts/01_download_data.sh

# 3. find which clip produces the skating/fall description -> results/find_video.csv
bash scripts/00_docker_run.sh bash scripts/02_find_video.sh

# 4. reproduce the contradiction on that clip (greedy, repeated for determinism)
bash scripts/00_docker_run.sh python3 -m src.reproduce \
    --video data/vlm-eval-videos/<the_clip>.mp4

# or just open an interactive shell in the container and run things by hand:
bash scripts/00_docker_run.sh
```

> No-Docker fallback: `python3 -m venv .venv && source .venv/bin/activate &&
> pip install -r requirements.txt`, then run the `scripts/0[123]_*.sh` directly.

## Layout

```
src/
  config.py        paths, model id, NUM_FRAMES=8, gen params
  video_utils.py   uniform 8-frame sampling (decord -> opencv fallback)
  model_runner.py  load model+processor; generate(); first_token_yes_no(); hidden-state hook
  prompts.py       the two canonical prompts + yes/no wording variants (for H1/H2)
  dataset_utils.py iterate the downloaded benchmark + metadata.csv
  find_video.py    run the describe-prompt over all clips, grep the target description
  reproduce.py     run both prompts greedy on one clip, dump outputs + yes/no probs
scripts/
  00_docker_build.sh / 00_docker_run.sh   build image / run in GPU container
  01_download_data.sh 02_find_video.sh 03_reproduce.sh
results/           outputs (gitignored)
data/              videos + model cache (gitignored)
Dockerfile         CUDA 12.4 runtime + deps
```
