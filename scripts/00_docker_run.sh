#!/usr/bin/env bash
# Drop into the GPU container with the repo + HF cache mounted.
#   ./scripts/00_docker_run.sh                 # interactive shell
#   ./scripts/00_docker_run.sh bash scripts/01_download_data.sh   # run a command directly
set -euo pipefail
cd "$(dirname "$0")/.."

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

docker run --rm -it \
  --gpus all \
  --shm-size=16g \
  -v "$PWD":/workspace \
  -v "$HF_CACHE":/root/.cache/huggingface \
  vlm-eval-repro "$@"
