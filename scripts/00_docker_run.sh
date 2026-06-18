#!/usr/bin/env bash
# Drop into the GPU container with the repo + HF cache mounted.
#   ./scripts/00_docker_run.sh                 # interactive shell
#   ./scripts/00_docker_run.sh bash scripts/01_download_data.sh   # run a command directly
set -euo pipefail
cd "$(dirname "$0")/.."

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

# GB10/DGX Spark: "--gpus all" works when the nvidia toolkit is installed.
# If the GPU isn't visible in the container, switch to CDI: GPU_FLAG="--device nvidia.com/gpu=all"
GPU_FLAG="${GPU_FLAG:---gpus all}"

docker run --rm -it \
  ${GPU_FLAG} \
  --shm-size=16g \
  -v "$PWD":/workspace \
  -v "$HF_CACHE":/root/.cache/huggingface \
  vlm-eval-repro "$@"
