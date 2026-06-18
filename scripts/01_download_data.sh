#!/usr/bin/env bash
# Download the benchmark videos (~1.15 GB) and warm the model cache.
set -euo pipefail
cd "$(dirname "$0")/.."

python -c "from src.dataset_utils import download_dataset; print('dataset ->', download_dataset())"

# Pre-download the model weights so the first run doesn't stall.
python -c "
from src import config
from huggingface_hub import snapshot_download
print('model ->', snapshot_download(config.MODEL_ID))
"
