#!/usr/bin/env bash
# Scan all clips with the describe-prompt and rank by keyword match.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m src.find_video "$@"
echo "inspect results/find_video.csv and pick the top match, then run scripts/03_reproduce.sh <video>"
