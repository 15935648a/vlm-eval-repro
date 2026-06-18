#!/usr/bin/env bash
# Check generality of the 'compute Yes mid-stack, flip to No late' pattern.
# Usage: scripts/08_flip_layers.sh [--top N]
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m src.flip_layers "$@"
