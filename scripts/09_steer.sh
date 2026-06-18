#!/usr/bin/env bash
# Activation-steering fix: build a fixed Yes-direction vector and sweep alpha, reporting
# recovery / retention / false-positive. Usage: scripts/09_steer.sh [--layer L --alphas 0 2 4 ...]
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m src.steer "$@"
