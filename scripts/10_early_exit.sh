#!/usr/bin/env bash
# Early-exit fix: decode the fall answer before the late 'No' override; sweep exit layer.
# Usage: scripts/10_early_exit.sh [--layers 26 27 28 ...]
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m src.early_exit "$@"
