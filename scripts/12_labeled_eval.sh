#!/usr/bin/env bash
# Net-accuracy eval (category as weak ground truth): final vs early-exit, per group.
# Usage: scripts/12_labeled_eval.sh [--layer 28 --cap 15]
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m src.labeled_eval "$@"
