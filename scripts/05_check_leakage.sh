#!/usr/bin/env bash
# Train/test leakage check (eval vs Kinetics-54K training). CPU-only, but runs in the
# same container as everything else. Usage: scripts/05_check_leakage.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m src.check_leakage "$@"
