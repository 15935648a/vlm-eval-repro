#!/usr/bin/env bash
# Join the contradictions against the benchmark's reference answers to see which side is
# actually wrong. CPU-only. Usage: scripts/11_check_labels.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m src.check_labels "$@"
