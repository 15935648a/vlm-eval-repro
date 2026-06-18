#!/usr/bin/env bash
# Mine results/find_video.csv for clips where describe=fall but yes/no=No.
# Usage: scripts/06_find_contradiction.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m src.find_contradiction "$@"
