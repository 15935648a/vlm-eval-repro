#!/usr/bin/env bash
# Subject-referent probe matrix on the top contradiction clips.
# Usage: scripts/07_diagnose.sh            # top-6 from results/contradictions.csv
#        scripts/07_diagnose.sh --videos data/.../clipA.mp4 data/.../clipB.mp4
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m src.diagnose "$@"
