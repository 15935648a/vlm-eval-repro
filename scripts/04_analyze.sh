#!/usr/bin/env bash
# Milestone 2: logit lens + activation patching. Usage: scripts/04_analyze.sh <video.mp4>
set -euo pipefail
cd "$(dirname "$0")/.."
if [ $# -lt 1 ]; then echo "usage: $0 <video.mp4>"; exit 1; fi
python3 -m src.analyze --video "$1"
