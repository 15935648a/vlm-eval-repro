#!/usr/bin/env bash
# Reproduce the contradiction on one clip. Usage: scripts/03_reproduce.sh <video.mp4>
set -euo pipefail
cd "$(dirname "$0")/.."
if [ $# -lt 1 ]; then echo "usage: $0 <video.mp4>"; exit 1; fi
python -m src.reproduce --video "$1"
