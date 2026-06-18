#!/usr/bin/env bash
# Build the experiment image.
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -t vlm-eval-repro .
