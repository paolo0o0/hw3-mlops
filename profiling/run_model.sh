#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/model-analyzer

docker build \
  -t triton-iris-model-analyzer \
  -f profiling/Dockerfile.model-analyzer .

docker run --rm --shm-size=1g \
  -v "$PWD:/workspace" \
  triton-iris-model-analyzer \
  bash -lc 'mkdir -p /tmp/model-analyzer && model-analyzer profile \
  -f /workspace/profiling/model-analyzer.yaml' \
  > results/model-analyzer/run.log 2>&1
