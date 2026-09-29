#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/perf
# Docker Desktop: host.docker.internal; Linux: --add-host maps to host gateway.
for batch in 1 4 8; do
  for concurrency in 1 8 16 32; do
    docker run --rm --add-host=host.docker.internal:host-gateway \
      -v "$PWD/results/perf:/results" nvcr.io/nvidia/tritonserver:25.11-py3-sdk \
      perf_analyzer -v -m iris_classifier -u host.docker.internal:8000 -i http \
      --shape "INPUT:$batch,4" --concurrency-range "$concurrency" \
      --measurement-interval 5000 --max-trials 20 --stability-percentage 20 \
      -f "/results/http_n${batch}_c${concurrency}.csv" \
      > "results/perf/http_n${batch}_c${concurrency}.log" 2>&1
    if grep -q 'Failed to obtain stable measurement' "results/perf/http_n${batch}_c${concurrency}.log"; then
      echo "Unstable measurement; inspect results/perf logs" >&2
      exit 1
    fi
    test -s "results/perf/http_n${batch}_c${concurrency}.csv"
  done
done

python3 profiling/summarize_perf.py
