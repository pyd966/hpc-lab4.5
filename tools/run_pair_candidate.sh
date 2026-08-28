#!/usr/bin/env bash
set -euo pipefail

/usr/local/cuda/bin/nvcc \
  -arch=sm_80 -O3 -std=c++17 -lineinfo \
  -Iinclude \
  tools/benchmark_my.cu tools/my_int8_fp64_candidate.cu utils.cu \
  -lcublas -lcudart -lcuda \
  -o tools/benchmark_pair_candidate

tools/benchmark_pair_candidate "${1:-4096}" "${2:-2,4,6,8}" "${3:-3}"
