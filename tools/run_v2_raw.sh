#!/usr/bin/env bash
set -euo pipefail

/usr/local/cuda/bin/nvcc \
  -arch=sm_80 -O3 -std=c++17 -lineinfo \
  -DLAB45_DIAGONAL_MODE=3 -DLAB45_USE_WMMA=1 -Iinclude -Itools \
  tools/benchmark_my.cu tools/my_int8_fp64_v2.cu utils.cu \
  -lcublas -lcudart -lcuda \
  -o tools/benchmark_v2_raw

tools/benchmark_v2_raw "${1:-1024}" "${2:-2,4,6,8}" "${3:-2}"
