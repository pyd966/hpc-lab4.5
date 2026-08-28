#!/usr/bin/env bash
set -euo pipefail

/usr/local/cuda/bin/nvcc \
  -arch=sm_80 -O3 -std=c++17 -lineinfo \
  tools/int8_lt_algo_scan.cu \
  -lcublasLt -lcublas -lcudart \
  -o tools/int8_lt_algo_scan

tools/int8_lt_algo_scan "${1:-4096}" "${2:-5}"
