# P1-P3 Candidate Results

Date: 2026-08-28 UTC

## Environment

- Partition: `lab4g10`
- GPU: NVIDIA A100 80GB PCIe MIG 1g.10gb
- Compute capability: 8.0
- Candidate source: `tools/my_int8_fp64_candidate.cu`
- Build target: `sm_80`, `-O3`, CUDA C++17
- The candidate was linked in place of `submit/my_int8_fp64.cu`; the submit
  source was not modified.

## Candidate Design

- One fused quantization launch per input generates every split.
- A splits are emitted directly as column-major transposed matrices.
- B splits remain in regular column-major layout.
- One persistent grow-only workspace holds all quantized matrices and one
  INT32 output matrix.
- INT8 GEMMs use `cublasGemmEx(CUBLAS_OP_T, CUBLAS_OP_N)`.
- Products on one anti-diagonal are accumulated in INT32 with `beta=0/1`,
  followed by one FP64 recombination kernel per anti-diagonal.
- All `splits * splits` products are retained in this candidate.

## Smoke Test

Command:

```text
./tools/benchmark_candidate 256,1024 2,4,6,8 1 --csv
```

The candidate matched the baseline L2 relative error for every tested size and
split count. At size 1024, candidate results were:

| splits | baseline ms | candidate ms | speedup | candidate GFLOPS | L2 relative error |
|---:|---:|---:|---:|---:|---:|
| 2 | 3.2881 | 0.9636 | 3.41x | 2228.64 | 2.193e-05 |
| 4 | 8.3036 | 2.3214 | 3.58x | 925.08 | 3.395e-10 |
| 6 | 16.2898 | 4.2578 | 3.83x | 504.37 | 5.307e-15 |
| 8 | 27.2015 | 6.7789 | 4.01x | 316.79 | 6.399e-16 |

Raw log: `tools/candidate_smoke_186127.out`.

## 4096 Results

Command:

```text
./tools/benchmark_candidate 4096 2,4,6,8 3 --csv
```

| splits | baseline ms | candidate ms | speedup | candidate GFLOPS | candidate L2 | baseline L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 88.5088 | 18.6952 | 4.73x | 7351.58 | 2.192e-05 | 2.192e-05 |
| 4 | 303.1385 | 58.3588 | 5.20x | 2355.07 | 3.397e-10 | 3.397e-10 |
| 6 | 655.0364 | 119.3847 | 5.49x | 1151.23 | 5.682e-15 | 5.685e-15 |
| 8 | 1146.8824 | 201.6338 | 5.69x | 681.63 | 2.139e-15 | 2.150e-15 |

Raw log: `tools/candidate_4096_186131.out`.

## Interpretation

The transposed layout and anti-diagonal accumulation preserve baseline-level
accuracy. Persistent storage and the fused quantizers remove repeated allocation
and residual traffic, while the TN path reduces total time by roughly 4.7-5.7x
at 4096. The full `S^2` GEMM count remains the dominant limitation: this
candidate does not reach the published 100-point checkpoints, especially for
splits 6 and 8. Pair pruning or a method that combines multiple products into
fewer Tensor Core GEMMs is still required.
