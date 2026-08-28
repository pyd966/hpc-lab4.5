/*
 * Reproducible extended-K anti-diagonal concatenation experiment.
 *
 * Quantized A splits are stored in reverse order within each M column while
 * B splits are stored in forward order within each N column.  Every set of
 * pairs with i+j=d is then a contiguous interval on both extended-K axes, so
 * one TN INT8 GEMM computes the whole diagonal without copying or duplication.
 * Reusable workspace allocation is outside the measured region.
 *
 * Build on the current A100 partition:
 *   nvcc -arch=sm_80 -O3 -std=c++17 -lineinfo -Iinclude \
 *     tools/diagonal_concat_probe.cu baseline/baseline_fp64.cu utils.cu \
 *     -lcublas -lcudart -lcuda -o tools/diagonal_concat_probe
 * Scan all D, or average one D over three iterations:
 *   ./tools/diagonal_concat_probe 4096 8
 *   ./tools/diagonal_concat_probe 8192 8 8 3
 */
#include "gemm_api.h"
#include "utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define PROBE_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t status_ = (call);                                        \
        if (status_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA failure at %s:%d: %s\n",            \
                         __FILE__, __LINE__, cudaGetErrorString(status_));    \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

#define PROBE_CUBLAS(call)                                                   \
    do {                                                                     \
        cublasStatus_t status_ = (call);                                     \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                              \
            std::fprintf(stderr, "cuBLAS failure at %s:%d: %d\n",          \
                         __FILE__, __LINE__, static_cast<int>(status_));      \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

constexpr int kTile = 32;
constexpr int kBlockRows = 8;
constexpr int kMaxSplits = 8;

__device__ __forceinline__ int8_t quantize_digit(double residual,
                                                 double scale) {
    double rounded = rint(residual / scale);
    rounded = fmin(127.0, fmax(-127.0, rounded));
    return static_cast<int8_t>(rounded);
}

/*
 * Output is a column-major (splits*K)-by-M matrix.  Split i occupies extended
 * K slot splits-1-i, making descending i intervals physically contiguous.
 */
static __global__ void quantize_a_reverse_interleaved_kernel(
    const double* input, int8_t* output, double initial_scale,
    int M, int K, int splits) {
    __shared__ int8_t tile[kMaxSplits][kTile][kTile + 1];

    const int input_m = blockIdx.x * kTile + threadIdx.x;
    const int input_k = blockIdx.y * kTile + threadIdx.y;

#pragma unroll
    for (int offset = 0; offset < kTile; offset += kBlockRows) {
        const int k = input_k + offset;
        if (input_m < M && k < K) {
            double residual = input[static_cast<size_t>(k) * M + input_m];
            double scale = initial_scale;
#pragma unroll
            for (int split = 0; split < kMaxSplits; ++split) {
                if (split < splits) {
                    const int8_t digit = quantize_digit(residual, scale);
                    tile[split][threadIdx.y + offset][threadIdx.x] = digit;
                    residual -= static_cast<double>(digit) * scale;
                    scale /= 254.0;
                }
            }
        }
    }
    __syncthreads();

    const int output_k = blockIdx.y * kTile + threadIdx.x;
    const int output_m = blockIdx.x * kTile + threadIdx.y;
    const size_t extended_k = static_cast<size_t>(splits) * K;
#pragma unroll
    for (int offset = 0; offset < kTile; offset += kBlockRows) {
        const int m = output_m + offset;
        if (m < M && output_k < K) {
#pragma unroll
            for (int split = 0; split < kMaxSplits; ++split) {
                if (split < splits) {
                    const int slot = splits - 1 - split;
                    output[static_cast<size_t>(m) * extended_k
                           + static_cast<size_t>(slot) * K + output_k]
                        = tile[split][threadIdx.x][threadIdx.y + offset];
                }
            }
        }
    }
}

/* Output is a column-major (splits*K)-by-N matrix in ascending split order. */
static __global__ void quantize_b_forward_interleaved_kernel(
    const double* input, int8_t* output, double initial_scale,
    int K, int N, int splits) {
    const size_t elements = static_cast<size_t>(K) * N;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x
                     + threadIdx.x;
    if (idx >= elements) return;

    const int k = static_cast<int>(idx % K);
    const int n = static_cast<int>(idx / K);
    const size_t extended_k = static_cast<size_t>(splits) * K;
    double residual = input[idx];
    double scale = initial_scale;
#pragma unroll
    for (int split = 0; split < kMaxSplits; ++split) {
        if (split < splits) {
            const int8_t digit = quantize_digit(residual, scale);
            output[static_cast<size_t>(n) * extended_k
                   + static_cast<size_t>(split) * K + k] = digit;
            residual -= static_cast<double>(digit) * scale;
            scale /= 254.0;
        }
    }
}

static __global__ void recombine_diagonal_kernel(const int32_t* partial,
                                                  double* output,
                                                  double scale,
                                                  size_t elements) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x
                     + threadIdx.x;
    if (idx < elements) {
        output[idx] += scale * static_cast<double>(partial[idx]);
    }
}

struct ProbeContext {
    int size;
    int splits;
    size_t elements;
    const double* a;
    const double* b;
    double* output;
    int8_t* aq_extended;
    int8_t* bq_extended;
    int32_t* diagonal_sum;
    cublasHandle_t handle;
    cudaStream_t stream;
};

/* End-to-end algorithm, excluding reusable workspace allocation. */
static void run_concatenated(const ProbeContext& ctx, int diagonal_limit) {
    double max_a = device_maxabs_fp64(ctx.a, ctx.elements);
    double max_b = device_maxabs_fp64(ctx.b, ctx.elements);
    if (max_a == 0.0) max_a = 1.0;
    if (max_b == 0.0) max_b = 1.0;
    const double scale_a0 = max_a / 127.0;
    const double scale_b0 = max_b / 127.0;

    const dim3 transpose_block(kTile, kBlockRows);
    const dim3 transpose_grid((ctx.size + kTile - 1) / kTile,
                              (ctx.size + kTile - 1) / kTile);
    quantize_a_reverse_interleaved_kernel<<<transpose_grid, transpose_block, 0,
                                             ctx.stream>>>(
        ctx.a, ctx.aq_extended, scale_a0,
        ctx.size, ctx.size, ctx.splits);

    constexpr int threads = 256;
    const int grid = static_cast<int>((ctx.elements + threads - 1) / threads);
    quantize_b_forward_interleaved_kernel<<<grid, threads, 0, ctx.stream>>>(
        ctx.b, ctx.bq_extended, scale_b0,
        ctx.size, ctx.size, ctx.splits);
    PROBE_CUDA(cudaMemsetAsync(ctx.output, 0,
                               ctx.elements * sizeof(double), ctx.stream));

    const int32_t alpha = 1;
    const int32_t beta = 0;
    const int extended_ld = ctx.splits * ctx.size;
    for (int diagonal = 0; diagonal < diagonal_limit; ++diagonal) {
        const int j_low = std::max(0, diagonal - (ctx.splits - 1));
        const int j_high = std::min(ctx.splits - 1, diagonal);
        const int pair_count = j_high - j_low + 1;
        const int i_high = diagonal - j_low;
        const int a_slot = ctx.splits - 1 - i_high;
        const int gemm_k = pair_count * ctx.size;

        PROBE_CUBLAS(cublasGemmEx(
            ctx.handle, CUBLAS_OP_T, CUBLAS_OP_N,
            ctx.size, ctx.size, gemm_k,
            &alpha,
            ctx.aq_extended + static_cast<size_t>(a_slot) * ctx.size,
            CUDA_R_8I, extended_ld,
            ctx.bq_extended + static_cast<size_t>(j_low) * ctx.size,
            CUDA_R_8I, extended_ld,
            &beta, ctx.diagonal_sum, CUDA_R_32I, ctx.size,
            CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));

        const double diagonal_scale = scale_a0 * scale_b0
                                    / std::pow(254.0, diagonal);
        recombine_diagonal_kernel<<<grid, threads, 0, ctx.stream>>>(
            ctx.diagonal_sum, ctx.output, diagonal_scale, ctx.elements);
    }
    PROBE_CUDA(cudaStreamSynchronize(ctx.stream));
}

static int retained_pairs(int splits, int diagonal_limit) {
    int pairs = 0;
    for (int diagonal = 0; diagonal < diagonal_limit; ++diagonal) {
        const int low = std::max(0, diagonal - (splits - 1));
        const int high = std::min(splits - 1, diagonal);
        pairs += high - low + 1;
    }
    return pairs;
}

int main(int argc, char** argv) {
    const int size = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int splits = argc > 2 ? std::atoi(argv[2]) : 8;
    const int requested_d = argc > 3 ? std::atoi(argv[3]) : 0;
    const int iterations = argc > 4 ? std::atoi(argv[4]) : 1;
    if (size <= 0 || splits <= 0 || splits > kMaxSplits
        || requested_d < 0 || requested_d > 2 * splits - 1
        || iterations <= 0) {
        std::fprintf(stderr,
                     "usage: %s [matrix-size] [splits<=8] [D] [iters]\n",
                     argv[0]);
        return 2;
    }

    const size_t elements = static_cast<size_t>(size) * size;
    std::vector<double> host_a(elements);
    std::vector<double> host_b(elements);
    fill_matrix(host_a.data(), elements, 0x1234ULL + 7 * size);
    fill_matrix(host_b.data(), elements, 0x5678ULL + 7 * size);

    double *a = nullptr, *b = nullptr, *reference = nullptr, *output = nullptr;
    int8_t *aq_extended = nullptr, *bq_extended = nullptr;
    int32_t* diagonal_sum = nullptr;
    PROBE_CUDA(cudaMalloc(&a, elements * sizeof(double)));
    PROBE_CUDA(cudaMalloc(&b, elements * sizeof(double)));
    PROBE_CUDA(cudaMalloc(&reference, elements * sizeof(double)));
    PROBE_CUDA(cudaMalloc(&output, elements * sizeof(double)));
    PROBE_CUDA(cudaMalloc(&aq_extended,
                          static_cast<size_t>(splits) * elements));
    PROBE_CUDA(cudaMalloc(&bq_extended,
                          static_cast<size_t>(splits) * elements));
    PROBE_CUDA(cudaMalloc(&diagonal_sum, elements * sizeof(int32_t)));
    PROBE_CUDA(cudaMemcpy(a, host_a.data(), elements * sizeof(double),
                          cudaMemcpyHostToDevice));
    PROBE_CUDA(cudaMemcpy(b, host_b.data(), elements * sizeof(double),
                          cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cudaStream_t stream;
    PROBE_CUBLAS(cublasCreate(&handle));
    PROBE_CUDA(cudaStreamCreate(&stream));
    PROBE_CUBLAS(cublasSetStream(handle, stream));
    PROBE_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    const double alpha = 1.0;
    const double beta = 0.0;
    PROBE_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             size, size, size, &alpha, a, size, b, size,
                             &beta, reference, size));
    PROBE_CUDA(cudaStreamSynchronize(stream));

    ProbeContext ctx{size, splits, elements, a, b, output,
                     aq_extended, bq_extended, diagonal_sum, handle, stream};

    run_concatenated(ctx, 2 * splits - 1);

    cudaDeviceProp properties{};
    int device = 0;
    PROBE_CUDA(cudaGetDevice(&device));
    PROBE_CUDA(cudaGetDeviceProperties(&properties, device));
    std::fprintf(stderr,
                 "GPU=%s CC=%d.%d size=%d splits=%d D=%d iters=%d\n",
                 properties.name, properties.major, properties.minor,
                 size, splits, requested_d, iterations);
    std::printf("size,splits,D,retained_pairs,total_pairs,time_ms,gflops,"
                "max_abs_err,l2_rel_err\n");

    const int first_d = requested_d ? requested_d : 1;
    const int last_d = requested_d ? requested_d : 2 * splits - 1;
    for (int diagonal_limit = first_d;
         diagonal_limit <= last_d; ++diagonal_limit) {
        const auto begin = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            run_concatenated(ctx, diagonal_limit);
        }
        const auto end = std::chrono::steady_clock::now();
        const double milliseconds =
            std::chrono::duration<double, std::milli>(end - begin).count()
            / iterations;

        AccuracyReport accuracy{};
        accuracy_compare(reference, output, size, size, &accuracy);
        const double flops = 2.0 * size * static_cast<double>(size) * size;
        const double gflops = flops / (milliseconds * 1.0e6);
        std::printf("%d,%d,%d,%d,%d,%.6f,%.2f,%.9e,%.9e\n",
                    size, splits, diagonal_limit,
                    retained_pairs(splits, diagonal_limit), splits * splits,
                    milliseconds, gflops,
                    accuracy.max_abs_err, accuracy.l2_rel_err);
        std::fflush(stdout);
    }

    PROBE_CUDA(cudaFree(diagonal_sum));
    PROBE_CUDA(cudaFree(bq_extended));
    PROBE_CUDA(cudaFree(aq_extended));
    PROBE_CUDA(cudaFree(output));
    PROBE_CUDA(cudaFree(reference));
    PROBE_CUDA(cudaFree(b));
    PROBE_CUDA(cudaFree(a));
    PROBE_CUDA(cudaStreamDestroy(stream));
    PROBE_CUBLAS(cublasDestroy(handle));
    return 0;
}
