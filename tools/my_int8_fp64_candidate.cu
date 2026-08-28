#include "gemm_api.h"
#include "utils.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kTile = 32;
constexpr int kBlockRows = 8;
constexpr int kLinearBlock = 256;

struct PersistentWorkspace {
    void* base = nullptr;
    size_t capacity = 0;
    int device = -1;
};

PersistentWorkspace g_workspace;

size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

int ensure_workspace(size_t bytes, cudaStream_t stream)
{
    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));

    if (g_workspace.base && g_workspace.device != device) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(g_workspace.base));
        g_workspace = {};
    }
    if (g_workspace.capacity >= bytes) return 0;

    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (g_workspace.base) CUDA_CHECK(cudaFree(g_workspace.base));

    void* replacement = nullptr;
    CUDA_CHECK(cudaMalloc(&replacement, bytes));
    g_workspace.base = replacement;
    g_workspace.capacity = bytes;
    g_workspace.device = device;
    return 0;
}

__device__ __forceinline__ int8_t quantize_one(double residual, double scale)
{
    double rounded = rint(residual / scale);
    rounded = fmin(127.0, fmax(-127.0, rounded));
    return static_cast<int8_t>(rounded);
}

template <int Splits>
__global__ void quantize_a_transposed(const double* __restrict__ input,
                                     int8_t* __restrict__ output,
                                     int M, int K, size_t split_stride,
                                     double initial_scale)
{
    __shared__ int8_t tile[Splits][kTile][kTile + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.x * kTile + tx;
    const int k_base = blockIdx.y * kTile;

#pragma unroll
    for (int j = 0; j < kTile; j += kBlockRows) {
        const int k = k_base + ty + j;
        if (m < M && k < K) {
            double residual = input[static_cast<size_t>(k) * M + m];
            double scale = initial_scale;
#pragma unroll
            for (int split = 0; split < Splits; ++split) {
                const int8_t q = quantize_one(residual, scale);
                tile[split][ty + j][tx] = q;
                residual = fma(-static_cast<double>(q), scale, residual);
                scale /= 254.0;
            }
        }
    }
    __syncthreads();

    const int k_out = k_base + tx;
#pragma unroll
    for (int j = 0; j < kTile; j += kBlockRows) {
        const int m_out = blockIdx.x * kTile + ty + j;
        if (k_out < K && m_out < M) {
            const size_t offset = static_cast<size_t>(m_out) * K + k_out;
#pragma unroll
            for (int split = 0; split < Splits; ++split) {
                output[static_cast<size_t>(split) * split_stride + offset] =
                    tile[split][tx][ty + j];
            }
        }
    }
}

template <int Splits>
__global__ void quantize_b_regular(const double* __restrict__ input,
                                   int8_t* __restrict__ output,
                                   size_t elements, size_t split_stride,
                                   double initial_scale)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    if (index >= elements) return;

    double residual = input[index];
    double scale = initial_scale;
#pragma unroll
    for (int split = 0; split < Splits; ++split) {
        const int8_t q = quantize_one(residual, scale);
        output[static_cast<size_t>(split) * split_stride + index] = q;
        residual = fma(-static_cast<double>(q), scale, residual);
        scale /= 254.0;
    }
}

__global__ void quantize_a_transposed_dynamic(
    const double* __restrict__ input, int8_t* __restrict__ output,
    int M, int K, size_t split_stride, int splits, double initial_scale)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    const size_t elements = static_cast<size_t>(M) * K;
    if (index >= elements) return;

    const int m = static_cast<int>(index % M);
    const int k = static_cast<int>(index / M);
    double residual = input[index];
    double scale = initial_scale;
    const size_t transposed = static_cast<size_t>(m) * K + k;
    for (int split = 0; split < splits; ++split) {
        const int8_t q = quantize_one(residual, scale);
        output[static_cast<size_t>(split) * split_stride + transposed] = q;
        residual = fma(-static_cast<double>(q), scale, residual);
        scale /= 254.0;
    }
}

__global__ void quantize_b_regular_dynamic(
    const double* __restrict__ input, int8_t* __restrict__ output,
    size_t elements, size_t split_stride, int splits, double initial_scale)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    if (index >= elements) return;

    double residual = input[index];
    double scale = initial_scale;
    for (int split = 0; split < splits; ++split) {
        const int8_t q = quantize_one(residual, scale);
        output[static_cast<size_t>(split) * split_stride + index] = q;
        residual = fma(-static_cast<double>(q), scale, residual);
        scale /= 254.0;
    }
}

__global__ void recombine_diagonal(const int32_t* __restrict__ temp,
                                   double* __restrict__ C, double scale,
                                   size_t elements)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    if (index < elements) {
        C[index] = fma(scale, static_cast<double>(temp[index]), C[index]);
    }
}

template <int Splits>
void launch_quantizers(const double* dA, const double* dB,
                       int8_t* dAq_transposed, int8_t* dBq,
                       int M, int N, int K, size_t elements_a,
                       size_t elements_b, double scale_a, double scale_b,
                       cudaStream_t stream)
{
    const dim3 block_a(kTile, kBlockRows);
    const dim3 grid_a((M + kTile - 1) / kTile,
                      (K + kTile - 1) / kTile);
    quantize_a_transposed<Splits><<<grid_a, block_a, 0, stream>>>(
        dA, dAq_transposed, M, K, elements_a, scale_a);

    const int grid_b = static_cast<int>(
        (elements_b + kLinearBlock - 1) / kLinearBlock);
    quantize_b_regular<Splits><<<grid_b, kLinearBlock, 0, stream>>>(
        dB, dBq, elements_b, elements_b, scale_b);
    (void)N;
}

void launch_quantizers_dynamic(const double* dA, const double* dB,
                               int8_t* dAq_transposed, int8_t* dBq,
                               int M, int N, int K, size_t elements_a,
                               size_t elements_b, int splits,
                               double scale_a, double scale_b,
                               cudaStream_t stream)
{
    const int grid_a = static_cast<int>(
        (elements_a + kLinearBlock - 1) / kLinearBlock);
    const int grid_b = static_cast<int>(
        (elements_b + kLinearBlock - 1) / kLinearBlock);
    quantize_a_transposed_dynamic<<<grid_a, kLinearBlock, 0, stream>>>(
        dA, dAq_transposed, M, K, elements_a, splits, scale_a);
    quantize_b_regular_dynamic<<<grid_b, kLinearBlock, 0, stream>>>(
        dB, dBq, elements_b, elements_b, splits, scale_b);
    (void)N;
}

double scale_for_split(double initial_scale, int split)
{
    while (split-- > 0) initial_scale /= 254.0;
    return initial_scale;
}

}  // namespace

extern "C" int gemm_my_int8_fp64(int M, int N, int K,
                                  const double* dA, const double* dB,
                                  double* dC, int splits,
                                  cublasHandle_t handle,
                                  cudaStream_t stream)
{
    if (splits < 1) splits = 1;
    if (M < 0 || N < 0 || K < 0 || !dA || !dB || !dC) return 1;

    const size_t elements_a = static_cast<size_t>(M) * K;
    const size_t elements_b = static_cast<size_t>(K) * N;
    const size_t elements_c = static_cast<size_t>(M) * N;

    double max_a = device_maxabs_fp64(dA, elements_a);
    double max_b = device_maxabs_fp64(dB, elements_b);
    if (max_a == 0.0) max_a = 1.0;
    if (max_b == 0.0) max_b = 1.0;
    const double scale_a0 = max_a / 127.0;
    const double scale_b0 = max_b / 127.0;

    const size_t a_bytes = static_cast<size_t>(splits) * elements_a;
    const size_t b_bytes = static_cast<size_t>(splits) * elements_b;
    const size_t temp_bytes = elements_c * sizeof(int32_t);
    const size_t a_offset = 0;
    const size_t b_offset = align_up(a_offset + a_bytes, 256);
    const size_t temp_offset = align_up(b_offset + b_bytes, 256);
    const size_t workspace_bytes = temp_offset + temp_bytes;
    const int workspace_status = ensure_workspace(workspace_bytes, stream);
    if (workspace_status != 0) return workspace_status;

    auto* base = static_cast<unsigned char*>(g_workspace.base);
    auto* dAq_transposed = reinterpret_cast<int8_t*>(base + a_offset);
    auto* dBq = reinterpret_cast<int8_t*>(base + b_offset);
    auto* temp = reinterpret_cast<int32_t*>(base + temp_offset);

    switch (splits) {
        case 1:
            launch_quantizers<1>(dA, dB, dAq_transposed, dBq, M, N, K,
                                 elements_a, elements_b, scale_a0, scale_b0,
                                 stream);
            break;
        case 2:
            launch_quantizers<2>(dA, dB, dAq_transposed, dBq, M, N, K,
                                 elements_a, elements_b, scale_a0, scale_b0,
                                 stream);
            break;
        case 3:
            launch_quantizers<3>(dA, dB, dAq_transposed, dBq, M, N, K,
                                 elements_a, elements_b, scale_a0, scale_b0,
                                 stream);
            break;
        case 4:
            launch_quantizers<4>(dA, dB, dAq_transposed, dBq, M, N, K,
                                 elements_a, elements_b, scale_a0, scale_b0,
                                 stream);
            break;
        case 5:
            launch_quantizers<5>(dA, dB, dAq_transposed, dBq, M, N, K,
                                 elements_a, elements_b, scale_a0, scale_b0,
                                 stream);
            break;
        case 6:
            launch_quantizers<6>(dA, dB, dAq_transposed, dBq, M, N, K,
                                 elements_a, elements_b, scale_a0, scale_b0,
                                 stream);
            break;
        case 7:
            launch_quantizers<7>(dA, dB, dAq_transposed, dBq, M, N, K,
                                 elements_a, elements_b, scale_a0, scale_b0,
                                 stream);
            break;
        case 8:
            launch_quantizers<8>(dA, dB, dAq_transposed, dBq, M, N, K,
                                 elements_a, elements_b, scale_a0, scale_b0,
                                 stream);
            break;
        default:
            launch_quantizers_dynamic(
                dA, dB, dAq_transposed, dBq, M, N, K, elements_a,
                elements_b, splits, scale_a0, scale_b0, stream);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(dC, 0, elements_c * sizeof(double), stream));

    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    const int32_t alpha = 1;
    const int32_t beta_zero = 0;
    const int32_t beta_one = 1;
    const int recombine_grid = static_cast<int>(
        (elements_c + kLinearBlock - 1) / kLinearBlock);

    const uint64_t max_diagonal_terms = static_cast<uint64_t>(splits);
    const uint64_t max_accumulator =
        max_diagonal_terms * static_cast<uint64_t>(K) * 127u * 127u;
    const bool diagonal_accumulation_safe =
        max_accumulator <= static_cast<uint64_t>(
                               std::numeric_limits<int32_t>::max());

    if (diagonal_accumulation_safe) {
        double diagonal_scale = scale_a0 * scale_b0;
        const int diagonal_limit = splits < 6 ? splits : 6;
        for (int diagonal = 0; diagonal < diagonal_limit; ++diagonal) {
            const int begin_i = diagonal < splits ? 0 : diagonal - splits + 1;
            const int end_i = diagonal < splits ? diagonal : splits - 1;
            bool first = true;
            for (int i = begin_i; i <= end_i; ++i) {
                const int j = diagonal - i;
                const int32_t* beta = first ? &beta_zero : &beta_one;
                CUBLAS_CHECK(cublasGemmEx(
                    handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha,
                    dAq_transposed + static_cast<size_t>(i) * elements_a,
                    CUDA_R_8I, K,
                    dBq + static_cast<size_t>(j) * elements_b, CUDA_R_8I, K,
                    beta, temp, CUDA_R_32I, M, CUBLAS_COMPUTE_32I,
                    CUBLAS_GEMM_DEFAULT));
                first = false;
            }
            recombine_diagonal<<<recombine_grid, kLinearBlock, 0, stream>>>(
                temp, dC, diagonal_scale, elements_c);
            diagonal_scale /= 254.0;
        }
    } else {
        for (int i = 0; i < splits; ++i) {
            const double scale_a = scale_for_split(scale_a0, i);
            for (int j = 0; j < splits; ++j) {
                const double scale_b = scale_for_split(scale_b0, j);
                CUBLAS_CHECK(cublasGemmEx(
                    handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha,
                    dAq_transposed + static_cast<size_t>(i) * elements_a,
                    CUDA_R_8I, K,
                    dBq + static_cast<size_t>(j) * elements_b, CUDA_R_8I, K,
                    &beta_zero, temp, CUDA_R_32I, M, CUBLAS_COMPUTE_32I,
                    CUBLAS_GEMM_DEFAULT));
                recombine_diagonal<<<recombine_grid, kLinearBlock, 0, stream>>>(
                    temp, dC, scale_a * scale_b, elements_c);
            }
        }
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
}
