#include "gemm_api.h"
#include "utils.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kMaxSplits = 8;
constexpr int kTile = 32;
constexpr int kBlockRows = 8;
constexpr size_t kVendorWorkspaceBytes =
    size_t{2} * 1024 * 1024 * 1024;

constexpr int kScaleA = 0;
constexpr int kInvScaleA = kScaleA + kMaxSplits;
constexpr int kScaleB = kInvScaleA + kMaxSplits;
constexpr int kInvScaleB = kScaleB + kMaxSplits;
constexpr int kDiagonalScale = kInvScaleB + kMaxSplits;
constexpr int kScaleCount = kDiagonalScale + 2 * kMaxSplits - 1;

struct Workspace {
    int device = -1;
    int compute_major = -1;
    int8_t* aq = nullptr;
    int8_t* bq = nullptr;
    int32_t* diagonal = nullptr;
    unsigned long long* max_bits = nullptr;
    double* scales = nullptr;
    void* vendor = nullptr;
    cudaEvent_t ready = nullptr;
    cudaStream_t last_stream = nullptr;
    bool event_recorded = false;
    size_t aq_capacity = 0;
    size_t bq_capacity = 0;
    size_t diagonal_capacity = 0;
    size_t vendor_capacity = 0;
};

thread_local Workspace workspace;

int initialize_device() {
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return static_cast<int>(status);
    if (workspace.device != -1 && workspace.device != device) {
        return static_cast<int>(cudaErrorInvalidDevice);
    }
    if (workspace.device == -1) {
        cudaDeviceProp properties{};
        status = cudaGetDeviceProperties(&properties, device);
        if (status != cudaSuccess) return static_cast<int>(status);
        workspace.device = device;
        workspace.compute_major = properties.major;
    }
    return 0;
}

int begin_workspace_use(cudaStream_t stream) {
    if (workspace.ready == nullptr) {
        cudaError_t status = cudaEventCreateWithFlags(
            &workspace.ready, cudaEventDisableTiming);
        if (status != cudaSuccess) return static_cast<int>(status);
    }
    if (workspace.event_recorded && workspace.last_stream != stream) {
        cudaError_t status = cudaStreamWaitEvent(stream, workspace.ready, 0);
        if (status != cudaSuccess) return static_cast<int>(status);
    }
    workspace.last_stream = stream;
    return 0;
}

int finish_workspace_use(cudaStream_t stream, int operation_result) {
    cudaError_t status = cudaEventRecord(workspace.ready, stream);
    workspace.event_recorded = status == cudaSuccess;
    if (operation_result != 0) return operation_result;
    return status == cudaSuccess ? 0 : static_cast<int>(status);
}

int grow_buffer(void** pointer, size_t* capacity, size_t bytes) {
    if (*capacity >= bytes) return 0;
    if (*pointer != nullptr) {
        cudaError_t status = cudaFree(*pointer);
        if (status != cudaSuccess) return static_cast<int>(status);
        *pointer = nullptr;
        *capacity = 0;
    }
    cudaError_t status = cudaMalloc(pointer, bytes);
    if (status != cudaSuccess) return static_cast<int>(status);
    *capacity = bytes;
    return 0;
}

int reserve_manual_workspace(size_t aq_bytes, size_t bq_bytes,
                             size_t diagonal_bytes) {
    int result = grow_buffer(reinterpret_cast<void**>(&workspace.aq),
                             &workspace.aq_capacity, aq_bytes);
    if (result != 0) return result;
    result = grow_buffer(reinterpret_cast<void**>(&workspace.bq),
                         &workspace.bq_capacity, bq_bytes);
    if (result != 0) return result;
    result = grow_buffer(reinterpret_cast<void**>(&workspace.diagonal),
                         &workspace.diagonal_capacity, diagonal_bytes);
    if (result != 0) return result;
    if (workspace.max_bits == nullptr) {
        cudaError_t status = cudaMalloc(&workspace.max_bits,
                                        2 * sizeof(unsigned long long));
        if (status != cudaSuccess) return static_cast<int>(status);
    }
    if (workspace.scales == nullptr) {
        cudaError_t status = cudaMalloc(&workspace.scales,
                                        kScaleCount * sizeof(double));
        if (status != cudaSuccess) return static_cast<int>(status);
    }
    return 0;
}

int reserve_vendor_workspace() {
    return grow_buffer(&workspace.vendor, &workspace.vendor_capacity,
                       kVendorWorkspaceBytes);
}

template <int BlockSize>
__global__ void maxabs_pair_kernel(const double* a, size_t elements_a,
                                   const double* b, size_t elements_b,
                                   unsigned long long* max_bits) {
    __shared__ double max_a[BlockSize];
    __shared__ double max_b[BlockSize];
    const size_t start = static_cast<size_t>(blockIdx.x) * BlockSize
                       + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * BlockSize;
    const size_t elements = elements_a > elements_b ? elements_a : elements_b;
    double local_a = 0.0;
    double local_b = 0.0;
    for (size_t index = start; index < elements; index += stride) {
        if (index < elements_a) local_a = fmax(local_a, fabs(a[index]));
        if (index < elements_b) local_b = fmax(local_b, fabs(b[index]));
    }
    max_a[threadIdx.x] = local_a;
    max_b[threadIdx.x] = local_b;
    __syncthreads();
    for (int offset = BlockSize / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            max_a[threadIdx.x] = fmax(max_a[threadIdx.x],
                                      max_a[threadIdx.x + offset]);
            max_b[threadIdx.x] = fmax(max_b[threadIdx.x],
                                      max_b[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMax(max_bits, static_cast<unsigned long long>(
                                __double_as_longlong(max_a[0])));
        atomicMax(max_bits + 1, static_cast<unsigned long long>(
                                    __double_as_longlong(max_b[0])));
    }
}

__global__ void prepare_scales_kernel(const unsigned long long* max_bits,
                                      double* scales) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double max_a = __longlong_as_double(static_cast<long long>(max_bits[0]));
    double max_b = __longlong_as_double(static_cast<long long>(max_bits[1]));
    if (max_a == 0.0) max_a = 1.0;
    if (max_b == 0.0) max_b = 1.0;

    double scale_a = max_a / 127.0;
    double scale_b = max_b / 127.0;
    double inverse_a = 1.0 / scale_a;
    double inverse_b = 1.0 / scale_b;
#pragma unroll
    for (int split = 0; split < kMaxSplits; ++split) {
        scales[kScaleA + split] = scale_a;
        scales[kInvScaleA + split] = inverse_a;
        scales[kScaleB + split] = scale_b;
        scales[kInvScaleB + split] = inverse_b;
        scale_a /= 254.0;
        scale_b /= 254.0;
        inverse_a *= 254.0;
        inverse_b *= 254.0;
    }

    double diagonal_scale = scales[kScaleA] * scales[kScaleB];
#pragma unroll
    for (int diagonal = 0; diagonal < 2 * kMaxSplits - 1; ++diagonal) {
        scales[kDiagonalScale + diagonal] = diagonal_scale;
        diagonal_scale /= 254.0;
    }
}

__device__ __forceinline__ int8_t quantize_digit(double residual,
                                                  double inverse_scale) {
    int digit = __double2int_rn(residual * inverse_scale);
    digit = digit > 127 ? 127 : digit;
    digit = digit < -127 ? -127 : digit;
    return static_cast<int8_t>(digit);
}

/*
 * Emit A as a column-major (S*K)-by-M matrix. Reversing split slots makes
 * every retained anti-diagonal addressable with the same leading dimension.
 */
template <int Splits>
__global__ void quantize_a_kernel(const double* input, int8_t* output,
                                  const double* scales, int M, int K) {
    __shared__ int8_t tile[Splits][kTile][kTile + 1];
    const int input_m = blockIdx.x * kTile + threadIdx.x;
    const int input_k = blockIdx.y * kTile + threadIdx.y;

#pragma unroll
    for (int offset = 0; offset < kTile; offset += kBlockRows) {
        const int k = input_k + offset;
        if (input_m < M && k < K) {
            double residual = input[static_cast<size_t>(k) * M + input_m];
#pragma unroll
            for (int split = 0; split < Splits; ++split) {
                const double scale = scales[kScaleA + split];
                const int8_t digit = quantize_digit(
                    residual, scales[kInvScaleA + split]);
                tile[split][threadIdx.y + offset][threadIdx.x] = digit;
                residual = fma(-static_cast<double>(digit), scale, residual);
            }
        }
    }
    __syncthreads();

    const int output_k = blockIdx.y * kTile + threadIdx.x;
    const int output_m = blockIdx.x * kTile + threadIdx.y;
    const size_t extended_k = static_cast<size_t>(Splits) * K;
#pragma unroll
    for (int offset = 0; offset < kTile; offset += kBlockRows) {
        const int m = output_m + offset;
        if (m < M && output_k < K) {
#pragma unroll
            for (int split = 0; split < Splits; ++split) {
                const int slot = Splits - 1 - split;
                output[static_cast<size_t>(m) * extended_k
                       + static_cast<size_t>(slot) * K + output_k] =
                    tile[split][threadIdx.x][threadIdx.y + offset];
            }
        }
    }
}

template <int Splits>
__global__ void quantize_b_kernel(const double* input, int8_t* output,
                                  const double* scales, size_t elements,
                                  int K) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x
                       + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t extended_k = static_cast<size_t>(Splits) * K;
    for (size_t index = start; index < elements; index += stride) {
        const int k = static_cast<int>(index % K);
        const size_t n = index / K;
        double residual = input[index];
#pragma unroll
        for (int split = 0; split < Splits; ++split) {
            const double scale = scales[kScaleB + split];
            const int8_t digit = quantize_digit(
                residual, scales[kInvScaleB + split]);
            output[n * extended_k + static_cast<size_t>(split) * K + k] =
                digit;
            residual = fma(-static_cast<double>(digit), scale, residual);
        }
    }
}

template <int Diagonals>
__global__ void recombine_kernel(const int32_t* partial, double* output,
                                 const double* scales, size_t elements) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x
                       + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = start; index < elements; index += stride) {
        double value = 0.0;
#pragma unroll
        for (int diagonal = 0; diagonal < Diagonals; ++diagonal) {
            value = fma(static_cast<double>(
                            partial[static_cast<size_t>(diagonal) * elements
                                    + index]),
                        scales[kDiagonalScale + diagonal], value);
        }
        output[index] = value;
    }
}

template <int Splits>
constexpr int retained_diagonals() {
    return Splits < 6 ? Splits : 6;
}

template <int Splits>
int run_manual(int M, int N, int K, const double* dA, const double* dB,
               double* dC, cublasHandle_t handle, cudaStream_t stream) {
    constexpr int Diagonals = retained_diagonals<Splits>();
    const size_t elements_a = static_cast<size_t>(M) * K;
    const size_t elements_b = static_cast<size_t>(K) * N;
    const size_t elements_c = static_cast<size_t>(M) * N;
    const size_t aq_bytes = static_cast<size_t>(Splits) * elements_a;
    const size_t bq_bytes = static_cast<size_t>(Splits) * elements_b;
    const size_t diagonal_bytes = static_cast<size_t>(Diagonals) * elements_c
                                * sizeof(int32_t);
    int result = reserve_manual_workspace(aq_bytes, bq_bytes,
                                          diagonal_bytes);
    if (result != 0) return result;

    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    CUDA_CHECK(cudaMemsetAsync(workspace.max_bits, 0,
                               2 * sizeof(unsigned long long), stream));

    constexpr int block = 256;
    const size_t max_elements = elements_a > elements_b
                              ? elements_a : elements_b;
    int reduction_grid = static_cast<int>((max_elements + block - 1) / block);
    if (reduction_grid > 1024) reduction_grid = 1024;
    maxabs_pair_kernel<block><<<reduction_grid, block, 0, stream>>>(
        dA, elements_a, dB, elements_b, workspace.max_bits);
    prepare_scales_kernel<<<1, 1, 0, stream>>>(workspace.max_bits,
                                               workspace.scales);

    const dim3 a_block(kTile, kBlockRows);
    const dim3 a_grid((M + kTile - 1) / kTile,
                      (K + kTile - 1) / kTile);
    quantize_a_kernel<Splits><<<a_grid, a_block, 0, stream>>>(
        dA, workspace.aq, workspace.scales, M, K);
    int b_grid = static_cast<int>((elements_b + block - 1) / block);
    if (b_grid > 4096) b_grid = 4096;
    quantize_b_kernel<Splits><<<b_grid, block, 0, stream>>>(
        dB, workspace.bq, workspace.scales, elements_b, K);
    CUDA_CHECK(cudaPeekAtLastError());

    const int32_t alpha = 1;
    const int32_t beta_zero = 0;
    const int32_t beta_one = 1;
    const int extended_k = Splits * K;
    for (int diagonal = 0; diagonal < Diagonals; ++diagonal) {
        const int j_low = diagonal >= Splits ? diagonal - Splits + 1 : 0;
        const int j_high = diagonal < Splits ? diagonal : Splits - 1;
        int32_t* output = workspace.diagonal
                        + static_cast<size_t>(diagonal) * elements_c;
        bool first = true;
        for (int j = j_low; j <= j_high; ++j) {
            const int i = diagonal - j;
            const int a_slot = Splits - 1 - i;
            const int32_t* beta = first ? &beta_zero : &beta_one;
            CUBLAS_CHECK(cublasGemmEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha,
                workspace.aq + static_cast<size_t>(a_slot) * K,
                CUDA_R_8I, extended_k,
                workspace.bq + static_cast<size_t>(j) * K,
                CUDA_R_8I, extended_k, beta, output, CUDA_R_32I, M,
                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
            first = false;
        }
    }

    int output_grid = static_cast<int>((elements_c + block - 1) / block);
    if (output_grid > 4096) output_grid = 4096;
    recombine_kernel<Diagonals><<<output_grid, block, 0, stream>>>(
        workspace.diagonal, dC, workspace.scales, elements_c);
    CUDA_CHECK(cudaPeekAtLastError());
    return 0;
}

int run_vendor(int M, int N, int K, const double* dA, const double* dB,
               double* dC, int splits, cublasHandle_t handle,
               cudaStream_t stream) {
    int result = reserve_vendor_workspace();
    if (result != 0) return result;

    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetWorkspace(handle, workspace.vendor,
                                    kVendorWorkspaceBytes));
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasSetMathMode(
        handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
    CUBLAS_CHECK(cublasSetEmulationStrategy(
        handle, CUBLAS_EMULATION_STRATEGY_EAGER));
    CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(
        handle, CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
    int max_mantissa_bits = 8 * splits;
    if (max_mantissa_bits > 55) max_mantissa_bits = 55;
    CUBLAS_CHECK(cublasSetFixedPointEmulationMaxMantissaBitCount(
        handle, max_mantissa_bits));

    const double alpha = 1.0;
    const double beta = 0.0;
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
        dA, CUDA_R_64F, M, dB, CUDA_R_64F, K, &beta, dC, CUDA_R_64F, M,
        CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_GEMM_DEFAULT));
    return 0;
}

}  // namespace

extern "C" int gemm_my_int8_fp64(
    int M, int N, int K, const double* dA, const double* dB, double* dC,
    int splits, cublasHandle_t handle, cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0 || dA == nullptr || dB == nullptr ||
        dC == nullptr || handle == nullptr || splits < 1 ||
        splits > kMaxSplits) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    int result = initialize_device();
    if (result != 0) return result;
    result = begin_workspace_use(stream);
    if (result != 0) return result;

    // lab3 is Hopper; retain the independently measured Ampere fallback for
    // compatibility with lab4g10 and other CC 8.x devices.
    if (workspace.compute_major >= 9) {
        result = run_vendor(M, N, K, dA, dB, dC, splits, handle, stream);
        return finish_workspace_use(stream, result);
    }

    switch (splits) {
        case 1: result = run_manual<1>(M, N, K, dA, dB, dC, handle, stream); break;
        case 2: result = run_manual<2>(M, N, K, dA, dB, dC, handle, stream); break;
        case 3: result = run_manual<3>(M, N, K, dA, dB, dC, handle, stream); break;
        case 4: result = run_manual<4>(M, N, K, dA, dB, dC, handle, stream); break;
        case 5: result = run_manual<5>(M, N, K, dA, dB, dC, handle, stream); break;
        case 6: result = run_manual<6>(M, N, K, dA, dB, dC, handle, stream); break;
        case 7: result = run_manual<7>(M, N, K, dA, dB, dC, handle, stream); break;
        case 8: result = run_manual<8>(M, N, K, dA, dB, dC, handle, stream); break;
        default: result = static_cast<int>(cudaErrorInvalidValue); break;
    }
    return finish_workspace_use(stream, result);
}
