#include "gemm_api.h"
#include "utils.h"

#include <cstddef>

namespace {

constexpr size_t kWorkspaceBytes = size_t{2} * 1024 * 1024 * 1024;

struct PersistentWorkspace {
    int device = -1;
    void* pointer = nullptr;
};

thread_local PersistentWorkspace workspace;

int reserve_workspace() {
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return static_cast<int>(status);
    if (workspace.device != -1 && workspace.device != device) {
        return static_cast<int>(cudaErrorInvalidDevice);
    }
    workspace.device = device;
    if (workspace.pointer == nullptr) {
        status = cudaMalloc(&workspace.pointer, kWorkspaceBytes);
        if (status != cudaSuccess) return static_cast<int>(status);
    }
    return 0;
}

}  // namespace

int gemm_my_int8_fp64(int M, int N, int K,
                      const double* dA, const double* dB, double* dC,
                      int splits, cublasHandle_t handle, cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0 || dA == nullptr || dB == nullptr ||
        dC == nullptr || handle == nullptr || splits < 1) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int workspace_status = reserve_workspace();
    if (workspace_status != 0) return workspace_status;

    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetWorkspace(handle, workspace.pointer,
                                    kWorkspaceBytes));
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
        dA, CUDA_R_64F, M, dB, CUDA_R_64F, K, &beta,
        dC, CUDA_R_64F, M, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT,
        CUBLAS_GEMM_DEFAULT));
    return 0;
}
