#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_OK(expr)                                                        \
    do {                                                                     \
        cudaError_t status = (expr);                                         \
        if (status != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA failure at %s:%d: %s\n", __FILE__,  \
                         __LINE__, cudaGetErrorString(status));              \
            return 1;                                                        \
        }                                                                    \
    } while (0)

#define CUBLAS_OK(expr)                                                      \
    do {                                                                     \
        cublasStatus_t status = (expr);                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                              \
            std::fprintf(stderr, "cuBLAS failure at %s:%d: %d\n",          \
                         __FILE__, __LINE__, static_cast<int>(status));       \
            return 1;                                                        \
        }                                                                    \
    } while (0)

static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop, int iters) {
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    return ms / iters;
}

int main(int argc, char** argv) {
    const int size = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 10;
    const int M = size, N = size, K = size;
    const size_t elems_a = static_cast<size_t>(M) * K;
    const size_t elems_b = static_cast<size_t>(K) * N;
    const size_t elems_c = static_cast<size_t>(M) * N;

    int8_t* A = nullptr;
    int8_t* B = nullptr;
    int32_t* C = nullptr;
    void* workspace = nullptr;
    constexpr size_t workspace_bytes = 64ull << 20;
    CUDA_OK(cudaMalloc(&A, elems_a));
    CUDA_OK(cudaMalloc(&B, elems_b));
    CUDA_OK(cudaMalloc(&C, elems_c * sizeof(int32_t)));
    CUDA_OK(cudaMalloc(&workspace, workspace_bytes));
    CUDA_OK(cudaMemset(A, 1, elems_a));
    CUDA_OK(cudaMemset(B, 1, elems_b));

    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));

    cublasHandle_t blas = nullptr;
    CUBLAS_OK(cublasCreate(&blas));
    const int32_t alpha = 1;
    const int32_t beta = 0;

    auto gemmex_nn = [&]() {
        return cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                            &alpha, A, CUDA_R_8I, M, B, CUDA_R_8I, K,
                            &beta, C, CUDA_R_32I, M, CUBLAS_COMPUTE_32I,
                            CUBLAS_GEMM_DEFAULT);
    };

    CUBLAS_OK(gemmex_nn());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) CUBLAS_OK(gemmex_nn());
    const float gemmex_ms = elapsed_ms(start, stop, iters);

    auto gemmex_tn = [&]() {
        return cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
                            &alpha, A, CUDA_R_8I, K, B, CUDA_R_8I, K,
                            &beta, C, CUDA_R_32I, M, CUBLAS_COMPUTE_32I,
                            CUBLAS_GEMM_DEFAULT);
    };

    CUBLAS_OK(gemmex_tn());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) CUBLAS_OK(gemmex_tn());
    const float gemmex_tn_ms = elapsed_ms(start, stop, iters);

    auto gemmex_tn_autotune = [&]() {
        return cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
                            &alpha, A, CUDA_R_8I, K, B, CUDA_R_8I, K,
                            &beta, C, CUDA_R_32I, M, CUBLAS_COMPUTE_32I,
                            CUBLAS_GEMM_AUTOTUNE);
    };

    CUBLAS_OK(gemmex_tn_autotune());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) CUBLAS_OK(gemmex_tn_autotune());
    const float gemmex_tn_autotune_ms = elapsed_ms(start, stop, iters);

    cublasLtHandle_t lt = nullptr;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t layout_a = nullptr;
    cublasLtMatrixLayout_t layout_b = nullptr;
    cublasLtMatrixLayout_t layout_c = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    CUBLAS_OK(cublasLtCreate(&lt));
    CUBLAS_OK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I,
                                      CUDA_R_32I));

    // A is treated as a column-major KxM buffer containing the physical
    // transpose of the original MxK matrix. A production quantizer can emit
    // this layout directly. This gives the regular TN layout required by the
    // cuBLASLt IMMA path without a separate transpose kernel.
    cublasOperation_t trans_a = CUBLAS_OP_T;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    CUBLAS_OK(cublasLtMatmulDescSetAttribute(
        op, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    CUBLAS_OK(cublasLtMatmulDescSetAttribute(
        op, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

    CUBLAS_OK(cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_8I, K, M, K));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_8I, K, N, K));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_32I, M, N, M));

    CUBLAS_OK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_OK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes, sizeof(workspace_bytes)));
    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    CUBLAS_OK(cublasLtMatmulAlgoGetHeuristic(
        lt, op, layout_a, layout_b, layout_c, layout_c, preference, 1,
        &heuristic, &returned));
    if (returned != 1) {
        std::fprintf(stderr, "No cuBLASLt algorithm supports the TN layout\n");
        return 2;
    }

    auto cublaslt_tn = [&]() {
        return cublasLtMatmul(lt, op, &alpha, A, layout_a, B, layout_b,
                              &beta, C, layout_c, C, layout_c,
                              &heuristic.algo, workspace, workspace_bytes, 0);
    };

    CUBLAS_OK(cublaslt_tn());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) CUBLAS_OK(cublaslt_tn());
    const float cublaslt_ms = elapsed_ms(start, stop, iters);

    int32_t first = 0;
    CUDA_OK(cudaMemcpy(&first, C, sizeof(first), cudaMemcpyDeviceToHost));
    if (first != K) {
        std::fprintf(stderr, "Incorrect result: C[0]=%d, expected %d\n", first,
                     K);
        return 3;
    }

    const double pair_ops = 2.0 * M * N * static_cast<double>(K);
    std::printf("size=%d iters=%d expected=%d\n", size, iters, first);
    std::printf("cublasGemmEx_NN_ms=%.4f int8_TOPS=%.2f\n", gemmex_ms,
                pair_ops / (gemmex_ms * 1.0e9));
    std::printf("cublasGemmEx_TN_ms=%.4f int8_TOPS=%.2f speedup=%.2fx\n",
                gemmex_tn_ms, pair_ops / (gemmex_tn_ms * 1.0e9),
                gemmex_ms / gemmex_tn_ms);
    std::printf("cublasGemmEx_TN_autotune_ms=%.4f int8_TOPS=%.2f "
                "speedup=%.2fx\n", gemmex_tn_autotune_ms,
                pair_ops / (gemmex_tn_autotune_ms * 1.0e9),
                gemmex_ms / gemmex_tn_autotune_ms);
    std::printf("cublasLt_TN_col_ms=%.4f int8_TOPS=%.2f speedup=%.2fx\n",
                cublaslt_ms, pair_ops / (cublaslt_ms * 1.0e9),
                gemmex_ms / cublaslt_ms);

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatmulDescDestroy(op);
    cublasLtDestroy(lt);
    cublasDestroy(blas);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(workspace);
    cudaFree(C);
    cudaFree(B);
    cudaFree(A);
    return 0;
}
