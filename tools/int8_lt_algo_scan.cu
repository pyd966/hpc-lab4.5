#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_OK(call)                                                        \
    do {                                                                     \
        cudaError_t status_ = (call);                                        \
        if (status_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,  \
                         cudaGetErrorString(status_));                        \
            return 1;                                                        \
        }                                                                    \
    } while (0)

#define CUBLAS_OK(call)                                                      \
    do {                                                                     \
        cublasStatus_t status_ = (call);                                     \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                              \
            std::fprintf(stderr, "cuBLAS %s:%d: %d\n", __FILE__,          \
                         __LINE__, static_cast<int>(status_));                \
            return 1;                                                        \
        }                                                                    \
    } while (0)

static int config_value(const cublasLtMatmulAlgo_t& algorithm,
                        cublasLtMatmulAlgoConfigAttributes_t attribute) {
    int value = -1;
    size_t written = 0;
    cublasStatus_t status = cublasLtMatmulAlgoConfigGetAttribute(
        &algorithm, attribute, &value, sizeof(value), &written);
    return status == CUBLAS_STATUS_SUCCESS ? value : -1;
}

int main(int argc, char** argv) {
    const int size = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 5;
    const size_t elements = static_cast<size_t>(size) * size;
    constexpr size_t workspace_bytes = size_t{256} << 20;

    int8_t* a = nullptr;
    int8_t* b = nullptr;
    int32_t* c = nullptr;
    void* workspace = nullptr;
    CUDA_OK(cudaMalloc(&a, elements));
    CUDA_OK(cudaMalloc(&b, elements));
    CUDA_OK(cudaMalloc(&c, elements * sizeof(int32_t)));
    CUDA_OK(cudaMalloc(&workspace, workspace_bytes));
    CUDA_OK(cudaMemset(a, 1, elements));
    CUDA_OK(cudaMemset(b, 1, elements));

    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t layout_a = nullptr;
    cublasLtMatrixLayout_t layout_b = nullptr;
    cublasLtMatrixLayout_t layout_c = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    CUBLAS_OK(cublasLtCreate(&handle));
    CUBLAS_OK(cublasLtMatmulDescCreate(
        &operation, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    const cublasOperation_t transpose = CUBLAS_OP_T;
    const cublasOperation_t no_transpose = CUBLAS_OP_N;
    CUBLAS_OK(cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_TRANSA,
        &transpose, sizeof(transpose)));
    CUBLAS_OK(cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_TRANSB,
        &no_transpose, sizeof(no_transpose)));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(
        &layout_a, CUDA_R_8I, size, size, size));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(
        &layout_b, CUDA_R_8I, size, size, size));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(
        &layout_c, CUDA_R_32I, size, size, size));
    CUBLAS_OK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_OK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes, sizeof(workspace_bytes)));

    constexpr int requested = 64;
    std::vector<cublasLtMatmulHeuristicResult_t> heuristics(requested);
    int returned = 0;
    CUBLAS_OK(cublasLtMatmulAlgoGetHeuristic(
        handle, operation, layout_a, layout_b, layout_c, layout_c,
        preference, requested, heuristics.data(), &returned));
    std::printf("size=%d returned=%d workspace_mib=%zu\n",
                size, returned, workspace_bytes >> 20);

    const int32_t alpha = 1;
    const int32_t beta = 0;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    struct Result { int index; float milliseconds; };
    std::vector<Result> results;

    for (int index = 0; index < returned; ++index) {
        const auto& heuristic = heuristics[index];
        if (heuristic.state != CUBLAS_STATUS_SUCCESS ||
            heuristic.workspaceSize > workspace_bytes) {
            continue;
        }
        cublasStatus_t status = cublasLtMatmul(
            handle, operation, &alpha,
            a, layout_a, b, layout_b, &beta,
            c, layout_c, c, layout_c, &heuristic.algo,
            workspace, workspace_bytes, nullptr);
        if (status != CUBLAS_STATUS_SUCCESS) continue;
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaEventRecord(start));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            CUBLAS_OK(cublasLtMatmul(
                handle, operation, &alpha,
                a, layout_a, b, layout_b, &beta,
                c, layout_c, c, layout_c, &heuristic.algo,
                workspace, workspace_bytes, nullptr));
        }
        CUDA_OK(cudaEventRecord(stop));
        CUDA_OK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CUDA_OK(cudaEventElapsedTime(&elapsed, start, stop));
        results.push_back({index, elapsed / iterations});
    }

    std::sort(results.begin(), results.end(),
              [](const Result& left, const Result& right) {
                  return left.milliseconds < right.milliseconds;
              });
    const double operations = 2.0 * size * static_cast<double>(size) * size;
    for (size_t rank = 0; rank < results.size(); ++rank) {
        const Result& result = results[rank];
        const auto& algorithm = heuristics[result.index].algo;
        std::printf(
            "rank=%zu heuristic=%d ms=%.6f tops=%.2f id=%d tile=%d "
            "stages=%d splitk=%d reduction=%d swizzle=%d custom=%d ws=%zu\n",
            rank, result.index, result.milliseconds,
            operations / (result.milliseconds * 1.0e9),
            config_value(algorithm, CUBLASLT_ALGO_CONFIG_ID),
            config_value(algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID),
            config_value(algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID),
            config_value(algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
            config_value(algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME),
            config_value(algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING),
            config_value(algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION),
            heuristics[result.index].workspaceSize);
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatmulDescDestroy(operation);
    cublasLtDestroy(handle);
    cudaFree(workspace);
    cudaFree(c);
    cudaFree(b);
    cudaFree(a);
    return 0;
}
