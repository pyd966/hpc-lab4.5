#include "gemm_api.h"
#include "utils.h"

#include <cstdio>
#include <vector>

int main() {
    constexpr int size = 1024;
    constexpr int splits = 4;
    const size_t elements = static_cast<size_t>(size) * size;
    std::vector<double> host_a(elements);
    std::vector<double> host_b(elements);
    fill_matrix(host_a.data(), elements, 0x1234);
    fill_matrix(host_b.data(), elements, 0x5678);

    double *a = nullptr, *b = nullptr, *reference = nullptr;
    double *output_a = nullptr, *output_b = nullptr;
    CUDA_CHECK(cudaMalloc(&a, elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&b, elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&reference, elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&output_a, elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&output_b, elements * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(a, host_a.data(), elements * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, host_b.data(), elements * sizeof(double),
                          cudaMemcpyHostToDevice));

    cublasHandle_t handle = nullptr;
    cudaStream_t stream_a = nullptr, stream_b = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaStreamCreate(&stream_a));
    CUDA_CHECK(cudaStreamCreate(&stream_b));
    CUBLAS_CHECK(cublasSetStream(handle, stream_a));
    const double alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             size, size, size, &alpha, a, size, b, size,
                             &beta, reference, size));
    CUDA_CHECK(cudaStreamSynchronize(stream_a));

    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    int status = gemm_my_int8_fp64(size, size, size, a, b, output_a,
                                   splits, handle, stream_a);
    if (status != 0) return status;
    status = gemm_my_int8_fp64(size, size, size, a, b, output_b,
                               splits, handle, stream_b);
    if (status != 0) return status;
    CUDA_CHECK(cudaStreamSynchronize(stream_a));
    CUDA_CHECK(cudaStreamSynchronize(stream_b));

    AccuracyReport report_a{}, report_b{};
    accuracy_compare(reference, output_a, size, size, &report_a);
    accuracy_compare(reference, output_b, size, size, &report_b);
    std::printf("stream_a_l2=%.9e stream_b_l2=%.9e\n",
                report_a.l2_rel_err, report_b.l2_rel_err);
    const bool passed = report_a.l2_rel_err < 1.0e-8 &&
                        report_b.l2_rel_err < 1.0e-8;
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream_a));
    CUDA_CHECK(cudaStreamDestroy(stream_b));
    CUDA_CHECK(cudaFree(output_b));
    CUDA_CHECK(cudaFree(output_a));
    CUDA_CHECK(cudaFree(reference));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(a));
    return passed ? 0 : 2;
}
