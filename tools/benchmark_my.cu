#include "gemm_api.h"
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::vector<int> parse_values(const char* text) {
    std::vector<int> values;
    const char* cursor = text;
    while (*cursor != '\0') {
        char* end = nullptr;
        values.push_back(static_cast<int>(std::strtol(cursor, &end, 10)));
        cursor = end;
        if (*cursor == ',') ++cursor;
    }
    return values;
}

int main(int argc, char** argv) {
    const std::vector<int> sizes = parse_values(argc > 1 ? argv[1] : "4096");
    const std::vector<int> split_values =
        parse_values(argc > 2 ? argv[2] : "2,4,6,8");
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 3;

    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    std::printf("size,splits,time_ms,gflops,max_abs_err,l2_rel_err\n");
    for (int size : sizes) {
        const size_t elements = static_cast<size_t>(size) * size;
        std::vector<double> host_a(elements);
        std::vector<double> host_b(elements);
        fill_matrix(host_a.data(), elements, 0x1234ULL + 7 * size);
        fill_matrix(host_b.data(), elements, 0x5678ULL + 7 * size);

        double* a = nullptr;
        double* b = nullptr;
        double* reference = nullptr;
        double* output = nullptr;
        CUDA_CHECK(cudaMalloc(&a, elements * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&b, elements * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&reference, elements * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&output, elements * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(a, host_a.data(), elements * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b, host_b.data(), elements * sizeof(double),
                              cudaMemcpyHostToDevice));
        const double alpha = 1.0;
        const double beta = 0.0;
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 size, size, size, &alpha, a, size, b, size,
                                 &beta, reference, size));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int splits : split_values) {
            int status = gemm_my_int8_fp64(size, size, size, a, b, output,
                                           splits, handle, stream);
            if (status != 0) {
                std::fprintf(stderr, "warmup failed: size=%d splits=%d status=%d\n",
                             size, splits, status);
                return status;
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CudaTimer timer;
            timer.begin(stream);
            for (int iteration = 0; iteration < iterations; ++iteration) {
                status = gemm_my_int8_fp64(size, size, size, a, b, output,
                                           splits, handle, stream);
                if (status != 0) return status;
            }
            timer.end(stream);
            const double milliseconds = timer.elapsed_ms() / iterations;
            AccuracyReport accuracy{};
            accuracy_compare(reference, output, size, size, &accuracy);
            const double operations = 2.0 * size * static_cast<double>(size)
                                    * size;
            const double gflops = operations / (milliseconds * 1.0e6);
            std::printf("%d,%d,%.6f,%.2f,%.9e,%.9e\n", size, splits,
                        milliseconds, gflops, accuracy.max_abs_err,
                        accuracy.l2_rel_err);
            std::fflush(stdout);
        }

        CUDA_CHECK(cudaFree(output));
        CUDA_CHECK(cudaFree(reference));
        CUDA_CHECK(cudaFree(b));
        CUDA_CHECK(cudaFree(a));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
