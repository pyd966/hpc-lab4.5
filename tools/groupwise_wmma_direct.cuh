#ifndef LAB45_GROUPWISE_WMMA_DIRECT_CUH
#define LAB45_GROUPWISE_WMMA_DIRECT_CUH

#include <mma.h>

namespace lab45_wmma_direct {

constexpr int kCtaM = 64;
constexpr int kCtaN = 64;
constexpr int kCtaK = 16;
constexpr int kWarpM = 16;
constexpr int kWarpN = 16;
constexpr int kWarpsM = kCtaM / kWarpM;
constexpr int kWarpsN = kCtaN / kWarpN;
constexpr int kWarps = kWarpsM * kWarpsN;
constexpr int kThreads = kWarps * 32;
constexpr int kOutputsPerThread = kCtaM * kCtaN / kThreads;

template <int Splits, int Diagonals>
__global__ __launch_bounds__(kThreads, 1)
void groupwise_wmma_kernel(const int8_t* __restrict__ aq,
                           const int8_t* __restrict__ bq,
                           double* __restrict__ output,
                           const double* __restrict__ diagonal_scales,
                           int M, int N, int K) {
    using namespace nvcuda;
    __shared__ __align__(16) int shared_c[kWarps * kWarpM * kWarpN];
    const int thread = threadIdx.x;
    const int warp = thread >> 5;
    const int warp_m = warp % kWarpsM;
    const int warp_n = warp / kWarpsM;
    const int tile_m = blockIdx.y * kCtaM;
    const int tile_n = blockIdx.x * kCtaN;
    const size_t extended_k = static_cast<size_t>(Splits) * K;

    double accumulators[kOutputsPerThread];
#pragma unroll
    for (int item = 0; item < kOutputsPerThread; ++item) {
        accumulators[item] = 0.0;
    }

#pragma unroll
    for (int diagonal = 0; diagonal < Diagonals; ++diagonal) {
        wmma::fragment<wmma::accumulator, kWarpM, kWarpN, kCtaK, int>
            accumulator;
        wmma::fill_fragment(accumulator, 0);
        const int begin_i = diagonal < Splits ? 0 : diagonal - Splits + 1;
        const int end_i = diagonal < Splits ? diagonal : Splits - 1;
        for (int i = begin_i; i <= end_i; ++i) {
            const int j = diagonal - i;
            const int a_slot = Splits - 1 - i;
            for (int k_base = 0; k_base < K; k_base += kCtaK) {
                wmma::fragment<wmma::matrix_a, kWarpM, kWarpN, kCtaK,
                               signed char, wmma::row_major> fragment_a;
                wmma::fragment<wmma::matrix_b, kWarpM, kWarpN, kCtaK,
                               signed char, wmma::col_major> fragment_b;
                const signed char* source_a =
                    reinterpret_cast<const signed char*>(aq)
                    + static_cast<size_t>(tile_m + warp_m * kWarpM)
                          * extended_k
                    + static_cast<size_t>(a_slot) * K + k_base;
                const signed char* source_b =
                    reinterpret_cast<const signed char*>(bq)
                    + static_cast<size_t>(tile_n + warp_n * kWarpN)
                          * extended_k
                    + static_cast<size_t>(j) * K + k_base;
                wmma::load_matrix_sync(fragment_a, source_a, extended_k);
                wmma::load_matrix_sync(fragment_b, source_b, extended_k);
                wmma::mma_sync(accumulator, fragment_a, fragment_b,
                               accumulator);
            }
        }

        wmma::store_matrix_sync(shared_c + warp * kWarpM * kWarpN,
                                accumulator, kWarpN, wmma::mem_row_major);
        __syncthreads();
        const double scale = diagonal_scales[diagonal];
#pragma unroll
        for (int item = 0; item < kOutputsPerThread; ++item) {
            const int linear = thread + item * kThreads;
            const int local_m = linear % kCtaM;
            const int local_n = linear / kCtaM;
            const int owner_warp = (local_n / kWarpN) * kWarpsM
                                 + local_m / kWarpM;
            const int owner_index = (local_m % kWarpM) * kWarpN
                                  + local_n % kWarpN;
            accumulators[item] = fma(
                static_cast<double>(
                    shared_c[owner_warp * kWarpM * kWarpN + owner_index]),
                scale, accumulators[item]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int item = 0; item < kOutputsPerThread; ++item) {
        const int linear = thread + item * kThreads;
        const int local_m = linear % kCtaM;
        const int local_n = linear / kCtaM;
        output[static_cast<size_t>(tile_n + local_n) * M
               + tile_m + local_m] = accumulators[item];
    }
    (void)N;
}

}  // namespace lab45_wmma_direct

#endif
