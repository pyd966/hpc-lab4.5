#ifndef LAB45_GROUPWISE_MMA_LARGE_CUH
#define LAB45_GROUPWISE_MMA_LARGE_CUH

#include <cstddef>
#include <cstdint>

namespace lab45_mma_large {

constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 32;
constexpr int kWarpM = 32;
constexpr int kWarpN = 32;
constexpr int kCtaM = 128;
constexpr int kCtaN = 64;
constexpr int kCtaK = 128;
constexpr int kSharedK = kCtaK + 16;
constexpr int kWarpTilesM = kWarpM / kMmaM;
constexpr int kWarpTilesN = kWarpN / kMmaN;
constexpr int kWarpsM = kCtaM / kWarpM;
constexpr int kWarpsN = kCtaN / kWarpN;
constexpr int kWarps = kWarpsM * kWarpsN;
constexpr int kThreads = kWarps * 32;

__device__ __forceinline__ uint32_t load_word(const int8_t* pointer) {
    return *reinterpret_cast<const uint32_t*>(pointer);
}

__device__ __forceinline__ void copy_async_16(
    int8_t* destination, const int8_t* source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned shared_address =
        static_cast<unsigned>(__cvta_generic_to_shared(destination));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(shared_address), "l"(source));
#else
    *reinterpret_cast<int4*>(destination) =
        *reinterpret_cast<const int4*>(source);
#endif
}

__device__ __forceinline__ void copy_async_commit_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__device__ __forceinline__ void mma_m16n8k32(
    int& c0, int& c1, int& c2, int& c3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

template <int Splits, int Diagonals>
__global__ __launch_bounds__(kThreads, 3)
void groupwise_mma_kernel(const int8_t* __restrict__ aq,
                          const int8_t* __restrict__ bq,
                          double* __restrict__ output,
                          const double* __restrict__ diagonal_scales,
                          int M, int N, int K) {
    __shared__ __align__(16) int8_t shared_a[kCtaM * kSharedK];
    __shared__ __align__(16) int8_t shared_b[kCtaN * kSharedK];

    const int thread = threadIdx.x;
    const int warp = thread >> 5;
    const int lane = thread & 31;
    const int group = lane >> 2;
    const int lane_in_group = lane & 3;
    const int warp_m = warp % kWarpsM;
    const int warp_n = warp / kWarpsM;
    const int cta_m = blockIdx.y * kCtaM;
    const int cta_n = blockIdx.x * kCtaN;
    const int tile_m = cta_m + warp_m * kWarpM;
    const int tile_n = cta_n + warp_n * kWarpN;
    const size_t extended_k = static_cast<size_t>(Splits) * K;

#pragma unroll 1
    for (int diagonal = 0; diagonal < Diagonals; ++diagonal) {
        int accumulators[kWarpTilesM][kWarpTilesN][4];
#pragma unroll
        for (int m = 0; m < kWarpTilesM; ++m) {
#pragma unroll
            for (int n = 0; n < kWarpTilesN; ++n) {
#pragma unroll
                for (int item = 0; item < 4; ++item) {
                    accumulators[m][n][item] = 0;
                }
            }
        }

        const int j_low = diagonal >= Splits ? diagonal - Splits + 1 : 0;
        const int j_high = diagonal < Splits ? diagonal : Splits - 1;
        const int pair_count = j_high - j_low + 1;
        const int i_high = diagonal - j_low;
        const int a_slot = Splits - 1 - i_high;
        const int total_k = pair_count * K;

        for (int k_base = 0; k_base < total_k; k_base += kCtaK) {
            for (int vector = thread;
                 vector < kCtaM * (kCtaK / 16);
                 vector += kThreads) {
                const int a_row = vector >> 3;
                const int a_column = (vector & 7) * 16;
                const int8_t* source_a = aq
                    + static_cast<size_t>(cta_m + a_row) * extended_k
                    + static_cast<size_t>(a_slot) * K
                    + k_base + a_column;
                copy_async_16(shared_a + a_row * kSharedK + a_column,
                              source_a);
            }
            for (int vector = thread;
                 vector < kCtaN * (kCtaK / 16);
                 vector += kThreads) {
                const int b_row = vector >> 3;
                const int b_column = (vector & 7) * 16;
                const int8_t* source_b = bq
                    + static_cast<size_t>(cta_n + b_row) * extended_k
                    + static_cast<size_t>(j_low) * K
                    + k_base + b_column;
                copy_async_16(shared_b + b_row * kSharedK + b_column,
                              source_b);
            }
            copy_async_commit_wait();
            __syncthreads();

#pragma unroll
            for (int k_inner = 0; k_inner < kCtaK; k_inner += kMmaK) {
                uint32_t fragment_a[kWarpTilesM][4];
                uint32_t fragment_b[kWarpTilesN][2];
                const int lane_k = k_inner + lane_in_group * 4;
#pragma unroll
                for (int m = 0; m < kWarpTilesM; ++m) {
                    const int8_t* row0 = shared_a
                        + (warp_m * kWarpM + m * kMmaM + group)
                              * kSharedK;
                    const int8_t* row1 = row0 + 8 * kSharedK;
                    fragment_a[m][0] = load_word(row0 + lane_k);
                    fragment_a[m][1] = load_word(row1 + lane_k);
                    fragment_a[m][2] = load_word(row0 + lane_k + 16);
                    fragment_a[m][3] = load_word(row1 + lane_k + 16);
                }
#pragma unroll
                for (int n = 0; n < kWarpTilesN; ++n) {
                    const int8_t* column = shared_b
                        + (warp_n * kWarpN + n * kMmaN + group)
                              * kSharedK;
                    fragment_b[n][0] = load_word(column + lane_k);
                    fragment_b[n][1] = load_word(column + lane_k + 16);
                }
#pragma unroll
                for (int m = 0; m < kWarpTilesM; ++m) {
#pragma unroll
                    for (int n = 0; n < kWarpTilesN; ++n) {
                        mma_m16n8k32(
                            accumulators[m][n][0], accumulators[m][n][1],
                            accumulators[m][n][2], accumulators[m][n][3],
                            fragment_a[m][0], fragment_a[m][1],
                            fragment_a[m][2], fragment_a[m][3],
                            fragment_b[n][0], fragment_b[n][1]);
                    }
                }
            }
            __syncthreads();
        }

        const double scale = diagonal_scales[diagonal];
#pragma unroll
        for (int m = 0; m < kWarpTilesM; ++m) {
#pragma unroll
            for (int n = 0; n < kWarpTilesN; ++n) {
                const int column0 = tile_n + n * kMmaN + lane_in_group * 2;
                const int row0 = tile_m + m * kMmaM + group;
                const int row1 = row0 + 8;
                const size_t index0 = static_cast<size_t>(column0) * M + row0;
                const size_t index1 = index0 + M;
                const size_t index2 = static_cast<size_t>(column0) * M + row1;
                const size_t index3 = index2 + M;
                const double value0 = static_cast<double>(accumulators[m][n][0]);
                const double value1 = static_cast<double>(accumulators[m][n][1]);
                const double value2 = static_cast<double>(accumulators[m][n][2]);
                const double value3 = static_cast<double>(accumulators[m][n][3]);
                if (diagonal == 0) {
                    output[index0] = value0 * scale;
                    output[index1] = value1 * scale;
                    output[index2] = value2 * scale;
                    output[index3] = value3 * scale;
                } else {
                    output[index0] = fma(value0, scale, output[index0]);
                    output[index1] = fma(value1, scale, output[index1]);
                    output[index2] = fma(value2, scale, output[index2]);
                    output[index3] = fma(value3, scale, output[index3]);
                }
            }
        }
    }
    (void)N;
}

}  // namespace lab45_mma_large

#endif
