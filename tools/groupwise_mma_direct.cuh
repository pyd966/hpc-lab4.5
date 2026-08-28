#ifndef LAB45_GROUPWISE_MMA_DIRECT_CUH
#define LAB45_GROUPWISE_MMA_DIRECT_CUH

#include <cstddef>
#include <cstdint>

namespace lab45_mma_direct {

constexpr int kWarpM = 16;
constexpr int kWarpN = 8;
constexpr int kMmaK = 32;
constexpr int kCtaM = 64;
constexpr int kCtaN = 32;
constexpr int kWarpsM = kCtaM / kWarpM;
constexpr int kWarpsN = kCtaN / kWarpN;
constexpr int kWarps = kWarpsM * kWarpsN;
constexpr int kThreads = kWarps * 32;

__device__ __forceinline__ uint32_t load_word(const int8_t* pointer) {
    return *reinterpret_cast<const uint32_t*>(pointer);
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
__global__ __launch_bounds__(kThreads, 2)
void groupwise_mma_kernel(const int8_t* __restrict__ aq,
                          const int8_t* __restrict__ bq,
                          double* __restrict__ output,
                          const double* __restrict__ diagonal_scales,
                          int M, int N, int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int group = lane >> 2;
    const int lane_in_group = lane & 3;
    const int warp_m = warp % kWarpsM;
    const int warp_n = warp / kWarpsM;
    const int tile_m = blockIdx.y * kCtaM + warp_m * kWarpM;
    const int tile_n = blockIdx.x * kCtaN + warp_n * kWarpN;
    const size_t extended_k = static_cast<size_t>(Splits) * K;

    double output0 = 0.0;
    double output1 = 0.0;
    double output2 = 0.0;
    double output3 = 0.0;

#pragma unroll
    for (int diagonal = 0; diagonal < Diagonals; ++diagonal) {
        int c0 = 0;
        int c1 = 0;
        int c2 = 0;
        int c3 = 0;
        const int begin_i = diagonal < Splits ? 0 : diagonal - Splits + 1;
        const int end_i = diagonal < Splits ? diagonal : Splits - 1;

#pragma unroll
        for (int i = begin_i; i <= end_i; ++i) {
            const int j = diagonal - i;
            const int a_slot = Splits - 1 - i;
            const int8_t* a_row0 = aq
                + static_cast<size_t>(tile_m + group) * extended_k
                + static_cast<size_t>(a_slot) * K;
            const int8_t* a_row1 = a_row0 + 8 * extended_k;
            const int8_t* b_column = bq
                + static_cast<size_t>(tile_n + group) * extended_k
                + static_cast<size_t>(j) * K;

            for (int k_base = 0; k_base < K; k_base += kMmaK) {
                const int lane_k = k_base + lane_in_group * 4;
                const uint32_t a0 = load_word(a_row0 + lane_k);
                const uint32_t a1 = load_word(a_row1 + lane_k);
                const uint32_t a2 = load_word(a_row0 + lane_k + 16);
                const uint32_t a3 = load_word(a_row1 + lane_k + 16);
                const uint32_t b0 = load_word(b_column + lane_k);
                const uint32_t b1 = load_word(b_column + lane_k + 16);
                mma_m16n8k32(c0, c1, c2, c3,
                             a0, a1, a2, a3, b0, b1);
            }
        }

        const double scale = diagonal_scales[diagonal];
        output0 = fma(static_cast<double>(c0), scale, output0);
        output1 = fma(static_cast<double>(c1), scale, output1);
        output2 = fma(static_cast<double>(c2), scale, output2);
        output3 = fma(static_cast<double>(c3), scale, output3);
    }

    const int column0 = tile_n + lane_in_group * 2;
    const int row0 = tile_m + group;
    const int row1 = row0 + 8;
    output[static_cast<size_t>(column0) * M + row0] = output0;
    output[static_cast<size_t>(column0 + 1) * M + row0] = output1;
    output[static_cast<size_t>(column0) * M + row1] = output2;
    output[static_cast<size_t>(column0 + 1) * M + row1] = output3;
    (void)N;
}

}  // namespace lab45_mma_direct

#endif
