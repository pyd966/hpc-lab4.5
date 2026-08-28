#ifndef LAB45_GROUPWISE_MMA_SHARED_N16_CUH
#define LAB45_GROUPWISE_MMA_SHARED_N16_CUH

#include <cstddef>
#include <cstdint>

namespace lab45_mma_shared_n16 {

constexpr int kWarpM = 16;
constexpr int kWarpN = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 32;
constexpr int kCtaM = 32;
constexpr int kCtaN = 64;
constexpr int kCtaK = 128;
constexpr int kSharedK = kCtaK + 16;
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
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
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

    double output0 = 0.0;
    double output1 = 0.0;
    double output2 = 0.0;
    double output3 = 0.0;
    double output4 = 0.0;
    double output5 = 0.0;
    double output6 = 0.0;
    double output7 = 0.0;

#pragma unroll 1
    for (int diagonal = 0; diagonal < Diagonals; ++diagonal) {
        int c0 = 0;
        int c1 = 0;
        int c2 = 0;
        int c3 = 0;
        int c4 = 0;
        int c5 = 0;
        int c6 = 0;
        int c7 = 0;
        const int j_low = diagonal >= Splits ? diagonal - Splits + 1 : 0;
        const int j_high = diagonal < Splits ? diagonal : Splits - 1;
        const int pair_count = j_high - j_low + 1;
        const int i_high = diagonal - j_low;
        const int a_slot = Splits - 1 - i_high;
        const int total_k = pair_count * K;
        for (int k_base = 0; k_base < total_k; k_base += kCtaK) {
                if (thread < kCtaM * (kCtaK / 16)) {
                    const int a_row = thread >> 3;
                    const int a_column = (thread & 7) * 16;
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
                    const int lane_k = k_inner + lane_in_group * 4;
                    const int8_t* a_row0 = shared_a
                        + (warp_m * kWarpM + group) * kSharedK;
                    const int8_t* a_row1 = a_row0 + 8 * kSharedK;
                    const int8_t* b_column0 = shared_b
                        + (warp_n * kWarpN + group) * kSharedK;
                    const int8_t* b_column1 = b_column0 + kMmaN * kSharedK;
                    const uint32_t a0 = load_word(a_row0 + lane_k);
                    const uint32_t a1 = load_word(a_row1 + lane_k);
                    const uint32_t a2 = load_word(a_row0 + lane_k + 16);
                    const uint32_t a3 = load_word(a_row1 + lane_k + 16);
                    const uint32_t b0 = load_word(b_column0 + lane_k);
                    const uint32_t b1 = load_word(b_column0 + lane_k + 16);
                    const uint32_t b2 = load_word(b_column1 + lane_k);
                    const uint32_t b3 = load_word(b_column1 + lane_k + 16);
                    mma_m16n8k32(c0, c1, c2, c3,
                                 a0, a1, a2, a3, b0, b1);
                    mma_m16n8k32(c4, c5, c6, c7,
                                 a0, a1, a2, a3, b2, b3);
                }
                __syncthreads();
        }

        const double scale = diagonal_scales[diagonal];
        output0 = fma(static_cast<double>(c0), scale, output0);
        output1 = fma(static_cast<double>(c1), scale, output1);
        output2 = fma(static_cast<double>(c2), scale, output2);
        output3 = fma(static_cast<double>(c3), scale, output3);
        output4 = fma(static_cast<double>(c4), scale, output4);
        output5 = fma(static_cast<double>(c5), scale, output5);
        output6 = fma(static_cast<double>(c6), scale, output6);
        output7 = fma(static_cast<double>(c7), scale, output7);
    }

    const int column0 = tile_n + lane_in_group * 2;
    const int column1 = column0 + kMmaN;
    const int row0 = tile_m + group;
    const int row1 = row0 + 8;
    output[static_cast<size_t>(column0) * M + row0] = output0;
    output[static_cast<size_t>(column0 + 1) * M + row0] = output1;
    output[static_cast<size_t>(column0) * M + row1] = output2;
    output[static_cast<size_t>(column0 + 1) * M + row1] = output3;
    output[static_cast<size_t>(column1) * M + row0] = output4;
    output[static_cast<size_t>(column1 + 1) * M + row0] = output5;
    output[static_cast<size_t>(column1) * M + row1] = output6;
    output[static_cast<size_t>(column1 + 1) * M + row1] = output7;
    (void)N;
}

}  // namespace lab45_mma_shared_n16

#endif
