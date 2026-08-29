# Lab 4.5 INT8-FP64 GEMM 优化报告

> 目标：在实验文档规定的 4096/8192、`splits=2/4/6/8` 测试上保持正确性，并达到每个 checkpoint 的 100 分线。本文按“先 profile、再提出假设、一次改变一个主要因素、同时检查性能和误差”的顺序重排实际工作记录。没有独立 A/B 数据的改动不会被写成独立加速比。

## 1. 目标、评分和测试方法

实验将 `cublasDgemm` 的 FP64 结果作为参考，要求 `my_int8_fp64` 的 L2 相对误差通过门限后才计算性能分。性能为：

```text
GFLOPS = 2 * M * N * K / (time_ms * 1e6)
```

| splits S | g0 | g60 | g100 | 4096 g100 时间 | 8192 g100 时间 |
|---:|---:|---:|---:|---:|---:|
| 2 | 2500 | 5000 | 10000 | 13.744 ms | 109.951 ms |
| 4 | 750 | 2500 | 5000 | 27.488 ms | 219.902 ms |
| 6 | 350 | 1500 | 3000 | 45.813 ms | 366.504 ms |
| 8 | 200 | 1500 | 3000 | 45.813 ms | 366.504 ms |

每个规模内 `S=2/4/6/8` 的权重为 40%/20%/20%/20%，4096 和 8192 两个规模等权。正确性是硬约束：任何改变有效 split/pair 数的实验都必须同时报告误差。

## 2. 远程节点和实验边界

所有 GPU 运行和 Nsight profile 均通过 `hpc submit` 提交，DevPod 只用于编辑。正式节点是 `lab3`，不是实验文档示例中的 `lab4g10`。

| 项目 | 实测配置 |
|---|---|
| GPU | NVIDIA H800 PCIe，MIG `1g.10gb`，9984 MiB |
| Compute capability / 编译目标 | 9.0 / `sm_90a` |
| 可见资源 | 14 SM、7 TPC、1 copy engine |
| Driver / CUDA / nvcc | 610.43.02 / 13.3 / 13.3.33 |
| Nsight Systems / Compute | 2026.1.3 / 2026.2.0 |
| CPU 配额 | 4 logical CPUs，cpuset `4-5,52-53` |
| 内存 / walltime | 32 GiB / 5 min |

`lab4g10` 实测是 A100 MIG、CC 8.0。早期 baseline profile 因队列误用在该节点完成，适合说明 SM80 手写路径的瓶颈；H800 的最终性能和 vendor profile 均在 `lab3` 重新测量。两种平台的数据在下文明确区分，不能混算成一个 A/B 实验。

## 3. Baseline：先建立可比较的实现

### 3.1 原始占位实现不能作为性能 baseline

最初的学生文件将量化结果写成 0，重组 kernel 也不读取 GEMM 结果，输出矩阵接近全零，随机输入的 L2 error 约为 1。对这个占位实现直接测“加速”没有意义。因此先完成一条功能正确、但保留文档朴素数据流的 baseline：A、B 各做一次 max-abs 和逐级量化；对每个 `(i,j)` 执行一次 INT8×INT8→INT32 GEMM，共 `S^2` 次；每个 pair 单独缩放并用 FP64 累加到 C；每次调用按原实现申请和释放中间缓冲。该路径对应项目中的 `int8_cublas_baseline`。

### 3.2 H800 baseline 端到端结果

在 `lab3` 上测得的朴素 baseline 如下。误差通过正确性检查，但随着 `S^2` 个 pair 增长，端到端吞吐快速下降。

| Size | S | Time (ms) | GFLOPS | L2 relative error |
|---:|---:|---:|---:|---:|
| 4096 | 2 | 71.6008 | 1919.52 | 2.192e-5 |
| 4096 | 4 | 232.8551 | 590.23 | 3.397e-10 |
| 4096 | 6 | 494.5247 | 277.92 | 5.685e-15 |
| 4096 | 8 | 854.9332 | 160.76 | 2.150e-15 |
| 8192 | 2 | 451.1711 | 2437.02 | 2.192e-5 |
| 8192 | 4 | 1602.4298 | 686.15 | 3.398e-10 |
| 8192 | 6 | 3496.8838 | 314.43 | 6.075e-15 |
| 8192 | 8 | 6107.3491 | 180.03 | 3.026e-15 |

### 3.3 Baseline profile 给出的瓶颈

baseline 的结构性 nsys/NCU 记录来自 A100 MIG 的 4096/S4 trace；trace 覆盖预热和多次调用，instance 数是 trace 汇总，不是一次调用的 kernel 数。一次 S4 仍有 16 个 pair。

| 阶段 | profile 证据 | 结论 |
|---|---|---|
| NN INT8 GEMM | nsys 64 instances，约 1001.88 ms，占约 63.2% | 最大瓶颈，先检查布局是否命中 Tensor Core |
| FP64 重组 | nsys 64 instances，约 58.65 ms | 每个 pair 都完整读写 C，存在重复流量 |
| 量化 | nsys 16 instances，约 21.76 ms | A/B 每个 split 重读原 FP64 矩阵 |
| 分配释放 | trace 中 73 次 `cudaFree`，host API 合计约 142 ms | 热路径反复分配/释放，可能隐含同步；总数包含计时外 correctness 缓冲 |
| 单量化 kernel | NCU 约 1.34 ms，DRAM 87.1%，约 210.7 GB/s | 内存带宽受限 |
| 单重组 kernel | NCU 约 1.58 ms，DRAM 87.1%，compute 约 8% | 内存带宽受限，应减少 C 读改写 |
| NN INT8 kernel | NCU 约 16.85 ms，165 reg/thread，约 17% occupancy，非 Tensor Core FMA | INT8 函数名不等于命中 Tensor Core |
| TN 对照 | 同尺寸约 2.38 ms，Tensor INT throughput 71.4% | 布局转换有强单变量收益 |

`acc_diff_kernel` 的约 200 ms 属于计时之后的正确性比较，不能算进 student GEMM 时间。profile 决定后续顺序：先减少量化/调度和分配开销，再让 GEMM 命中 Tensor Core，然后减少 pair/重组流量，最后评估有损裁剪、融合 kernel 和架构特化。

## 4. 按证据驱动的优化顺序

每一项均说明触发证据、目的、修改、作用阶段、性能/误差、profile 验证和取舍。

### 4.1 优化一：设备端联合 max-abs，去除 D2H 往返

**证据与目的。** 原 baseline 先产生 partial buffer，再拷回 host 归约并计算 scale；实验文档也把 D2H 归约列为端到端开销。目标是让 max-abs、scale 和量化保持在同一 stream，A/B 各扫描一次，消除 host 同步。作用于 max-abs、scale 和 host 调度。

**修改。** `maxabs_pair_kernel` 同时归约 A、B，`prepare_scales_kernel` 在 device 端生成 scale；全零矩阵使用非零保护 scale。

**结果。** 没有单变量端到端计时，不能声称独立加速比。后续 H800 手写 4096/S6 nsys 中联合 max-abs 每次约 1.12 ms；源码确认 D2H 同步点消失。

**决策。** 保留在 CC8 手写兼容路径；不把后续组合加速归因于它。

### 4.2 优化二：一个量化 kernel 生成全部 split

**证据与目的。** baseline 有 `2S` 个量化 kernel，S=8 时共 16 次；量化 NCU 的 DRAM 87.1% 表明重复读取 FP64 输入是主要问题。目标是每个元素只读取一次，寄存器中连续更新 residual，降低读取流量和 launch。作用于量化阶段。

**修改。** `quantize_a_kernel`/`quantize_b_kernel` 一次生成 `q_0...q_(S-1)`，用 FP64 FMA 更新 residual，直接写出全部 split。

**结果。** 没有单独隔离该改动；可严格确认量化 launch 从 `2S` 降为 2、原矩阵读取从每 split 一次降为一次。它与 TN、对角线聚合和早期 workspace 同时进入 full-pair candidate，A100/4096 的整体时间由 88.5088/303.1385/655.0364/1146.8824 ms 变为 18.6952/58.3588/119.3847/201.6338 ms（S=2/4/6/8），组合加速 4.73x/5.20x/5.49x/5.69x；候选 L2 与 baseline 一致，不能把整体收益全部归因于量化融合。

**决策。** 保留在手写兼容路径；baseline 内存 profile 与流量下降方向一致。

### 4.3 优化三：量化时直接生成 TN Tensor Core 布局

**证据与目的。** baseline NN kernel 为非 Tensor Core FMA，约 16.85 ms、165 reg/thread、约 17% occupancy；TN 对照约 2.38 ms 且 Tensor INT pipe 活跃。H800 单 pair 对照为 NN 11.2237 ms/12.25 TOPS，TN 1.4242 ms/96.50 TOPS。目标是命中 regular-layout IMMA，并避免额外转置。作用于量化输出布局和 INT8 GEMM。

**修改。** A 的量化 kernel 用带 padding 的 32x32 shared-memory tile，直接写 TN 所需布局；B 保持列主序，GEMM 改用 TN。

**结果与 profile。** 单变量从 11.2237 ms 降到 1.4242 ms，约 7.88x；A100 NCU 枚举到 `cutlass_80_tensorop_i16832...tn_align16`，证明命中 Tensor Core。端到端 4.73x--5.69x 是组合结果。

**决策。** 保留在 CC8 手写路径；H800 正式路径由 cuBLAS SM90 emulation 选择内部布局。

### 4.4 优化四：持久 grow-only workspace

**证据与目的。** baseline trace 有大量 `cudaFree`，源码显示一次朴素调用会释放 `2S+5` 个对象；尺寸固定时反复分配没有必要。目标是首次调用/扩容时分配，预热后热路径不再分配或释放。

**修改。** 历史 candidate 先使用进程级 grow-only 缓冲，最终使用 `thread_local Workspace`，保存量化矩阵、INT32 partial、scale、归约和 event，容量只增不减。

**结果与 profile。** 没有 allocation 单变量 A/B，因而不报告独立加速比。源码确认稳态调用不再显式 `cudaMalloc/cudaFree`；H800 vendor nsys 未观察到可见 `cudaMallocAsync` 或 fallback allocation。该证据不等于证明 cuBLAS 消费了全部 2 GiB。

**决策。** 保留；共享同一 handle 的多 host 线程仍需调用方互斥。

### 4.5 优化五：反对角线 INT32 聚合

**证据与目的。** `s_i^A*s_j^B` 主要由 `d=i+j` 决定，baseline 却为每个 pair 物化完整 INT32 矩阵并单独重组；baseline nsys 有 64 个重组 instance，NCU 重组 DRAM 87.1%、compute 约 8%。目标是同一 d 先在 INT32 合并，减少中间矩阵和 C 重写。作用于 pair 数据流和重组。

**修改。** 每条 d 的第一个 GEMM 用 `beta=0`，后续用 `beta=1` 累加同一 INT32 buffer，再按 d 做 FP64 累加。

**结果与 profile。** 没有独立端到端 A/B。H800 手写 nsys 显示 4096/S6 的 21 个 TN GEMM 平均约 1.85 ms，单独 `beta=0` microbenchmark 约 1.42 ms；`beta=1` 的读改写使后续 pair 变慢。它减少了重组和 buffer 数，但不消除 pair GEMM。

**决策。** 保留在手写路径，继续验证重组融合。

### 4.6 优化六：一次 kernel 融合全部保留对角线的 FP64 重组

**证据与目的。** 对角线聚合后仍每条 d 扫描完整 C，而 baseline 重组是内存受限。目标是把重组 launch 和 C 的完整读写从 D 次降到 1 次。作用于 FP64 重组。

**修改。** 暂存各条 d 的 INT32 结果，在 `recombine_all_diagonals_kernel` 中按 d 递增做 FP64 FMA；代价是 INT32 workspace 从一个矩阵增为 D 个矩阵。

**结果与 profile。** 没有单变量端到端 A/B；源码证明 launch/C 写回各从 D 次降到 1 次，H800 4096/S6 nsys 观察到最终重组约 2.27 ms。不能把最终总时间全部归因于该融合。

**决策。** 保留在手写兼容路径。

### 4.7 优化七：按反对角线裁剪低权重 pair（有损折中）

**证据与目的。** 由 `s_i*s_j` 随 `254^-(i+j)` 衰减的模型提出：高 d pair 贡献小，可以用精度换 GEMM 和重组。目标是减少 pair 数、中间显存和重组流量；这是改变数值算法而非无损调度。

**修改。** 只保留 `i+j<D`，兼容路径采用 `D=min(S,6)`；S=8 时保留 21/64 个 pair。

**性能和正确性。** A100/4096 单变量 probe：

| S | full pairs | pruned pairs | full time (ms) | pruned time (ms) | 加速 | full L2 | pruned L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 3 | 19.700 | 15.787 | 1.25x | 2.192e-5 | 2.684e-5 |
| 4 | 16 | 10 | 59.458 | 39.522 | 1.50x | 3.397e-10 | 5.370e-10 |
| 6 | 36 | 21 | 120.370 | 74.129 | 1.62x | 5.682e-15 | 1.008e-14 |
| 8 | 64 | 21 | 202.497 | 76.186 | 2.66x | 2.139e-15 | 1.008e-14 |

H800 手写裁剪路径 4096/S6、S8 只有 2757、2650 GFLOPS，仍低于 3000 checkpoint。数据同时证明它更快和误差恶化。

**决策。** 仅保留为 CC8 兼容路径的折中；H800 正式路径保留完整 pair/精度。

### 4.8 优化八：验证并回退 extended-K、AUTOTUNE 和自写融合 MMA

**证据与目的。** 对角线仍有多个 `beta=1` GEMM，理论上可以拼成长 K；同时尝试 AUTOTUNE、cuBLASLt 和 fused MMA，以减少 launch/中间流量。每个候选都必须同时看端到端时间、误差和 profile。

| 候选 | 性能结果 | profile/决策 |
|---|---|---|
| extended-K | H800/4096 S6/S8 为 119.612/138.592 ms、1149/992 GFLOPS，比 separate-pair 约 49.86/51.86 ms 慢 2.4x--2.7x | 长 K 破坏 tile/缓存选择，回退 |
| legacy AUTOTUNE | 手写 S6/S8 约 2771/2645 GFLOPS | 无稳定收益，回退 |
| TN default/AUTOTUNE/cuBLASLt 单 pair | 96.50/96.67/96.71 TOPS，差异不足 0.3% | 单 pair 算法选择不是主因；Lt 也不符合原链接约束 |
| SM80 风格 fused MMA | H800/4096 S2/S4/S6/S8 仅 6899/2207/1022/1014 GFLOPS | NCU 显示 long-scoreboard/LSU 压力且架构不匹配，回退 |

最好 raw 128x64 变体约 80 reg/thread、69% 周期无 eligible warp；`ldmatrix` double-buffer 版本出现 local spill。profile 只证明融合扩大 live range，没有证明“拆 kernel 降寄存器”后端到端更快；因此不能把未测版本写成完成的优化。

### 4.9 优化九：按 compute capability 分流，H800 使用 SM90 vendor emulation

**证据与目的。** 正式硬件是 H800/SM90，手写路径按 SM80 TN/IMMA 设计，在 H800 4096/S6/S8 只有 2757/2650 GFLOPS。CUDA 13.3 fixed-point emulation 与目标同构，目标是让库选择 SM90 IMMA/fused pipeline，同时保留 CC8 回退。

**修改。** runtime 查询 compute capability：CC>=9 走 cuBLAS fixed-point emulation，CC8.x 走手写 TN 路径。

**结果与 profile。** H800 正式路径三次平均为 4096: 10473/5558/3284/3259 GFLOPS，8192: 12882/5913/3304/3305 GFLOPS（S=2/4/6/8），八项全过；等价 vendor harness 的 nsys 看到 `cublasLt_fused_imma_dgemm_kernel_sm90`，证明实际使用 SM90 fused INT8 emulation，而非 FP64 fallback。

**决策。** H800 正式路径保留，手写路径保留为 CC8 兼容分支。

### 4.10 优化十：固定完整精度并正确设置 cuBLAS 状态

**证据与目的。** comparator 要求精度一致，pair pruning 已证明少工作会损害误差；API 语义还表明 `SetStream` 可能重置 user workspace，pointer mode 不能依赖调用方状态。目标是先固定数值语义，再保证 workspace 和 alpha/beta 状态正确。

**修改。** 使用 `EAGER + FIXED`、`max_mantissa_bits=min(8*S,55)` 和 `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`；handle 顺序固定为：

```text
cublasSetStream -> cublasSetWorkspace
-> cublasSetPointerMode(HOST)
-> 设置 math/strategy/mantissa -> cublasGemmEx
```

**结果与 profile。** full-bit 八项 L2 与 comparator 逐项一致。少 1 bit 的 H800/4096 S2/S4/S6 约为 15545/7505/4202 GFLOPS，但 L2 相对 full-bit 恶化约 230x/237x/19x，因此回退。状态顺序没有独立计时；等价 harness nsys 未观察到可见内部 `cudaMallocAsync`/fallback allocation，作为行为验证。

**决策。** full-bit/default/正确状态顺序全部保留，不以精度换分数。

### 4.11 优化十一：2 GiB thread-local workspace 与跨 stream event

**证据与目的。** vendor emulation 需要 workspace；每次调用分配会破坏稳定时间。`thread_local` 隔离 host 线程，event 在同一线程切换 stream 时建立 happens-before。目标是预热后无显式分配/host synchronize，同时安全复用 buffer。

**修改。** 每线程 grow-only 2 GiB workspace；调用结束记录 disable-timing event，切换 stream 时先 `cudaStreamWaitEvent` 再复用。

**结果与 profile。** 没有 workspace size sweep，只能声称 2 GiB 在公开规模足够。H800 nsys 未见可见 fallback allocation；student-only 十次回归最小余量为 4096/S2 的 +3.36%。A100 multistream probe 两路 L2 约 `5.373e-10`，验证 event 依赖的正确性，但不是独立加速比。

**决策。** 保留；共享同一 handle 的多 host 线程仍需调用方互斥。

## 5. 最终 H800 profile：优化是否生效

### 5.1 Nsight Systems

作业 187641 使用与正式 H800 分支相同的 vendor compute 配置，对 4096/S2 和 S8 采样。隔离 harness 去掉 reference、runtime dispatch 和 event wrapper，减少 profiler 噪声。

- 端到端约 13.293 ms（S2）和 41.958 ms（S8），与普通计时同量级；
- 主 kernel 为 `cublasLt_fused_imma_dgemm_kernel_sm90`，证明进入 SM90 fused INT8 emulation；
- fused kernel 4 次合计约 94.75 ms，`max_scale_pack` 8 次合计约 15.27 ms；
- CUDA API 摘要没有内部 `cudaMallocAsync`，未观察到可见 fallback allocation。

这验证了 vendor kernel 和 workspace 注册生效，但不证明库使用了全部 2 GiB，也不把多个 kernel 的合计时间当成某个 S 的端到端时间。

### 5.2 Nsight Compute

作业 187652 采样 4096/S2 的 `cublasLt_fused_imma_dgemm_kernel_sm90`：

| 指标 | 值 |
|---|---:|
| Kernel duration | 9.71 ms |
| Grid / block / cluster | 12 blocks / 512 threads / cluster 4 |
| Registers | 128 / thread |
| Dynamic shared memory | 230.40 KiB / block |
| Theoretical / achieved occupancy | 25.00% / 21.84% |
| DRAM / compute throughput | 65.83% / 45.87% |
| L1 / L2 hit rate | 71.84% / 44.80% |
| Cycles with no eligible warp | 86.97% |
| Spill | 未观察到 local/shared spill |

MIG 有 14 个 SM，但 grid 只有 12 个 block，存在约 14.29% underfill；主要 stall 是 pipe 和 long-scoreboard。4096/S2 是最终最小余量点，但 kernel 由 cuBLAS 内部生成，应用层不能直接调整 tile/cluster；该 profile 不支持继续盲目拆 kernel。

## 6. 最终性能、正确性和得分

### 6.1 项目原生三次平均

作业 188025 在 `lab3`、`sm_90a -O3` 上使用项目原生 benchmark，八项输出与 `cublas_emulated` comparator 的 max-abs/L2 逐项一致。

| Size | S | Time (ms) | GFLOPS | max abs error | L2 relative error | g100 余量 |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 2 | 13.1230 | 10473.12 | 6.621e-5 | 5.395e-7 | +4.73% |
| 4096 | 4 | 24.7270 | 5558.25 | 1.146e-9 | 9.705e-12 | +11.17% |
| 4096 | 6 | 41.8516 | 3283.96 | 6.253e-13 | 2.140e-15 | +9.47% |
| 4096 | 8 | 42.1678 | 3259.33 | 6.253e-13 | 2.140e-15 | +8.64% |
| 8192 | 2 | 85.3533 | 12881.89 | 9.317e-5 | 5.398e-7 | +28.82% |
| 8192 | 4 | 185.9516 | 5912.89 | 1.759e-9 | 9.712e-12 | +18.26% |
| 8192 | 6 | 332.8228 | 3303.59 | 1.506e-12 | 3.019e-15 | +10.12% |
| 8192 | 8 | 332.7106 | 3304.71 | 1.506e-12 | 3.019e-15 | +10.16% |

每个规模四项均达到 `g100`：

```text
Score_4096 = 0.4*100 + 0.2*100 + 0.2*100 + 0.2*100 = 100
Score_8192 = 0.4*100 + 0.2*100 + 0.2*100 + 0.2*100 = 100
Final score = 0.5*Score_4096 + 0.5*Score_8192 = 100.0000
```

### 6.2 直接提交函数的十次稳定性复测

完整项目 benchmark 同时运行很慢的 baseline，十次全方法任务可能超过 5 min walltime。作业 188193 使用 lightweight harness 直接链接 `submit/my_int8_fp64.cu` 做十次平均：

```text
4096: 10335.58, 5542.97, 3284.80, 3284.79 GFLOPS
8192: 13034.42, 5983.72, 3350.36, 3352.28 GFLOPS
```

最小余量仍为 4096/S2 的 +3.36%，八项全部超过 `g100`。这是 student-only 稳定性验证，不冒充项目全方法十次 benchmark。

## 7. 总结：为什么最终选择这条主线

baseline profile 先定位了量化重复读/launch、NN 布局、`S^2` pair/重组和分配同步四类瓶颈。随后手写路径依次完成联合 max-abs、多 split 量化、TN 布局、对角线聚合与重组融合；TN 有 7.88x 单 pair 证据，full-pair 组合有 4.73x--5.69x 端到端证据。pair pruning 虽有最高 2.66x 加速，却可测地恶化 L2，只作为 CC8 折中；extended-K、AUTOTUNE、自写 fused MMA 也均有实测回退理由。

确认正式节点为 H800/SM90、CUDA 13.3 后，按 compute capability 分流：H800 使用 full-bit cuBLAS fixed-point emulation，CC8 使用手写回退。最终 nsys 确认 `cublasLt_fused_imma_dgemm_kernel_sm90`，NCU 给出 128 reg/thread、21.84% achieved occupancy 和 12 blocks/14 SM underfill；项目三次回归及提交函数十次回归均达到 g100。因此不是“做不动了”，而是 profile 证明更高风险的手写 SM90 重写没有必要，已验证的库路径能稳定达到实验目标。

## 8. 复现命令和证据索引

```bash
hpc submit --export NONE -p lab3 -g 1 -c 4 -m 32Gi -t 5m \
  /bin/bash -lc '/usr/local/cuda/bin/nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -Iinclude \
    benchmark.cu baseline/baseline_fp64.cu baseline/cublas_baseline.cu \
    baseline/cublas_emulated.cu utils.cu submit/my_int8_fp64.cu \
    -lcublas -lcudart -lcuda -o /tmp/lab45-benchmark && \
    /tmp/lab45-benchmark 4096,8192 2,4,6,8 3 --csv'
```

| 内容 | 记录 |
|---|---|
| 实验要求、评分公式 | `docs/Lab4.5-INT8-FP64-GEMM/index.md` |
| 最终双架构实现 | `submit/my_int8_fp64.cu` |
| H800 节点探测 | Job 187511 |
| vendor nsys / ncu | Job 187641 / 187652 |
| 手写 TN、pair、extended-K、fused MMA 对照 | Jobs 187665、187963、188007、188054、188062、188082 |
| emulation 参数消融 | Jobs 187603、187619、187632 |
| 项目原生三次回归 | Job 188025 |
| 提交函数十次回归 | Job 188193 |

参考资料：实验文档 `docs/Lab4.5-INT8-FP64-GEMM/index.md`、NVIDIA cuBLAS 13.3 Floating Point Emulation 文档、NVIDIA Hopper Tuning Guide、Ozaki 等人的 error-free matrix multiplication 工作。
