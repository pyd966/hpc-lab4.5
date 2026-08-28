# Lab 4.5 INT8 Tensor Core 模拟 FP64 GEMM：优化过程与 100 分结果

> 本文是一份可供正式实验报告参考的完整技术记录。它按因果关系重新组织优化，而不是照搬实际试错顺序。所有时间与 profile 数据均标注测试平台；组合实验不会被拆成虚假的单项加速比。

## 0. 摘要

本实验的目标是在 H800 上把 FP64 矩阵逐级量化为 INT8 分量，利用 INT8 Tensor Core 完成部分积，再以 FP64 比例尺重组结果。评分同时受正确性门控与端到端吞吐量约束：测试规模为 4096 和 8192，`splits` 为 2、4、6、8，对应满分 checkpoint 为 10000、5000、3000、3000 GFLOPS。

优化工作最终形成两条按架构分流的路径：

1. CC 8.x 兼容路径保留手写 Ozaki 实现。它依次完成了设备端 max-abs、一次生成全部 split、量化时生成 TN 布局、反对角线 INT32 聚合、pair 裁剪和持久 workspace 等优化。在 A100 MIG 上从功能基线提升到约 81.25 分；在 H800 上能通过 6/8 个 `g100` 点，但 4096/S6 和 4096/S8 仍未过线。
2. CC >= 9 正式路径使用 CUDA 13.3 cuBLAS fixed-point emulation。它通过完整 mantissa bit 配置、正确的 stream/workspace 状态顺序、2 GiB thread-local workspace 和跨 stream event 保护；采用相同 vendor 调用配置的隔离 profile harness 实际进入 `cublasLt_fused_imma_dgemm_kernel_sm90`。正式提交文件在 H800 上八项全部达到 `g100`，最终得分 100.0000。

最终 H800 三次平均结果为：

| Size | S | Time (ms) | GFLOPS | L2 relative error | `g100` 余量 |
|---:|---:|---:|---:|---:|---:|
| 4096 | 2 | 13.1230 | 10473.12 | 5.395e-7 | +4.73% |
| 4096 | 4 | 24.7270 | 5558.25 | 9.705e-12 | +11.17% |
| 4096 | 6 | 41.8516 | 3283.96 | 2.140e-15 | +9.47% |
| 4096 | 8 | 42.1678 | 3259.33 | 2.140e-15 | +8.64% |
| 8192 | 2 | 85.3533 | 12881.89 | 5.398e-7 | +28.82% |
| 8192 | 4 | 185.9516 | 5912.89 | 9.712e-12 | +18.26% |
| 8192 | 6 | 332.8228 | 3303.59 | 3.019e-15 | +10.12% |
| 8192 | 8 | 332.7106 | 3304.71 | 3.019e-15 | +10.16% |

## 1. 报告的逻辑主线

实际开发中先后尝试过布局、pair 裁剪、extended-K、自写 MMA、cuBLAS 参数和不同队列，因此 git 历史并不是适合报告的叙事顺序。本文把它们重排为下面的闭环：

```text
正确性与成本模型
        |
        v
建立可测的朴素 Ozaki 基线
        |
        v
减少量化与内存管理开销
        |
        v
修正 GEMM 布局，真正命中 INT8 Tensor Core
        |
        v
减少 pair、重组次数和中间矩阵流量
        |
        v
尝试 extended-K / groupwise fusion，确认手写路径上限
        |
        v
在正式 lab3/H800 上重新 profile，进行架构选择
        |
        v
SM90 fixed-point emulation + 状态/workspace 工程优化
        |
        v
参数消融、最终 profile、八项回归、100 分核验
```

这不是 MPI/OMP 类型的 CPU 并行实验，因此本文不人为套用“MPI -> OMP”的层次。与之对应的优化层次是：host 调度与资源管理、量化 kernel、GEMM 布局与算法、部分积数据流、架构特化。

每项优化统一回答七个问题：

- **触发证据**：由源码观察、成本模型、nSys、NCU 还是硬件/API 知识提出；
- **目的**：想减少什么时间、流量、launch、资源压力或精度风险；
- **修改**：具体改变了什么；
- **作用阶段**：max-abs、量化、GEMM、重组、调度或内存管理；
- **正确性**：误差机制是否变化，测试是否通过；
- **性能与 profile**：before/after，以及 profiler 是否验证原假设；
- **决策**：正式保留、仅兼容路径保留，还是回退。

## 2. 问题定义、算法和评分约束

### 2.1 逐级量化

对 FP64 矩阵中的元素 `x`，使用 S 个 INT8 分量近似：

```text
x ~= sum(i=0..S-1) s_i q_i
```

令矩阵最大绝对值为 `Xmax`，首级比例尺为：

```text
s_0 = Xmax / 127
```

逐级提取分量：

```text
r_0 = x
q_i = round(r_i / s_i)
r_(i+1) = r_i - s_i q_i
s_(i+1) = s_i / 254
```

选择 254 的原因不是经验常数。round 后有 `|r_(i+1)| <= s_i/2`；为了让下一层仍落在 `[-127,127]`，最大缩减率正好是 254。缩减率更大可能溢出，更小则浪费下一层动态范围。

在理想实数模型中，相对矩阵最大值的分解误差上界约为 `1/254^S`：

| S | 归一化理论上界 | 约等效精度 |
|---:|---:|---:|
| 1 | 3.9e-3 | 8 bit |
| 2 | 1.5e-5 | 16 bit |
| 4 | 2.4e-10 | 32 bit |
| 6 | 3.8e-15 | 48 bit |
| 8 | 5.8e-20 | 64 bit |

这不是逐元素相对误差界，也不包含 INT32 GEMM 与 FP64 重组的舍入。S=8 的理想量化界已低于 FP64 unit roundoff，因此实际误差会由 FP64 运算主导，不会继续按 `1/254^S` 下降。

对 A、B 分别量化后：

```text
C = A B
  ~= sum(i=0..S-1) sum(j=0..S-1)
      (s_i^A s_j^B) * (A_i^INT8 B_j^INT8)
```

因此朴素实现需要 `S^2` 次 INT8xINT8->INT32 GEMM。INT32 部分积最终按 FP64 权重 `s_i^A s_j^B` 累加到 C。

接口采用列主序：`C[M*N] = A[M*K] * B[K*N]`，对应 `A[k*M+m]`、`B[n*K+k]`、`C[n*M+m]`。benchmark 传入的 S 不能被静默改小；任何 pair 裁剪都会改变有效算法，必须作为精度折中单独披露。

### 2.2 端到端成本模型

实验文档明确指出，单次 INT8 GEMM 快并不等于整个 FP64 模拟路径快。朴素 S=8 路径包括：

- 2 次 max-abs 归约；
- A、B 共 `2S=16` 次量化；
- `S^2=64` 次 INT8 GEMM；
- 最多 `S^2=64` 次 FP64 重组；
- 超过 140 次 kernel launch；
- 若每个 pair 都物化一个 INT32 矩阵，仅部分积写入量就是 `64*M*N*4` 字节。

由此得到最初的瓶颈排序：

1. 先消除 host 同步、重复分配和量化阶段对原矩阵的重复读取；
2. 再确保 GEMM 真正命中 Tensor Core，而不是只看到函数名为 INT8 GEMM；
3. 然后减少 pair 数、部分积落地和 C 的重复读改写；
4. 最后才值得自写融合 MMA 或调 SM90 pipeline。

### 2.3 正确性门控和评分

结果相对 `cublasDgemm` FP64 reference 计算 max-abs error 和 L2 relative error。正确性失败的测试点直接为 0 分，因此任何减少 bit、split 或 pair 的优化都必须同时报告误差。

| S | `g0` | `g60` | `g100` | `g100` 对应时间上限（4096 / 8192） |
|---:|---:|---:|---:|---:|
| 2 | 2500 | 5000 | 10000 | 13.744 / 109.951 ms |
| 4 | 750 | 2500 | 5000 | 27.488 / 219.902 ms |
| 6 | 350 | 1500 | 3000 | 45.813 / 366.504 ms |
| 8 | 200 | 1500 | 3000 | 45.813 / 366.504 ms |

每个规模内 S=2/4/6/8 的权重为 40%/20%/20%/20%，两个规模等权。八项都达到 `g100` 时总分为 100。

### 2.4 测量原则

为避免把噪声或组合改动写成优化结论，本文采用以下证据等级：

| 标记 | 含义 |
|---|---|
| 单变量证据 | 相同平台和输入下只改变一个因素，可报告该因素的加速比 |
| 组合收益 | 多项修改一起进入候选，只能报告整体 before/after |
| Profile 验证 | profiler 中 kernel 类型、时间、带宽、stall 或 API 行为按预期变化 |
| 源码/模型证据 | 能证明 launch 数、流量或状态语义改变，但不能单独声称某个加速比 |
| 失败实验 | 假设合理但实测变慢或精度风险过高，正式回退 |

## 3. 平台与基线

### 3.1 正式远程节点

GPU 测试均通过 `hpc submit` 运行，DevPod 只用于编辑。正式队列是 `lab3`，不是文档命令示例中的 `lab4g10`。

| 项目 | lab3 正式环境 |
|---|---|
| GPU | NVIDIA H800 PCIe MIG 1g.10gb，9984 MiB |
| Compute capability | 9.0，编译目标 `sm_90a` |
| 可见计算资源 | 14 SM，7 TPC，1 copy engine |
| Driver / CUDA | 610.43.02 / CUDA 13.3 |
| nvcc / NCU / nsys | 13.3.33 / 2026.2.0 / 2026.1.3 |
| CPU | 2 x Xeon Gold 5418Y；任务 cpuset 实际为 4 logical CPUs |
| 任务内存 / walltime | 32 GiB / 5 min |

`lab4g10` 实际是 A100 MIG、CC 8.0。早期 A100 profile 仍有价值，但只能用于解释 SM80 手写路径，不能外推为 H800 最终结论。

### 3.2 原始占位实现与功能基线

最初的学生文件不是一个可比较的低性能实现：量化结果写 0，重组 kernel 不读取 GEMM 结果，C 最终为 0，随机输入 L2 error 约为 1。直接优化这个占位实现得到的“加速”没有意义。

因此第一步不是性能优化，而是建立功能基线：按照实验文档完成 max-abs、逐级量化、全部 `S^2` INT8 GEMM 和 FP64 重组，并使用项目的 `int8_cublas_baseline` 作为可测对照。

H800 上朴素 baseline 的端到端结果如下：

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

这组结果同时说明两件事：量化算法已经正确工作；随 S 增长的 `S^2` 数据流使端到端吞吐急剧下降。

### 3.3 基线 profile 给出的瓶颈

早期 A100/4096/S4 nsys 记录提供了清晰的结构性证据。该 trace 覆盖了 benchmark 的预热和多次调用，下面的 instance 数与时间是 trace 汇总值，不是单次 S4 调用的 kernel 数；按算法，单次 S4 仍是 16 个 pair：

| 阶段 | 次数或总时间 | 观察 |
|---|---:|---|
| NN INT8 GEMM | trace 中 64 instances，约 1001.88 ms | 占约 63.2%，是最大项 |
| 重组 | trace 中 64 instances，约 58.65 ms | 完整 C 被重复读写 |
| 量化 | trace 中 16 instances，约 21.76 ms | 原矩阵为每个 split 重读 |
| `cudaFree` | 整条 trace 73 次，host API 合计约 142 ms | 同时含计时内 baseline 与计时外 accuracy 的释放 |

trace 中另有约 200.07 ms 的 `acc_diff_kernel`，它来自 `utils.cu::accuracy_compare`，位于 GEMM 计时区间之后，只用于正确性比较。因此它不能计入 student 实现的端到端时间，也不能作为部分积或 FP64 重组瓶颈。

同理，73 次 `cudaFree` 不能全部算给 baseline。源码能精确证明的是：单次朴素 S 路径在计时函数内部由两次 `device_maxabs_fp64` 释放 2 个 partial buffer，并在结尾释放 `2S` 个 quant buffer、2 个 residual buffer 和 1 个 INT32 temp，合计 `2S+5` 次；`accuracy_compare` 在计时外还会释放 3 个统计 buffer。因此约 142 ms 的 API 汇总只能说明同步式 allocation/free 广泛存在，不能整体加入某个 kernel 阶段时间。

NCU 进一步区分了不同瓶颈：

- 单个量化 kernel 约 1.34 ms，DRAM throughput 87.1%，约 210.7 GB/s，是内存带宽受限；
- 单个重组 kernel 约 1.58 ms，DRAM throughput 87.1%，compute throughput 仅约 8%，同样是内存受限；
- 原 NN INT8 kernel 约 16.85 ms，compute throughput 88.3%，165 reg/thread、约 17% occupancy，执行的是非 Tensor Core FMA 路径；
- 改为 TN 的对照 kernel 约 2.38 ms，Tensor INT throughput 71.4%，说明布局选择决定了能否真正使用 Tensor Core。

这一 profile 决定了后续顺序：资源管理和融合量化解决前后处理，TN 布局解决最大 GEMM 项，反对角线聚合解决中间矩阵与重组，最后再讨论 pair 裁剪和自写融合。

## 4. 手写 Ozaki 路径：前后处理与布局

### 4.1 优化 1：设备端联合 max-abs，去除 D2H 标量往返

**触发证据。** 原实现对 A、B 分别启动归约、分配 partial buffer，再把结果拷回 host 决定比例尺。源码调用链中存在 device-to-host 传输和同步；实验文档也明确把“从 device 拷回归约结果并 host-side reduce”列为端到端开销。

**目的。** 消除量化开始前的 host 同步点，使 max-abs、scale 生成和量化都留在同一 CUDA stream 的依赖链中，同时只扫描 A、B 各一次。

**修改。** `maxabs_pair_kernel` 在设备端同时归约 A、B 的最大绝对值，`prepare_scales_kernel` 直接生成两组 FP64 scale。全零矩阵使用安全的非零 scale，避免除零。后续 kernel 从 device scale 读取，不经过 host。

**作用阶段。** max-abs、scale 准备和 host 调度。

**正确性。** scale 和残差仍使用 FP64，未改变逐级量化公式。多规模随机矩阵及多 stream probe 均通过；全零输入的保护逻辑由源码检查覆盖。

**性能与 profile。** 此项没有单独隔离计时，且不在 4.6 节 commit `6c9760c` 的历史组合数据中，因此不能声称独立或组合加速比。它进入了后来的最终手写路径；H800/4096/S6 的 nsys 中，联合 max-abs 每次约 1.12 ms，说明它已不再是主项，但也不是零成本。

**决策。** 保留在手写兼容路径。证据是源码数据流、D2H 同步点消失和最终 profile；没有为它虚构加速比。

### 4.2 优化 2：一次 kernel 生成同一矩阵的全部 split

**触发证据。** 基线为 A、B 各启动 S 个量化 kernel；S=8 共 16 次。量化 kernel 的 DRAM throughput 达 87.1%，表明重复读取 FP64 输入而不是算术本身限制性能。

**目的。** 把输入矩阵的读取次数从 S 次降为 1 次，把量化 launch 从 `2S` 降为 2，并让一个线程在寄存器中连续更新 FP64 residual。

**修改。** `quantize_a_kernel` 和 `quantize_b_kernel` 每个线程加载一个或一组 FP64 元素后，在寄存器中依次生成 `q_0...q_(S-1)`。分量选择后用 FP64 FMA 更新残差，避免低阶 split 因 FP32 更新而丢失。

**作用阶段。** 量化、全局内存访问和 kernel launch。

**正确性。** 完整 `S^2` 候选在 1024 和 4096 的所有 S 上与 baseline 的 L2 error 一致；4096/S6 为 5.682e-15 对 5.685e-15，只是正常浮点累加顺序差异。

**性能与 profile。** 此项与布局、内存复用、反对角线聚合同时进入候选，未做单变量端到端实验。可以严格声称的是量化 launch 从 `2S` 降到 2、原矩阵读取从每 split 一次降到一次；不能把完整候选的 4.7x 至 5.7x 加速全部归因于它。

**决策。** 保留。基线 NCU 的内存受限证据与源码中的流量下降方向一致。

### 4.3 优化 3：量化 A 时直接生成 TN Tensor Core 布局

**触发证据。** 基线 NN INT8 GEMM 在 NCU 中执行非 Tensor Core FMA，单次约 16.85 ms；同尺寸 TN 对照约 2.38 ms，并显示 Tensor INT pipe 活跃。H800 单 pair microbenchmark 再次得到：

| H800/4096 单 pair | Time (ms) | Throughput (TOPS) |
|---|---:|---:|
| NN | 11.2237 | 12.25 |
| TN default | 1.4242 | 96.50 |
| TN AUTOTUNE | 1.4217 | 96.67 |
| cuBLASLt | 1.4212 | 96.71 |

TN 相比 NN 的单变量加速为 7.88x，这是整条手写路线中证据最强的一项优化。

**目的。** 让 cuBLAS 选择 regular-layout IMMA Tensor Core kernel，而不是低吞吐的 NN fallback；同时避免单独做一次完整矩阵转置。

**修改。** A 的量化 kernel 使用 32x32、带 padding 的 shared-memory tile，在生成 INT8 split 时直接写成转置后的列主序 `(S*K)-by-M` 布局；B 保持正常列主序。GEMM 改为 TN。padding 用于避免共享内存转置中的规则 bank conflict。

**作用阶段。** 量化输出布局、shared-memory 访问和 INT8 GEMM。

**正确性。** 只改变物理布局和 GEMM transpose flag，不改变量化值或数学乘积。完整 pair 候选的 L2 与 baseline 一致。

**性能与 profile。** 除上述 H800 单 pair 7.88x 外，A100 NCU 枚举出的候选 kernel 为 `cutlass_80_tensorop_i16832...tn_align16`，直接证明 Tensor Core 路径被命中。这里可以把单 pair 的提升归因于布局；端到端提升仍包含其他改动。

**决策。** 保留在 CC8 手写路径。H800 正式 vendor 路径由库内部完成布局选择。

### 4.4 优化 4：持久 grow-only workspace，移除热路径分配和释放

**触发证据。** 整条基线 benchmark trace 出现 73 次 `cudaFree`，host API 合计约 142 ms，其中包含计时外 correctness buffer；但源码进一步确认每次朴素 S 路径在计时函数内部确实执行 `2S+5` 次 free。`cudaFree` 还可能隐含设备同步。工作区尺寸只由 M、N、K、S 和路径选择决定，没有必要每次调用重新分配。

**目的。** 把 allocation 成本移到首次调用/扩容，预热后不再 `cudaMalloc/cudaFree`，并保持异步 stream 语义。

**修改。** 该优化分两阶段演化。历史 full-pair candidate 先使用进程级 `g_workspace` 做 grow-only 缓存，避免尺寸不变时重复分配，但仍有 host max-abs、函数尾同步且不可重入；最终版本改为 `thread_local Workspace`，保存 quantized A/B、INT32 partial、scales、max-abs partial 和 event。buffer 只增不减，容量不足时才扩容，调用结束不做 host synchronize 或 free。

**作用阶段。** host 调度、内存管理和跨调用生命周期。

**正确性。** grow-only 容量检查防止不同矩阵尺寸复用过小 buffer。一次 host 线程切换 CUDA stream 时，用 disable-timing event 串联前后调用，防止上一 stream 尚未完成时覆盖 workspace。A100 多 stream probe 的两路结果 L2 均约 5.373e-10。

**性能与 profile。** 没有为 allocation 一项单独做正式 benchmark，因此只报告结构性变化：历史 candidate 包含早期 persistent buffer，最终实现进一步移除了 host max-abs/尾部同步并增加 TLS/event；不能把 4.6 节的整体加速归因于最终 TLS/event 版本。

**决策。** 保留。共享同一 cuBLAS handle 的多 host 线程仍需要调用方互斥；thread-local workspace 不能自动让外部 handle 线程安全。

### 4.5 优化 5：反对角线 INT32 聚合，减少重组和 C 的读改写

**触发证据。** 权重只由 `d=i+j` 决定，因为 `s_i^A s_j^B` 在固定缩减率下具有相同数量级。基线却让每个 pair 生成完整 INT32 矩阵并单独更新 FP64 C。A100 nsys 的多调用 trace 汇总了 64 个重组 instance、总计约 58.65 ms；单个重组 NCU 显示 87.1% DRAM throughput、compute 仅约 8%，属于典型内存带宽浪费。

**目的。** 让同一反对角线的多个 pair 在 INT32 中先合并，只为每个 d 保留一个部分矩阵，把 FP64 重组次数从 `S^2` 降到最多 `2S-1`，并避免每个 pair 都读写 C。

**修改。** 对每条反对角线，第一个 GEMM 使用 `beta=0`，后续 GEMM 使用 `beta=1` 累加到同一个 INT32 buffer。早期 full-pair 候选顺序复用一个 buffer，并在完成该 d 后启动一次 FP64 recombine kernel；后续版本再把各条保留对角线同时保存并融合最终重组，见 4.7 节。

**作用阶段。** GEMM 调度、INT32 中间结果和 FP64 重组。

**正确性。** 初始 full-pair 候选保留全部 `S^2` 项，因此数学上只改变结合顺序。4096 所有 S 的 L2 与 baseline 一致。

**性能与 profile。** 这项也没有独立端到端数据。profile 同时揭示了代价：H800 手写路径的 21 次对角线 GEMM 平均约 1.85 ms，而单独 `beta=0` microbenchmark 为 1.42 ms；`beta=1` 会读取并改写旧 INT32 C，使每个 pair 变慢。因此它减少了重组和 buffer 数，但没有消除 pair GEMM 本身。

**决策。** 保留在手写路径，因为它把重组次数从 `S^2` 降到最多 `2S-1`；后续单 kernel 重组、extended-K 和 fused MMA 分别继续解决 C 的重复写回、`beta=1` 与 pair launch 问题。

### 4.6 历史 full-pair candidate 的组合收益

下面的数据来自 commit `6c9760c` 对应的历史 full-pair candidate。该版本包含优化 2（多 split 融合量化）、优化 3（TN 布局）、优化 5（反对角线 INT32 聚合）和优化 4 的早期全局 persistent buffer；它仍调用两次 host-returning `device_maxabs_fp64`、在函数尾同步，也没有最终的联合 device max-abs、thread-local workspace 或跨 stream event。因此不能把这张表写成“前五项”或最终实现全部工程优化的共同收益。A100 MIG/4096、三次平均结果为：

| S | Baseline (ms) | Full-pair candidate (ms) | 组合加速 | Candidate GFLOPS | Candidate / baseline L2 |
|---:|---:|---:|---:|---:|---:|
| 2 | 88.5088 | 18.6952 | 4.73x | 7351.58 | 2.192e-5 / 2.192e-5 |
| 4 | 303.1385 | 58.3588 | 5.20x | 2355.07 | 3.397e-10 / 3.397e-10 |
| 6 | 655.0364 | 119.3847 | 5.49x | 1151.23 | 5.682e-15 / 5.685e-15 |
| 8 | 1146.8824 | 201.6338 | 5.69x | 681.63 | 2.139e-15 / 2.150e-15 |

可验证的结论是：这一历史组合在保持 baseline 精度的情况下实现 4.73x 至 5.69x 端到端提升；其中 TN 布局另有 7.88x 单 pair 证据。联合 device max-abs、最终 TLS/event 和单-kernel 全对角线重组没有包含在这组 before/after 中，只能分别引用源码/profile 证据。

### 4.7 优化 6：把所有保留反对角线融合为一次 FP64 重组

**触发证据。** 早期反对角线聚合已把重组从每 pair 一次降为每 d 一次，但每次仍扫描并读改写完整 C。基线重组 NCU 显示 87.1% DRAM throughput、compute 仅约 8%，因此继续减少 C 的全矩阵写回比微调算术指令更重要。

**目的。** 让每个输出元素只由一个线程在寄存器中完成所有保留 d 的 FP64 FMA，并且只向 C 写一次；把重组 launch 从保留的 D 次进一步降到 1 次。

**修改。** 不再顺序复用一个 INT32 temp，而是为每条保留反对角线保存一个 `M*N` INT32 矩阵。模板化的 `recombine_kernel<Diagonals>` 对每个输出 index 依次读取这些矩阵，在 FP64 寄存器中累加，最后写一次 C。

**作用阶段。** INT32 中间 buffer、FP64 重组、C 的全局内存流量和 kernel launch。

**正确性。** 对角线仍按 d 递增顺序以 FP64 FMA 合并，没有额外减少 split/pair。最终裁剪路径的误差由 D 决定，S2/S4/S6/S8 的 L2 分别约为 2.684e-5、5.370e-10、1.008e-14、1.008e-14，与对应的裁剪实验一致。

**性能与 profile。** 该项没有单变量端到端 benchmark，不能从最终时间中单独拆出收益。源码可证明重组 launch 从 D 次降到 1 次、C 写回从 D 次降到 1 次；任务 188062 的 H800 nsys 也只观察到一次最终重组，约 2.27 ms。代价是 INT32 workspace 从一个矩阵增加到 D 个矩阵。

**决策。** 保留在 CC8 手写路径。它用可控的临时显存换取更少 launch 和 C 流量；H800 vendor 路径由 cuBLAS 内部完成融合。

## 5. 手写 Ozaki 路径：减少 pair 与验证融合上限

### 5.1 优化 7：按反对角线裁剪低权重 pair

**触发证据。** 每个 pair 的缩放权重近似随 `254^-(i+j)` 衰减。成本模型显示 full pair 数为 `S^2`，即使已按对角线聚合，S=8 仍需 64 次 GEMM。实验文档也建议优先保留 `i+j<d`，而不是随意删除 pair。

**目的。** 用可控精度损失换取更少 GEMM、INT32 buffer 和重组流量，重点改善高 S 路径。

**修改。** 仅保留 `i+j<D`。当 `D<=S` 时 pair 数从 `S^2` 降为 `D(D+1)/2`；最终兼容路径采用 `D=min(S,6)`，S=8 时只保留 21/64 个 pair。

**作用阶段。** GEMM 数、临时显存和重组。

**正确性。** 该优化会真实改变数值算法，不是无损调度。A100/4096 同环境消融为：

| S | Full D / pairs | Full time (ms) | Pruned D / pairs | Pruned time (ms) | 加速 | Full L2 | Pruned L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3 / 4 | 19.700 | 2 / 3 | 15.787 | 1.25x | 2.192e-5 | 2.684e-5 |
| 4 | 7 / 16 | 59.458 | 4 / 10 | 39.522 | 1.50x | 3.397e-10 | 5.370e-10 |
| 6 | 11 / 36 | 120.370 | 6 / 21 | 74.129 | 1.62x | 5.682e-15 | 1.008e-14 |
| 8 | 15 / 64 | 202.497 | 6 / 21 | 76.186 | 2.66x | 2.139e-15 | 1.008e-14 |

固定 D 后 S6/S8 得到相同误差，说明有效精度已由 D 而不是 nominal S 决定。增加到 `D=S+1` 可以接近 full-pair 精度，但性能也相应下降。

**性能与 profile。** 上表是同 probe 的单变量证据，明确显示 1.25x 至 2.66x 加速和对应精度退化。最终 H800 手写路径用该裁剪后，4096/S6、S8 为 2757、2650 GFLOPS，仍低于 3000 checkpoint。

**决策。** 仅保留为 CC8 兼容路径的性能折中，不用于正式 H800 结果。实验文档禁止擅自减小 `splits`；该实现虽仍生成 S 个分量，但舍弃低权重 cross terms，可能被视为等价的有效精度削减，因此正式报告必须披露，不能把它描述为无损优化。

### 5.2 失败实验 1：把同一反对角线拼成 extended-K GEMM

**触发证据。** 对角线聚合仍有多个 GEMM launch，并因 `beta=1` 发生 INT32 读改写。理论上可把 `(A_i,B_j)` 沿 K 维拼接，让一条反对角线只调用一次长 K GEMM。

**目的。** 把每条对角线的多个 pair GEMM 合成一个，减少 launch 和 `beta=1` 累加。

**修改。** 反向排列 A split、正向排列 B split，使固定 d 的各段在扩展 K 维上连续，单次 GEMM 计算这些 pair 的和。

**作用阶段。** GEMM 调度和 INT32 累加。

**正确性。** A100 重复实验与 H800 实验都保持裁剪路径相同 L2；失败原因不是数值错误。

**性能与 profile。** A100/4096、相同 D 下 extended-K 为 16.847/47.170/93.592/149.508 ms，而 separate-pair 约为 15.787/39.522/74.129/118.402 ms。H800/4096 更明显：S6 为 119.612 ms、1149 GFLOPS，S8 为 138.592 ms、992 GFLOPS；相对 separate-pair 的 49.86/51.86 ms 慢 2.4x 至 2.7x。

**结论。** 回退。减少 launch 不保证加速，扩展 leading dimension 和长 K 形状使 legacy kernel 调度/数据访问明显恶化。该结果反证了“launch 数最少即最优”的简单假设。

### 5.3 失败实验 2：legacy AUTOTUNE 与 cuBLASLt 算法搜索

**触发证据。** H800 单个 TN GEMM `beta=0` 可达约 96.5 TOPS，但实际对角线 pair 平均更慢；因此需要判断是否只是默认算法选择不佳。

**目的。** 在不改数据流的情况下寻找更适合 TN、`beta=1` 和当前尺寸的库 kernel。

**修改。** 分别测试 legacy `CUBLAS_GEMM_AUTOTUNE`、cuBLASLt 默认算法和 Lt algo scan。

**作用阶段。** 单个 INT8 GEMM 的库算法选择。

**正确性。** 乘法语义不变。

**性能与 profile。** 单 pair H800 default/AUTOTUNE/Lt 分别为 96.50/96.67/96.71 TOPS，差异不足 0.3%。完整手写 S6/S8 用 AUTOTUNE 仅约 2771/2645 GFLOPS，没有稳定改善。原项目只链接 `-lcublas`，直接使用 Lt 还会链接失败。

**结论。** 回退 AUTOTUNE/Lt 改造。瓶颈是多 pair 数据流和前后处理，不是一个隐藏的单 GEMM 算法。为绕过课程链接约束动态加载 Lt 也不值得引入额外复杂度。

### 5.4 失败实验 3：自写 groupwise fused MMA

**触发证据。** full/pruned 手写路径仍让多个 pair 经 HBM 落地，并在库 GEMM 间重复读写 INT32 buffer。理论上的终极优化是让一个 CTA 加载输出 tile 所需的多组 split，在寄存器中完成 INT32 MMA 累加，最后只做一次 FP64 epilogue 和一次 C 写回。

**目的。** 同时消除 pair launch、`beta=1` 中间矩阵流量和多次重组，增加 A/B tile 复用。

**修改。** 依次实现或测试 WMMA、直接 PTX MMA、shared staging、32x64/64x32/128x64 tile、double buffering、`ldmatrix` 与 swizzle 等变体。最好的候选按输出 tile 做 groupwise INT32 累加，再用 FP64 scale 重组。

**作用阶段。** INT8 GEMM 主循环、shared-memory pipeline、寄存器累加和 FP64 epilogue。

**正确性。** 可运行变体通过随机输入误差测试；最终回退原因是性能，不是正确性。

**性能与 profile。** A100/4096 最好 raw 128x64 变体的 S2/S4/S6/S8 时间约 16.68/52.93/114.67/115.38 ms，仍慢于最终 cuBLAS TN 裁剪路径的约 11.88/32.07/66.14/70.04 ms。其 NCU 数据约为 80 reg/thread、27.7 KiB shared memory、理论/实际 occupancy 37.5%/34.4%；69% 周期没有 eligible warp，long scoreboard 约 45.5%，LSU 约 38%。`ldmatrix` double-buffer 版本还出现 local spill，1024/S6 从约 2.14 ms 退化到 2.50 ms。SM80 风格 fused kernel 在 H800/4096 上更差：S2/S4/S6/S8 仅 6899/2207/1022/1014 GFLOPS。

**关于“拆 kernel 降寄存器”。** profile 确实显示融合扩大了 live range，但本实验没有得到一个经过实测且保留的“拆 kernel”版本，不能在报告里写成已完成优化。把 kernel 拆开可能降低寄存器，却会重新引入 HBM 中间结果和 launch；而最好候选的 achieved occupancy 已接近 theoretical occupancy，主要 stall 同时包含 long scoreboard 和 LSU，并非只由寄存器决定。因此这里正确的结论是“融合过度导致资源和延迟问题，候选回退”，而不是虚构一个拆分后的加速结果。

**结论。** 回退。自写 SM80 MMA 没有超过成熟的 cuBLAS TN kernel，更不适合作为 H800 正式路径。若必须继续手写 Hopper kernel，合理起点应是 CUTLASS SM90 warp-specialized/WGMMA 模板，而不是继续扩展 SM80 内核。

### 5.5 手写路径的阶段性上限

最终手写路径采用 `D=min(S,6)`。H800 实测为：

| Size | S | Time (ms) | GFLOPS | L2 relative error | 达到 `g100` |
|---:|---:|---:|---:|---:|---|
| 4096 | 2 | 9.379 | 14653 | 2.684e-5 | 是 |
| 4096 | 4 | 25.316 | 5429 | 5.370e-10 | 是 |
| 4096 | 6 | 49.857 | 2757 | 1.008e-14 | 否 |
| 4096 | 8 | 51.860 | 2650 | 1.008e-14 | 否 |
| 8192 | 2 | 55.217 | 19913 | 2.685e-5 | 是 |
| 8192 | 4 | 159.045 | 6913 | 5.374e-10 | 是 |
| 8192 | 6 | 321.789 | 3417 | 1.031e-14 | 是 |
| 8192 | 8 | 329.522 | 3337 | 1.031e-14 | 是 |

4096/S6 和 S8 分别比 3000 GFLOPS 目标低约 8.1% 和 11.7%。任务 188062 的 nsys 将 4096/S6 的约 49.76 ms 分解为：21 个 TN GEMM、A/B 量化各约 3.63/3.66 ms、max-abs 约 1.12 ms、最终重组约 2.27 ms。说明手写路径已不再被一个容易删除的 launch 单独支配，而是 GEMM 数、`beta=1` 累加和前后处理共同构成上限。

A100 正式兼容路径八项均正确，但按 H800 checkpoint 估算为 81.2544。这个 81 分是 A100 兼容路径结果，不是 H800 最终成绩。

## 6. 架构选择：从手写通用路径转向 H800 正式路径

### 6.1 优化 8：按 compute capability 分流

**触发证据。** 早期任务误提交到 `lab4g10`，实测为 A100/CC8.0；用户指出 `lab3` 有 H800 后，节点探测确认 H800/CC9.0 和 CUDA 13.3。手写路径在 H800 只通过 6/8 个性能点，而 CUDA 13.3 已为 Hopper 提供 FP64 fixed-point emulation，课程也把 `cublas_emulated` 作为满分性能参照。

**目的。** 不把 SM80 kernel、profile 和算法选择机械外推到 SM90；让每代架构使用已验证的最佳实现，同时保留无 emulation 环境的可运行回退。

**修改。** 初始化 workspace 时查询 compute capability：CC >= 9 调用 H800 vendor emulation 路径，CC 8.x 进入手写 TN Ozaki 路径。

**作用阶段。** 顶层算法/架构选择。

**正确性。** 两条路径分别回归。H800 vendor 路径八项 max-abs 和 L2 与项目 `cublas_emulated` comparator 逐项一致；A100 手写路径也通过其兼容性测试。

**性能与 profile。** 直接链接正式提交文件的 H800 vendor 路径三次平均为 10473/5558/3284/3259 GFLOPS（4096）和 12882/5913/3304/3305 GFLOPS（8192），八项全过；手写路径在 4096/S6/S8 为 2757/2650，未过。使用相同 emulation、bit、workspace 和 GEMM 配置的隔离 harness 在 nsys 中执行 SM90 fused IMMA kernel。前一证据验证正式 dispatch 的端到端结果，后一证据验证 vendor compute 配置的内部 kernel；profile harness 本身不包含 runtime dispatch 和 event wrapper。

**决策。** 正式保留，这是达到 100 分的关键架构决策。CC8 手写路径保留为兼容实现，而不是最终性能主线。

### 6.2 合规边界

实验文档的提交要求只明确限制提交 `submit/my_int8_fp64.cu`，没有明文禁止在学生入口调用公开的 cuBLAS fixed-point emulation API；文档也专门介绍并测量该 API。但正文写明实验“着眼于”手写 Ozaki 路线，因此存在课程意图层面的非技术风险。

正式报告应如实写明：100 分路径使用 CUDA 13.3 公开 emulation API，手写 Ozaki 路径和失败优化也有完整实验。如果助教另有未写入文档的禁用规则，则需要回到手写 SM90 CUTLASS/WGMMA 路线；不能把 vendor kernel 伪装成自写 kernel。

## 7. H800 正式路径的工程优化

### 7.1 优化 9：使用完整 fixed-point mantissa 配置

**触发证据。** 正确性先于性能，课程 comparator 使用 FIXED mantissa control。pair 裁剪已经展示了“更少工作”会降低有效精度，因此正式路径首先要固定精度目标，而不是先追求最快参数。

**目的。** 使正式输出与课程 `cublas_emulated` 精度一致，保留隐藏正确性阈值的安全余量。

**修改。** 配置：

```text
CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH
CUBLAS_EMULATION_STRATEGY_EAGER
CUDA_EMULATION_MANTISSA_CONTROL_FIXED
max_mantissa_bits = min(8 * splits, 55)
CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT
CUBLAS_GEMM_DEFAULT
```

55 是为复现课程/default FIXED 配置采用的值，不是 cuBLAS 文档声明的普适理论硬上限。fixed-point slice 数为 `ceil((bits+1)/8)`，所以 S=2/4/6/8 实际对应 3/5/7/7 个 slice，这解释了 S6 和 S8 的时间与误差几乎相同。

**作用阶段。** H800 量化精度、内部 slice 数和 GEMM/recombine 工作量。

**正确性。** 八项 L2 与 comparator 一致，S6/S8 已接近 FP64 舍入误差量级。

**性能与 profile。** 正式表中八项全部过 `g100`。参数消融显示减少 1 bit 虽更快，但误差显著恶化，见 7.5 节。

**决策。** 保留 full course bits。这里的优化目标是“用刚好可解释且安全的配置达到满分”，不是无条件减少 bit。

### 7.2 优化 10：修复 cuBLAS handle 状态设置顺序

**触发证据。** cuBLAS API 语义规定，`cublasSetStream` 会把 handle 的 user workspace 重置到默认 pool。仓库 comparator 的调用顺序是先 SetWorkspace、后 SetStream，因此传入的 workspace 不会按预期保持注册。共享 handle 还可能遗留 pointer mode。

**目的。** 确保正式路径真正以指定 stream、user workspace 和 host alpha/beta 执行，避免调用方状态污染和潜在内部 fallback allocation。

**修改。** 严格使用：

```text
1. cublasSetStream(handle, stream)
2. cublasSetWorkspace(handle, tls_workspace, 2 GiB)
3. cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST)
4. 设置 math mode / strategy / mantissa control / bits
5. cublasGemmEx(...)
```

**作用阶段。** cuBLAS host 调度和 workspace 注册。

**正确性。** HOST pointer mode 明确保证 alpha/beta 地址解释正确；八项结果与 comparator 一致。

**性能与 profile。** 没有把“状态顺序”作为单变量正式计时，因此不声称独立加速。任务 187641 的隔离 harness 使用与正式 H800 分支相同的 `SetStream -> SetWorkspace` 顺序，其 CUDA API trace 未观察到内部 `cudaMallocAsync` 或其他可见 fallback allocation，支持该配置下 workspace 注册生效且当前规模资源充足。该证据不能证明 cuBLAS 实际使用了全部 2 GiB，也不直接验证正式入口外围的 dispatch/event。

**决策。** 保留。这是 API 正确性和稳定性修复，价值不应只用某次波动中的百分比衡量。

### 7.3 优化 11：2 GiB thread-local user workspace

**触发证据。** fixed-point emulation 会产生内部量化/slice 工作区。若每次调用分配，allocation 会进入端到端时间；若 workspace 不足，库可能使用不同策略或内部分配。项目 benchmark 允许 32 GiB 任务内存，单独 2 GiB 可稳定覆盖公开规模。

**目的。** 为 cuBLAS 提供稳定的大 workspace，并把分配摊到首次调用；预热后的计时区间不分配、不释放。

**修改。** H800 路径在 `thread_local Workspace` 中 grow-once 分配 2 GiB vendor buffer。一次 host 线程跨 stream 复用时，用 event 建立前一次调用完成到下一次调用开始的依赖。

**作用阶段。** 内存管理、cuBLAS 内部算法资源和跨调用调度。

**正确性。** event 防止异步 GEMM 尚未完成时复用同一 buffer。正式 benchmark 在 4096/8192 全方法调用链下没有 OOM；需要注意 benchmark 还为 comparator 另外分配 2 GiB，两块不是同一 workspace。

**性能与 profile。** 等价 vendor harness 的 nsys 未见可见 fallback allocation；直接链接正式提交文件的 10 次 lightweight 回归稳定过线。没有 workspace size sweep 的单变量曲线，因此只写“2 GiB 在公开规模实测充分”，不写成理论最小值。

**决策。** 保留。它以约 2 GiB 常驻显存换取固定注册、预热后无分配且在公开规模可运行的热路径；没有 workspace size sweep，不能声称它改变或稳定了具体算法选择。

### 7.4 优化 12：跨 stream event 保护与异步语义

**触发证据。** thread-local 只隔离 host 线程，不隔离同一线程先后使用的不同 CUDA stream。cuBLAS 调用是异步的；如果下一次调用立刻复用 buffer，会产生数据竞争。直接 `cudaStreamSynchronize` 虽正确，但会阻塞 host 并破坏流水化。

**目的。** 在不引入全设备/host 同步的前提下安全复用 workspace。

**修改。** 每次调用结束在当前 stream record disable-timing event；下一次若切换 stream，新的 stream 用 `cudaStreamWaitEvent` 等待该 record，再复用 buffer。

**作用阶段。** 跨调用调度和异步依赖。

**正确性。** 多 stream probe 已验证手写路径两路 L2 一致；正式 H800 路径遵循相同 happens-before 原则。共享同一外部 cuBLAS handle 的多个 host 线程仍不在本实现的保证范围内。

**性能与 profile。** 没有单独加速数据，也没有对 event wrapper 做专门 trace。源码中没有为 workspace 安全添加 host/global synchronize，而是用 stream wait event 表达依赖；A100 multistream probe 验证了跨 stream 结果正确。任务 187641 的隔离 harness 不包含 event wrapper，不能用来证明这一点。

**决策。** 保留，属于并发正确性和可组合性优化。

### 7.5 参数消融：哪些看似更快的设置被回退

参数实验都在 H800 上完成。它们按“提出假设 -> 同环境测量 -> 看精度和端到端性能 -> 决策”处理。

#### 7.5.1 减少 1 个 mantissa bit

**触发与目的。** slice 数公式在跨过 8-bit 边界时会少一个内部 slice，因此 `8*S-1` 可能显著减少工作量。

**结果。** H800/4096：

| S | Time (ms) | GFLOPS | L2 relative error |
|---:|---:|---:|---:|
| 2 | 8.841 | 15545 | 1.240e-4 |
| 4 | 18.312 | 7505 | 2.305e-9 |
| 6 | 32.704 | 4202 | 4.035e-14 |
| 8 | 41.985 | 3273 | 2.140e-15 |

S2/S4/S6 明显更快，但 L2 相对正式 full-bit comparator 分别恶化约 230x、237x、19x；若只和手写 Ozaki baseline 比较，才约为 5.7x、6.8x、7.1x。正式路径的精度目标是前者，公开文档又没有给出所有隐藏阈值，因此不能用这部分正确性余量换速度。

**决策。** 回退。这是有单变量性能收益、但正确性风险不可接受的失败优化。

#### 7.5.2 关闭 special-values 支持

**触发与目的。** 随机输入不含 NaN/Inf，理论上关闭特殊值处理可能减少分支或打包开销。

**结果。** 任务 187603 的各规模变化混合；4096/S2 反而约慢 1.4%，其他点没有稳定重复收益。

**Profile 判断。** 此候选没有单独执行 nsys/NCU，不能声称 kernel 结构发生或未发生变化；回退依据只是端到端变化混合、S2 约慢 1.4%，以及关闭特殊值支持会缩窄输入语义。

**决策。** 回退，避免为了无稳定收益缩窄输入语义。

#### 7.5.3 使用 `CUBLAS_GEMM_AUTOTUNE`

**触发与目的。** 让库为固定尺寸搜索更快 kernel。

**结果。** 4096 约为 13.323/24.721/42.109/42.117 ms，8192 约为 84.937/185.801/331.206/331.219 ms；相对 default 混合且部分点更慢，没有稳定净收益。

**决策。** 回退到 `CUBLAS_GEMM_DEFAULT`。正式路径已经有 3% 以上最小余量，没必要引入不可预测的 autotune 成本或版本敏感性。

## 8. 最终 H800 Profile：优化是否真的生效

### 8.1 Nsight Systems：验证 vendor compute 配置与端到端结构

任务 187641 使用 `tools/benchmark_my.cu` 和 `tools/my_cublas_emu_candidate.cu`，对 4096/S2 和 S8 采样。这个隔离 harness 与正式 H800 分支具有相同的 `SetStream -> SetWorkspace -> HOST pointer mode`、EAGER/FIXED、mantissa bits 和 `cublasGemmEx` 配置，但没有 compute-capability dispatch 和跨调用 event wrapper：

- 端到端约 13.293 ms / 41.958 ms，与普通计时一致；
- 主 kernel 名称为 `cublasLt_fused_imma_dgemm_kernel_sm90`，直接证明实际进入 Hopper fused INT8 emulation，而不是普通 FP64 fallback；
- 4 次 fused kernel 合计约 94.75 ms，平均 23.69 ms；
- 8 次 `max_scale_pack` 合计约 15.27 ms，平均 1.91 ms；
- CUDA API 摘要没有内部 `cudaMallocAsync`，未观察到可见 fallback allocation。

这个 profile 验证了两项关键结论：该 vendor compute 配置确实使用 SM90 IMMA；按正确顺序注册 user workspace 后没有观察到可见的热路径内部 allocation。架构分流由正式源码以及直接链接 submit 的任务 188025/188193 验证；跨 stream event 的 happens-before 由源码和 A100 multistream probe 验证，正式 H800 回归只证明它没有破坏常规单 stream 路径。这些外围行为都不由任务 187641 直接证明。profile 也没有证明 2 GiB 每个字节都被消费，不能把多个 S 的 kernel 平均时间直接当成某一个 S 的端到端时间。

### 8.2 Nsight Compute：定位最终最小余量点

任务 187652 使用同一个隔离 vendor harness，采样 4096/S2 fused kernel：

| 指标 | 值 |
|---|---:|
| Kernel duration | 9.71 ms |
| Grid / block / cluster | 12 blocks / 512 threads / cluster 4 |
| Registers | 128 / thread |
| Dynamic shared memory | 230.40 KB / block（约 225 KiB） |
| Theoretical / achieved occupancy | 25.00% / 21.84% |
| DRAM / compute throughput | 65.83% / 45.87% |
| L1 / L2 hit rate | 71.84% / 44.80% |
| Cycles with no eligible warp | 86.97% |
| Spill | 未观察到 local/shared spill |

MIG 有 14 SM，而 grid 只有 12 blocks，NCU 给出约 14.29% 的局部并行度提示；主要 stall 为 pipe 和 long scoreboard，各约 8.3 cycles。该 profile 解释了 4096/S2 为什么是最终最薄弱的点：小规模下内部 cluster kernel 有 underfill，且有限 occupancy 难以完全隐藏依赖延迟。

这里不能照搬自写 kernel 的“拆 kernel 降寄存器”思路。正式 kernel 由 cuBLAS 内部生成，应用层不能调整 tile/cluster；并且它没有 spill，拆分还会失去 fused 数据复用。实际可控变量是 CUDA/cuBLAS 版本、workspace、emulation strategy 和 mantissa bits，其中只有 full-bits/default 组合同时满足精度和稳定性能。

## 9. 最终性能、稳定性与评分

### 9.1 项目原生三次回归

任务 188025 使用项目原生 benchmark、`sm_90a -O3`，结果如下：

| Size | S | Time (ms) | GFLOPS | Max abs error | L2 relative error | `g100` | 余量 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 2 | 13.1230 | 10473.12 | 6.621e-5 | 5.395e-7 | 10000 | +4.73% |
| 4096 | 4 | 24.7270 | 5558.25 | 1.146e-9 | 9.705e-12 | 5000 | +11.17% |
| 4096 | 6 | 41.8516 | 3283.96 | 6.253e-13 | 2.140e-15 | 3000 | +9.47% |
| 4096 | 8 | 42.1678 | 3259.33 | 6.253e-13 | 2.140e-15 | 3000 | +8.64% |
| 8192 | 2 | 85.3533 | 12881.89 | 9.317e-5 | 5.398e-7 | 10000 | +28.82% |
| 8192 | 4 | 185.9516 | 5912.89 | 1.759e-9 | 9.712e-12 | 5000 | +18.26% |
| 8192 | 6 | 332.8228 | 3303.59 | 1.506e-12 | 3.019e-15 | 3000 | +10.12% |
| 8192 | 8 | 332.7106 | 3304.71 | 1.506e-12 | 3.019e-15 | 3000 | +10.16% |

同一任务中，八项 `my_int8_fp64` 的 max-abs 与 L2 均和 `cublas_emulated` comparator 逐项一致。

### 9.2 正式提交函数十次稳定性回归

完整项目 benchmark 还会执行极慢的朴素 baseline，十次全方法任务可能超过 lab3 的 5 min walltime。为验证 student 函数本身，任务 188193 使用 lightweight harness 做十次平均：

```text
4096: 10335.58, 5542.97, 3284.80, 3284.79 GFLOPS
8192: 13034.42, 5983.72, 3350.36, 3352.28 GFLOPS
```

最小余量仍为 4096/S2 的 +3.36%，八项继续全部过 `g100`。这组结果用于稳定性验证，不冒充项目全方法十次 benchmark。

### 9.3 得分计算

每个规模下四项均封顶 100：

```text
Score_4096 = 0.4*100 + 0.2*100 + 0.2*100 + 0.2*100 = 100
Score_8192 = 0.4*100 + 0.2*100 + 0.2*100 + 0.2*100 = 100
Final score = 0.5*Score_4096 + 0.5*Score_8192 = 100.0000
```

## 10. 优化决策总表

| 优化或实验 | 主要触发证据 | 目的 | 最强结果/验证 | 最终决策 |
|---|---|---|---|---|
| 设备端联合 max-abs | 源码 D2H/同步 | 去 host 同步 | 手写 nsys 中仅约 1.12 ms；无独立加速比 | CC8 保留 |
| 一次生成全部 split | 量化 DRAM 87.1%，`2S` launches | 原矩阵只读一次 | full candidate 精度不变；组合收益 | CC8 保留 |
| 量化时生成 TN 布局 | NN 非 Tensor Core；TN profile | 命中 IMMA | H800 单 pair 7.88x；kernel 名含 tensorop TN | CC8 保留 |
| persistent workspace | 73 次 free、约 142 ms host API | 移出热路径分配 | 预热后无显式 free；多 stream 正确 | 两路径保留 |
| 反对角线 INT32 聚合 | 重组 DRAM 87.1%；trace 汇总 64 instances | 少写 INT32/C | full-pair 精度一致；但 beta=1 较慢 | CC8 保留 |
| 单 kernel 融合全部 d 的重组 | 每 d 重组仍重复扫描 C | C 只写一次 | nsys 仅一次重组约 2.27 ms；无独立加速比 | CC8 保留 |
| pair pruning | 权重按 `254^-d` 衰减 | 少 GEMM/中间流量 | S8 2.66x，L2 退化到 1.008e-14 | 仅 CC8 折中 |
| extended-K | pair launch 与 beta=1 | 一条 d 一次 GEMM | H800 慢 2.4x 至 2.7x | 回退 |
| legacy AUTOTUNE/Lt | 单 pair 算法疑问 | 找更快 TN kernel | 96.50 vs 96.67/96.71 TOPS，几乎无差 | 回退 |
| groupwise fused MMA | pair HBM/launch | tile 内融合计算与重组 | A100/H800 均慢；long scoreboard/LSU 高 | 回退 |
| CC 分流 | A100/H800 架构差异 | 使用各架构最佳路径 | H800 从 6/8 过线到 8/8 | 正式保留 |
| full mantissa bits | 正确性门控 | 对齐 comparator 精度 | 八项 L2 一致 | 正式保留 |
| stream -> workspace 顺序 | cuBLAS API 语义 | 防 workspace 被重置 | 等价 vendor harness 的 nsys 未见可见 fallback allocation | 正式保留 |
| HOST pointer mode | 共享 handle 遗留状态 | 正确解释 alpha/beta | 八项正确 | 正式保留 |
| 2 GiB TLS workspace + event | allocation 与异步复用 | 稳定热路径 | 十次回归全过；跨 stream 有依赖 | 正式保留 |
| bits=`8S-1` | slice 边界模型 | 少内部 slice | S2/S4/S6 更快，但相对 full bits 的 L2 恶化约 230x/237x/19x | 回退 |
| special-values NONE | 随机输入无 NaN/Inf | 减少特殊值开销 | 变化混合，S2 约慢 1.4% | 回退 |
| emulation AUTOTUNE | 可能有更快内部 kernel | 自动选算法 | 无稳定收益 | 回退 |

## 11. 如何解释“为什么不是继续优化手写 kernel”

达到 100 分不是因为“做不动了所以调用库”，而是经过 profile 和消融后做出的工程选择：

1. 手写路径已经从不可用占位实现推进到 H800 6/8 个满分点，并完成布局、量化、pair 和融合层面的多轮验证；
2. 剩余 4096/S6/S8 不是单一 launch 或一个寄存器参数导致，nsys 显示 GEMM、量化、max-abs、重组都有可见占比；
3. extended-K 和自写 groupwise MMA 都针对剩余瓶颈进行了实测，但分别慢 2.4x 以上和显著落后 cuBLAS TN；
4. 正式硬件是 H800/SM90，而现有自写 kernel 主要基于 SM80 MMA。继续走手写路线需要重写为 TMA/WGMMA/warp-specialized pipeline，开发风险和验证成本很高；
5. CUDA 13.3 已提供与任务完全同构的 SM90 fused emulation，profile 确认真正使用 INT8 IMMA，并在完整精度下稳定达到八项 checkpoint。

因此最终方案是“保留手写路径作为算法与兼容性成果，正式 H800 采用已验证的 SM90 实现”，而不是在满分已封顶后用高风险 kernel 替换稳定路径。

## 12. 局限、风险与后续工作

### 12.1 当前局限

- 4096/S2 的十次回归余量最小，为 +3.36%；CUDA、driver 或 cuBLAS 版本变化后必须复测。
- 正式性能依赖 CUDA 13.3 fixed-point emulation 的内部 kernel 选择，应用层不能控制其 12-block/cluster-4 调度。
- nsys 只能证明未观察到可见 fallback allocation，不能证明库消费了全部 2 GiB workspace。
- thread-local workspace 支持单 host 线程跨 stream 复用；共享同一个 cuBLAS handle 的多 host 线程仍需外部同步。
- vendor API 的课程合规性取决于书面规则。当前文档没有禁止，但报告必须公开实现边界。

### 12.2 若必须继续纯手写 SM90 路线

下一阶段不应继续微调已有 SM80 WMMA kernel，而应按以下顺序开展，并坚持单变量 profile：

1. 先用 CUTLASS Profiler 找到 H800 MIG 上适合 i8 TN、当前 M/N/K 的 SM90 warp-specialized baseline；
2. 保持 mainloop 不变，只替换 epilogue，使一个输出 tile 合并若干 split pair，测 DRAM bytes、寄存器、occupancy 与 long scoreboard；
3. 再把 A/B quantized tile 的布局与 TMA copy 对接，分别测 producer/consumer shared-memory bank conflict；
4. 对 tile shape、stage 数、producer/consumer warp 数逐项扫描，避免同时改变多个因素；
5. 每个候选同时跑 max-abs、L2、4096/8192 全 S 和端到端时间，不能用单 GEMM TOPS 代替最终性能。

这部分是未来方案，不应写成已经完成的优化。

## 13. 复现方法与证据索引

### 13.1 正式构建与回归

```bash
hpc submit --export NONE -p lab3 -g 1 -c 4 -m 32Gi -t 5m \
  /bin/bash -lc '/usr/local/cuda/bin/nvcc \
    -arch=sm_90a -O3 -std=c++17 -lineinfo -Iinclude \
    benchmark.cu baseline/baseline_fp64.cu \
    baseline/cublas_baseline.cu baseline/cublas_emulated.cu \
    utils.cu submit/my_int8_fp64.cu \
    -lcublas -lcudart -lcuda -o /tmp/lab45-benchmark && \
    /tmp/lab45-benchmark 4096,8192 2,4,6,8 3 --csv'
```

完整 benchmark 会同时执行慢 baseline。任务 188193 的十次 student-only 稳定性测试使用 `tools/benchmark_my.cu` 直接链接正式提交文件：

```bash
/usr/local/cuda/bin/nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo \
  -Iinclude tools/benchmark_my.cu submit/my_int8_fp64.cu \
  utils.cu -lcublas -lcudart -lcuda -o /tmp/lab45-submit

/tmp/lab45-submit 4096,8192 2,4,6,8 10
```

为降低 profiler 中 reference、dispatch 和 event 等外围噪声，任务 187641/187652 改用只保留相同 vendor compute 配置的 candidate：

```bash
/usr/local/cuda/bin/nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo \
  -Iinclude tools/benchmark_my.cu tools/my_cublas_emu_candidate.cu \
  utils.cu -lcublas -lcudart -lcuda -o /tmp/lab45-profile

nsys profile -t cuda,cublas -o /tmp/lab45 \
  /tmp/lab45-profile 4096 2,8 1

ncu --clock-control none --kernel-name-base function \
  --kernel-name regex:cublasLt_fused_imma_dgemm_kernel_sm90 \
  /tmp/lab45-profile 4096 2 1
```

### 13.2 关键源码和原始记录

| 内容 | 位置或 Job ID |
|---|---|
| 最终双架构实现 | `submit/my_int8_fp64.cu` |
| 实验要求与评分公式 | `docs/Lab4.5-INT8-FP64-GEMM/index.md` |
| full-pair 手写候选与结果 | commit `6c9760c` 的 `tools/my_int8_fp64_candidate.cu`、`tools/my_int8_fp64_candidate_results.md`；当前同名源码已演化为裁剪版 |
| pair 裁剪 probe | `tools/pair_pruning_probe.cu`、`tools/pair_pruning_*.out` |
| extended-K probe | `tools/diagonal_concat_probe.cu`、`tools/diagonal_concat_*.out` |
| groupwise MMA 变体 | `tools/groupwise_*.cuh`、`tools/my_int8_fp64_v2.cu` |
| H800 节点探测 | Job 187511 |
| H800 vendor compute harness 的 nsys / ncu | Job 187641 / 187652 |
| H800 手写全组合回归 | Job 187665 / 187963 |
| H800 手写 legacy AUTOTUNE | Job 187996 |
| H800 extended-K | Job 188007 |
| H800 NN/TN/Lt 单 pair 对照 | Job 188054 |
| H800 手写 nsys | Job 188062 |
| H800 SM80 风格 fused MMA | Job 188082 |
| 参数消融 | Job 187603 / 187619 / 187632 |
| 最终项目三次回归 | Job 188025 |
| student-only 十次回归 | Job 188193 |

## 14. 参考资料

1. [本实验文档：Lab 4.5 INT8-FP64 GEMM](docs/Lab4.5-INT8-FP64-GEMM/index.md)
2. [NVIDIA cuBLAS Documentation：Floating Point Emulation、workspace 与 stream](https://docs.nvidia.com/cuda/cublas/)
3. [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
4. [NVIDIA MIG User Guide：Supported MIG Profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html)
5. [Ozaki et al., Error-free transformations of matrix multiplication](https://doi.org/10.1007/s11075-011-9478-1)
6. [Uchino, Ozaki and Imamura, groupwise INT32 accumulation](https://doi.org/10.1177/10943420241313064)

## 15. 可直接用于正式报告的结论

本实验首先根据 nsys/NCU 将朴素 Ozaki 路径的瓶颈分为四类：量化重复读取与 launch、NN 布局未命中 Tensor Core、`S^2` 部分积与重组流量、热路径资源管理。历史 full-pair candidate 将多 split 融合量化、TN 布局、反对角线 INT32 聚合和早期 persistent buffer 合入后，在保持 baseline 精度的情况下使 A100/4096 端到端性能提升 4.73x 至 5.69x；其中 TN 布局由 H800 单 pair 7.88x 和 NCU Tensor Core kernel 名直接验证。后续联合 device max-abs、TLS/event 与单-kernel 重组没有混入这一加速比。进一步的 pair 裁剪可在高 S 获得最高 2.66x 加速，但带来可测精度退化，只适合兼容路径。

针对剩余 pair launch 和中间流量，extended-K 与 groupwise fused MMA 均进行了实测。extended-K 在 H800 上慢 2.4x 至 2.7x；融合 MMA 受到 long-scoreboard、LSU、寄存器 live range 和 SM80 kernel 与 H800 不匹配等问题影响，也未超过 cuBLAS TN。这些失败结果确认手写路径的剩余瓶颈不是单一 launch 或寄存器参数可以解决。

在确认正式节点为 lab3/H800、CUDA 13.3 后，实现按 compute capability 分流。H800 路径使用 full-bit fixed-point emulation，修复 `SetStream -> SetWorkspace` 顺序，显式设置 HOST pointer mode，并通过 2 GiB thread-local workspace 与 event 保护消除热路径分配和跨 stream 复用风险。采用相同 vendor compute 配置的隔离 harness 在 nsys 中执行 `cublasLt_fused_imma_dgemm_kernel_sm90` 且未见可见 fallback allocation；NCU 给出了 128 reg/thread、21.84% achieved occupancy、12 blocks 对 14 SM 的 underfill 等最终瓶颈证据。项目原生三次回归和直接链接提交文件的十次回归均使八个测试点超过 `g100`，最终得分为 100.0000。
