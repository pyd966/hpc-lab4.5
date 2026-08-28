# Lab 4.5 远程配置、Profile 与 100 分优化方案

> 记录时间：2026-08-28（UTC）
>
> 目标：在 4096^3、8192^3 和 splits=2/4/6/8 的全部组合上通过正确性，并达到实验文档的 100 分 GFLOPS checkpoint。
> 范围：本文记录实际远程节点、基线与 profile 证据，并给出可执行的优化路线。当前尚未修改 `submit/my_int8_fp64.cu` 的占位实现。

## 0. 结论先行

1. **当前远程节点不是实验文档所写的 H800。** 2026-08-28 实际 `lab4g10` 分区是 NVIDIA A100 80GB PCIe 的 `1g.10gb` MIG，CC 8.0、14 SM、9728 MiB；因此必须用 `sm_80` 构建。文档中的 H800、`sm_90a`、TMA、WGMMA 路线不能用于本次实际节点。
2. **当前学生实现一定得 0 分。** 两个 TODO 仍是占位逻辑，量化恒为 0，重组不写 C，实测 L2 相对误差为 1。
3. **当前 NN INT8 GEMM 没有走 Ampere Tensor Core。** Nsight Compute 显示 `cutlass1x` kernel 由普通 FMA/整数管线主导；4096^3 单 pair 约 15.6 ms，仅 8.8 TOPS。
4. **把 A 在量化时直接写成转置布局，再调用 `cublasGemmEx(T,N)`，即可命中 SM80 IMMA。** 实测单 pair 为 2.40 ms/57.3 TOPS（4096）和 23.57 ms/46.6 TOPS（8192，AUTOTUNE），相对 NN 快约 5.3 至 6.6 倍。无需 cuBLASLt，也不改变最终链接依赖。
5. **pair 裁剪只能作为辅助手段。** 保留 `i+j<D` 时，`D<S` 会直接退化成 D 级精度；`D=S` 增大误差常数；`D=S+1` 才基本恢复 full-pair 误差。当前逐 pair cuBLAS 方案即使裁剪也无法让所有 splits 满分。
6. **100 分主路线必须做到 tile 级 groupwise accumulation。** 同一反对角线的所有 INT8 GEMM 应在同一个 SM80 MMA mainloop 中累加到 INT32 fragment，随后在 epilogue 转成 FP64、乘 scale，并让一个 CTA 在寄存器中跨反对角线累加最终 C。这样才能消除逐 pair 的完整 INT32 中间矩阵和反复 C 读写。
7. **当前平台无法直接验证“文档宣称的 100 分等于 cuBLAS emulation 性能”。** 当前 A100 上 vendor emulation 只有约 0.83 至 3.83 TFLOPS，远低于固定满分 checkpoint 3 至 10 TFLOPS。提交前必须向助教确认评测硬件究竟是 H800 还是当前 A100。

## 1. 远程节点实测

### 1.1 提交方式

所有 GPU 性能与 profile 均通过 `hpc submit -p lab4g10 -g 1 ...` 在计算节点完成，没有在 DevPod 直接运行。主要任务如下：

| Job ID | 用途 |
|---:|---|
| 185358 | 节点、GPU、CUDA、CPU、内存和 profiler 探测 |
| 185377 | `make ARCH=sm_80 -j16` |
| 185394 | 4096/8192、全部 splits 的完整基线 |
| 185404 | Nsight Systems，4096、splits=4 |
| 185411/185413/185418/185423 | Nsight Compute：量化、重组、NN IGEMM、vendor emulation |
| 185521/185526 | TN IMMA microbenchmark 的 nsys/ncu |
| 185614 | 两种规模的完整反对角线裁剪扫描 |
| 185658 | NN、TN、TN AUTOTUNE、cuBLASLt 最终对照 |

节点探测命令的核心部分：

```bash
hpc submit -p lab4g10 -g 1 -t 5m \
  "hostname; lscpu; nvidia-smi -L; nvidia-smi; nvcc --version; \
   ncu --version; nsys --version"
```

### 1.2 实际配置

| 项目 | 实测值 |
|---|---|
| 分区显示名 | `Lab 4 A100 MIG 10G` |
| 物理 GPU | NVIDIA A100 80GB PCIe |
| MIG profile | `1g.10gb` |
| CUDA 设备名 | `NVIDIA A100 80GB PCIe MIG 1g.10gb` |
| Compute capability | 8.0 |
| SM 数 | 14 |
| 可见显存 | 9728 MiB |
| MIG copy engine | 1 |
| GPU 最大 SM 时钟 | 1410 MHz |
| GPU 最大显存时钟 | 1512 MHz |
| Driver / CUDA UMD | 610.43.02 / 13.3 |
| CUDA Toolkit | 13.3，nvcc 13.3.73 |
| Nsight Compute | 2026.2.1.0 |
| Nsight Systems | 2026.1.3 |
| 物理 CPU | 2 x Intel Xeon Gold 5320，26 核/Socket，SMT2 |
| Job CPU cgroup | 16 个逻辑 CPU，cpuset `0-7,52-59` |
| Job 内存 cgroup | 24 GiB |
| 分区 walltime / 并发限制 | 30 min / 每用户 1 个 active job |

`lscpu` 和 `free` 显示的是宿主机 104 逻辑 CPU、503 GiB 内存；任务实际可用量必须看 cgroup 和 `hpc limits`，不能把宿主机总量写成 job 配额。

### 1.3 与实验文档的冲突

| 项目 | 实验文档 | 2026-08-28 实测 |
|---|---|---|
| GPU | H800 PCIe | A100 80GB PCIe |
| 架构 | Hopper，CC 9.0a | Ampere，CC 8.0 |
| 编译目标 | `sm_90a` | `sm_80` |
| 关键 MMA | WGMMA | `mma.sync` IMMA |
| CPU | Xeon Gold 5418Y | Xeon Gold 5320 |
| 分区显示 | 文档称 H800 | CLI 明确显示 A100 MIG 10G |

因此当前构建必须显式使用：

```bash
hpc submit -p lab4g10 -g 1 "make ARCH=sm_80 -j16"
```

仓库 Makefile 默认 `sm_90a` 与实际 CC 8.0 不匹配。若正式评测恢复为 H800，则本文的 SM80 custom kernel 需要单独保留 `sm_90a` 分支，不能把 A100 profile 结果外推到 H800。

## 2. 源码与评分约束

### 2.1 当前调用链

朴素路径为：

```text
2 x max-abs reduction
  -> 2S x 逐级量化 kernel
  -> S^2 x INT8 GEMM
  -> S^2 x INT32 -> FP64 重组 kernel
  -> stream synchronize
  -> cudaFree 全部 workspace
```

`splits=8` 时至少有 2 次归约、16 次量化、64 次 GEMM、64 次重组，共 146 个 kernel，且热路径中有 19 次 device allocation/free。

### 2.2 当前提交的致命问题

- `quantize_split_kernel` 把所有 `q[i]` 写为 0。
- `recombine_add_kernel` 不读 `temp`，也不更新 C。
- 主函数预先清零 C，因此随机输入结果恒为 0，实测 L2 相对误差为 1。
- `device_maxabs_fp64` 固定使用默认 stream，执行阻塞 D2H，并在每次调用中 malloc/free。
- `cublasSetStream`、kernel launch、部分 allocation 没有完整错误处理。
- 函数末尾同步并释放所有 workspace，无法利用 benchmark 的预热阶段摊薄分配成本。
- 当前 `cublasGemmEx(N,N)` 的布局不满足 A100 regular-layout IMMA 的 TN 要求，实际回退到低吞吐 kernel。

### 2.3 满分预算

文档的固定 checkpoint 和对应时间上限：

| splits | 100 分 GFLOPS | 4096^3 时间上限 | 8192^3 时间上限 |
|---:|---:|---:|---:|
| 2 | 10000 | 13.744 ms | 109.951 ms |
| 4 | 5000 | 27.488 ms | 219.902 ms |
| 6 | 3000 | 45.813 ms | 366.504 ms |
| 8 | 3000 | 45.813 ms | 366.504 ms |

最终 100 分要求八个组合全部正确且全部达到对应 checkpoint。`splits=2` 占 40%，其余各占 20%，但任何一个组合不到 100 都不能得到总分 100。

实验文档只写“L2 相对误差低于阈值”，没有公布阈值数值。这意味着任何 pair 裁剪都必须以 full-pair 误差为基准留出余量，不能只凭均匀随机输入的一次结果宣称正确。

## 3. 完整基线

构建：`sm_80 -O3`。命令：

```bash
hpc submit -p lab4g10 -g 1 -t 20m \
  "./benchmark 4096,8192 2,4,6,8 3 --csv"
```

### 3.1 正确的朴素 baseline

| Size | S | time (ms) | GFLOPS | L2 relative error |
|---:|---:|---:|---:|---:|
| 4096 | 2 | 88.142 | 1559.3 | 2.192e-5 |
| 4096 | 4 | 303.399 | 453.0 | 3.397e-10 |
| 4096 | 6 | 654.679 | 209.9 | 5.685e-15 |
| 4096 | 8 | 1145.278 | 120.0 | 2.150e-15 |
| 8192 | 2 | 586.151 | 1875.8 | 2.192e-5 |
| 8192 | 4 | 2216.997 | 496.0 | 3.398e-10 |
| 8192 | 6 | 5101.100 | 215.5 | 6.075e-15 |
| 8192 | 8 | 9284.363 | 118.4 | 3.026e-15 |

八项性能都低于对应 `g0`，即使正确也为 0 分。

### 3.2 当前学生占位实现

所有组合的 L2 相对误差均为 `1.0`。吞吐数字没有评分意义，因为量化与重组没有执行真实工作。

### 3.3 cuBLAS fixed-point emulation

| Size | S | time (ms) | GFLOPS | L2 relative error |
|---:|---:|---:|---:|---:|
| 4096 | 2 | 43.987 | 3124.5 | 5.395e-7 |
| 4096 | 4 | 92.473 | 1486.3 | 9.705e-12 |
| 4096 | 6 | 161.707 | 849.9 | 2.140e-15 |
| 4096 | 8 | 161.600 | 850.5 | 2.140e-15 |
| 8192 | 2 | 287.312 | 3826.9 | 5.398e-7 |
| 8192 | 4 | 690.117 | 1593.2 | 9.712e-12 |
| 8192 | 6 | 1295.703 | 848.6 | 3.019e-15 |
| 8192 | 8 | 1329.980 | 826.7 | 3.019e-15 |

这组数据进一步证明当前分区与文档制定 checkpoint 时的目标平台不一致：vendor emulation 在实际 A100 上同样远低于 `g100`。

## 4. Profile 结果

### 4.1 Nsight Systems

任务 185404：`benchmark 4096 4 1`。摘要覆盖 baseline、学生实现和 vendor emulation 的预热与正式调用，以及精度比较。

| Kernel | Instances | Total time | Time share | 说明 |
|---|---:|---:|---:|---|
| `cutlass1x::...Igemm...` | 64 | 1001.88 ms | 63.2% | baseline + student 的 NN INT8 GEMM |
| `acc_diff_kernel` | 4 | 200.07 ms | 12.6% | 正确性比较，不在被测 GEMM 内 |
| `cublasLt_fused_imma_dgemm_kernel_sm80` | 2 | 175.01 ms | 11.0% | vendor emulation |
| FP64 DGEMM kernel | 2 | 109.89 ms | 6.9% | reference |
| `recombine_add_kernel` | 64 | 58.65 ms | 3.7% | baseline 真重组 + student 空 kernel |
| `quantize_one_split_kernel` | 16 | 21.76 ms | 1.4% | 正确 baseline 的量化 |

CUDA API 摘要中，`cudaStreamSynchronize`、`cudaDeviceSynchronize`、`cudaEventSynchronize` 占大部分 host 等待；73 次 `cudaFree` 总计约 142 ms host API 时间。workspace 复用与移除热路径同步是必要优化，但从 GPU 时间看，第一瓶颈仍然是错误的 GEMM 路径。

### 4.2 Nsight Compute

MIG 不允许 profiler 锁频，命令必须带 `--clock-control none`。PCIe shared-unit 指标也无法在 MIG 中采集。

| Kernel | Duration | 关键指标 | 判断 |
|---|---:|---|---|
| 单 split 量化 | 1.34 ms | DRAM 87.1%，210.7 GB/s；compute 23.3%；occupancy 89.1% | DRAM bound |
| 单 pair 重组 | 1.58 ms | DRAM 87.1%，210.6 GB/s；compute 8.0%；occupancy 92.5% | DRAM bound |
| NN INT8 GEMM | 16.85 ms | compute 88.3%；165 reg/thread；occupancy 17.0%；FMA 管线主导 | 未走 Tensor(INT) |
| TN INT8 GEMM | 2.38 ms | Tensor(INT) 71.4%；L2 hit 87.0%；148 reg/thread；73.7 KiB smem/CTA | 正确 IMMA 路径 |
| vendor fused S=4 | 95.69 ms | grid 8，实际 14 SM；255 reg/thread；100.35 KiB smem；occupancy 12.5% | Tensor(INT)，但 grid 太小且 spill |

量化 kernel 的 long-scoreboard stall 占约 83.9%，重组约 95.3%。因此在切换 IMMA 后，应把多个 split 的量化融合成一次内存读取，并把重组并入 MMA epilogue。

TN kernel 的 NCU 还报告 shared store 平均约 2.4-way bank conflict，占 shared-store wavefront 的约 17.2%，估算仍有约 10% 的局部优化空间。自定义 mainloop 应使用 padded/swizzled shared layout，并逐项扫描 tile 和 pipeline stage。

## 5. NN 与 TN 的决定性对照

可复现探针：`tools/int8_gemm_probe.cu`。它用全 1 输入验证布局结果，并分别测 NN、legacy TN、TN AUTOTUNE 和 cuBLASLt TN。正式数值正确性另由随机矩阵的 pair-pruning 探针覆盖。

```bash
hpc submit -p lab4g10 -g 1 -t 10m \
  "nvcc -arch=sm_80 -O3 -std=c++17 tools/int8_gemm_probe.cu \
   -lcublas -lcublasLt -o /tmp/int8_gemm_probe && \
   /tmp/int8_gemm_probe 4096 10"
```

任务 185658：

| Size | NN default | TN default | TN AUTOTUNE | cuBLASLt TN |
|---:|---:|---:|---:|---:|
| 4096 | 15.715 ms / 8.75 TOPS | 2.406 / 57.13 | 2.397 / 57.34 | 2.397 / 57.33 |
| 8192 | 124.907 ms / 8.80 TOPS | 24.308 / 45.23 | 23.574 / 46.64 | 24.309 / 45.23 |

nsys 的实际 kernel 名称：

```text
NN: cutlass1x::gemm::...Igemm...
TN: cutlass_80_tensorop_i16832gemm_s8_128x64_128x3_tn_align16
```

legacy cuBLAS TN 与 cuBLASLt 性能相同，而 Makefile 只显式链接 `-lcublas`，所以最终提交首选 legacy API：

```cpp
// Quantizer directly emits AqT[k + m*K] = Aq[m + k*M].
cublasGemmEx(handle,
             CUBLAS_OP_T, CUBLAS_OP_N,
             M, N, K,
             &one,
             AqT, CUDA_R_8I, K,
             Bq,  CUDA_R_8I, K,
             &beta,
             temp, CUDA_R_32I, M,
             CUBLAS_COMPUTE_32I,
             CUBLAS_GEMM_AUTOTUNE);
```

`CUBLAS_GEMM_AUTOTUNE` 在 8192 上仅再快约 3%，不是高 splits 满分的充分条件，但预热阶段可缓存算法，保留它是合理的低风险优化。

## 6. 反对角线裁剪实验

可复现文件：

- `tools/pair_pruning_probe.cu`
- `tools/pair_pruning_4096.out`
- `tools/pair_pruning_8192.out`

```bash
hpc submit -p lab4g10 -g 1 -t 20m \
  "nvcc -arch=sm_80 -O3 -std=c++17 -Iinclude \
   tools/pair_pruning_probe.cu utils.cu -lcublas -o /tmp/pair_pruning_probe && \
   /tmp/pair_pruning_probe 4096 8"
```

探针实现了：一次生成全部 FP64 residual split；量化 A 时用 32x32 padded shared tile 直接写 `Aq^T`；TN IMMA；同一 `d=i+j` 用 INT32 `beta=0/1` 聚合；每条反对角线仅一次 FP64 重组。计时覆盖 max-abs、量化、GEMM、重组和同步，仅排除可复用 workspace allocation。

定义保留条件 `i+j<D`。

### 6.1 `D=S` 的激进三角裁剪

| S | pairs | 4096 GFLOPS / L2 | 8192 GFLOPS / L2 | g100 |
|---:|---:|---:|---:|---:|
| 2 | 3/4 | 8706 / 2.684e-5 | 9964 / 2.685e-5 | 10000 |
| 4 | 10/16 | 3478 / 5.370e-10 | 3317 / 5.374e-10 | 5000 |
| 6 | 21/36 | 1854 / 1.008e-14 | 1629 / 1.031e-14 | 3000 |
| 8 | 36/64 | 1161 / 2.139e-15 | 970 / 3.018e-15 | 3000 |

`D=S` 保持相同误差阶，但 S=2/4/6 的误差比 full-pair 高约 22% 至 77%。文档没有公开阈值，因此不能默认它可过隐藏正确性。

### 6.2 `D=S+1` 的保守裁剪

| S | pairs | 4096 GFLOPS / L2 | 8192 GFLOPS / L2 | full-pair L2 (8192) |
|---:|---:|---:|---:|---:|
| 2 | 4/4 | 6976 / 2.192e-5 | 7789 / 2.192e-5 | 2.192e-5 |
| 4 | 13/16 | 2817 / 3.397e-10 | 2599 / 3.398e-10 | 3.398e-10 |
| 6 | 26/36 | 1548 / 5.682e-15 | 1331 / 6.072e-15 | 6.071e-15 |
| 8 | 43/64 | 993 / 2.139e-15 | 816 / 3.018e-15 | 3.018e-15 |

`D=S+1` 基本恢复 full-pair 误差，是不知道隐藏阈值时的安全起点。但逐 pair cuBLAS 吞吐仍远低于高 splits 满分目标。

### 6.3 不可采用 `D<S`

固定 D 时，不同传入 splits 的误差逐行相同。例如 8192：

| D | L2 relative error |
|---:|---:|
| 3 | 1.221e-7 |
| 4 | 5.374e-10 |
| 5 | 2.317e-12 |
| 6 | 1.031e-14 |

这说明 `D<S` 实质上把有效精度降成 D 级。即便某一随机输入碰巧触及 FP64 误差底限，也不能把它当成一般正确性保证。

## 7. A100 roofline 与可达性

官方 A100 规格给出 624 TOPS dense INT8 Tensor Core。按实际 14/108 SM 线性折算，当前 MIG 的计算上界约 80.9 TOPS；1/8 HBM/L2 分片对应的内存带宽量级约 242 GB/s，与 NCU 反推值一致。

若保留 P 个等尺寸 INT8 GEMM，评分吞吐的理想上限约为 `80.9/P` TFLOPS：

| S | full pairs | full 上限 | `D=S` pairs | `D=S` 上限 | g100 |
|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 20.2 T | 3 | 27.0 T | 10 T |
| 4 | 16 | 5.06 T | 10 | 8.09 T | 5 T |
| 6 | 36 | 2.25 T | 21 | 3.85 T | 3 T |
| 8 | 64 | 1.26 T | 36 | 2.25 T | 3 T |

结论：

- S=6/8 的 full `S^2` 展开在理论上已经不能满分。
- S=8 的 `D=S` 36 pair 仍低于 3 T 理论目标。
- 必须减少有效 pair 数或改变 mainloop 组织；同时隐藏精度不允许盲目减少 D。
- 当前逐 pair cuBLAS 只达到约 45 至 57 TOPS，离 80.9 TOPS 上界仍有空间，但仅靠 AUTOTUNE 不够。

## 8. 面向 100 分的实施方案

### P0：锁定环境与正确性

1. 每轮先记录 `hpc partitions`、`nvidia-smi -L` 和 benchmark 的 CC，按 `sm_80` 或 `sm_90a` 分支编译。
2. 先完成 full-pair 正确版本：FP64 `rint`、`[-127,127]` clamp、FP64 residual 更新、FP64 scale 和重组。
3. 使用 `fma(-digit, scale, residual)` 更新 residual，禁止在正确性基线阶段启用 fast-math 或 FP32 residual。
4. 覆盖随机矩阵、全零、量化边界、不同动态范围和 4096/8192 全部 splits。
5. 正确性验收：L2 不劣于本文 full-pair 基线；任何后续优化必须同时打印 max-abs 和 L2。

### P1：一次量化全部 split，并直接生成 IMMA 布局

1. A 使用 32x32 padded shared tile：沿 `A[k*M+m]` 合并读取，在寄存器中生成全部 split，再转置并沿 `AqT[m*K+k]` 合并写出。
2. B 一次读取生成全部 split，保持 KxN column-major。
3. A/B 各只启动一个量化 kernel，把 8192、S=8 的 residual 流量从约 18 GiB 降到约 2 GiB。
4. 自写 stream-aware 两级 max-abs reduction，把 max 和 scale 留在 device，移除 helper 的 D2H 和默认 stream 依赖。
5. 复用一个 grow-only workspace；容量键至少包含 device、M/N/K、splits。不得为每个 shape 永久缓存一套 10 GB 级 buffer。
6. 移除函数末尾同步。所有操作排入传入 stream，由 benchmark 的 event 建立计时依赖。

验收：量化总时间显著低于当前 `2S x 1.34 ms`；NCU 无 spill，DRAM bytes 接近“读一次 FP64 + 写 S 份 INT8”。

### P2：强制 TN IMMA

1. 改用上文的 `cublasGemmEx(T,N)`，A 指向量化时已转置的 `AqT`。
2. 首次 warmup 使用 `CUBLAS_GEMM_AUTOTUNE`，正式调用复用同一 handle 的算法缓存。
3. NCU 必须看到 `tensorop_i16832gemm` 和 Tensor(INT) 管线非零；仅检查 API 返回成功不够。
4. 不优先引入 cuBLASLt：legacy TN 已达到相同性能，而最终 Makefile 未显式链接 `-lcublasLt`。

验收：单 pair 在 4096 达到约 2.4 ms，在 8192 达到约 23.6 ms 或更好。

### P3：反对角线 groupwise INT32 accumulation

1. 按 `d=0,1,...`（即 `i+j=d`）从大权重到小权重处理。
2. 同一 d 的首个 GEMM 用 `beta=0`，其余用 `beta=1`，只生成一个 INT32 `temp_d`。
3. 每条反对角线只执行一次 INT32 -> FP64 转换、scale 和 C 累加。
4. 未获知隐藏阈值前使用 `D=min(2S-1,S+1)`；`D=S` 只作为经完整精度验证后的激进候选；禁止 `D<S`。
5. 当前最大 INT32 上界为 `8*8192*127^2 = 1,057,030,144 < INT32_MAX`，评分规模安全。通用 K 仍需显式检查。

这一阶段是可靠过渡版本，不是全部满分终点。

### P4：自定义 SM80 groupwise MMA + FP64 epilogue

这是当前 A100 上达到高 splits 满分的核心。

1. 使用 `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`，候选 CTA tile 从 `128x64x64`、`128x128x64` 开始。
2. 用 `cp.async` 和 3/4-stage double buffering 搬运 A/B tile；shared layout 做 padding/swizzle，优先消除现有 2.4-way shared-store conflict。
3. **一个 CTA 独占一个输出 tile。** 对每条 d，把其 L 个 pair 视作逻辑扩展 K 维 `K'=L*K`，在同一个 INT32 fragment 中连续 MMA。不要让每个 pair 落地完整 INT32 矩阵。
4. 每条 d 完成后，把 INT32 fragment 转为 FP64，乘该 d 的公共 scale，累加到线程持有的 FP64 C fragment。
5. 所有 d 完成后只写一次 FP64 C。若 FP64 fragment 导致寄存器 spill，则降低 CTA N tile 或分两批 d；以端到端时间而非 kernel 数量决定边界。
6. scale 用 FP64 预计算并按 d 存放；累加顺序从大到小。不要用 FP32 scale。
7. 当前镜像未安装 CUTLASS headers，且最终只收一个 `.cu` 文件，所以方案不能依赖外部 CUTLASS 安装。可使用单文件内联 PTX/最小 MMA wrapper；如需 vendoring，必须先确认提交规则允许。

预期收益来源：

- `P` 次完整 INT32 写回降为 0；
- `D` 次或 `P` 次 C 读改写降为 1；
- pair launch 从 P 降为 1 个持久化 kernel；
- 同反对角线的 INT32 聚合在寄存器中精确完成。

### P5：最后处理调度固定成本

1. 若仍保留多 kernel 路径，workspace 稳定后再用 CUDA Graph 捕获；key 必须包含 device、stream、handle、shape、splits 和所有指针。
2. 使用 `cudaMallocAsync` 只有在配置保留阈值并避免每次同步时才有意义；首选显式 workspace 复用。
3. 不建议 split-K：M/N 已提供足够 CTA，parallel split-K 会新增 `slices*M*N*sizeof(int32)` workspace 和 reduction 流量。

## 9. 分阶段验收表

每次只改变一个因素，并记录：`M,S,D,pairs,time_ms,GFLOPS,maxAbs,L2,kernel count,Tensor(INT)%,DRAM bytes,registers,smem,occupancy,spill`。

| 阶段 | 必须满足的退出条件 |
|---|---|
| P0 | 八个组合正确；误差达到 full-pair 基线 |
| P1 | A/B 各一次量化；无 host D2H/sync；workspace 不在热路径分配 |
| P2 | NCU 明确显示 `i16832` Tensor(INT)；单 pair 达到本文 TN 水平 |
| P3 | `D=S+1` 误差与 full 基本一致；输出完整 D-性能-误差曲线 |
| P4-S2 | 4096 <13.744 ms，8192 <109.951 ms |
| P4-S4 | 4096 <27.488 ms，8192 <219.902 ms |
| P4-S6/S8 | 两规模均 <45.813/366.504 ms |
| 最终 | 10 次迭代复测八项均过线，并保留至少 5% 时间余量应对 MIG 时钟波动 |

优先顺序按评分权重为 S2 -> S4 -> S6 -> S8，但最终 100 分必须完成全部组合。

## 10. 风险与需要确认的问题

1. **评测硬件冲突：最高优先级。** 当前 A100 与文档 H800 不一致。必须向助教确认最终评测节点、arch 和 checkpoint 是否同步更新。
2. **隐藏精度阈值。** 文档没有数值。默认采用 `D=S+1`，并以 full-pair L2 为门槛。
3. **单文件提交。** 实验文档只收 `submit/my_int8_fp64.cu`；不要依赖仓库外 CUTLASS 或额外链接参数。
4. **MIG profile 波动。** ncu 无法锁频，最终时间需留余量并多轮复测。
5. **workspace 可重入性。** 静态全局 cache 会破坏多 stream/多线程调用；若为评分做 thread-local cache，必须记录其限制。
6. **scale 的逐位一致性。** 同反对角线理论权重相同，但分别做 FP64 除法可能产生最后几 bit 差异。统一用同一递推生成的 diagonal scale，并与 full baseline 做误差对照。
7. **benchmark 实现细节。** 注释写“取最小值”，实际计时循环取批量平均；所有报告应按平均时间解释。

## 11. 参考资料

1. NVIDIA cuBLAS 13.3，INT8/IMMA layout 与 FP64 fixed-point emulation：<https://docs.nvidia.com/cuda/cublas/>
2. NVIDIA A100 规格，dense INT8 624 TOPS：<https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-a4-nvidia-1485612-r12-web.pdf>
3. NVIDIA MIG profile 表：<https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html>
4. NVIDIA Ampere Tuning Guide，`cp.async`、SM80 和 IMMA：<https://docs.nvidia.com/cuda/ampere-tuning-guide/>
5. NVIDIA PTX ISA，warp-level `mma.sync`：<https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma>
6. CUDA Graph：<https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>
7. Stream-ordered allocator：<https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html>
8. Uchino, Ozaki, Imamura, groupwise INT32 accumulation：<https://journals.sagepub.com/doi/pdf/10.1177/10943420241313064>
