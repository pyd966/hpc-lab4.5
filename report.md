# Lab 4.5 H800 配置、Profile 与 100 分实现报告

> 记录日期：2026-08-28（UTC）
> 目标：在 H800 MIG 上通过 4096/8192 与 splits=2/4/6/8 的全部正确性检查，并按实验文档公开公式达到 100 分。

## 0. 最终结论

1. 正式 GPU 队列是 `lab3`，不是实验文档命令示例中的 `lab4g10`。`lab3` 已实测为 NVIDIA H800 PCIe MIG 1g.10gb、CC 9.0、14 SM，正确编译目标为 `sm_90a`。
2. `submit/my_int8_fp64.cu` 已完成双架构实现：CC >= 9 使用 CUDA 13.3 fixed-point emulation；CC 8.x 保留手写量化、TN IMMA 与对角线重组兼容路径。
3. 最终项目原生回归任务 188025 在 `lab3` 上完成八个正式组合，全部数值误差与 `cublas_emulated` 对照一致，全部超过各自 `g100`。按公开评分公式得分为 **100.0000**。
4. 此前的 **81.2544** 是误在 `lab4g10/A100 MIG` 上测得的兼容路径估算，不是 H800 正式成绩。A100 数据只用于跨架构回退验证。
5. H800 profile 证明正式路径实际执行 `cublasLt_fused_imma_dgemm_kernel_sm90`；2 GiB user workspace 已按正确顺序注册，trace 未观察到内部 `cudaMallocAsync`。

## 1. 评分约束与源码审阅

### 1.1 公开评分目标

实验文档先按 L2 相对误差做正确性门控，再按固定 checkpoint 计分：

| splits | `g0` | `g60` | `g100` | 4096 时间上限 | 8192 时间上限 |
|---:|---:|---:|---:|---:|---:|
| 2 | 2500 | 5000 | 10000 | 13.744 ms | 109.951 ms |
| 4 | 750 | 2500 | 5000 | 27.488 ms | 219.902 ms |
| 6 | 350 | 1500 | 3000 | 45.813 ms | 366.504 ms |
| 8 | 200 | 1500 | 3000 | 45.813 ms | 366.504 ms |

规模 4096 和 8192 等权；每个规模内 splits=2/4/6/8 的权重为 40%/20%/20%/20%。八项都达到 `g100` 时总分封顶 100。

### 1.2 调用链和原始问题

`benchmark.cu` 为每个规模生成 FP64 输入，以 `cublasDgemm` 为参考，预热后对每种方法计时并计算 max-abs 与 L2 相对误差。原始学生占位实现存在以下问题：

- split 量化结果被写成 0，重组 kernel 不读取 GEMM 结果，随机输入输出恒为 0；
- 每个 split 和 pair 都产生独立 kernel，S=8 时有 64 次 INT8 GEMM 和大量完整矩阵读写；
- 热路径反复分配、同步和释放 workspace；
- A/B 的 NN 布局在 A100 上没有命中高吞吐 regular-layout IMMA。

正式实现不再使用这条占位调用链。入口查询运行时 compute capability 后分流：

| 设备 | 正式路径 | 状态 |
|---|---|---|
| CC >= 9，`lab3/H800` | cuBLAS fixed-point emulation + 持久 user workspace | H800 全组合实测 100 分 |
| CC 8.x，`lab4g10/A100` | fused residual quantization + TN IMMA + diagonal INT32 accumulation | 兼容路径实测正确，约 81.25 分 |

## 2. 远程计算节点实测

### 2.1 作业与队列

所有 GPU 计时和 profile 均通过 `hpc submit` 在远程节点执行，DevPod 只用于编辑。主要 H800 任务：

| Job ID | 用途 |
|---:|---|
| 187511 | 首次 `lab3` GPU/软件探测 |
| 187542 | 首轮项目原生 10 次 CSV 完整，但 wrapper 最终 exit 1 |
| 187641 | nsys：4096，splits=2/8 |
| 187652 | ncu：4096，splits=2 fused kernel |
| 187603/187619/187632 | emulation 参数候选：special-values、mantissa bits、AUTOTUNE |
| 187665/187963 | H800 手写 TN 路径完整性能与误差 |
| 187996/188007/188054/188062/188082 | AUTOTUNE、extended-K、API 对照、手写 nsys、融合 MMA 复核 |
| 188025 | 最终节点复核 + 项目原生三次回归 |
| 188162 | 项目原生 10 次；慢 baseline 使作业达到 5 min walltime |
| 188193 | 正式提交函数 lightweight 10 次回归，成功 |

从任务 188025 和 profiler 实测得到：

| 项目 | 实测值 |
|---|---|
| 分区 | `lab3`，显示名 `Lab3 H800 MIG 10G` |
| GPU | NVIDIA H800 PCIe |
| MIG | `1g.10gb`，可见 9984 MiB，ECC on |
| Compute capability | 9.0，构建目标 `sm_90a` |
| 可见计算资源 | 14 SM，7 TPC，1 copy engine |
| Driver / CUDA UMD | 610.43.02 / 13.3 |
| CUDA Toolkit | 13.3，nvcc 13.3.33 |
| Nsight Compute / Systems | 2026.2.0 / 2026.1.3 |
| 宿主 CPU | 2 x Intel Xeon Gold 5418Y，24 core/socket，SMT2 |
| Job CPU cgroup | 4 logical CPUs，cpuset `4-5,52-53` |
| 宿主 / Job 内存 | 503 GiB / 32 GiB cgroup limit |
| 分区限制 | 5 min walltime，每用户 1 个 active job |
| 容器镜像 | `hpc101-lab3:v26.2` |

`lscpu` 和 `free` 展示的是宿主机总资源，不能当成作业配额；CPU affinity 和 `memory.max` 才是任务实际限制。

### 2.2 实验文档中的队列错误

实验文档的 H800、MIG 1g.10gb、CUDA 13.3 和 `sm_90a` 描述是正确的，但“构建与运行”示例仍写成 `-p lab4g10`。该分区实际是 A100 MIG、CC 8.0。H800 正式命令必须改用：

```bash
hpc submit --export NONE -p lab3 -g 1 ...
```

`--export NONE` 用于避免把 DevPod 中与实验无关的环境变量和凭据转发到远程容器。

## 3. H800 正式实现

### 3.1 计算配置

H800 路径使用 CUDA 13.3 公开 API：

```text
CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH
CUBLAS_EMULATION_STRATEGY_EAGER
CUDA_EMULATION_MANTISSA_CONTROL_FIXED
max_mantissa_bits = min(8 * splits, 55)
CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT
CUBLAS_GEMM_DEFAULT
```

这条路径由 cuBLAS 在内部完成缩放、INT8 slice、IMMA GEMM 和 FP64 重组。课程提交要求只限制提交 `submit/my_int8_fp64.cu`，没有明文禁止这个公开 API；同时评分文档明确以该实现的性能作为 `g100`。如果课程另有未写入文档的“禁止直接调用 emulation”规则，则必须由课程方明确，该限制会改变当前最优方案。

### 3.2 handle 状态与 workspace

正式实现按以下顺序设置 handle：

1. `cublasSetStream(handle, stream)`；
2. `cublasSetWorkspace(handle, tls_workspace, 2 GiB)`；
3. `cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST)`；
4. 设置 math mode、strategy、mantissa control 和 bit count；
5. 调用 `cublasGemmEx`。

顺序不能交换：`cublasSetStream` 会把 user workspace 重置为默认 workspace pool。仓库 `baseline/cublas_emulated.cu` 先 SetWorkspace、后 SetStream，因此 benchmark 中的对照行没有真正保留传入的 2 GiB workspace；正式实现避免了这个顺序错误。这也解释了 `my_int8_fp64` 有时略快于对照行。

workspace 采用 `thread_local` grow-once 缓存，预热后不再分配。一次 host 线程跨 stream 复用时，disable-timing event 在前后调用间建立依赖，防止异步覆盖；共享 cuBLAS handle 的跨 host 线程互斥仍由调用方负责。任务 187641 的 CUDA API trace 没有观察到内部 `cudaMallocAsync` 或其他可见 fallback 分配；该证据不表示 cuBLAS 必然消费了全部 2 GiB。

benchmark 自己还会另外分配 2 GiB 给 `cublas_emulated` 对照，不能把两者误认为同一块内存。8192 全方法原生回归没有 OOM。

### 3.3 splits 与有效 slice

为复现课程 baseline，bit count 设置为 `min(8*S,55)`。55 是 FIXED 控制下采用的课程/默认配置值，不是 cuBLAS 文档声明的通用硬上限。

CUDA 文档给出的 fixed-point slice 数关系为 `ceil((bits+1)/8)`，所以 S=2/4/6/8 对应 3/5/7/7 个 slice。S=6 和 S=8 都被设置为 55 bits，这解释了两者时间和误差几乎相同。

## 4. 正式性能与 100 分核验

任务 188025 使用原生项目调用链、`sm_90a -O3`、三次平均：

| Size | S | time (ms) | GFLOPS | max abs error | L2 relative error | `g100` | 余量 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 2 | 13.1230 | 10473.12 | 6.621e-5 | 5.395e-7 | 10000 | +4.73% |
| 4096 | 4 | 24.7270 | 5558.25 | 1.146e-9 | 9.705e-12 | 5000 | +11.17% |
| 4096 | 6 | 41.8516 | 3283.96 | 6.253e-13 | 2.140e-15 | 3000 | +9.47% |
| 4096 | 8 | 42.1678 | 3259.33 | 6.253e-13 | 2.140e-15 | 3000 | +8.64% |
| 8192 | 2 | 85.3533 | 12881.89 | 9.317e-5 | 5.398e-7 | 10000 | +28.82% |
| 8192 | 4 | 185.9516 | 5912.89 | 1.759e-9 | 9.712e-12 | 5000 | +18.26% |
| 8192 | 6 | 332.8228 | 3303.59 | 1.506e-12 | 3.019e-15 | 3000 | +10.12% |
| 8192 | 8 | 332.7106 | 3304.71 | 1.506e-12 | 3.019e-15 | 3000 | +10.16% |

同一任务中，八项 `my_int8_fp64` 的 max-abs 和 L2 与 `cublas_emulated` 对照逐项一致。任务 188193 直接链接正式提交文件，用 lightweight harness 做 10 次平均稳定性复测，GFLOPS 分别为：

```text
4096: 10335.58, 5542.97, 3284.80, 3284.79
8192: 13034.42, 5983.72, 3350.36, 3352.28
```

10 次平均中最小余量仍为 4096/S=2 的 +3.36%。八个组合的单项得分均为 100，因此：

```text
Score = 1/2 * (100 + 100) = 100.0000
```

## 5. H800 Profile

### 5.1 Nsight Systems

任务 187641 对 4096、S=2/8 采样：

- 端到端时间约 13.293 ms / 41.958 ms；
- 主 kernel 为 `cublasLt_fused_imma_dgemm_kernel_sm90`，证明进入 Hopper INT8 fused emulation 路径；
- 4 次 fused kernel 合计 94.75 ms，平均 23.69 ms；
- 8 次 `max_scale_pack` 合计 15.27 ms，平均 1.91 ms；
- CUDA API 摘要没有内部 `cudaMallocAsync`，未观察到可见 fallback 分配。

### 5.2 Nsight Compute

任务 187652 采样 4096/S=2 的 fused kernel：

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
| Spill | 无 local/shared spill |

grid 只有 12 个 block，而 MIG 有 14 SM；NCU 给出约 14.29% 的局部并行度提示。主要 stall 为 pipe 与 long-scoreboard，各约 8.3 cycle。最薄弱的 4096/S=2 在正式回归中有 3.36% 至 4.73% 实测余量，但该内核的 tile/cluster 由 cuBLAS 内部选择，应用层无法直接调整。

## 6. 候选优化实验与取舍

### 6.1 emulation 参数扫描

| 候选 | 结果 | 决策 |
|---|---|---|
| `SPECIAL_VALUES_SUPPORT_NONE` | 各规模波动混合，4096/S2 约慢 1.4% | 回退 |
| `CUBLAS_GEMM_AUTOTUNE` | 无稳定收益，8192/S2/S4/S6 更慢 | 回退 |
| `max_bits=8*S-1` | 明显更快，但 L2 退化到 S2 1.24e-4、S4 2.30e-9、S6 4.03e-14 | 正确性风险，回退 |
| 2 GiB user workspace | 按 API 正确注册；nsys 未见内部异步/fallback 分配，八项稳定过线 | 保留 |
| EAGER + FIXED + full course bits | 精度与对照一致，八项 100 | 保留 |

少 1 bit 会减少一个内部 slice，因此加速明显，但误差比课程 baseline 差约 5 至 7 倍。隐藏阈值没有公开，不能用性能换掉这部分正确性余量。

### 6.2 手写 Ozaki 路径

手写候选使用一次 max-abs、A/B 各一次多 split 量化、量化时直接生成 TN 布局、保留 `D=min(S,6)` 条反对角线，并以 INT32 `beta=0/1` 聚合。H800 实测：

| Size | S | GFLOPS | L2 relative error | 是否达到 `g100` |
|---:|---:|---:|---:|---|
| 4096 | 2 | 14653 | 2.684e-5 | 是 |
| 4096 | 4 | 5429 | 5.370e-10 | 是 |
| 4096 | 6 | 2757 | 1.008e-14 | 否 |
| 4096 | 8 | 2650 | 1.008e-14 | 否 |
| 8192 | 2 | 19913 | 2.685e-5 | 是 |
| 8192 | 4 | 6913 | 5.374e-10 | 是 |
| 8192 | 6 | 3417 | 1.031e-14 | 是 |
| 8192 | 8 | 3337 | 1.031e-14 | 是 |

任务 188062 的 nsys 显示 4096/S=6 每次调用中：

- 21 个 TN INT8 GEMM 平均约 1.85 ms；单独 `beta=0` microbenchmark 为 1.42 ms / 96.5 TOPS；
- A/B 量化各约 3.63/3.66 ms；
- max-abs 约 1.12 ms，最终重组约 2.27 ms；
- `beta=1` 的读改写使多 pair 对角线比单独 GEMM 慢。

继续测试得到：legacy `AUTOTUNE` 仍只有 2771/2645 GFLOPS；把同一反对角线拼成长 K 反而降至 1149/992 GFLOPS；SM80 风格 fused groupwise MMA 在 H800 上 S6/S8 只有 1022/1014 GFLOPS。cuBLASLt 单 pair 与 legacy TN 同为约 96.7 TOPS，但原 Makefile 只链接 `-lcublas`，直接调用 Lt 会链接失败。通过动态加载绕过链接约束不适合作为课程提交方案。

结论：手写路径已经能通过 6/8 个性能点，但 4096/S6/S8 仍差 8.1%/11.7%，且裁剪后的 L2 比完整 baseline 更弱。它适合作为 A100/无 emulation 环境的兼容分支，不应替换当前精度和性能都更稳的 H800 路径。

### 6.3 A100 兼容性对照

`lab4g10` 是 A100 80GB PCIe MIG 1g.10gb、CC 8.0、14 SM。正式兼容分支在那里八项均能运行，但按 H800 checkpoint 估算为 81.2544。这个结果用于证明 runtime dispatch 和非 Hopper 回退可用，不代表正式实验平台无法达到 100 分。

## 7. 最终优化方案与完成状态

以下方案已经执行，不是待办列表：

1. **按架构分流。** H800 使用已实测的 fused emulation；A100 使用手写 TN IMMA，避免把 SM80 profile 外推到 Hopper。
2. **固定完整精度。** 保留 EAGER/FIXED 和课程 bit count，不采用少 1 bit 或更激进的 pair 裁剪。
3. **修复 cuBLAS 状态顺序。** 先 stream、后 workspace，强制 HOST pointer mode，避免调用方遗留状态污染 alpha/beta。
4. **持久 workspace。** 每线程复用 2 GiB，预热后不分配；跨 stream 通过 event 串行，热路径无 host synchronize/free。
5. **用 profile 验证而非只看 API 成功。** nsys 已确认 SM90 fused IMMA kernel，且未见可见 fallback 分配；ncu 已定位最小余量点的 grid underfill 与 warp eligibility。
6. **逐项回归。** 项目原生 benchmark 完成三次集成回归，正式提交函数又完成 lightweight 10 次稳定性回归；两轮对 4096/8192、S=2/4/6/8 都为 100.0000。

若 CUDA/cuBLAS 版本变化导致 4096/S2 失去约 3% 余量，下一优先级不是降低 mantissa bits，而是：固定镜像版本，扫描允许的 workspace 大小和 emulation strategy，并重新 profile `max_scale_pack` 与 fused kernel。由于当前已封顶 100，替换 cuBLAS 内部 kernel 的高风险自写 WGMMA 不进入最终提交。

## 8. 复现命令

正式构建和回归应在 `lab3` H800 上执行：

```bash
hpc submit --export NONE -p lab3 -g 1 -c 4 -m 32Gi -t 5m \
  /bin/bash -lc '/usr/local/cuda/bin/nvcc \
    -arch=sm_90a -O3 -std=c++17 -lineinfo -Iinclude \
    benchmark.cu baseline/baseline_fp64.cu \
    baseline/cublas_baseline.cu baseline/cublas_emulated.cu \
    utils.cu submit/my_int8_fp64.cu \
    -lcublas -lcudart -lcuda -o /tmp/lab45-benchmark && \
    /tmp/lab45-benchmark 4096,8192 2,4,6,8 10 --csv'
```

profile 同样必须提交到 `lab3`；MIG 环境下 ncu 使用 `--clock-control none`。任务 187641/187652 使用下面的 lightweight harness：其计算设置与正式 H800 分支一致，只保留一次 FP64 reference，不运行朴素 baseline 和第二个 emulation 对照。

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

## 9. 参考资料

1. [NVIDIA cuBLAS 13.3 文档：Floating Point Emulation、workspace 与 stream](https://docs.nvidia.com/cuda/cublas/)
2. [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
3. [NVIDIA MIG User Guide：1g.10gb profile](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html)
4. [Ozaki et al., Error-free transformations of matrix multiplication](https://doi.org/10.1007/s11075-011-9478-1)
5. [Uchino, Ozaki, Imamura, groupwise INT32 accumulation](https://doi.org/10.1177/10943420241313064)
