我正在完成学校 HPC 课程的 lab，当前这个 lab 的内容是 基于 INT8 张量核的 FP64 GEMM 模拟详细信息你可以查看 docs/Lab4.5-INT8-FP64-GEMM/ 下的文件。

需要注意的是，你目前正在运行在一台 x86 机器上，但是这台机器仅作为 devpod 进行开发使用。如果你需要真正测试运行时间或者进行 profile，你必须使用 `hpc` 命令提交到远程集群上进行工作。远程集群的使用方法以及提交方式可以在 docs/ 找到。

注意，当前目录已经初始化了 git 仓库，并且这个仓库添加了 remote github repo。请你在完成一轮修改后，自动进行 commit，并且 push 到远程 repo。