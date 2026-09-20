# cuda/ —— 纯 CUDA C++ 练习

和 `../triton/` 一一对应的 CUDA 版本。**同样的三个问题，用 CUDA 再做一遍**，
目的是看清两件事：

1. Triton 到底替你做了什么（shared memory、bank conflict、
   寄存器分块、向量化、边界 mask —— 全是编译器的活）；
2. Triton 替你做不了什么（grid 怎么开、数据在存储层次里怎么走、
   什么时候该换一套 kernel）。

每个练习都是 **`v0` 朴素实现 → 逐步优化 → 官方库基线** 的一条链，
每一步的收益都实测过，写进各自的 `README.md`。

测试环境：A100-SXM4-40GB（108 SM，40 MB L2，带宽 **1555.2 GB/s**，
fp32 算力 **19.5 TFLOP/s**），CUDA 12.4，`-arch=sm_80`，`CUDA_VISIBLE_DEVICES=7`。

---

## 1. 练习一览

| 目录 | 问题 | 瓶颈类型 | 优化链 | 结果 |
|------|------|---------|-------|------|
| [`01_vector_add/`](01_vector_add/README.md) | `out = x + y` | memory-bound | 合并访存 → float4 → grid-stride | 84.8% 峰值带宽（≈ 天花板） |
| [`02_reduction/`](02_reduction/README.md) | `sum(x)` | memory-bound | divergence → bank conflict → shuffle → grid-stride | **5.58x**，CUB 的 96.9% |
| [`03_matmul/`](03_matmul/README.md) | `C = A × B` | **compute-bound** | shared 分块 → 寄存器分块 → float4 | **6.7x**，cuBLAS 的 88.5% |

三个练习是**递进**的，建议按顺序做：

- **01** 建立基本功：什么是合并访存、怎么量带宽、为什么 `~85%` 就是上限。
  这里的结论是"优化空间很小"——**最简单的写法已经接近最优**。
- **02** 才是真正的优化练习：同样的 memory-bound 问题，
  朴素写法只有 **15.7%** 峰值，因为归约是"多对一"，结构本身要设计。
- **03** 换赛道：第一个 compute-bound 的问题，
  整个思路从"怎么少搬数据"变成"**怎么提高算术强度**"，
  连 occupancy 的结论都反过来了。

---

## 2. 快速开始

```bash
cd 01_vector_add    # 或 02_reduction / 03_matmul
make                # 编译
make run            # 跑全部对比和扫描
make quick          # 只跑正确性 + 主对比（快）
make clean
```

所有练习共用 [`common.mk`](common.mk)（编译选项）和
[`common.cuh`](common.cuh)（计时、校验、设备信息）。三个统一约定：

| | |
|---|---|
| **换卡** | `make GPU=3 run` —— 默认用 7 号卡，跑之前先 `nvidia-smi` 确认是空的 |
| **换架构** | `make ARCH=sm_90` |
| **只测正确性** | `./bench --check` —— sanitizer 目标用的就是它 |

每个练习的 `Makefile` 还有一组**分析目标**，不需要任何特权就能跑：

```bash
make sass       # cuobjdump -sass，统计关键指令条数（FFMA / LDS / LDG / SHFL / BAR）
make ptx        # 导出 PTX
make regs       # nvcc -Xptxas -v，看寄存器和 shared memory 用量
make sanitize   # compute-sanitizer memcheck，查越界
make racecheck  # compute-sanitizer racecheck，查漏掉的 __syncthreads()（02 / 03）
```

`01_vector_add` 没有 `racecheck` 目标 —— 它根本不用 shared memory，
没有可竞争的东西。

> `make sass` 是这套练习里最有用的一个目标。**"我加的这个优化，
> 编译器是不是早就做了？"** 这个问题只有反汇编能回答 ——
> 03 里就发现编译器自己把 shared 读向量化成了 `LDS.128`
> （见 `03_matmul/README.md` §3.2）。

---

## 3. 三个必须知道的坑

这三条是做完三个练习后总结的，**不知道的话测出来的数全是错的**。

### 3.1 计时前必须预热 GPU 时钟

`common.cuh` 里的 `warmup_gpu()` 会空转 200 ms 把 SM 时钟拉到稳态。
**不做这一步，一个进程里第一个被测的变体会虚慢 20~30%。**

`bench()` 自带的那几次 warmup 只够热模块加载和 cache，**热不了时钟**。
这个坑在 02 里真实地污染过一版 README：v0 排在第一个被测，
替整个进程背了时钟爬坡的锅，表里写成 12.2%，实际是 15.7%，
连带 v0→v1 的收益从 1.63x 虚报成 2.09x。

> 副产品是一个有用的诊断信号：**如果一个 kernel 的耗时随时钟明显变化，
> 它就还不是带宽受限的。** 真正打满带宽的 kernel 对时钟不敏感。

### 3.2 `warpSize` 不是编译期常量

它是运行时特殊寄存器（SASS 里的 `WARP_SZ`），编译器**没法据此展开循环**。
`for (int o = warpSize/2; o > 0; o >>= 1)` 写成 `constexpr int WARP = 32`
之后，SASS 里的 `SHFL` 从 2 条（循环体）变成 10 条（完全展开），
实测 **1.19x**，改一个 token。详见 `02_reduction/README.md` §3.5。

### 3.3 卡被别人占着的话，绝对数字全废（但比值还能用）

这是**最容易忘、后果最大**的一条。同一台机器、同一份代码、同一天，
GPU 空闲和被占时实测：

| | 空闲时 | 被占时 | 比 |
|---|---|---|---|
| `03_matmul` cuBLAS | 18.43 TFLOP/s（94.5% 峰值） | 8.68 TFLOP/s（44.5%） | 2.1x |
| `03_matmul` v3 | 16.30 TFLOP/s（83.6%） | 7.06 TFLOP/s（36.2%） | 2.3x |
| `triton/01` torch.add | 151 us | 331 us | 2.2x |

**但优化链的比值基本不变**：v0→v3 空闲时 6.7x，被占时 6.9x；
三步的单步收益 2.19/2.02/1.52 vs 2.07/2.22/1.51。

> 判断依据很简单：**看"占峰值"。** cuBLAS 跑不到 90%+、
> 向量加跑不到 80%+，就说明卡不是你一个人的，别记这次的数。
> 各练习 README 里的数字都是空闲卡上测的。

`nvidia-smi` 在某些 shell 里跑不起来（缺 loader，报
`No such file or directory`）。用 NVML 查是更可靠的办法：

```bash
python -c "
import pynvml as N; N.nvmlInit()
for i in range(N.nvmlDeviceGetCount()):
    h=N.nvmlDeviceGetHandleByIndex(i); u=N.nvmlDeviceGetUtilizationRates(h)
    p=N.nvmlDeviceGetComputeRunningProcesses(h)
    print(f'GPU{i} util={u.gpu:3d}% mem={u.memory:3d}% '
          f'sm={N.nvmlDeviceGetClockInfo(h,N.NVML_CLOCK_SM)}MHz '
          f'procs={[(x.pid, x.usedGpuMemory//2**20) for x in p]}')
"
```

`util` 接近 0 且 `procs` 为空，才是干净的卡。
（`throttle` 位 `0x4` 是 SW power cap，满载时出现是正常的。）

### 3.4 正确性容差要跟着规模走

fp32 累加 K 次，随机符号下误差按 `sqrt(K)` 增长。固定 `1e-5`
在 K=4096 时必然误报。参考值一律用 **CPU double 累加** ——
2²⁶ 个数用 float 顺序累加会严重丢精度，**反而是 GPU 的树形归约更准**，
拿 float 参考值会得出完全相反的结论。

还有一条：跑之前把输出缓冲填成 **NaN**（`cudaMemset(dst, 0x7F, n)`），
别填 0。grid 算错导致某些位置没被写到时，填 0 可能"恰好接近对"，
填 NaN 一定会被抓到。

---

## 4. 三个练习各自最反直觉的结论

| 练习 | 结论 |
|------|------|
| 01 | **向量化（float4）在常规 block 下毫无收益** —— v0/v1/v2 差距在 ±1% 噪声内。它真正的价值在小 block 下（block=32 时 2.6x）。 |
| 02 | **结构上的浪费比微观的访存冲突值钱得多**：修 bank conflict 只有 1.29x，而"少开一半 block"有 1.86x。 |
| 03 | **占用率和性能是反着的**：v0/v1 占用率 100%，v2/v3 只有 25% —— 快的是后者。compute-bound kernel 靠 ILP，不靠 TLP。 |

第三条尤其值得注意，因为它**推翻了前两个练习建立的直觉**。
"occupancy 要拉满"是个有前提的建议，前提是 memory-bound。

---

## 5. 和 `../triton/` 的分工

每个练习的 README 都有一节 "§7 和 Triton 版的对照"。总的来说：

| 层次 | 谁负责 |
|------|-------|
| block 内的归约策略、shared memory 分配与同步、bank conflict swizzle、寄存器分块、向量化 | **Triton 编译器** |
| grid 开多大、每线程吃多少数据、分块尺寸怎么选、什么 shape 换什么 kernel | **还是你自己** |

一句话：**Triton 替你写 kernel，但不替你想 roofline。**

⚠️ **两边的绝对数字不能直接比。** 尤其是 matmul：
`triton/03_matmul` 是 fp16 + Tensor Core（峰值 312 TFLOP/s），
`cuda/03_matmul` 是纯 fp32 CUDA core（峰值 19.5 TFLOP/s）。
只看绝对 TFLOP/s 会得出"Triton 快 11 倍"这种毫无意义的结论。
**看"占峰值"和"占官方库"才是可比的。**

---

## 6. 相关文档

| 想干什么 | 去哪 |
|---|---|
| 查 `nvcc` / `ncu` / `nsys` 在哪、怎么跑 | [`../README.md`](../README.md) |
| 解决 `ncu` 的 `ERR_NVGPUCTRPERM` | [`../README.md` §6.2](../README.md)（Docker + `--cap-add=SYS_ADMIN`） |
| 学 profiler 怎么用、报告怎么读 | [`../综合练习/vector_mul2/`](../综合练习/vector_mul2/README.md) |
| 对比 PyTorch / Triton / CUDA 三种写法 | [`../综合练习/vector_mul2/`](../综合练习/vector_mul2/README.md) §2 |
| 练 Triton 本身 | [`../triton/`](../triton/README.md) |

> 本目录里的所有分析（`make sass` / `make regs` / Occupancy API）
> **都不需要 profiling 权限**。`ncu` 在这台机器上要走 Docker，
> 但大部分问题不用 `ncu` 也能回答 —— 这一点在
> `../综合练习/vector_mul2/README.md` §7 里展开过。
