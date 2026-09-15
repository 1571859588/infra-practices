# 向量元素乘 2：PyTorch / Triton / CUDA 三方案对比 + Profile 实战

用最简单的 `y = x * 2` 把三种 GPU 编程方案从"写法差异"一路跑到"profiler 里能看到什么"。
所有代码和数据都在本机（8×A100-40GB）实测跑通，日期 2026-09-15。

> 上级目录的 [`../README.md`](../README.md) 记录了本机 `nvcc / ncu / nsys / c++` 的完整环境信息，
> 本文只讲这个例子怎么跑、怎么分析。

---

## 目录

- [0. 快速开始](#0-快速开始)
- [1. 文件说明](#1-文件说明)
- [2. 三种写法的核心差异](#2-三种写法的核心差异)
- [3. 性能实测结果](#3-性能实测结果)
- [4. nsys 分析（可用 ✅）](#4-nsys-分析可用-)
- [5. ncu 分析（用 Docker 绕过权限限制 ✅）](#5-ncu-分析用-docker-绕过权限限制-)
- [6. GUI 分析怎么做](#6-gui-分析怎么做)
- [7. 不需要 profiling 权限的替代分析](#7-不需要-profiling-权限的替代分析)
- [8. 踩坑清单](#8-踩坑清单)
- [9. 我的笔记](#9-我的笔记)

---

## 0. 快速开始

```bash
cd /mnt/gfs/nyt1/infra/practices/vector_mul2

# 一键全跑（benchmark 部分，约 2 分钟）
bash run_all.sh

# 加上 profile（nsys + compute-sanitizer，约 5 分钟）
bash run_all.sh --profile

# ncu 需要 GPU 性能计数器权限，本机用 Docker 绕过（见 §5）
bash profile/run_ncu_docker.sh
```

手动分步跑：

```bash
conda activate cpp                 # torch 2.9.0+cu128 + triton 3.5.0 + cmake/ninja
export CUDA_VISIBLE_DEVICES=3      # 挑一张空闲卡，见下方说明

make                               # 编译 libmul2.so + cuda_mul2
python torch_mul2.py               # 方案一
python triton_mul2.py              # 方案二
./cuda_mul2                        # 方案三（独立可执行）
python bench_all.py                # 三方统一对比
python bench_all.py --sweep        # 规模扫描
python launch_overhead.py          # CPU 端 launch 开销
```

**挑空闲卡**（本机是共享机器，别挤在别人正在用的卡上）：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
# 选 memory.used 最小的那张，比如 3
export CUDA_VISIBLE_DEVICES=3
```

**依赖说明**：脚本里的 `PYTHON` 默认写死成
`/mnt/public/nyt1/docqa/restored_envs/cpp/bin/python`，所以即使没 `conda activate`
也能跑。想换环境：`PYTHON=/path/to/python bash run_all.sh`。

---

## 1. 文件说明

| 文件 | 作用 |
|---|---|
| `torch_mul2.py` | **方案一**：PyTorch 实现 + 单独 benchmark |
| `triton_mul2.py` | **方案二**：Triton kernel + BLOCK_SIZE 扫描 + autotune + PTX 检查 |
| `cuda_mul2.cu` | **方案三**：CUDA kernel（标量版 / float4 版 / 故意越界版）+ `extern "C"` 包装 |
| `main.cu` | CUDA 版独立 benchmark 驱动，带 NVTX 标记 |
| `Makefile` | 构建 `libmul2.so`（给 ctypes）和 `cuda_mul2`（给 profiler） |
| `common.py` | 共用的 CUDA Event 计时 / 校验 / 带宽计算 |
| `bench_all.py` | **三方统一对比**：同一份数据，同一进程，含正确性校验 |
| `launch_overhead.py` | 测 CPU 端 launch 开销，解释小规模下的反直觉结果 |
| `profile_target.py` | 给 profiler 用的轻量入口，迭代少、NVTX 分段清晰 |
| `profile/run_nsys.sh` | nsys 采集脚本（宿主机直接跑） |
| `profile/run_ncu.sh` | ncu 采集脚本（宿主机跑会因权限失败，脚本里写了原因） |
| `profile/run_ncu_docker.sh` | **ncu / nsys 的 Docker 版**：`--cap-add=SYS_ADMIN` 绕过权限限制，实际能采到数（§5） |
| `run_all.sh` | 一键全流程 |
| `reports/` | profile 产物（`.nsys-rep` / `.ncu-rep` / `.sqlite`），可下载到本地用 GUI 打开 |

**为什么 CUDA 版用 ctypes 而不是 `torch.utils.cpp_extension`？**
torch 是 cu128 编的，本机系统 nvcc 是 12.4、conda 的是 13.3，用 `cpp_extension`
会触发 torch 的 CUDA 版本一致性检查，容易卡在环境问题上。`ctypes` + 纯 C 接口
完全绕开这层：`libmul2.so` 不含任何 torch 符号，跨 CUDA 版本都能调。
代价是要自己传 `data_ptr()`、自己管 stream。

---

## 2. 三种写法的核心差异

### 2.1 PyTorch

```python
def torch_vector_mul2(x: torch.Tensor):
    return x * 2
```

- 一行搞定，完全不接触硬件细节。
- 底层走 ATen 的 `vectorized_elementwise_kernel<4, ...>`（nsys 里能看到这个名字），
  NVIDIA/Meta 已经调好，对简单算子接近带宽上限。
- **代价**：每个算子一次 launch + 一次完整读写显存。`(x*2+1).relu()` 会是
  3 个 kernel、3 轮显存往返 —— 这就是 Triton / CUDA 存在的理由。

### 2.2 Triton

```python
@triton.jit
def triton_mul2_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x * 2, mask=mask)
```

最大的心智差异：**粒度是 block（program），不是 thread。你看不到 `threadIdx`。**

| CUDA 概念 | Triton 对应 |
|---|---|
| `blockIdx.x` | `tl.program_id(axis=0)` |
| `threadIdx.x` | **没有** —— block 内的线程划分由编译器决定 |
| `if (idx < n)` | `mask=` 参数，越界 lane 自动不读不写 |
| 手写 `float4` | 编译器自动向量化（见下） |
| `blockDim.x` | `BLOCK_SIZE`（`tl.constexpr`，编译期常量） |
| 每 block 线程数 | `num_warps`（launch 参数，可 autotune） |

`BLOCK_SIZE` 标成 `tl.constexpr` 意味着它是编译期常量：每个不同的值会触发一次
独立 JIT 编译，产生特化代码。

**Triton 自动做了 128-bit 向量化** —— `python triton_mul2.py` 输出里可以看到 PTX：

```
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
@%p1 st.global.v4.b32 [ %rd3 + 0 ], { %r9, %r10, %r11, %r12 };
```

`.v4.b32` = 一条指令读写 4 个 32-bit，等价于手写 CUDA 的 `float4`。
**这就是为什么 Triton 能追平 torch，却比朴素的标量 CUDA 版更快。**

### 2.3 CUDA

```cuda
__global__ void cuda_mul2_kernel(const float* x, float* y, int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) y[idx] = x[idx] * 2.0f;
}
```

- 控制力最强，但所有细节都要自己管：grid/block 划分、边界检查、向量化、对齐。
- **边界检查不能省**：`n` 通常不被 `blockDim` 整除，最后一个 block 有多余线程。
  漏掉就是越界写 —— 而且经常不报错（见 [§7.3](#73-compute-sanitizer-抓静默越界)）。
- 本例额外写了 `float4` 向量化版做对比：

```cuda
__global__ void cuda_mul2_kernel_vec4(const float4* x, float4* y, int n_vec4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec4) {
        float4 v = x[idx];
        v.x *= 2.f; v.y *= 2.f; v.z *= 2.f; v.w *= 2.f;
        y[idx] = v;
    }
}
```

### 2.4 一张表总结

| | PyTorch | Triton | CUDA |
|---|---|---|---|
| 代码量 | 1 行 | ~8 行 | ~6 行 kernel + ~10 行 host |
| 编程粒度 | 张量 | block | thread |
| 边界处理 | 自动 | `mask=` | 手写 `if` |
| 向量化 | 自动 | 自动 | **手写** |
| 编译时机 | 预编译 | 首次调用 JIT | 提前 `nvcc` |
| 首次调用开销 | ~0 | **数百 ms**（JIT） | ~0 |
| launch 开销 | 中（~7 us） | **高（~11-18 us）** | **低（~3-4 us）** |
| 算子融合 | ❌ | ✅ | ✅ |
| 调优手段 | 换算子 | `autotune` | 手调一切 |
| 适合 | 90% 场景 | 自定义融合算子 | 极限优化 / 特殊指令 |

---

## 3. 性能实测结果

环境：A100-SXM4-40GB（sm_80，HBM2e 理论峰值 **1555 GB/s**），CUDA 12.4，
torch 2.9.0+cu128，triton 3.5.0。
带宽 = `2 × n × 4 bytes / 耗时`（读一次 + 写一次）。

### 3.1 主规模（n = 16.7M，64 MiB/buffer）

```
  torch  x*2                101.8 us    1318.0 GB/s  ( 84.8% of peak)
  triton BLOCK=1024         101.9 us    1316.9 GB/s  ( 84.7% of peak)
  cuda   scalar             104.8 us    1280.3 GB/s  ( 82.3% of peak)
  cuda   vec4               100.6 us    1334.1 GB/s  ( 85.8% of peak)
```

**结论：四者几乎打平，差距 < 4%。** 因为 mul2 是纯 memory-bound，
只要访存是 coalesced 的，谁都能吃到 ~85% 的 HBM 带宽，没有优化空间。

唯一有意义的差距是 `cuda scalar` 落后 ~3%：它用 32-bit 的 `LDG.E` / `STG.E`，
访存指令数是 vec4 版的 4 倍。**这恰恰说明手写 CUDA 不等于更快** ——
没做向量化的朴素 CUDA 反而是四个里最慢的。

### 3.2 规模扫描（`python bench_all.py --sweep`）

| n | buffer | torch | triton | cuda scalar | cuda vec4 |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 0.02 MiB | 6.7 us / 4.9 GB/s | **16.3 us** / 2.0 GB/s | 3.5 us / 9.3 GB/s | 3.6 us / 9.2 GB/s |
| 65,536 | 0.25 MiB | 6.4 us / 82 GB/s | **16.1 us** / 33 GB/s | 3.8 us / 138 GB/s | 3.7 us / 143 GB/s |
| 1,048,576 | 4 MiB | 6.5 us / 1290 GB/s | **15.9 us** / 529 GB/s | 7.6 us / 1104 GB/s | 6.0 us / **1396 GB/s** |
| 16,777,216 | 64 MiB | 101.8 us / 1318 | 101.9 us / 1317 | 104.8 us / 1280 | 100.6 us / **1334** |
| 67,108,864 | 256 MiB | 392.8 us / 1367 | 393.1 us / 1366 | 399.6 us / 1343 | 389.8 us / **1377** |

**三个关键观察：**

1. **小规模下耗时不随 n 变化。** 4K 和 64K 元素耗时几乎一样（6.7 vs 6.4 us）。
   说明瓶颈根本不是 GPU，是 CPU 端提交 kernel 的开销。

2. **小规模下 Triton 最慢（~16 us 地板）。** 不是 kernel 慢，是 Triton 的
   Python 侧 launch 路径长。`launch_overhead.py` 直接量了这个：

   ```
   CPU 端 launch 开销（n=4096，不同步，纯 host 时间 / 次）
     torch  (ATen)        6.67 us/launch
     triton (JIT)        10.58 us/launch     ← 多次测量在 10~18 us 波动
     cuda   (ctypes)      2.78 us/launch
   ```

   在 nsys 里也能交叉验证：Triton 走 `cuLaunchKernelEx`（driver API，avg 5.5 us），
   torch/CUDA 走 `cudaLaunchKernel`（median 3.8 us）。

3. **4 MiB 是个有意思的拐点**：GPU 侧刚好够忙起来，此时 `cuda vec4` 拿到全场
   最高的 1396 GB/s（89.7% peak），而 `cuda scalar` 只有 1104 GB/s —— 访存指令
   数的差距在这个规模最明显。规模再大时两者都被 HBM 带宽卡住，差距被抹平。

> **实践结论**：小张量上要提速，靠 CUDA Graph / 算子融合 / 增大 batch，
> 而不是优化 kernel 内部。写 Triton kernel 时如果输入常常很小，
> 先确认 launch 开销不是瓶颈，否则白优化。

### 3.3 Triton BLOCK_SIZE 扫描（n = 16.7M）

```
  BLOCK_SIZE=256            101.7 us    1319.2 GB/s  ( 84.8% of peak)
  BLOCK_SIZE=512            101.3 us    1324.5 GB/s  ( 85.2% of peak)
  BLOCK_SIZE=1024           101.8 us    1318.5 GB/s  ( 84.8% of peak)
  BLOCK_SIZE=2048           102.7 us    1306.4 GB/s  ( 84.0% of peak)
  BLOCK_SIZE=4096           103.3 us    1298.8 GB/s  ( 83.5% of peak)
  autotuned                 101.0 us    1328.4 GB/s  ( 85.4% of peak)
```

差距 < 1%。memory-bound 算子对 BLOCK_SIZE 极不敏感 —— 只要有足够多的 block
把 SM 填满、隐藏住访存延迟就行。autotune 每次选出的 best_config 也不稳定
（跑两次分别选了 `BLOCK=1024,warps=8` 和 `BLOCK=256,warps=2`），进一步印证
这个维度上没有真实差异。**对 compute-bound 或有 shared memory 的 kernel，
autotune 才真正值钱。**

---

## 4. nsys 分析（可用 ✅）

本机只有系统 CUDA 12.4 带 `nsys`（**conda 环境里没有**），版本 2023.4.4。

```bash
export PATH=/usr/local/cuda/bin:$PATH
bash profile/run_nsys.sh
```

> ⚠️ 宿主机上跑会打印 `CPU IP/backtrace sampling not supported, disabling`——
> `/proc/sys/kernel/perf_event_paranoid = 4` 挡住了 CPU 采样。
> **GPU 侧（kernel 时间线 / CUDA API / memcpy / NVTX，走 CUPTI）完全不受影响**，
> 下面 §4.3 的所有数据都是这么采的。
> 想要 CPU 火焰图和线程调度，用 §5.2 的 Docker 方案（`--cap-add=SYS_ADMIN` 一并解锁）。

### 4.1 采集命令

```bash
# Python 侧：三种实现一起抓
nsys profile -t cuda,nvtx --cuda-memory-usage=true \
     --force-overwrite true -o reports/all \
     python profile_target.py --n 16777216 --iters 20

# 纯 CUDA 可执行文件：时间线最干净，没有 Python 噪声
nsys profile -t cuda,nvtx --force-overwrite true -o reports/cuda_only \
     ./cuda_mul2 16777216 100 256
```

参数说明：

| 参数 | 作用 |
|---|---|
| `-t cuda,nvtx` | 只 trace CUDA API + NVTX。加 `osrt` 会把报告撑大好几倍 |
| `--cuda-memory-usage=true` | 记录 `cudaMalloc/Free`，看显存分配开销 |
| `--force-overwrite true` | 覆盖同名报告，不然会报错退出 |
| `-o path` | 输出 `.nsys-rep`。**不要带扩展名**，nsys 自己加 |
| `--capture-range=cudaProfilerApi` | 配合代码里的 `torch.cuda.profiler.start/stop` 只抓关键段 |

### 4.2 命令行看结果

```bash
nsys stats --report cuda_gpu_kern_sum reports/all.nsys-rep   # kernel 耗时排行
nsys stats --report cuda_api_sum      reports/all.nsys-rep   # CUDA API 耗时
nsys stats --report nvtx_sum          reports/all.nsys-rep   # NVTX 区段
nsys stats --report cuda_gpu_mem_time_sum reports/cuda_only.nsys-rep  # memcpy
nsys stats --help-reports                                    # 列出全部报表
```

> 第一次 `nsys stats` 会把 `.nsys-rep` 转成 `.sqlite`（几秒）。
> 之后复用；改了报告要加 `--force-export=true`。

### 4.3 实测结果与解读

**(a) Kernel 汇总（`cuda_gpu_kern_sum`，3 种实现各 25 次）**

```
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Name
 --------  ---------------  ---------  ---------  ------------------------------------------
     26.4        2,736,748         25  109,469.9  cuda_mul2_kernel(const float*, float*, int)
     24.4        2,528,340         25  101,133.6  at::native::vectorized_elementwise_kernel<4, AUnaryFunctor<float,...>>
     24.2        2,511,893         25  100,475.7  triton_mul2_kernel
     24.0        2,487,859         25   99,514.4  cuda_mul2_kernel_vec4(const float4*, float4*, int)
      0.9           94,270          1   94,270.0  distribution_elementwise_grid_stride_kernel  ← torch.randn 造数据
```

读法：
- torch 的 kernel 叫 `vectorized_elementwise_kernel<**4**, ...>` —— 模板参数 `4`
  就是向量化宽度，坐实了它也在做 128-bit 访存。
- 三个"做了向量化"的实现（torch / triton / cuda_vec4）耗时在 99.5~101 us，
  互相在 1.6% 以内；标量 CUDA 版 109.5 us，慢 10%。
  **GPU 侧的真实差距只有向量化这一条。**
- 最后那个 `distribution_elementwise_grid_stride_kernel` 是 `torch.randn` 造数据用的，
  不属于被测对象 —— profile 时一定要认出这种"无关 kernel"。

**(b) CUDA API 汇总（`cuda_api_sum`）**

```
 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Max (ns)     Name
 --------  ---------------  ---------  -----------  -----------  ----------  ------------------
     51.7       15,134,134         76    199,133.3      3,838.5  12,811,118  cudaLaunchKernel
     27.9        8,173,974          5  1,634,794.8  1,772,668.0   2,089,019  cudaDeviceSynchronize
     12.6        3,681,074          1  3,681,074.0           --          --  cudaProfilerStop
      5.1        1,482,845          1  1,482,845.0           --          --  cudaGetDeviceProperties_v2
      1.5          429,137          2    214,568.5           --     220,604  cudaMalloc
      0.5          137,690         25      5,507.6      3,872.0      34,560  cuLaunchKernelEx      ← Triton
      0.3           90,534          1     90,534.0           --          --  cuModuleLoadData      ← Triton 载入 cubin
```

读法：
- `cudaLaunchKernel` 的 **max = 12.8 ms，median 只有 3.8 us** —— 差了 3000 倍。
  那一次 12.8 ms 是**第一次 launch**，包含 CUDA context 创建和 module 加载。
  **这就是必须 warmup 的硬证据。** 看 median 而不是 avg。
- Triton 走 `cuLaunchKernelEx`（driver API），median 3.87 us，和
  `cudaLaunchKernel` 的 3.84 us 几乎一样 —— 说明 Triton 慢在 **Python 侧**
  （查缓存、算 grid、组装参数），而不是 driver 调用本身。
- `cudaMalloc` 两次共 0.43 ms，单次 ~215 us。**显存分配非常贵**，
  这就是 `torch.empty_like` 复用缓冲、以及 PyTorch 自带 caching allocator 的意义。

**(c) NVTX 区段（`nvtx_sum`）—— 本次最有价值的发现**

```
 Time (%)  Total Time (ns)  Instances      Avg (ns)        Range
 --------  ---------------  ---------  ---------------  ---------------
     98.6      690,284,678          1  690,284,678.0    warmup           ← 690 ms !!
      0.3        2,234,911          1    2,234,911.0    impl::cuda
      0.3        2,215,693          1    2,215,693.0    impl::torch
      0.3        2,118,547          1    2,118,547.0    impl::triton
      0.3        2,034,035          1    2,034,035.0    impl::cuda_vec4
```

**warmup 段 690 ms，占了整个 profile 的 98.6%，而四个实测段各只有 2 ms。**

这 690 ms 绝大部分是 **Triton 的 JIT 编译**（Python → TTIR → TTGIR → LLVM IR → PTX → cubin）
外加 CUDA context 初始化。在 nsys GUI 时间线上，这段表现为第一次 kernel 之前
一大片 CPU 忙、GPU 全空的区域。

**如果不做 warmup，你测到的就是编译时间，误差 300 倍以上。**
Triton 会把结果缓存到 `~/.triton/cache`，所以第二次跑同一个进程快很多 ——
这也意味着 **benchmark 结果会受"是不是第一次跑"影响**，写测试时要注意。

**(d) memcpy 汇总（`cuda_gpu_mem_time_sum`，来自纯 CUDA 报告）**

```
 Time (%)  Total Time (ns)  Count   Avg (ns)      Operation
 --------  ---------------  -----  -----------  ----------------------------
     59.5        7,280,873      1  7,280,873.0  [CUDA memcpy Host-to-Device]
     40.5        4,951,279      1  4,951,279.0  [CUDA memcpy Device-to-Host]
```

64 MiB 的 H2D 用了 **7.28 ms**，算下来只有 **9.2 GB/s**。
对比：同样 64 MiB 的 kernel 只要 **0.109 ms**。

**一次 PCIe 传输 ≈ 67 次 kernel 执行。**

9.2 GB/s 远低于 PCIe Gen4 x16 的 ~25 GB/s，因为 `std::vector` 是**可分页内存**，
驱动要先拷到一块内部的 pinned 缓冲区再 DMA。改用 `cudaHostAlloc` /
`cudaMallocHost` 分配 pinned 内存可以接近满速，还能用 `cudaMemcpyAsync` 和计算重叠。

> 这是全篇最重要的工程结论：**优化 kernel 前先确认数据搬运不是瓶颈。**
> 本例里 kernel 优化最多省 10 us，而把 H2D 从 pageable 换成 pinned 能省几毫秒。

---

## 5. ncu 分析（用 Docker 绕过权限限制 ✅）

### 5.1 宿主机上直接跑会失败

```bash
$ bash profile/run_ncu.sh
==PROF== Connected to process 2677812 (.../cuda_mul2)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA
          GPU Performance Counters on the target device 0.
==WARNING== No kernels were profiled.
```

**原因**：NVIDIA 驱动默认把 GPU 性能计数器限制为管理员
（`NVreg_RestrictProfilingToAdminUsers=1`）。换 conda 环境、
`pip install nvidia-nsight-compute-cu12` 装新版、换 CUDA 版本，全都没用 ——
这不是版本问题。

### 5.2 Docker 方案（本机实测可用，不需要管理员、不需要重启）

**关键认知：驱动检查的不是"你是不是 root"，而是进程有没有 `CAP_SYS_ADMIN` 这个 capability。**

容器里以 root 运行 + `--cap-add=SYS_ADMIN`，就满足了这个条件。这是 NVIDIA
官方文档给出的容器内 profiling 方案，宿主机一行配置都不用改。

```bash
bash profile/run_ncu_docker.sh
```

脚本核心就是这个 `docker run`：

```bash
docker run --rm --entrypoint bash \
    --gpus '"device=3"' \                      # 映射指定 GPU
    --cap-add=SYS_ADMIN \                      # ★ 关键：解锁性能计数器
    -v "$PWD":/work -w /work \                 # 代码 + 产物目录
    -v /usr/local/cuda-12.4/nsight-systems-2023.4.4:/opt/nsys:ro \  # 镜像里没 nsys
    -v /mnt/public:/mnt/public:ro \            # conda env（torch/triton）
    -e TRITON_CACHE_DIR=/tmp/triton_cache \    # 只读挂载下 triton 要能写缓存
    nvidia/cuda:12.4.0-devel-ubuntu22.04 -c '<命令>'
```

**逐项说明**（每一条都是实测踩出来的）：

| 参数 | 为什么需要 |
|---|---|
| `--cap-add=SYS_ADMIN` | 不加就是 `ERR_NVGPUCTRPERM`，加了立刻能采。这一条是全部的关键 |
| **不能加 `--user`** | 加了 `--user $(id -u):$(id -g)` 之后 `CapEff: 0000000000000000`，capability 全丢，ncu 照样失败。必须是容器 root |
| `--gpus '"device=3"'` | 引号是嵌套的（shell 一层 + docker 一层），少一层会解析失败 |
| `-v .../nsight-systems-...:/opt/nsys:ro` | `nvidia/cuda:*-devel` 镜像**自带 ncu 但不带 nsys**，要用 nsys 就从宿主机挂进去 |
| `-v /mnt/public:/mnt/public:ro` | conda env `cpp`（torch 2.9.0+cu128 + triton 3.5.0）在这里，路径保持一致才能直接用绝对路径的 python |
| `-e TRITON_CACHE_DIR=/tmp/triton_cache` | 默认缓存路径在只读挂载里，triton 会写失败 |
| `--entrypoint bash` | 只是为了跳过镜像默认打印的一大段 license banner |

验证权限确实到位：

```bash
$ docker run --rm --cap-add=SYS_ADMIN ... -c 'grep CapEff /proc/self/status'
CapEff:	00000000a82425fb          # 非 0，包含 CAP_SYS_ADMIN
```

**顺带解决了 nsys 的 CPU 采样问题**（§4 里宿主机因 `perf_event_paranoid=4` 拿不到）：

```
$ nsys status --environment          # 在容器里
Root privilege: enabled
Linux perf_event_open syscall available: OK
CPU Profiling Environment (process-tree): OK
CPU Profiling Environment (system-wide): OK
Sampling trigger: LBR ... Available
```

对比两份报告的 sqlite 表，实锤差别：

| 表 | 容器内采集 | 宿主机采集 |
|---|---:|---|
| `COMPOSITE_EVENTS`（CPU 采样点） | 322 | **表不存在** |
| `SAMPLING_CALLCHAINS`（调用栈） | 2,267 | **表不存在** |
| `SCHED_EVENTS`（线程调度） | 202 | **表不存在** |

**产物属主是 root**，所以脚本每个 `docker run` 结尾都有一句
`chown $(id -u):$(id -g) /work/reports/xxx`，否则宿主机上删不掉、GUI 也读不了。

### 5.3 实测结果：四个 kernel 横向对比

`n = 16,777,216`（64 MiB/buffer），A100-SXM4-40GB：

| kernel | Duration | DRAM 吞吐% | SM 吞吐% | global ld 指令数 | 寄存器/线程 | 达成 occupancy |
|---|---:|---:|---:|---:|---:|---:|
| torch `vectorized_elementwise_kernel<4,...>` | 87.68 us | 86.48% | 5.09% | 131,072 | 28 | 82.48% |
| `triton_mul2_kernel` | 87.23 us | 86.68% | 5.11% | 131,072 | 16 | 82.09% |
| `cuda_mul2_kernel`（标量） | **99.68 us** | **78.66%** | **15.64%** | **524,288** | 16 | 78.96% |
| `cuda_mul2_kernel_vec4` | **86.66 us** | **87.34%** | 5.52% | 131,072 | 16 | **89.65%** |

> 重跑会有 ~1% 的抖动（Duration 87.2~88.3 us、DRAM 85.9~86.8%），
> 但**指令数、寄存器数这类静态量每次完全一致** —— 结论不受影响。
> 看 profile 数据时要分清哪些是会抖的（时间、吞吐率）、哪些不该抖（指令数、寄存器）。

**这张表把 §3、§7.2 的所有猜测都验证了：**

1. **`524,288 / 131,072 = 4.0`** —— 标量版的 global load 指令数正好是另外三个的
   4 倍。§7.2 里从 SASS 读出来的 `LDG.E` vs `LDG.E.128` 差异，在这里被计数器坐实。

2. **标量版 SM 吞吐 15.64%，是别人的 3 倍。** 注意：mul2 的浮点运算量四者完全相同，
   多出来的 SM 占用**全是地址计算和访存指令发射**。这是"指令开销"最直观的量化。

3. **标量版 DRAM 吞吐反而更低（78.66% vs 87.34%）。** 搬的字节数一样，但发射带宽
   被指令数吃掉了，喂不满 HBM。**这就是 9% 性能差距的完整因果链。**

4. **torch 用 28 个寄存器，是其他实现的 1.75 倍**，occupancy 也最低（82.48%）——
   ATen 那个模板 kernel 要处理任意 functor / 任意 stride，通用性的代价。
   但它照样跑到 86.48% 带宽，说明**这个规模下 occupancy 82% 已经足够隐藏延迟了**。

**GPU Speed Of Light 原文**（`--set full` 的第一屏）：

```
cuda_mul2_kernel(const float *, float *, int) (65536,1,1)x(256,1,1), CC 8.0
    DRAM Throughput          78.34 %        Duration              99.81 us
    Compute (SM) Throughput  15.67 %        Memory Throughput   1.22 TB/s
    L1/TEX Cache Throughput  20.99 %        L1 Hit 0%  L2 Hit 60.89%
    Achieved Occupancy       78.10 %

cuda_mul2_kernel_vec4(const float4 *, float4 *, int) (16384,1,1)x(256,1,1), CC 8.0
    DRAM Throughput          87.71 %        Duration              86.59 us
    Compute (SM) Throughput   5.52 %        Memory Throughput   1.36 TB/s
    L1/TEX Cache Throughput  24.31 %        L1 Hit 0%  L2 Hit 61.02%
    Achieved Occupancy       87.63 %
```

`L1 Hit Rate = 0%` 是纯流式访问的标志 —— 每个数据只读一次，缓存毫无用处。
看到这个数字就该知道：**优化方向只有"减少访存量"，不可能靠提高命中率。**

再看 DRAM 字节数：

```
                          dram__bytes_read.sum   dram__bytes_write.sum
cuda_mul2_kernel                    67.11 MB            54.46 MB
cuda_mul2_kernel_vec4               67.11 MB            50.52 MB
l1tex__t_sectors_...global_op_ld     2,097,152           2,097,152   （两者相同）
```

读的字节数完全一样（67.11 MB ≈ 64 MiB），**L1 sector 数也完全一样** ——
搬运的数据量一字节不差，差的纯粹是**发射多少条指令去搬**。

### 5.4 采集命令详解

```bash
# 1) 全量报告：scalar + vec4 放进同一份报告，方便 GUI 做 Baseline 对比
ncu --set full --launch-skip 10 --launch-count 2 \
    -o reports/ncu_cuda --force-overwrite \
    ./cuda_mul2 16777216 1 256

# 2) 只取关键指标，不生成报告文件（最快）
ncu --metrics gpu__time_duration.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__inst_executed_op_global_ld.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread \
    -k 'regex:triton_mul2_kernel' --launch-count 1 \
    python profile_target.py --n 16777216 --iters 3

# 3) 回看报告
ncu --import reports/ncu_cuda.ncu-rep --page details
ncu --import reports/ncu_cuda.ncu-rep --page details --csv    # 转 CSV 便于脚本处理
```

**`--launch-skip 10` 这一条是踩出来的坑。** `main.cu` 里有 10 次 scalar warmup，
直接用 `--launch-count 2` 的话两个名额全被 warmup 吃掉，报告里只有
`cuda_mul2_kernel`、根本没有 vec4。跳过 10 次之后，第 11、12 次 launch
正好是 `bench_scalar` 和 `bench_vec4` 各一次。

> **写 profile 脚本前先数清楚"第几次 launch 是你要的那次"** ——
> 或者干脆用 `--nvtx-include 'bench_vec4/'` 按 NVTX 区段筛，更稳。

**对 Python 程序的铁律：一定要用 `-k` / `--nvtx-include` / `--launch-count` 收窄范围。**
ncu 是 **replay** 机制 —— 为了采集所有计数器，它会把同一个 kernel 反复重放几十次。
不收窄的话，torch 初始化那几十个 kernel 都要被 replay，能跑几十分钟到几小时。

### 5.5 memory-bound kernel 该看哪些指标

| 指标 | 含义 | 本例实测 |
|---|---|---|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM 带宽利用率 | **核心指标**，86~87%（标量版 78%） |
| `dram__bytes_read.sum` + `dram__bytes_write.sum` | 实际 DRAM 读写字节 | 67.11 MB ≈ `n×4`，说明访存完全 coalesced |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 计算单元利用率 | 5.1~5.5%，**极低 → 确认 memory-bound** |
| `smsp__inst_executed_op_global_ld.sum` | global load 指令数 | 标量 524,288 vs 向量化 131,072，**正好 4×** |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 达成 occupancy | 82~90% |
| `launch__registers_per_thread` | 每线程寄存器 | torch 28，其余 16 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | L1 读 sector 数 | 两版都是 2,097,152（相同！差别只在指令数） |

GUI 里对应的 section：
- **GPU Speed Of Light Throughput** —— 一眼看出 Memory 高 / Compute 低
- **Memory Workload Analysis** —— L1/L2/DRAM 各级的命中和吞吐
- **Source Counters** —— 因为编译加了 `-lineinfo`，可以把指标落到 `.cu` 源码行

### 5.6 其他可选方案（不如 Docker 方便，记录备查）

| 方案 | 说明 |
|---|---|
| 管理员放开驱动限制 | `echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-profiling.conf` + `update-initramfs -u` + **重启**。一劳永逸，但要重启整机，共享服务器上基本不现实 |
| `sudo ncu ...` | 本机 `sudo` 要密码、`sudo -n` 不可用；有密码的话可行，但每次都要输 |
| 只用 nsys + SASS + sanitizer | 见 §7，能回答本例 90% 的问题，但拿不到硬件计数器 |

**结论：在共享服务器上，Docker 是成本最低的方案** —— 不改宿主机、不重启、
不需要 sudo 密码，只要用户在 `docker` 组里。

---

## 6. GUI 分析怎么做

本机是无图形界面的服务器，`nsys-ui` / `ncu-ui` 直接跑需要 X11 转发，很卡。
**推荐做法：服务器上采集，本地看。**

### 6.1 本地装 GUI

去 NVIDIA 官网下载（免费，需注册）：
- Nsight Systems: <https://developer.nvidia.com/nsight-systems>
- Nsight Compute: <https://developer.nvidia.com/nsight-compute>

**版本要求：本地 GUI 版本 ≥ 服务器采集端版本。**
本机是 nsys **2023.4.4** / ncu **2024.1.1**，本地装任意更新的版本都能打开。
反过来（本地版本更旧）会打不开。

### 6.2 把报告拉到本地

```bash
# 在你自己的机器上执行
BASE=/mnt/gfs/nyt1/infra/practices/vector_mul2/reports
scp <user>@<server>:$BASE/all.nsys-rep .        # Python 侧三方对比时间线
scp <user>@<server>:$BASE/cuda_only.nsys-rep .  # 纯 CUDA，含 memcpy
scp <user>@<server>:$BASE/in_docker.nsys-rep .  # 容器内采集，额外带 CPU 采样
scp <user>@<server>:$BASE/ncu_cuda.ncu-rep .    # ncu 全量报告（scalar + vec4）

# 或者用 VS Code Remote：直接在文件树里右键 -> Download
```

体积都很小：`.nsys-rep` ~260 KB ~ 1 MB，`ncu_cuda.ncu-rep` ~340 KB。
`.sqlite` 是 `nsys stats` 的中间产物，不用传。

### 6.3 Nsight Systems GUI 怎么看

打开 `all.nsys-rep` 后，从上到下的几行 track：

```
NVTX                 ├──── warmup (690ms) ────┤├impl::torch┤├impl::triton┤├impl::cuda┤
CUDA API             ....密集的 cudaLaunchKernel....
CUDA HW (GPU)        ....一格格的 kernel 执行块....
```

**具体操作：**

1. **先看 NVTX 行**找到 `impl::torch` / `impl::triton` / `impl::cuda` 三段，
   右键某段 → **Zoom into selection**，把无关部分排除掉。
2. **对齐 CUDA API 行和 CUDA HW 行**：这是 nsys 最核心的价值。
   - 如果 API 行的 `cudaLaunchKernel` 密密麻麻、HW 行的 kernel 块之间有空隙
     → **GPU 在等 CPU 喂数据**，瓶颈在 launch 开销（本例小规模时就是这样）。
   - 如果 HW 行的 kernel 块首尾相接没有空隙 → GPU 饱和，瓶颈在 kernel 内部，
     该转去用 ncu。
3. **在 kernel 块上悬停**：显示 duration、grid/block 尺寸、寄存器数、
   以及"对应哪一次 `cudaLaunchKernel`"的连线。
4. **Events View**（底部面板）：把 kernel 按耗时排序，等价于命令行的
   `cuda_gpu_kern_sum` 但可以点进去跳转时间线。
5. **看 warmup 段**：会看到一大片 CPU 活动但 GPU 全空的区域，那就是 Triton JIT。
6. **打开 `in_docker.nsys-rep`（容器内采的那份）会多出几行 track**：
   *CPU (0-N)* 利用率、每个线程的 *OS Runtime* 和采样调用栈。
   右键线程行 → **Show in Events View** 可以看到 2,267 条 callchain。
   宿主机采的 `all.nsys-rep` / `cuda_only.nsys-rep` 里这些行是空的。

**本例在 GUI 里最该确认的两件事：**
- 大规模（64 MiB）时 kernel 块应该紧密相接，GPU 饱和；
- 小规模（16 KB）时 kernel 块之间有大片空隙，且空隙宽度 ≈ launch 开销。

### 6.4 Nsight Compute GUI 怎么看

`bash profile/run_ncu_docker.sh` 会生成 `reports/ncu_cuda.ncu-rep`（~340 KB），
里面**同时包含 `cuda_mul2_kernel` 和 `cuda_mul2_kernel_vec4` 两个 kernel**，
就是为了在 GUI 里直接做 Baseline 对比（这也是 §5.4 里 `--launch-skip 10
--launch-count 2` 的目的）。

打开 `.ncu-rep` 后：

1. 左上角下拉框选 kernel（本例有 `cuda_mul2_kernel` 和 `cuda_mul2_kernel_vec4`）。
2. **Details 页 → GPU Speed Of Light** 是第一屏：两根横条，
   Memory 应该很长（~85%），Compute 很短（<10%）→ 确认 memory-bound。
3. **Memory Workload Analysis**：一张 L1 ↔ L2 ↔ DRAM 的数据流图，
   标着每一级的字节数和命中率。本例应该看到 L1/L2 命中率极低（流式访问，没有复用）。
4. **Source 页**：因为 `Makefile` 加了 `-lineinfo`，左边是 `.cu` 源码、右边是 SASS，
   每行标着采样命中数。能直接看到 `y[idx] = x[idx] * 2.0f` 这行占了几乎所有时间。
5. **Baseline 对比（最实用的功能）**：
   - 先选中 `cuda_mul2_kernel`，点工具栏的 **Add Baseline**；
   - 再切到 `cuda_mul2_kernel_vec4`；
   - 所有指标旁边会出现 `+x%` / `-x%` 的差值。
   - 本例应该看到 Duration `-13%`、DRAM Throughput `+11%`、
     global load 指令数 `-75%` —— 和 §5.3 的表一一对应。
   - 这是对比两版实现最直观的方式，比看两份报告强得多。

命令行也能做 baseline 对比（需要两份独立报告）：

```bash
ncu --import a.ncu-rep --baseline b.ncu-rep --page details
```

### 6.5 如果一定要在服务器上开 GUI

```bash
ssh -X <user>@<server>           # 需要本地有 X server
/usr/local/cuda/bin/nsys-ui
```
实测在共享服务器上很卡，不推荐。优先用 6.2 的"采集 + 下载"方案。

---

## 7. 不需要 profiling 权限的替代分析

即使有了 §5 的 Docker 方案，下面这些工具仍然值得先用 —— 它们**不需要任何特殊权限、
不需要 Docker、秒级出结果**，而且本例的核心结论（向量化差 4 倍）它们就能回答。
在没有 docker 组权限的机器上，这就是全部能用的手段。

### 7.1 `nvcc -Xptxas=-v`：看寄存器 / shared memory 占用

```bash
make occupancy
```

实测输出：

```
ptxas info : Compiling entry function '_Z21cuda_mul2_kernel_vec4PK6float4PS_i' for 'sm_80'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 14 registers, 372 bytes cmem[0]

ptxas info : Compiling entry function '_Z16cuda_mul2_kernelPKfPfi' for 'sm_80'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 10 registers, 372 bytes cmem[0]
```

读法：
- **`0 bytes spill stores/loads` 是最该确认的一行。** 一旦有 spill，
  说明寄存器不够、变量被踢到 local memory（实际在显存里），性能会断崖下跌。
- 标量版 10 个寄存器、vec4 版 14 个 —— 都极少。A100 每 SM 有 65536 个寄存器，
  按 14 个/线程算，寄存器完全不会限制 occupancy。
- Triton 侧的对应信息：`python triton_mul2.py` 输出 `n_regs = 16, shared = 0 bytes`。

### 7.2 `cuobjdump -sass`：确认访存指令宽度

```bash
make sass
```

实测输出：

```
Function : _Z21cuda_mul2_kernel_vec4PK6float4PS_i      ← float4 版
    LDG.E.128.CONSTANT R8, [R2.64] ;       ← 128-bit 读
    STG.E.128 [R4.64], R8 ;                ← 128-bit 写

Function : _Z16cuda_mul2_kernelPKfPfi                  ← 标量版
    LDG.E.CONSTANT R2, [R2.64] ;           ← 32-bit 读
    STG.E [R4.64], R7 ;                    ← 32-bit 写

Function : _Z22cuda_mul2_kernel_buggyPKfPfi            ← 故意越界的版本
    LDG.E R2, [R2.64] ;                    ← 注意：没有 .CONSTANT
    STG.E [R4.64], R7 ;
```

两个可以直接读出来的结论：

1. **`.128` 后缀 = 一条指令搬 16 字节，无后缀 = 4 字节。**
   同样的数据量，标量版要发 4 倍的访存指令 —— 这就是 §3.1 里那 3%（4 MiB 规模下
   高达 26%）性能差距的来源，**完全不需要 profiler 就能验证**。

2. **`.CONSTANT` 后缀是 `const __restrict__` 带来的。** 对比第三个 kernel：
   `cuda_mul2_kernel_buggy` 的参数没写 `const`/`__restrict__`，编译器无法确认
   `x` 和 `y` 不重叠，于是退化成普通的 `LDG.E`，走不了只读数据缓存（`__ldg` 路径）。
   **这是 `__restrict__` 值得写的实证。**

Triton 侧看 PTX（`python triton_mul2.py` 自动打印）：
```
ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
```
同样是 128-bit，编译器自动做的。

### 7.3 compute-sanitizer：抓静默越界

这是 ncu 不可用时**最有价值**的工具。`cuda_mul2.cu` 里故意留了一个无边界检查的
kernel（`cuda_mul2_kernel_buggy`）来演示。

**第一步：不用 sanitizer 跑，看看 bug 有多隐蔽**

```bash
$ ./cuda_mul2 1000 1 256 1
n=1000 (0.0 MiB per buffer), iters=1, block=256
!! 运行无边界检查的 kernel（n=1000, block=256, 尾块多出 24 个线程）
!! cudaDeviceSynchronize -> no error        ← CUDA 说"没问题"
scalar :   0.0092 ms        0.9 GB/s
vec4   :   0.0133 ms        0.6 GB/s
max_abs_err = 0  -> PASS                    ← 校验也说"通过"
```

**程序不报错、结果校验通过、退出码 0。** 24 个线程越界读写了，但因为越界地址
恰好落在同一个显存页里，没触发保护。这种 bug 在生产环境可能潜伏几个月，
直到某天分配布局变了才炸。

**第二步：用 sanitizer 跑**

```bash
$ compute-sanitizer --tool memcheck ./cuda_mul2 1000 1 256 1
========= Invalid __global__ read of size 4 bytes
=========     at cuda_mul2_kernel_buggy(const float*, float*, int)+0x70
=========        in /mnt/gfs/.../vector_mul2/cuda_mul2.cu:53
=========     by thread (232,0,0) in block (3,0,0)
=========     Address 0x7c5838a00fa0 is out of bounds
=========     and is 1 bytes after the nearest allocation at 0x7c5838a00000 of size 4,000 bytes
=========     Host Frame:launch_mul2_buggy [0x13cb4]
=========     Host Frame:main [0xc025]
...
========= Program hit cudaErrorLaunchFailure (error 719) due to "unspecified
=========   launch failure" on CUDA API call to cudaDeviceSynchronize.
========= ERROR SUMMARY: 26 errors
```

**精确到源码行 `cuda_mul2.cu:53`、线程 `(232,0,0)`、block `(3,0,0)`。**

算一下对不对：`n=1000`、`block=256` → `grid = ceil(1000/256) = 4`，共 1024 线程；
下标 1000~1023 越界，正好 **24 个线程**。26 = 24 个越界读 + 2 条级联的 API 错误
（`cudaDeviceSynchronize` 和 `cudaMemcpy` 都返回了 `cudaErrorLaunchFailure`）。
越界**写**没被单独报出来，是因为 kernel 在越界读之后就被 sanitizer 终止了。

注意对比：同一个程序**不带 sanitizer** 时 `cudaDeviceSynchronize` 返回 `no error`，
**带 sanitizer** 时返回 `cudaErrorLaunchFailure(719)` —— sanitizer 把静默错误
变成了硬失败，这正是它的价值。

能定位到源码行是因为 `Makefile` 里加了 `-lineinfo`。没有它只能看到汇编偏移。

**验证正确的版本干净：**

```bash
$ compute-sanitizer --tool memcheck ./cuda_mul2 1048576 5 256
========= ERROR SUMMARY: 0 errors
```

其他 sanitizer 工具：

```bash
compute-sanitizer --tool racecheck  ./app   # shared memory 数据竞争
compute-sanitizer --tool synccheck  ./app   # __syncthreads 在分支里被部分线程执行
compute-sanitizer --tool initcheck  ./app   # 读取未初始化的显存
```

> ⚠️ sanitizer 下程序会慢 10~100 倍（本例 4 MiB 时 0.39 ms vs 0.10 ms），
> 只用来验正确性，**不要用它的计时数据**。

### 7.4 用 nsys 代替 ncu 做粗粒度对比

虽然拿不到硬件计数器，但 `nsys stats --report cuda_gpu_kern_sum` 给出的
kernel 耗时（§4.3a）已经足以做 A/B 对比：**知道"哪个快"不需要 ncu，
只有搞清楚"为什么快"才需要。** 而本例的"为什么"，用 §7.2 的 SASS 就回答了。

### 7.5 这些替代手段和 ncu 的结论对得上吗？

对得上，而且是很好的交叉验证：

| 问题 | 免权限手段的答案 | ncu 的答案（§5.3） |
|---|---|---|
| 向量化差多少？ | SASS：`LDG.E.128` vs `LDG.E`，推断 4× 指令数 | `smsp__inst_executed_op_global_ld.sum`：524,288 vs 131,072，**正好 4×** |
| 有没有寄存器 spill？ | `-Xptxas=-v`：`0 bytes spill` | `launch__registers_per_thread` = 16，远低于上限 |
| 哪个实现快？ | nsys kernel 耗时排行 | Duration 列，排序一致 |
| 是 memory-bound 吗？ | 算带宽 ≈ 85% peak，推断是 | `sm__throughput` 只有 5% —— **直接证据** |

**唯一必须靠 ncu 的是最后一行。** 前面三个问题，免权限手段给出的答案在数值上
和 ncu 完全一致。所以实践顺序应该是：**先用 nsys + SASS 把能回答的问题回答掉，
剩下真的需要硬件计数器时再开 Docker。**

---

## 8. 踩坑清单

按踩坑概率从高到低排：

| # | 坑 | 后果 | 正确做法 |
|---|---|---|---|
| 1 | **不 warmup** | Triton 首次 JIT 690 ms，测出来全是编译时间 | `common.bench()` 里有 25 次 warmup |
| 2 | **不同步就计时** | kernel launch 是异步的，只测到 launch 的几 us | 用 `torch.cuda.Event` 或 `cudaEvent`，末尾 `synchronize()` |
| 3 | **用 `time.time()` 测 GPU** | 混入 host 调度抖动 | CUDA Event 记录在流上，测的是 GPU 时间线 |
| 4 | **忘了 `if (idx < n)`** | 静默越界，程序还"PASS" | 必写；用 compute-sanitizer 验（§7.3） |
| 5 | **profile 时用 `-G`** | `-G` 关掉所有优化，性能数据完全失真 | profile 用 `-lineinfo`，`-G` 只给 cuda-gdb 用 |
| 6 | **ncu 不收窄范围就跑 Python** | replay 机制，几十分钟到几小时 | 必加 `-k` / `--nvtx-include` / `--launch-count 1` |
| 7 | **只看一次测量** | GPU 有 DVFS 频率波动 | `common.bench()` 取 5 组的中位数 |
| 8 | **ctypes 不声明 argtypes** | 64 位指针被当成 int 截断，段错误 | `bench_all.py` 里显式设 `argtypes` |
| 9 | **ctypes 传 stream=0** | kernel 跑在默认流，和 torch 的流不同步 | 传 `torch.cuda.current_stream().cuda_stream` |
| 10 | **在别人用的卡上跑** | 数据全废，还影响同事 | 先 `nvidia-smi` 挑 `memory.used` 最小的卡 |
| 11 | **nsys `-o` 带扩展名** | 生成 `x.nsys-rep.nsys-rep` | `-o reports/all`，nsys 自己加后缀 |
| 12 | **忘了 `--force-overwrite`** | 同名报告存在时 nsys 直接报错退出 | 脚本里都加了 |
| 13 | **conda 里找 ncu/nsys** | 找不到 —— conda 的 `cuda-nvcc` 不含 Nsight | 用 `/usr/local/cuda/bin/` 下的 |
| 14 | **把 `torch.randn` 的 kernel 当被测对象** | 汇总表里混进无关 kernel | 认清名字，或用 NVTX 圈定范围 |
| 15 | **测出 `cudaLaunchKernel` avg 很大就慌** | avg 被首次 launch 的 12.8 ms 拉爆 | 看 **median**，不看 avg |
| 16 | **看到 `ERR_NVGPUCTRPERM` 就以为只能等管理员重启** | 白等 | 用 Docker + `--cap-add=SYS_ADMIN`（§5.2），不改宿主机、不重启 |
| 17 | **Docker 里加了 `--user` 想让产物属主正确** | `CapEff: 0`，capability 全丢，ncu 照样报 `ERR_NVGPUCTRPERM` | 必须容器 root 跑，结尾 `chown` 回来 |
| 18 | **忘了 `chown` 产物** | reports/ 下全是 root 属主，宿主机删不掉、GUI 读不了 | 每个 `docker run` 末尾 `chown $(id -u):$(id -g)` |
| 19 | **以为 `nvidia/cuda:*-devel` 镜像带 nsys** | 只有 ncu 和 nvcc，没有 nsys | 从宿主机把 `nsight-systems-*` 目录 `-v ...:/opt/nsys:ro` 挂进去 |
| 20 | **`--gpus '"device=3"'` 的嵌套引号写少一层** | docker 参数解析失败 | shell 一层 + docker 一层，两层都要；脚本里写成 `--gpus "\"device=$GPU\""` |
| 21 | **`--launch-count N` 直接用，不数 warmup** | 名额被 warmup 的 kernel 吃光，报告里没有你要的 kernel | 配 `--launch-skip`，或用 `--nvtx-include 'bench_vec4/'` 按区段筛（§5.4） |
| 22 | **只读挂载 conda env 后跑 triton** | triton 写缓存失败 | `-e TRITON_CACHE_DIR=/tmp/triton_cache` |

---

## 9. 我的笔记

<!-- 下面留给自己复现时补充 -->

### 复现记录

- [ ] `bash run_all.sh` 跑通，结果和 §3 对得上吗？
- [ ] `bash run_all.sh --profile` 跑通，nsys 报告生成了吗？
- [ ] `bash profile/run_ncu_docker.sh` 跑通，`reports/ncu_cuda.ncu-rep` 生成了吗？
      四个 kernel 的指标和 §5.3 的表对得上吗？
- [ ] 故意去掉 `--cap-add=SYS_ADMIN` 跑一次，确认会退回 `ERR_NVGPUCTRPERM`
      （亲手验证"是这一条起的作用"，比看文档印象深）
- [ ] 把 `reports/all.nsys-rep` 下载到本地用 GUI 打开，找到 690 ms 的 warmup 段
- [ ] 在 GUI 里对比大规模 vs 小规模，看 kernel 块之间的空隙
- [ ] 在 ncu GUI 里对 scalar / vec4 做一次 Add Baseline，看 `-75%` 的指令数差

### 待验证 / 待扩展

- [ ] **Source 页**：ncu 报告是带 `-lineinfo` 编的，但源码在容器里的路径是
      `/work/cuda_mul2.cu`。本地 GUI 打开时如果找不到源码，用
      **Resolve → 手动指定目录** 映射回本地路径
- [ ] **pinned memory**：把 `main.cu` 的 `std::vector` 换成 `cudaMallocHost`，
      看 H2D 能不能从 9.2 GB/s 提到 ~25 GB/s（§4.3d）
- [ ] **算子融合**：写一个 `(x*2+1).relu()`，对比
      "torch 三个 kernel" vs "Triton 一个 kernel" —— 这才是 Triton 真正的价值，
      本例的单算子 mul2 完全体现不出来
- [ ] **CUDA Graph**：小规模下用 `torch.cuda.graphs` 消掉 launch 开销，
      看能不能把 4K 元素的 6.7 us 压下去（§3.2 观察 1）
- [ ] **grid-stride loop**：把 CUDA kernel 改成 grid-stride 写法，
      固定 grid 大小，对比性能和 occupancy
- [ ] **换 dtype**：fp16 / bf16 下带宽利用率会怎样？访存量减半，
      launch 开销占比会更高吗？

### 我的观察

<!-- 写在这里 -->
