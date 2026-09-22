# 纯 Triton 练习

只写 Triton，不掺 CUDA C++、不做 profiler 全流程 —— 目标是把 Triton 的编程模型
本身练熟。想看「同一个算子用三种方案写 + 完整 ncu/nsys 分析流程」，去
[`../综合练习/vector_mul2/`](../综合练习/vector_mul2/README.md)。

所有代码在本机（8×A100-40GB，triton 3.5.0）实测跑通，README 里的数字都是实测值。

---

## 目录

- [0. 快速开始](#0-快速开始)
- [1. 练习列表](#1-练习列表)
- [2. Triton 编程模型速查](#2-triton-编程模型速查)
- [3. 实测结果与核心结论](#3-实测结果与核心结论)
- [4. 调试与排查手段](#4-调试与排查手段)
- [5. 踩坑清单](#5-踩坑清单)
- [6. 后续练习方向](#6-后续练习方向)
- [7. 我的笔记](#7-我的笔记)

---

## 0. 快速开始

每个练习是一个**目录**：`v*.py` 是各个变体（`v0` 朴素 → 逐步优化），
`bench.py` 把它们串起来对比，`README.md` 记录效果、原理和结论，
`_shared.py` 放该练习共用的测试用例和报告函数。

```bash
cd /mnt/gfs/nyt1/infra/practices/triton

# 一键跑完三个练习
bash run_all.sh                 # 完整（含参数扫描）
bash run_all.sh --quick         # 跳过扫描，只跑正确性 + 变体对比

# 或单独跑某个练习的全部变体
export CUDA_VISIBLE_DEVICES=7   # 先挑一张空闲卡
PY=/mnt/public/nyt1/docqa/restored_envs/cpp/bin/python
cd 01_vector_add && $PY bench.py          # 或 bench.py --quick

# 或只跑单个变体（每个 v*.py 自带正确性 + 单点性能）
cd 01_vector_add && $PY v0_naive.py
```

**环境**：conda env `cpp`（`/mnt/public/nyt1/docqa/restored_envs/cpp`），
自带 torch 2.9.0+cu128 + triton 3.5.0。脚本里写了绝对路径，**不需要
`conda activate` 也能跑**。想换环境：`PYTHON=/path/to/python bash run_all.sh`。

**挑空闲卡**（共享机器，别挤在别人正在用的卡上）：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=7       # 选 memory.used 最小的
```

`nvidia-smi` 在某些 shell 里跑不起来（缺 loader，报 `No such file or directory`），
用 NVML 查更可靠 —— 命令见 [`../cuda/README.md` §3.3](../cuda/README.md)。
**卡被别人占着的话绝对数字全废**（实测慢 2.1~2.3x），但优化链的**比值**基本不变。

---

## 1. 练习列表

难度递增，每个变体都可独立运行，末尾都有「[练习] 试试看」的自测题。

| 目录 | 主题 | 新学到的东西 | 类型 |
|---|---|---|---|
| [`01_vector_add/`](01_vector_add/README.md) | 向量加法 | `program_id` / `arange` / `mask` / grid lambda | memory-bound |
| [`02_fused_softmax/`](02_fused_softmax/README.md) | 融合 Softmax | `tl.max`/`tl.sum` 规约、`other=`、`num_warps`、**算子融合** | memory-bound |
| [`03_matmul/`](03_matmul/README.md) | 分块矩阵乘 | 2D 分块、K 循环、寄存器累加器、`tl.dot`（Tensor Core）、group-major | **compute-bound** |
| `common.py` | — | 共用的 CUDA Event 计时 / 校验 / 带宽算力换算 | — |
| `run_all.sh` | — | 一键全跑 | — |

各目录里的变体：

| | `v0` | `v1` | `v2` | `v3` |
|---|---|---|---|---|
| **01_vector_add** | 朴素（连续分段） | **反面教材**：跨步访存 | 循环展开 | autotune |
| **02_fused_softmax** | torch 手工拼（未融合） | 融合版（一 program 一行） | persistent | online（分块规约） |
| **03_matmul** | 朴素分块 | 调优分块 + group-major | autotune | 融合 epilogue |

**推荐顺序**：01 → 02 → 03。01 建立基本手感，02 是 Triton **最有说服力**的
用例（融合带来 4.3× 提速），03 才开始碰真正的性能调优。

> **CUDA 对照版在 [`../cuda/`](../cuda/README.md)**，同样的三个问题用
> CUDA C++ 再做一遍。每个练习的 README 末尾都有一节 "和 CUDA/Triton
> 版的对照"，讲清楚哪些活是编译器干的、哪些还得自己想。

---

## 2. Triton 编程模型速查

### 2.1 和 CUDA 的对照

最大的心智差异：**Triton 的编程粒度是 block（program），不是 thread。
你看不到 `threadIdx`。**

| CUDA | Triton | 说明 |
|---|---|---|
| `blockIdx.x` | `tl.program_id(axis=0)` | 我是第几个 program |
| `threadIdx.x` | **没有** | block 内怎么分给线程，编译器决定 |
| `blockDim.x` | `BLOCK_SIZE`（`tl.constexpr`） | 编译期常量，每个值触发一次独立 JIT |
| 每 block 线程数 | `num_warps`（launch 参数） | 和 `BLOCK_SIZE` 是**两个独立概念** |
| `if (idx < n)` | `mask=` 参数 | 越界 lane 自动不读不写 |
| 手写 `float4` | 自动向量化 | 编译器生成 `.v4.b32`，见 §3.1 |
| shared memory + `__shfl` 手写规约 | `tl.max(x, axis=0)` | 一行搞定 |
| `__syncthreads()` | **通常不需要** | block 内的同步由编译器插 |
| 标量运算 | **全是向量运算** | `offsets` 是长度 BLOCK_SIZE 的向量 |

### 2.2 一个 kernel 的标准骨架

```python
@triton.jit
def my_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                              # ① 我是谁
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)       # ② 我负责哪段
    mask = offs < n                                          # ③ 边界保护
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)          # ④ 读
    tl.store(out_ptr + offs, x * 2, mask=mask)               # ⑤ 算 + 写

grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)    # ⑥ 要几个 program
my_kernel[grid](x, out, n, BLOCK_SIZE=1024)
```

### 2.3 `BLOCK_SIZE` vs `num_warps`

这两个最容易混：

- **`BLOCK_SIZE`**：一个 program 处理**多少数据元素**。`tl.constexpr`，编译期常量，
  必须是 2 的幂。
- **`num_warps`**：编译器用**多少个 warp（32 线程）**去处理这些数据。launch 参数。

`BLOCK_SIZE=4096, num_warps=4` 意味着 128 个线程处理 4096 个元素，
每线程 32 个。改 `num_warps` 不改变每个 program 处理的数据量，
只改变并行处理它的线程数 —— 对规约类 kernel 影响很大（`02` 里能看到）。

### 2.4 `mask` 和 `other` 的配合

`other=` 指定被 mask 掉的 lane 读到什么值。**选错会静默算错**：

| 场景 | 正确的 `other` | 用错会怎样 |
|---|---|---|
| 求和 / 累加 | `0.0` | 非 0 会污染结果 |
| 求最大值 | `-float("inf")` | `0.0` 会让全负数的行算出 max=0 |
| softmax | `-float("inf")` | `exp(-inf)=0`，既不影响 max 也不影响 sum |
| 求最小值 | `float("inf")` | 同理 |

---

## 3. 实测结果与核心结论

环境：A100-SXM4-40GB（sm_80），HBM2e 峰值 **1555 GB/s**，
fp16 Tensor Core 密集峰值 **312 TFLOP/s**。

> 下面是**跨练习**的横向结论。每个练习自己的四个变体怎么一步步优化上来、
> 每一步收益多少，在各自目录的 README 里：
> [01](01_vector_add/README.md) / [02](02_fused_softmax/README.md) /
> [03](03_matmul/README.md)。

### 3.1 练习 01：memory-bound 的天花板

`n = 16,777,216`，读 x + 读 y + 写 out = `3 × n × 4` 字节：

```
  torch x + y                   148.9 us    1352.3 GB/s  ( 87.0% of peak)
  triton BLOCK_SIZE=128         152.2 us    1322.4 GB/s  ( 85.0% of peak)
  triton BLOCK_SIZE=256         151.1 us    1332.2 GB/s  ( 85.7% of peak)
  triton BLOCK_SIZE=1024        151.4 us    1329.9 GB/s  ( 85.5% of peak)
  triton BLOCK_SIZE=4096        149.4 us    1347.8 GB/s  ( 86.7% of peak)
```

**结论：`BLOCK_SIZE` 几乎没有影响（差距 <2%），Triton 和 torch 打平。**
纯 memory-bound 算子只要访存是 coalesced 的，谁都能吃到 ~85% HBM 带宽，
没有优化空间。**这类算子上 Triton 的价值不是"更快"，而是"能融合"。**

Triton 自动做了 128-bit 向量化，PTX 里能直接看到（脚本会打印）：

```
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
```

`.v4.b32` = 一条指令读 4 个 32-bit，等价于手写 CUDA 的 `float4`。

> ⚠️ **但别把追平 torch 归功于向量化。** 后来做了一个隔离实验（固定合并访存模式，
> 只切换向量化开关）：**向量化单独值 0%**（1346.6 vs 1342.1 GB/s）。
> 在已经打满带宽的 kernel 上，少发指令没有地方可省 —— 真正决定成败的是
> **合并访存**。详见 [`01_vector_add/README.md` §3.2](01_vector_add/README.md)
> 和 [`01_vector_add/v0_naive.md` §2.7](01_vector_add/v0_naive.md)。

### 3.2 练习 02：融合才是 Triton 的价值 ⭐

`shape = (4096, 4096)`，按理想访存量（读一次写一次）折算：

```
  torch 手工拼（未融合）         459.8 us     291.9 GB/s  ( 18.8% of peak)
  torch.softmax（已融合）        106.9 us    1255.2 GB/s  ( 80.7% of peak)
  triton 融合版                  101.9 us    1316.8 GB/s  ( 84.7% of peak)
```

**triton 融合版 vs torch 手工拼算子：4.31×。**

这是全目录最重要的一个数字。手工拼的版本

```python
z = x - x.max(dim=1, keepdim=True)[0]   # kernel 1+2，中间结果落显存
numerator = torch.exp(z)                 # kernel 3，中间结果落显存
denominator = numerator.sum(dim=1, ...)  # kernel 4
return numerator / denominator           # kernel 5
```

每一步都是独立 kernel，每个都要完整读一遍写一遍显存。融合版把整行读进
寄存器，**中间结果一次都不写回显存**，一次读写完成全部计算。

> **这才是写 Triton 的理由。** 单算子（练习 01）只能追平 torch；
> 多算子融合才有 4× 这种量级的提升。挑 Triton 改写的目标时，
> 优先找「连续好几个 elementwise/reduction 算子」的地方。

`torch.softmax` 自己也是融合实现，所以和 triton 打平很正常 ——
**torch 有的算子别去重写，要重写的是 torch 没有的融合组合。**

行宽扫描（M=4096 固定）：

```
  triton N=256                   22.2 us     377.9 GB/s  ( 24.3% of peak) BLOCK=256, warps=4
  triton N=1024                  22.9 us    1466.8 GB/s  ( 94.3% of peak) BLOCK=1024, warps=4
  triton N=4096                 101.9 us    1316.5 GB/s  ( 84.7% of peak) BLOCK=4096, warps=16
  triton N=16384                396.7 us    1353.4 GB/s  ( 87.0% of peak) BLOCK=16384, warps=16
```

- `N=256` 带宽只有 24%：总共才 4 MiB，GPU 还没忙起来就结束了，
  瓶颈是 launch 开销。**规模太小时带宽利用率这个指标本身没意义。**
- `N=1024` 拿到全场最高 94.3% —— 一行刚好装满寄存器，规模又够大。
- `N` 再大要靠 `num_warps=16` 撑住，寄存器压力上升，带宽略降。
- **`N` 大到一行装不进寄存器时，这个「一 program 一行」的写法就失效了**，
  必须改成 online softmax（分块规约）—— 这正是 FlashAttention 的核心技巧。

### 3.3 练习 03：compute-bound 才需要真调优

方阵 fp16，`2 × M × N × K` FLOP：

```
  torch  512x512                  9.8 us      27.5 TFLOP/s  (  8.8% of peak)
  triton 512x512                 27.7 us       9.7 TFLOP/s  (  3.1% of peak) (35% of torch)
  torch  1024x1024               21.2 us     101.3 TFLOP/s  ( 32.5% of peak)
  triton 1024x1024               47.0 us      45.7 TFLOP/s  ( 14.7% of peak) (45% of torch)
  torch  2048x2048               93.9 us     183.0 TFLOP/s  ( 58.6% of peak)
  triton 2048x2048              167.4 us     102.6 TFLOP/s  ( 32.9% of peak) (56% of torch)
  torch  4096x4096              647.4 us     212.3 TFLOP/s  ( 68.0% of peak)
  triton 4096x4096              736.3 us     186.7 TFLOP/s  ( 59.8% of peak) (88% of torch)
```

**别指望在 matmul 上打赢 cuBLAS。** 小矩阵上只有 cuBLAS 的 35%，
到 4096 才追到 88%。cuBLAS 对每个尺寸段都有手调 kernel 和启发式选择，
这是 NVIDIA 投了十几年的东西。**写 Triton matmul 的意义在于能融合
epilogue（bias/激活/量化），而不是裸 matmul 更快。**

分块配置扫描（4096×4096）—— 这才是 matmul 真正要调的东西：

```
  BM=64  BN=64  BK=32          1040.0 us     132.1 TFLOP/s  ( 42.4%) warps=4 stages=5
  BM=64  BN=128 BK=32           837.9 us     164.0 TFLOP/s  ( 52.6%) warps=4 stages=4
  BM=128 BN=64  BK=32           878.7 us     156.4 TFLOP/s  ( 50.1%) warps=4 stages=4
  BM=128 BN=128 BK=32           732.5 us     187.6 TFLOP/s  ( 60.1%) warps=4 stages=4
  BM=128 BN=128 BK=64           830.2 us     165.5 TFLOP/s  ( 53.1%) warps=4 stages=4
  BM=128 BN=256 BK=64           735.0 us     187.0 TFLOP/s  ( 59.9%) warps=8 stages=3
  BM=256 BN=128 BK=64           714.5 us     192.4 TFLOP/s  ( 61.7%) warps=8 stages=3
```

**最好和最差差 1.46×**（`256×128×64` vs `64×64×32`）。
对比练习 01 里 `BLOCK_SIZE` 的 <2% 差距 —— **compute-bound 和 memory-bound
的优化重点完全不同**：前者调分块，后者基本无可调。

### 3.4 一个反直觉的实测：GROUP_M 在 A100 上几乎没用

`group-major` 排序是 Triton 官方 matmul 教程强调的技巧，理论上能提高 L2 命中率。
实测 4096×4096：

```
  GROUP_M=1                     727.2 us     189.0 TFLOP/s
  GROUP_M=2                     724.5 us     189.7 TFLOP/s
  GROUP_M=4                     734.2 us     187.2 TFLOP/s
  GROUP_M=8                     735.2 us     187.0 TFLOP/s
  GROUP_M=16                    726.3 us     189.2 TFLOP/s
```

**全在噪声内，毫无影响。** 原因：A100 的 L2 有 **40 MB**，而 B 矩阵
（4096×4096×2 字节）才 **32 MB** —— 整个矩阵基本就待在 L2 里，
怎么排都命中，group-major 没有用武之地。

换到 8192×8192（B 矩阵 128 MB，装不下）才看出区别：

```
  GROUP_M=1     5715.5 us    192.4 TFLOP/s
  GROUP_M=16    5498.7 us    200.0 TFLOP/s     ← +4%
```

> **教训：优化技巧是否有效，取决于硬件参数和问题规模。**
> 照搬别人（尤其是不同代 GPU 上）的调优结论而不实测，很容易做无用功。
> 官方教程用的是 L2 更小的卡，结论在 A100 上就打了折扣。

---

## 4. 调试与排查手段

### 4.1 看编译产物（寄存器 / shared memory / PTX）

`common.py` 里的 `kernel_info()` 封装了这个。手动写法：

```python
c = my_kernel.warmup(x, out, n, BLOCK_SIZE=1024, grid=(1,))
print(c.metadata.shared, c.metadata.num_warps, c.metadata.num_stages)
print(getattr(c, "n_regs", "n/a"))     # 注意：warmup 后可能还取不到
print(c.asm["ptx"])                    # 完整 PTX
print(c.asm.keys())                    # ttir / ttgir / llir / ptx / cubin
```

> ⚠️ `n_regs` **只有 kernel 真正被加载到设备之后才有值**。刚 `warmup()` 出来的
> 对象上取会是 `None`/报错，所以要 `getattr(c, 'n_regs', 'n/a')` 兜一下。
> 实测：`num_warps=4/8` 时取不到，`num_warps=16` 时能取到 —— 不稳定，别依赖。

看中间 IR 对理解编译过程很有用：`ttir`（Triton IR）→ `ttgir`（带 layout 的）
→ `llir`（LLVM IR）→ `ptx` → `cubin`。

### 4.2 `TRITON_INTERPRET=1`：用 Python 调试 kernel

```bash
TRITON_INTERPRET=1 python 01_vector_add/v0_naive.py
```

在解释器模式下跑 kernel，**可以在 `@triton.jit` 函数里加 `print()` 和断点**，
`tl.load` 的结果是真的 numpy 数组。极慢，但排查逻辑错误（尤其是 mask 和
下标算错）非常好用。

### 4.3 缓存与环境变量

| 变量 | 作用 |
|---|---|
| `TRITON_CACHE_DIR` | JIT 缓存目录，默认 `~/.triton/cache`。只读文件系统里必须改 |
| `TRITON_INTERPRET=1` | 解释器模式，见上 |
| `TRITON_PRINT_AUTOTUNING=1` | 打印 autotune 选中的配置 |
| `MLIR_ENABLE_DUMP=1` | dump 每一遍 pass 之后的 IR（很啰嗦） |

**Triton 的 JIT 结果会缓存**，所以「第一次跑」和「第二次跑」性能差异巨大
（首次编译可达数百毫秒）。写 benchmark 一定要 warmup，见 §5 第 1 条。

### 4.4 配合 profiler

本目录不做完整的 profiler 流程，但要用的话：

```bash
# nsys：看时间线、确认 kernel 名字和 launch 开销
/usr/local/cuda/bin/nsys profile -t cuda,nvtx -o rep --force-overwrite true \
    $PY 02_fused_softmax/v1_fused.py
/usr/local/cuda/bin/nsys stats --report cuda_gpu_kern_sum rep.nsys-rep

# ncu：看硬件计数器。本机宿主机上会报 ERR_NVGPUCTRPERM，要走 Docker
#      完整方案见 ../综合练习/vector_mul2/README.md §5
```

对 Triton kernel 用 ncu 时**必须收窄范围**（`-k regex:softmax_kernel
--launch-count 1`），否则 torch 初始化的几十个 kernel 都要被 replay。

实测（容器内，`--cap-add=SYS_ADMIN`）：

```bash
ncu --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread \
    -k 'regex:softmax_kernel' --launch-count 1 python 02_fused_softmax/v1_fused.py
```
```
  softmax_kernel (4, 1, 1)x(128, 1, 1), CC 8.0
    dram__throughput...                    %      0.10
    gpu__time_duration.sum           usecond      3.87
    launch__registers_per_thread   register/       16
    sm__throughput...                      %      0.09
```

> ⚠️ **注意 grid 是 `(4,1,1)`** —— 抓到的是脚本开头 `shape=(4,128)` 那次
> **正确性检查**的 launch，不是后面的性能测试！所以吞吐量只有 0.1%。
> 这就是「`--launch-count N` 之前先数清楚第几次 launch 是你要的那次」这条
> 铁律的现场演示。要抓性能测试那次，得配 `--launch-skip`，
> 或者用 `--nvtx-include` 按区段筛。

matmul 值得看的额外指标（已验证指标名有效）：

```bash
ncu --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    -k 'regex:matmul_kernel' --launch-skip 3 --launch-count 1 python 03_matmul/v1_tiled.py
```
```
  matmul_kernel (32, 1, 1)x(256, 1, 1), CC 8.0
    gpu__time_duration.sum                                  usecond    47.20
    sm__pipe_tensor_op_hmma_cycles_active...                      %    67.47   ← Tensor Core
    sm__throughput.avg.pct_of_peak_sustained_elapsed              %    18.95
```

**Tensor Core 利用率 67.47%，而整体 SM 吞吐只有 18.95%。** 这两个数字放一起
才有意义：说明 SM 大部分时间不是在算，而是在等数据 / 做地址计算 ——
matmul kernel 的优化方向就是把 Tensor Core 喂得更满（调分块、调 `num_stages`
让 software pipelining 更深）。

> ⚠️ 上面两段输出是在**拆目录之前**的单文件脚本上抓的，
> `--launch-skip/--launch-count` 的序号对现在的 `v*.py` **不一定还对得上**
> （每个变体自己的正确性用例数量不同）。这恰好又印证了上一条：
> **序号要自己数，别抄。** 先用 `--print-summary per-kernel` 或 nsys
> 看一眼一共 launch 了几次、哪次是你要的。

其他有用的：

```
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum      # shared memory 流量
launch__occupancy_limit_registers                   # 是不是寄存器限制了 occupancy
smsp__inst_executed_op_global_ld.sum                # global load 指令数
```

---

## 5. 踩坑清单

| # | 坑 | 后果 | 正确做法 |
|---|---|---|---|
| 1 | **不 warmup** | 首次 JIT 几百毫秒，测出来全是编译时间 | `common.bench()` 里有 25 次 warmup |
| 2 | **不同步就计时** | launch 是异步的，只测到几微秒的 launch 开销 | 用 `torch.cuda.Event` + `synchronize()` |
| 3 | **`BLOCK_SIZE` 不是 2 的幂** | 编译报错 `arange's range must be a power of 2` | 用 `triton.next_power_of_2(n)`。约束是 2 的幂，**不是 32 的倍数** —— `96`/`192` 也过不了 |
| 4 | **忘了 `mask=`** | 静默越界读写，结果可能还"看着对" | 一律写 mask。**查越界要用 `compute-sanitizer` + `PYTORCH_NO_CUDA_MEMORY_CACHING=1`** —— 不加这个环境变量会报 0 errors（torch 缓存分配器挡住了），实测见 [`01_vector_add/v0_naive.md` §1](01_vector_add/v0_naive.md)。`TRITON_INTERPRET=1` 只适合查逻辑错，查越界会崩在 host 堆上且无定位信息 |
| 5 | **`other=` 选错** | 静默算错（求 max 时用 `other=0`，全负数的行就错） | 按 §2.4 的表选 |
| 6 | **混淆 `BLOCK_SIZE` 和 `num_warps`** | 以为改了并行度其实没改 | 见 §2.3：一个是数据量，一个是线程数 |
| 7 | **规约时 `axis` 写错** | 结果 shape 不对或规约错方向 | 1D 数据用 `axis=0`；2D 想按行规约是 `axis=1` |
| 8 | **对非 contiguous 张量直接用指针算术** | 结果错乱 | 传 `stride()` 进 kernel，或先 `.contiguous()` |
| 9 | **fp16 累加** | 精度崩坏，K 大时尤甚 | accumulator 用 `tl.float32`，最后再 `.to(tl.float16)` |
| 10 | **fp16 matmul 要求完全相等** | 校验必然 FAIL | `atol` 随 K 放大（误差 ~ `sqrt(K)·eps`） |
| 11 | **`n_regs` 当成稳定接口用** | `AttributeError` 或 `None` | 只有 kernel 上设备后才有值，用 `getattr` 兜 |
| 12 | **只看一次测量** | GPU 有 DVFS 频率波动 | `common.bench()` 取 5 组中位数 |
| 13 | **在别人用的卡上跑** | 绝对数字全废（实测慢 2.1~2.3x），还影响同事 | 先挑空闲卡，见 §0；比值还能用，绝对值不能记 |
| 14 | **只读文件系统里跑 triton** | JIT 写缓存失败 | 设 `TRITON_CACHE_DIR=/tmp/triton_cache` |
| 15 | **照搬别人的调优参数** | 可能完全无效（见 §3.4 的 GROUP_M） | 在自己的卡和自己的规模上实测 |
| 16 | **以为重写 torch 已有算子能变快** | 白干（见 §3.2 的 `torch.softmax`） | 挑 torch **没有**的融合组合下手 |

---

## 6. 后续练习方向

拆成目录之后，原来列在这里的三项已经作为变体做掉了：

- [x] **融合 epilogue** → [`03_matmul/v3_fused_relu.py`](03_matmul/README.md)
      —— `relu(A@B + bias)`，对比「triton 一个 kernel」vs「torch 三个 kernel」。
      这就是 §3.3 说的「Triton matmul 的真正意义」
- [x] **online softmax** → [`02_fused_softmax/v3_online.py`](02_fused_softmax/README.md)
      —— 解决 §3.2 里「一行装不进寄存器」的问题
- [x] **`@triton.autotune` 系统化调优** → `01_vector_add/v3_autotune.py`
      和 `03_matmul/v2_autotune.py`，配 `TRITON_PRINT_AUTOTUNING=1` 看它选了什么

剩下的，按「学到的东西 / 投入时间」排序：

- [ ] **04：LayerNorm（含反向）** —— 前向是规约练习，**反向要处理跨 program 的
      梯度累加**，会用到 `tl.atomic_add` 和锁，难度陡增
- [ ] **05：FlashAttention 前向** —— online softmax + 分块 matmul 的综合，
      Triton 最有代表性的应用
- [ ] **06：dropout + 随机数** —— `tl.rand`，理解 seed/offset 的用法
- [ ] **07：量化 matmul** —— int8/fp8，epilogue 里做 dequant

配合 [`../综合练习/vector_mul2/`](../综合练习/vector_mul2/README.md) 的
profiler 流程，可以给上面每一个练习都做一遍 nsys + ncu 分析；
想把同一个问题用 CUDA C++ 再写一遍，去 [`../cuda/`](../cuda/README.md)。

---

## 7. 我的笔记

<!-- 下面留给自己复现时补充 -->

### 复现记录

- [ ] `bash run_all.sh` 跑通，三个练习的数字和 §3 对得上吗？
- [x] 练习 01 的三道题 → [`01_vector_add/v0_naive.md`](01_vector_add/v0_naive.md)
      （去掉 mask 后 compute-sanitizer 默认报 **0 errors**，原因和正确姿势都在里面）
- [ ] 练习 02 第 1 题：去掉 `- tl.max(...)`，观察 nan 怎么出现
- [ ] 练习 02 第 2 题：`other=-inf` 改成 `other=0`，哪种输入会算错？
- [ ] 练习 03 第 1 题：accumulator 改 fp16，误差变多少？
- [ ] 试一次 `TRITON_INTERPRET=1`，在 kernel 里 print 出 offsets 和 mask

### 我的观察

<!-- 写在这里 -->
