# 01_vector_add —— 向量加法

`out = x + y`。Triton 的 "Hello World"，也是**memory-bound 算子的基准参照**。

这个练习要建立的核心认知：**纯 elementwise 算子几乎没有优化空间，
但有巨大的「写坏」空间。** 写对了就是 ~86% 带宽（和 torch 打平），
访存模式写坏了直接掉到 15%。

---

## 1. 变体一览

| 文件 | 做了什么 | 实测带宽 | vs torch |
|---|---|---|---|
| `v0_naive.py` | 连续分段，每个 program 一个 tile | 1332.6 GB/s (85.7%) | 100% |
| `v1_strided_bad.py` | **反面教材**：改成循环分配，访存打散 | 229.1 GB/s (14.7%) | 17% |
| `v2_unroll.py` | 每个 program 处理 UNROLL 个 tile | 1344.6 GB/s (86.5%) | 101% |
| `v3_autotune.py` | `@triton.autotune` 自动选配置 | 1342.8 GB/s (86.4%) | 101% |
| `bench.py` | 全变体对比 + 参数扫描 + PTX 反汇编 | — | — |
| `_shared.py` | 共用的用例 / 计时口径 / sys.path 处理 | — | — |

基线：`torch.add` = 151.0 us / 1333.7 GB/s (85.8%)。
A100-SXM4-40GB，HBM2e 峰值 1555 GB/s。

---

## 2. 用法

```bash
export CUDA_VISIBLE_DEVICES=7          # 先挑张空闲卡
PY=/mnt/public/nyt1/docqa/restored_envs/cpp/bin/python

$PY bench.py                # 推荐：全变体对比（约 2 分钟）
$PY bench.py --quick        # 跳过参数扫描

$PY v0_naive.py             # 单个变体：正确性 + 单点性能 + 练习题
$PY v1_strided_bad.py
TRITON_PRINT_AUTOTUNING=1 $PY v3_autotune.py     # 看 autotune 选了啥
```

每个变体都能独立跑，`_shared.py` 负责把 `triton/` 加进 `sys.path`，
所以不需要装包、也不依赖从哪个目录启动。

---

## 3. 原理与实测结论

### 3.1 memory-bound 的天花板：~86%，且和配置无关

```
  v0 BLOCK_SIZE=128             151.8 us    1326.4 GB/s  ( 85.3% of peak)
  v0 BLOCK_SIZE=256             150.9 us    1334.0 GB/s  ( 85.8% of peak)
  v0 BLOCK_SIZE=1024            151.2 us    1331.2 GB/s  ( 85.6% of peak)
  v0 BLOCK_SIZE=4096            149.2 us    1348.9 GB/s  ( 86.7% of peak)
```

`BLOCK_SIZE` 差 32 倍，性能差 **1.7%**。原因：这个 kernel 每读 8 字节只做
1 次加法，算术强度 = 1 FLOP / 12 Byte，彻底被 DRAM 带宽卡死。
只要 program 数量足够把所有 SM 填满、把访存延迟藏住，剩下的都由 HBM 决定。

**推论：在这类算子上做「调参优化」是浪费时间。** Triton 的价值不在这里
（见 [`../02_fused_softmax/`](../02_fused_softmax/README.md) 的融合）。

> 为什么是 86% 而不是 100%？1555 GB/s 是理论峰值，实际要扣掉刷新、
> ECC、行切换等开销。**85–90% 基本就是 A100 上 streaming kernel 的实际上限**，
> 达到这个数就可以停止优化了。

### 3.2 v1 反面教材：访存模式值 5.8 倍 ⭐

这是本练习最重要的一个数字。v0 和 v1 **算的东西完全一样、program 数完全一样、
结果完全正确**，唯一区别是下标怎么分：

```python
# v0：连续分段（contiguous / block 分配）
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#   program 0 → [0, 1023]，program 1 → [1024, 2047] …

# v1：循环分配（cyclic）
offsets = pid + tl.arange(0, BLOCK_SIZE) * n_programs
#   program 0 → [0, 16384, 32768, …]，program 1 → [1, 16385, …]
```

结果：

```
  v0 朴素版          151.1 us    1332.6 GB/s  ( 85.7%)
  v1 跨步访存         878.8 us     229.1 GB/s  ( 14.7%)    ← 5.8x 慢
```

**两件事同时坏掉了：**

| | v0 | v1 |
|---|---|---|
| per-thread 地址连续？ | 是 → 编译器发 128-bit 访存 | 否 → 只能 32-bit |
| per-warp 地址连续？（合并访存） | 是 → 32 lane 共用 ~4 条 cache line | 否 → 32 lane 打到 32 条不同 cache line |

PTX 里看得一清二楚（`bench.py` 会打印）：

```
  v0 连续      ld.global 共  4 条：128-bit × 4
    @%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
  v1 跨步      ld.global 共 16 条：32-bit × 16
    @%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ];
```

同样一个 1024 元素的 tile：v0 用 4 条指令读完，v1 要 16 条。
`.v4.b32` 就是「一条指令读 4 个 32-bit」，**等价于手写 CUDA 的 `float4`** ——
对照 [`../../cuda/01_vector_add/`](../../cuda/01_vector_add/README.md)，
那边是手写 `float4` 才拿到的东西，Triton 在访存模式允许时自动就做了。

> ⚠️ **在 Triton 里你不能直接「开启向量化」。** 没有 `float4` 类型、没有 pragma。
> 你能控制的只有**访存模式**；模式连续，编译器就向量化，模式打散，它无能为力。
> 所以 v1 这个实验测的是「访存模式」的总价值（向量化 + 合并访存），
> 没法把两者拆开单独归因 —— 这是 Triton 抽象层次带来的取舍。

跨步大小的影响（跨步被正确性绑死 = program 数，只能通过 `BLOCK_SIZE` 间接调）：

```
  v1 BLOCK=256     1882.1 us   107.0 GB/s ( 6.9%)   跨步 65536 元素 = 256 KB
  v1 BLOCK=1024     874.5 us   230.2 GB/s (14.8%)   跨步 16384 元素 =  64 KB
  v1 BLOCK=4096     926.6 us   217.3 GB/s (14.0%)   跨步  4096 元素 =  16 KB
```

跨步 256 KB 时最惨（6.9%）—— 跨步越大，同一个 warp 摸到的页越多，
L2 和 TLB 都开始失效。16 KB 和 64 KB 之间反而差不多，说明到这个量级
主要成本已经是「32 条 lane 打 32 条 cache line」，跨步再变化影响不大。

### 3.3 v2 循环展开：+0.8%，基本是噪声

```
  v2 UNROLL=1      151.2 us   1331.6 GB/s (85.6%)   16384 programs
  v2 UNROLL=2      151.0 us   1333.6 GB/s (85.8%)    8192 programs
  v2 UNROLL=4      149.8 us   1343.9 GB/s (86.4%)    4096 programs   ← 最好
  v2 UNROLL=8      150.4 us   1338.3 GB/s (86.1%)    2048 programs
  v2 UNROLL=16     155.3 us   1296.7 GB/s (83.4%)    1024 programs   ← 开始变差
```

思路是对的：`tl.static_range` 编译期展开，多个 tile 的访存能在指令级重叠，
提高 memory level parallelism。但 v0 本来就已经 85.7%，天花板只剩 14% ——
**优化空间不够，收益就只能是噪声级别。**

UNROLL=16 开始掉，是因为 program 数降到 1024，而 A100 有 108 个 SM，
每个 SM 能同时驻留多个 block —— program 太少就填不满机器了。
**这是 UNROLL 的代价：它用「更少的并行度」换「更多的 ILP」，过头就亏。**

### 3.4 v3 autotune：和手调打平，价值在可移植性

autotune 从 7 个候选里选，结果 1342.8 GB/s，和手调的 v2（1344.6）打平。

**在这个练习里 autotune 没有带来性能收益** —— 因为所有配置本来就都在
85–86%，选哪个都一样。它的真正价值是**换卡 / 换规模时不用重新人肉扫**：

```bash
TRITON_PRINT_AUTOTUNING=1 python v3_autotune.py
```

代价是**第一次调用要把所有 config 各跑一遍**。候选 7 个还好，
matmul 那种动辄几十个候选的，第一次调用能到几十秒
（见 [`../03_matmul/README.md`](../03_matmul/README.md) §3.3）。

---

## 4. 一句话总结

| 优化手段 | 这个 kernel 上的效果 | 为什么 |
|---|---|---|
| 调 `BLOCK_SIZE` | +1.7%（噪声） | 已经打满带宽 |
| 循环展开 | +0.8%（噪声） | 同上，且过头会掉并行度 |
| autotune | ±0% | 所有候选都一样好 |
| **写坏访存模式** | **−83%** ⭐ | 向量化和合并访存全丢 |

> **memory-bound 算子的功夫全在「别写坏」，不在「怎么调优」。**
> 保证访存连续 → 拿到 85% 带宽 → 收工。想要更快只有一条路：**减少访存量**，
> 也就是算子融合，见练习 02。

---

## 5. 踩坑记录

| 坑 | 现象 | 正确做法 |
|---|---|---|
| `BLOCK_SIZE` 开到 65536 以上 | **编译卡死**（不是报错，是几分钟不返回） | `BLOCK_SIZE` 是一个 tile 的元素数，整个要进寄存器；上限几千 |
| 拿 v1 当「向量化开关」 | 归因错误 | 它同时破坏了合并访存，数字是两者叠加 |
| 不 warmup 就测 autotune | 测到的是 tuning 时间（几百 ms） | `_shared.bench()` 里有 25 次 warmup |
| `BLOCK_SIZE` 非 2 的幂 | 编译报错 | `triton.next_power_of_2()` |
| 忘了 `mask=` | 静默越界，`n=1000` 时读到别人的显存 | 一律写 mask；用 `compute-sanitizer` 验 |

---

## 6. 对照阅读

- [`../../cuda/01_vector_add/`](../../cuda/01_vector_add/README.md) —— 同一个算子的 CUDA 版，
  那边 `float4` 要手写、grid-stride 要手写，可以直接对比两种抽象的代价
- [`../02_fused_softmax/`](../02_fused_softmax/README.md) —— memory-bound 算子真正的优化方向：融合
- [`../../综合练习/vector_mul2/`](../../综合练习/vector_mul2/README.md) —— 同类算子的完整 ncu / nsys profile 流程

---

## 7. 我的笔记

<!-- 复现时写在这里 -->

- [ ] `bench.py` 的数字和 §3 对得上吗？（卡不同、其他人在跑任务都会影响）
- [ ] v0 第 1 题：去掉 mask 用 n=1000 跑，`compute-sanitizer` 报什么？
- [ ] v1 第 1 题：用 ncu 比 v0/v1 的 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`，
      比例是不是接近 4:16？
- [ ] v2 第 2 题：`tl.static_range` 换成 `range`，性能掉多少？
