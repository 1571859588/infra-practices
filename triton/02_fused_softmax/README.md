# 02_fused_softmax —— 融合 Softmax

行方向的 `softmax(x)`。**本目录最有价值的一个练习** —— 它回答了
「为什么要写 Triton」这个问题。

练习 01 的结论是 elementwise 算子没有优化空间。那 Triton 到底赢在哪？
答案就在这里：**赢在能把多个算子融成一个，把访存量直接砍掉。**

---

## 1. 变体一览

| 文件 | 做了什么 | 访存量 | 实测带宽 | vs torch |
|---|---|---|---|---|
| `v0_unfused_torch.py` | 纯 torch 手工拼 4 个算子，**不含 Triton** | ~8MN | 290.8 GB/s (18.7%) | 26% |
| `v1_fused.py` | 一 program 一行，整行进寄存器 | 2MN | **1318.9 GB/s (84.8%)** | **119%** |
| `v2_persistent.py` | 固定 program 数 + 行循环（**负收益**） | 2MN | 1209.8 GB/s (77.8%) | 109% |
| `v3_online.py` | 分块规约，行宽无上限 | 3MN | 921.9 GB/s (59.3%) | 83% |
| `bench.py` | 全变体对比 + 4 组参数扫描 + 能力边界 | — | — | — |
| `_shared.py` | 共用用例 / 计时口径 | — | — | — |

基线：`torch.softmax`（自己也是融合实现）= 121.0 us / 1109.3 GB/s。
`shape = (4096, 4096)` fp32，A100-SXM4-40GB（GPU 7），HBM2e 峰值 1555 GB/s。

> **带宽口径**：所有变体都按**理想访存量** `2*M*N*4`（读一次 + 写一次）折算，
> 不是各自实际搬的字节数。这样融合版的数字≈真实带宽利用率，未融合版会
> 显得远低于峰值 —— 因为它实际搬了 4 倍的数据。**按各自实际访存量算的话，
> 未融合版也能显得「带宽利用率很高」，问题就被掩盖了。**
> 这是 benchmark 口径最容易骗人的地方。

---

## 2. 用法

```bash
export CUDA_VISIBLE_DEVICES=7
PY=/mnt/public/nyt1/docqa/restored_envs/cpp/bin/python

$PY bench.py                # 推荐：全变体 + 扫描（约 3 分钟）
$PY bench.py --quick        # 只要变体对比

$PY v0_unfused_torch.py     # 单跑某个变体
$PY v1_fused.py             # 额外打印 num_warps 对寄存器的影响
$PY v3_online.py            # 额外演示 n_cols=262144
```

---

## 3. 原理与实测结论

### 3.1 ★ 融合 vs 未融合：4.54 倍

**全目录最重要的一个数字。**

```
  v0 torch 手工拼（未融合）    461.5 us    290.8 GB/s  ( 18.7% of peak)
  v1 融合版                   101.8 us   1318.9 GB/s  ( 84.8% of peak)   ← 4.54x
```

v0 的代码是：

```python
z = x - x.max(dim=1, keepdim=True)[0]     # kernel 1(max) + kernel 2(sub)
numerator = torch.exp(z)                   # kernel 3
denominator = numerator.sum(dim=1, ...)    # kernel 4
return numerator / denominator             # kernel 5
```

**每个 torch 算子都是一个独立的 CUDA kernel，kernel 之间只能通过显存传数据。**
所以每一步都要把整个 4096×4096 矩阵读一遍、写一遍：

```
读 x            (MN)
写 z = x - max  (MN)   ← 中间结果落显存
读 z            (MN)
写 num = exp(z) (MN)   ← 中间结果落显存
读 num          (MN)
写 out          (MN)
外加 max / sum 两次规约各读一遍
                       ≈ 8MN
```

v1 把整行读进寄存器，减最大值、exp、求和、除法全在片上完成，
**中间结果一次都不写回显存**，总访存量 2MN。

> ### 这才是写 Triton 的理由
>
> - 单算子（练习 01）：Triton 只能**追平** torch，因为大家都打满了带宽
> - 多算子融合：Triton 能快 **4 倍以上**，因为它减少的是访存**量**
>
> 挑 Triton 改写目标时，优先找「连续好几个 elementwise / reduction 算子」
> 的地方。**不要去重写 torch 已有的单个算子** —— 见 §3.2。

### 3.2 别重写 torch 已经融合好的算子

```
  torch.softmax (已融合)     121.0 us   1109.3 GB/s  ( 71.3%)
  v1 融合版                  101.8 us   1318.9 GB/s  ( 84.8%)   119% of torch
```

`torch.softmax` 自己就是融合实现，所以 v1 只比它快 19%（而不是 4 倍）。
这 19% 可能来自 `torch.softmax` 要处理任意 dim / 任意 dtype / 半精度累加等
通用情况，而 v1 只管「2D、fp32、dim=1」这一种。

**结论：torch 有的算子别去重写，要重写的是 torch 没有的融合组合。**

### 3.3 v2 persistent：负收益，而且扫描直接指出了原因 ⭐

persistent kernel 是 Triton 官方 tutorial 02 强调的写法：program 数固定成
「刚好填满 GPU」，每个 program 用 `tl.range` 循环处理多行，省掉 program
反复创建/退出的开销，还能做 software pipelining。

**实测在 A100 上是负收益，每个配置都比 v1 慢：**

```
  v1 融合版          101.8 us   1318.9 GB/s  ( 84.8%)     4096 programs
  v2 persistent      110.9 us   1209.8 GB/s  ( 77.8%)      432 programs
```

occupancy 扫描把原因说得很清楚：

```
  v2 occupancy=1     113.8 us   1179.9 GB/s  ( 75.9%)      108 programs
  v2 occupancy=2     112.3 us   1195.5 GB/s  ( 76.9%)      216 programs
  v2 occupancy=4     110.9 us   1210.2 GB/s  ( 77.8%)      432 programs
  v2 occupancy=8     106.6 us   1259.2 GB/s  ( 81.0%)      864 programs
  v2 occupancy=16    104.4 us   1285.1 GB/s  ( 82.6%)     1728 programs
  v1（等价于 program 数 = 4096）                            84.8%
```

**program 数越多越快，单调趋向 v1。** 这就是判决书：persistent 化在这个
kernel 上唯一的净效果是**降低了并行度**，性能随着你把并行度还回去而恢复。
省下的调度开销根本补不上损失。

`num_stages`（pipelining 深度）也没用：

```
  v2 num_stages=1    112.4 us  ( 76.8%)
  v2 num_stages=2    113.5 us  ( 76.0%)
  v2 num_stages=4    110.7 us  ( 78.0%)
  v2 num_stages=8    110.9 us  ( 77.8%)
```

**为什么官方教程的技巧在这里不灵？**

persistent 化解决的是「program 创建/退出开销占比高」的问题。而这个 softmax
kernel 里，每个 program 要搬 4096×4 = 16 KB 数据、做两次 4096 元素的规约 ——
**单个 program 的工作量足够大，launch 开销本来就可以忽略。** GPU 的硬件
block 调度器非常便宜，4096 个 block 排队上 108 个 SM 这件事几乎不要钱。

小规模时也一样（persistent 本该在这里更有优势，因为并行度损失最小）：

```
  M=128    v1  22.9 us      v2  26.8 us   (86% of v1)
  M=1024   v1  22.7 us      v2  26.1 us   (87% of v1)
  M=8192   v1  52.7 us      v2  56.7 us   (93% of v1)
  M=65536  v1 392.2 us      v2 414.2 us   (95% of v1)
```

M=128 时 v1 只有 22.9 us / 2.9% 带宽 —— 这时候确实是 launch-bound，
但 persistent **也没能改善**，说明瓶颈不是「program 太多」，
而是「总工作量太小，GPU 根本没热起来」。

> **教训和 `../03_matmul/` 的 GROUP_M 一样：**
> 官方教程里的优化技巧是针对**特定 kernel 特征**的，不是万能招式。
> 照搬之前先问一句「我这个 kernel 真的有这个瓶颈吗」，然后实测。
> 留着这个变体不是因为它快，是因为**它的扫描曲线教会了怎么给一个
> 优化手段做归因**。

### 3.4 v3 online：用 1.5 倍访存换掉行宽限制

v1 有个硬前提：`BLOCK_SIZE >= n_cols`，**一整行必须装进寄存器**。
`n_cols = 1<<18` 时 BLOCK_SIZE 就是 262144，一个 tile 要 1 MB 寄存器，
编译器直接卡死（不是报错，是几分钟不返回）。

online softmax 把行切成固定大小的块，只维护两个标量状态：

```
m = -inf, d = 0
for each block:
    m_new = max(m, max(block))
    d     = d * exp(m - m_new) + sum(exp(block - m_new))   # ★ 修正项
    m     = m_new
```

那个 `exp(m - m_new)` **修正因子**是全部的关键：最大值一变，之前累加的 d
是按旧基准归一化的，要整体乘上 `exp(m_old - m_new)` 换算过来。
**这一步就是 FlashAttention 里 rescale 的来源。**

代价是**要读两遍 x**：第一遍算 (m, d)，第二遍才能算 `exp(x-m)/d` 写出去。
访存量 3MN 而不是 2MN，理论上限只有 v1 的 2/3。实测吻合得很好：

```
  v1 融合版      1318.9 GB/s
  v3 online       921.9 GB/s      = 70% of v1（理论预期 67%）
```

block_size 扫描有个有意思的现象：

```
  v3 block_size=256     154.8 us   867.1 GB/s  ( 55.8%)
  v3 block_size=1024    145.8 us   920.8 GB/s  ( 59.2%)
  v3 block_size=4096    111.7 us  1201.5 GB/s  ( 77.3%)   ← 接近 v1
```

`block_size=4096` 时 n_cols 恰好一块装完，两遍循环各只有一次迭代，
**第二遍的读命中 L2**（4096×4 = 16 KB，刚读过），所以并没有付满 1.5 倍的
DRAM 代价。这说明「读两遍」的真实成本取决于**两遍之间的数据还在不在 cache 里**，
不能简单按字节数估。

能力边界（v1 在这个行宽上编译不出来）：

```
  [PASS] v3 n_cols=262144: max_abs_err = 2.32831e-10
  v3 n_cols=262144    504.8 us    265.9 GB/s  ( 17.1% of peak)
```

带宽只有 17%：行宽 1 MB，两遍之间早被挤出 L2，真的读了两遍 DRAM，
而且只有 64 行 → 只有 64 个 program，填不满 108 个 SM。
**但它能跑出正确结果，而 v1 根本跑不了 —— 这时候「能跑」比「跑得快」重要。**

### 3.5 行宽 N 的影响（v1）

```
  v1 N=256      22.1 us    379.8 GB/s  ( 24.4%)  BLOCK=256,   warps=4
  v1 N=1024     23.1 us   1455.1 GB/s  ( 93.6%)  BLOCK=1024,  warps=4    ← 全场最高
  v1 N=4096    101.9 us   1317.2 GB/s  ( 84.7%)  BLOCK=4096,  warps=16
  v1 N=16384   396.5 us   1353.9 GB/s  ( 87.1%)  BLOCK=16384, warps=16
```

- **N=256 只有 24%**：总数据量才 4096×256×4×2 = 8 MiB，GPU 还没忙起来就结束了。
  瓶颈是 launch 开销和 program 工作量不足。**规模太小时，
  「带宽利用率」这个指标本身没有意义** —— 别对着它调优。
- **N=1024 拿到 93.6%**：一行刚好装满寄存器（`num_warps=4` 就够），
  规模又足够大。这是这个写法的甜点。
- N 再大要靠 `num_warps=16` 撑，寄存器压力上升，带宽略降。

---

## 4. 一句话总结

| 优化手段 | 效果 | 为什么 |
|---|---|---|
| **算子融合**（v0→v1） | **+354%** ⭐ | 访存量从 8MN 降到 2MN |
| 替换 torch 已融合的算子 | +19% | 只是省掉了通用性开销 |
| persistent kernel | **−8%** | 这个 kernel 没有 launch 瓶颈，只是丢了并行度 |
| online 分块（v3） | −30%，但解锁任意行宽 | 3MN vs 2MN 的必然代价 |

> **memory-bound 算子唯一真正有效的优化是「减少访存量」。**
> 调块大小、调并行度、调 pipelining 都是在 85% 那条线上下抖动；
> 融合是唯一能把线本身往上抬的手段。

---

## 5. 踩坑记录

| 坑 | 现象 | 正确做法 |
|---|---|---|
| `other=0` 而不是 `-inf` | 全负数的行算出 max=0，结果静默错误 | 求 max 一律 `other=-float("inf")` |
| 忘记减最大值 | 大输入 exp 溢出成 inf，再相除得 nan | 先 `- tl.max(...)`；用 `x*100` 的用例守住 |
| `n_cols` 很大时直接用 v1 | **编译卡死**，不报错 | `BLOCK_SIZE` 上限几千；超了换 v3 |
| 按各变体实际访存量算带宽 | 未融合版也显得「利用率高」，掩盖问题 | 统一按理想访存量折算（见 `_shared.py` 注释） |
| 循环携带的标量用 python float | 循环内外类型不一致，编译报错 | 用 `tl.zeros([1], dtype=tl.float32)` |
| 拿 `num_stages` 当万能加速 | 这里 1 和 8 没区别 | 只有循环体真的在等访存时才有用 |
| 在 N=256 这种规模上调优 | 优化半天带宽还是 24% | 先确认规模够大，再看带宽指标 |

---

## 6. 对照阅读

- [`../../cuda/02_reduction/`](../../cuda/02_reduction/README.md) —— `tl.max` / `tl.sum`
  这一行背后，CUDA 里要手写的 shared memory + warp shuffle 规约，共 5 个版本
- [`../01_vector_add/`](../01_vector_add/README.md) —— 为什么单算子没有优化空间
- [`../03_matmul/`](../03_matmul/README.md) —— compute-bound 的另一套优化逻辑

---

## 7. 我的笔记

<!-- 复现时写在这里 -->

- [ ] `bench.py` 的 4.54x 能复现吗？
- [ ] v0 第 1 题：用 nsys 数一下 `naive_softmax` 到底 launch 了几个 kernel
- [ ] v1 第 1 题：去掉 `- tl.max(...)`，用 `x*100` 跑，nan 从哪一步开始出现
- [ ] v1 第 2 题：`other=-inf` 改成 `other=0`，构造一个会算错的输入
- [ ] v3 第 1 题：去掉 `d * tl.exp(m - m_new)` 的修正因子，
      构造「后面的块最大值更大」的输入让它出错
- [ ] v3 第 4 题：改成 `softmax(QK^T)V`，写出 FlashAttention 前向
