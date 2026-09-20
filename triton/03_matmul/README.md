# 03 matmul —— 分块矩阵乘，以及「打不过 cuBLAS 之后干什么」

`C = A @ B`，fp16 输入 / fp32 累加 / fp16 输出。这是 Triton 练习里第一个
**compute-bound** 的 kernel：前两个练习的天花板是显存带宽，这个练习的天花板是
Tensor Core 算力。优化思路完全不同。

测试环境：A100-SXM4-40GB（108 SM，40 MB L2，fp16 dense Tensor Core 峰值
312 TFLOP/s），CUDA_VISIBLE_DEVICES=7，Triton 3.x。

---

## 1. 变体一览

| 文件 | 做了什么 | 4096² 实测 | 占 cuBLAS |
|------|---------|-----------|----------|
| `v0_naive.py` | row-major pid 映射，分块 64×64×32 | 112.6 TFLOP/s | 55% |
| `v1_tiled.py` | 分块调到 128×256×64 + group-major 排序 | 181.1 TFLOP/s | 88% |
| `v2_autotune.py` | `@triton.autotune` 扫 12 个候选 | 186.8 TFLOP/s | 91% |
| `v3_fused_relu.py` | 融合 epilogue：`relu(A@B + bias)` | —— | 见 §3.5 |

`_shared.py` 放公共的测试用例、`atol_for(K)`、计时和报告函数。
`bench.py` 把所有变体和所有参数扫描串起来跑。

---

## 2. 使用方法

```bash
# 单个变体（自带正确性 + 单点性能）
CUDA_VISIBLE_DEVICES=7 python v0_naive.py
CUDA_VISIBLE_DEVICES=7 python v1_tiled.py
CUDA_VISIBLE_DEVICES=7 python v2_autotune.py
CUDA_VISIBLE_DEVICES=7 python v3_fused_relu.py

# 全部对比 + 参数扫描
CUDA_VISIBLE_DEVICES=7 python -u bench.py
CUDA_VISIBLE_DEVICES=7 python -u bench.py --quick   # 跳过扫描
CUDA_VISIBLE_DEVICES=7 python -u bench.py --big     # 额外跑 8192²（GROUP_M 要这个尺寸才看得出）

# 看 autotune 每个尺寸选了什么
CUDA_VISIBLE_DEVICES=7 TRITON_PRINT_AUTOTUNING=1 python v2_autotune.py

# 看冷编译缓存的代价
CUDA_VISIBLE_DEVICES=7 TRITON_CACHE_DIR=/tmp/cold python v2_autotune.py
```

> `python -u`：不加的话 stdout 会被缓冲，扫描跑十几秒看不到任何输出，
> 容易误以为卡死。这个坑在 `../01_vector_add/README.md` 里也记了一次。

---

## 3. 实测结果

### 3.1 先看整体：没有一个尺寸上都最优的配置

| 尺寸 | cuBLAS | v0 (64×64×32) | v1 (128×256×64) | v2 (autotune) |
|------|--------|---------------|-----------------|---------------|
| 512² | 22.6 | 7.8 (35%) | 7.7 (34%) | 8.7 (39%) |
| 1024² | 131.2 | **93.5 (71%)** | 58.9 (45%) | 70.2 (54%) |
| 2048² | 217.9 | 132.4 (61%) | 106.2 (49%) | **173.5 (80%)** |
| 4096² | 205.3 | 112.6 (55%) | 181.1 (88%) | **186.8 (91%)** |

单位 TFLOP/s，括号里是占 cuBLAS 的比例。

**v1「优化版」在 1024² 和 2048² 上反而比 v0 慢**，这是本练习最该记住的一条。
原因是并行度：128×256 的分块在 1024² 上只切出 `8 × 4 = 32` 个 program，
而 A100 有 108 个 SM —— 七成的 SM 全程空转。v0 的 64×64 切出
`16 × 16 = 256` 个，反而喂得饱。到 4096² 时 v1 能切出 `32 × 16 = 512` 个，
并行度不再是瓶颈，大分块的算术强度优势才显现出来。

> 教训：**「调优后的分块」是对某一个尺寸调优的**，不是普遍更好。
> 这正是 cuBLAS 要为不同尺寸段准备不同 kernel 的原因，也是 autotune
> 在这个练习里真正有价值的原因（对比 `../01_vector_add/`，那里
> autotune 白跑，因为所有配置都一样好）。

### 3.2 分块尺寸：4096² 上 1.47 倍的差距

```
BM=64  BN=64  BK=32  w4 s5    1077.6 us   127.5 TFLOP/s   40.9%
BM=64  BN=128 BK=32  w4 s4     873.2 us   157.4 TFLOP/s   50.4%
BM=128 BN=64  BK=32  w4 s4     895.2 us   153.5 TFLOP/s   49.2%
BM=128 BN=128 BK=32  w4 s4     746.9 us   184.0 TFLOP/s   59.0%
BM=128 BN=128 BK=64  w4 s4     852.3 us   161.3 TFLOP/s   51.7%   ← 变大反而慢
BM=128 BN=256 BK=64  w8 s3     744.4 us   184.6 TFLOP/s   59.2%
BM=256 BN=128 BK=64  w8 s3     733.9 us   187.3 TFLOP/s   60.0%   ← 最快
```

最好 / 最差 = **1.47×**。对比 `../01_vector_add/`：那个 memory-bound kernel
上 BLOCK_SIZE 从 128 扫到 16384 的差距 **不到 2%**。这是 compute-bound 和
memory-bound 最直观的分野 —— **compute-bound kernel 里分块尺寸是第一优化项，
memory-bound kernel 里它几乎无所谓**。

注意 `128×128×32 → 128×128×64` 是**变慢**的（184.0 → 161.3）：BK 翻倍
让每个 stage 的 shared memory 需求翻倍，而 `num_warps` 还是 4，搬进来的数据
没有足够的 warp 去消化。分块参数不是单调的，四个维度（BM/BN/BK/warps）要一起调
—— 这也是人肉调参很快就调不动、需要 autotune 的原因。

### 3.3 num_stages：软件流水的深度

```
num_stages=2   902.6 us   152.3 TFLOP/s
num_stages=3   748.0 us   183.8 TFLOP/s     ← +21%
num_stages=4   744.7 us   184.6 TFLOP/s     ← 饱和
num_stages=5   失败：OutOfResources
               Required: 196608, Hardware limit: 166912
```

`num_stages` 是 Triton 自动做的 software pipelining 深度：在算第 k 块的时候
预取第 k+1、k+2…块，用 shared memory 换访存延迟的隐藏。2 → 3 有 21% 的提升，
3 → 4 就饱和了（延迟已经藏住，再深没意义）。

到 5 直接编译失败，错误信息给得很清楚：需要 192 KB shared memory，A100 每个
SM 只有 164 KB 可用。**这是一个硬约束，不是性能问题** ——
`BLOCK_M × BLOCK_K + BLOCK_K × BLOCK_N` 乘以 `num_stages` 乘以 2 字节
就是需求量，可以直接算出来：`(128×64 + 64×256) × 5 × 2 = 196608`，
和报错数字完全吻合。

### 3.4 GROUP_M：一个几乎为零的优化（本练习最有教育意义的负结果）

group-major 排序是 Triton 官方 matmul 教程着重讲的技巧：改变 program 走查 C
的顺序，让同时在跑的 program 复用相同的 A 行块和 B 列块，提高 L2 命中率。
理论很漂亮。实测：

```
@ 4096²（B 矩阵 32 MB）
  GROUP_M=1    744.0 us   184.7 TFLOP/s    ← row-major，等于不优化
  GROUP_M=2    746.5 us   184.1 TFLOP/s
  GROUP_M=4    746.7 us   184.1 TFLOP/s
  GROUP_M=8    748.2 us   183.7 TFLOP/s    ← v1 的默认值，比 GROUP_M=1 还慢一点
  GROUP_M=16   737.7 us   186.3 TFLOP/s
```

**完全没有效果**，GROUP_M=1 和 GROUP_M=8 的差别（0.6%）在测量噪声里。

归因很直接：A100 的 L2 有 **40 MB**，而 4096×4096 的 fp16 B 矩阵只有
**32 MB —— 整个装得进 L2**。既然 B 全在 L2 里，program 按什么顺序访问它都是
L2 命中，优化访问顺序自然没有收益。

把矩阵放大到 L2 装不下再看：

```
@ 8192²（B 矩阵 128 MB，远超 40 MB L2）
  GROUP_M=1   5803.3 us   189.5 TFLOP/s
  GROUP_M=2   5802.6 us   189.5 TFLOP/s
  GROUP_M=4   5723.3 us   192.1 TFLOP/s
  GROUP_M=8   5643.1 us   194.8 TFLOP/s    ← +2.8%
  GROUP_M=16  5642.7 us   194.9 TFLOP/s
```

这才出现了单调的、可复现的 **+2.8%**。方向对，但量级远小于教程给人的印象。

> **这一条是全套练习里最重要的方法论**：一个优化「原理上成立」不代表
> 「在你的硬件上、你的规模下有收益」。L2 容量、矩阵大小、访问模式三者
> 一比，就能提前判断出 4096² 上这个优化必然无效 —— 教程里没写这个前提，
> 是因为它成书时的卡 L2 小得多（V100 只有 6 MB）。
>
> 判断方法：**先算工作集和 cache 容量的比值，再决定值不值得做 cache 优化。**

### 3.5 epilogue 融合：Triton 写 matmul 的真正理由

裸 matmul 打不过 cuBLAS（最好 91%），那为什么还要写 Triton matmul？
因为 **cuBLAS 只会给你一个 `C = A@B`**，后面接的 bias、激活、量化，
每一个都是独立 kernel，每一个都要把整个 C 完整读一遍写一遍。

`v3_fused_relu.py` 在 accumulator 还在寄存器里的时候顺手做完 bias 和 relu。

#### ⚠️ 先说一个测量陷阱（我第一版就踩了）

第一版我拿「triton 融合版」直接和「`torch.matmul` + bias + relu」比，
测出来融合版在 512² 上只有 **0.54×** —— 融合居然让性能掉了一半。
结论当然是错的：这个对比里换掉了**两样东西**，epilogue 融合的收益，
**加上**我的 matmul 打不过 cuBLAS 的亏损。小矩阵上后者远大于前者。

要隔离 epilogue 的贡献，必须**固定 matmul 实现不变**：

- A. `triton matmul` + torch bias + torch relu ← 同一个 matmul，不融合
- B. `triton matmul` 融合 bias + relu ← 同一个 matmul，融合

这才是合法的 A/B。`v3_fused_relu.py` 里两个基线都留着，
`torch_unfused()` 标注了「仅供参考」，`triton_unfused()` 才是对照组。

#### 方阵：融合收益随尺寸下降

| 尺寸 | torch 3 kernels<br>(cuBLAS，参考) | triton 不融合 | triton 融合 | **epilogue 收益** |
|------|------|------|------|------|
| 512² | 19.5 us | 35.4 us | 25.5 us | **1.39×** |
| 1024² | 29.8 us | 50.8 us | 38.6 us | **1.32×** |
| 2048² | 104.4 us | 171.1 us | 141.5 us | **1.21×** |
| 4096² | 776.6 us | 871.2 us | 761.1 us | **1.14×** |

融合稳定有收益，且**矩阵越大收益越小**。算一下就知道为什么：
epilogue 省掉的访存是 `4MN` 字节（add 读写 C 一遍、relu 读写 C 一遍），
而 matmul 的计算量是 `2MNK`。方阵下 K 跟着长，计算量按 N³ 涨、
省掉的访存按 N² 涨，占比自然下降。

**两端的收益来源还不一样**，值得分开算：

- 4096²：C = 32 MB，额外访存 = 4 × 32 MB = **128 MB**，按 1555 GB/s 算
  ≈ **82 us**。实测差值 110 us（871.2 − 761.1）—— 同一量级，
  torch 的 elementwise kernel 达不到峰值带宽，对得上。**这里是访存主导。**
- 512²：C = 0.5 MB，额外访存 4 MN = 1 MB，按峰值算只要 **0.7 us**。
  但实测差值是 **9.9 us**，差 14 倍。所以小矩阵上的 1.39×
  **根本不是省访存省出来的，是省掉了 2 次 kernel launch + torch 的
  dispatch 开销**。

> 别把「融合有收益」一律归因成「省了访存」。小 kernel 上 launch 开销
> 经常是大头，两者的量级估算方法完全不同。

#### 扁矩阵：融合真正的主场

按上面的公式，K 越小 epilogue 占比越高。固定 M=N=4096 扫 K：

| K | 不融合 | 融合 | 收益 |
|---|--------|------|------|
| 4096 | 860.7 us | 747.7 us | 1.15× |
| 1024 | 335.0 us | 208.8 us | **1.60×** |
| 256 | 206.7 us | 77.7 us | **2.66×** |
| 64 | 169.9 us | 45.8 us | **3.71×** |

K=64 时融合快 **3.7 倍**。这时 `2MNK` 的计算量已经很小，整个 kernel 变成
memory-bound，省掉的 128 MB 访存就是几乎全部的耗时。

这个形状在推理里极其常见（LoRA 的低秩分支、attention 的 output proj、
分类头），也正是 FlashAttention、量化 GEMM、LoRA 融合这些工作的共同套路 ——
和练习 02 的 softmax 融合是同一个道理，只是融的位置从「整个算子」
换成了「matmul 的尾巴」。

### 3.6 autotune 选了什么，以及它的两种开销

autotune 对每个 `(M, N, K)` 分别 tune 并缓存：

| (M, N, K) | 选中的配置 |
|-----------|-----------|
| 256³ | 32×64×32, w2, s5 |
| 512³ | 64×64×32, w4, s5 |
| 1024³ | 64×64×32, w4, s5 |
| 2048³ | 64×128×32, w4, s4 |
| 4096³ | **256×128×64, w8, s3** |

最后一行和 §3.2 手动扫出来的最优配置（187.3 TFLOP/s）**完全一致** ——
autotune 独立复现了人肉调参的结果。小尺寸上它选小分块，也印证了 §3.1
「大分块在小矩阵上填不满 SM」的分析。

**开销一：首次调用**（要把 12 个候选各跑一遍）

- 编译缓存冷（`TRITON_CACHE_DIR` 指向空目录）：**8.0 s**
- 编译缓存冷（刚改过这个文件，缓存被 invalidate）：**8.9 s**
- 编译缓存热（连跑 3 次，源码没动）：**2.1 / 1.9 / 2.1 s**

这里有**两层缓存**，很容易搞混：

1. **编译产物缓存** —— 落在磁盘上（`TRITON_CACHE_DIR`），跨进程有效。
   冷的时候 12 个 kernel 从头编译，这是 8 s 里的大头。
2. **autotune 的选择缓存** —— 只存在 `kernel.cache` 这个 python dict 里，
   **进程一退就没了**。所以即使编译缓存是热的，每次起进程还是要把 12 个候选
   各跑一遍才知道谁最快 —— 那 2.1 s 就是 benchmark 本身。

注意第二行：**改一下源码，编译缓存就失效，又回到 8 s**。
迭代开发的时候实际一直在付这个钱，不是「第一次 8 秒之后就好了」。

**开销二：每次调用的 host 侧派发**

```
512²:  v1 写死 64×64×32 w4 s5    21.5 us
       v2 autotune（同一配置）     29.1 us    +7.6 us
```

两者在 GPU 上跑的是**同一个 kernel**（配置是从 autotune 自己的 cache 里读出来
再喂给 v1 的），所以 7.6 us 全部是 python 侧的取 key、查 cache 开销。

这个数字顺手解释了 §3.1 表里一个看起来矛盾的地方：**1024² 上 v2 选的配置和
v0 一模一样（64×64×32），为什么 v2 是 30.6 us 而 v0 只有 23.0 us？**
`23.0 + 7.6 = 30.6` —— 差的正好就是 autotune 的固定开销。

> 所以 autotune 不是无代价的。kernel 本身只有几十微秒时，7.6 us 的固定开销
> 就吃掉 25% 了。生产里的做法是 tune 完之后把选中的配置**写死**进代码。

---

## 4. 一句话总结

| 优化 | 收益 | 前提 |
|------|------|------|
| 调分块尺寸 | 4096² 上 **1.47×** | compute-bound 才值得；参数对尺寸敏感 |
| `num_stages` 2→3 | **+21%** | 再深就饱和，且受 shared memory 硬限制 |
| group-major (`GROUP_M`) | 4096² **0%** / 8192² **+2.8%** | 工作集必须装不进 L2 |
| autotune | 最优配置，但每次调用 **+7.6 us** | 候选里得有好配置；小 kernel 上开销显眼 |
| **融合 epilogue** | 方阵 **1.14~1.39×**，扁矩阵 **最高 3.71×** | K 越小越值；这才是写 Triton matmul 的理由 |

大方向：**裸 matmul 别想赢 cuBLAS（最好 91%），赢的地方在融合。**

---

## 5. 踩坑记录

1. **拿 cuBLAS 当融合的对照组** —— §3.5 详述。测「优化 X 的收益」时，
   对照组必须只差 X 这一件事。我第一版同时换了 matmul 实现和 epilogue，
   测出「融合让性能掉一半」的荒谬结论，而且**它和我自己写在同一个文件里的
   预测正好相反**，才发现搞错了。
   → 自己的测量结果和自己的理论预测冲突时，先怀疑测量设计。

2. **`num_stages=5` 直接编译失败**，`OutOfResources: Required: 196608,
   Hardware limit: 166912`。不是 bug，是 shared memory 真的不够。
   扫参数的代码要用 try/except 包住，否则整个扫描挂在中间。
   需求量可以手算：`(BM×BK + BK×BN) × num_stages × 2 字节`。

3. **误以为 autotune 首次开销是 20 s**。上一轮我随手记了这个数，这次实测
   热缓存 2.1 s、冷缓存 8.0 s。20 s 大概是某次冷缓存 + 机器忙时测到的。
   而且中间还差点又记错一次：改完文件后第一次跑是 8.9 s，看着像「热缓存也要
   8 秒」，连跑三次才确认改源码会 invalidate 编译缓存（见 §3.6）。
   → 性能数字必须带测量条件，而且**单次测量不算测量** ——
     一个数字重复不出来，就说明你还没搞清它依赖什么。

4. **`% M` / `% N` 回绕只能用在 load，不能用在 store**。
   读越界回绕只是读到无关数据（反正会被 mask 掉或乘 0），
   写越界回绕会**覆盖别人算好的结果**。store 必须用真正的边界 mask。
   拿 `M=300` 这种非整除尺寸跑才测得出来 —— 所以 `_shared.py` 的
   `CASES` 里专门放了 `(300, 500, 177)`。

5. **fp16 的容差要随 K 放大**。累加误差大致按 `sqrt(K) × eps` 涨，
   固定 `atol` 在 K=1024 时必然误报。`_shared.py` 里用
   `atol_for(K) = sqrt(K) × 1e-2`。

6. **autotune 的 cache key 包含 dtype**，实际是
   `(M, N, K, 'torch.float16', 'torch.float16', 'torch.float16')` 六元组，
   不是 `key=["M","N","K"]` 字面上的三元组。想从 cache 里捞配置得注意这点
   （`bench.py` 的 `autotune_overhead()` 用 `k[:3]` 匹配）。

---

## 6. 和其他练习的对照

- `../01_vector_add/` —— memory-bound 的极端：分块尺寸无所谓（<2%），
  访存模式决定一切（5.8× 差距）。和本练习正好相反。
- `../02_fused_softmax/` —— 同样是融合，但融的是整个算子。
  那里的天花板是带宽，融合降低访存总量；这里的天花板是算力，
  融合消除的是纯浪费的访存。
- `../../cuda/03_matmul/` —— 同一个问题用 CUDA 手写：shared memory
  分块、register 分块、`float4` 向量化，能看到 Triton 替你做了什么。

## 7. 自己的笔记

<!-- 复现时把自己机器上的数据和观察记在这里 -->
