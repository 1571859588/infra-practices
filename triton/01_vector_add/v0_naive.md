# `v0_naive.py` 练习题详解

对应 [`v0_naive.py`](v0_naive.py) 末尾的三道题。每道题都给出**改哪一行、怎么跑、
实测输出、为什么**，你可以照着自己复现一遍。

本文所有输出都是实测的（A100-SXM4-40GB / GPU7 空闲 / triton 3.5.0 / torch 2.9.0+cu128），
不是推演出来的。三道题各自都藏着一个比题面更值钱的结论：

| 题 | 题面答案 | 真正值钱的结论 |
|---|---|---|
| 1 | 越界读写 | **`compute-sanitizer` 默认抓不到它** —— 被 torch 的缓存分配器挡住了 |
| 2 | 要改签名 | **`x * a + y` 会被融合成一条 FMA**，于是 `torch.allclose` 默认容差会判你错 |
| 3 | 不能编译 | 约束是 **2 的幂**，不是「32 的倍数」；`96` 和 `192` 一样过不了 |

---

## 0. 准备

```bash
cd /mnt/gfs/nyt1/infra/practices/triton/01_vector_add
export CUDA_VISIBLE_DEVICES=7                                  # 先挑一张空闲卡
PY=/mnt/public/nyt1/docqa/restored_envs/cpp/bin/python
```

挑卡很重要：卡被别人占着时带宽会掉一半以上，本文的性能数字就对不上了。
`nvidia-smi` 在某些 shell 里跑不起来，用 NVML 更可靠：

```bash
$PY -c "
import pynvml as N; N.nvmlInit()
for i in range(N.nvmlDeviceGetCount()):
    h=N.nvmlDeviceGetHandleByIndex(i); u=N.nvmlDeviceGetUtilizationRates(h)
    p=N.nvmlDeviceGetComputeRunningProcesses(h)
    print(f'GPU{i} util={u.gpu:3d}% procs={[(x.pid, x.usedGpuMemory//2**20) for x in p]}')
"
```

`util=0%` 且 `procs=[]` 才算干净。

> ⚠️ **别在仓库里改 `v0_naive.py`。** 下面每道题都让你新建一个临时文件，
> 建议统一放 `/tmp/vecadd_ex/`，做完直接删。所有临时脚本开头加一行
> `sys.path.insert(0, "…/01_vector_add")` 就能复用仓库里的 `_shared.py`。

先跑一遍原版，拿到基线：

```bash
$PY v0_naive.py
```

```
[正确性]
  [PASS] n=1024    : max_abs_err = 0
  [PASS] n=1000    : max_abs_err = 0
  [PASS] n=98765   : max_abs_err = 0
  [PASS] n=1048576 : max_abs_err = 0

[性能] n = 16,777,216（每个 buffer 64 MiB）
  torch x + y                   148.7 us    1353.7 GB/s  ( 87.1% of peak)
  v0 朴素版                       148.9 us    1352.4 GB/s  ( 87.0% of peak)
```

**和 torch 打平，87% 峰值带宽。** 记住这两个数，后面全靠它们做对照。

---

## 1. 第 1 题：把 mask 去掉，用 n=1000 跑会发生什么？

> 题面提示是「加 compute-sanitizer」。**这个提示会把你带到坑里** ——
> 按最自然的方式跑，sanitizer 报 0 errors。这一题真正要学的就是这件事。

### 1.1 改哪里

`v0_naive.py:48-52` 原本是：

```python
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

去掉 mask，就是三处 `mask=mask` 全删：

```python
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(out_ptr + offsets, x + y)
```

`n=1000`、`BLOCK_SIZE=1024` → `grid = cdiv(1000, 1024) = 1` 个 program，
它会老老实实处理 `offsets = 0..1023`，也就是**越界 24 个元素 = 96 字节**。

### 1.2 步骤一：直接跑 —— 它「正确」

```bash
mkdir -p /tmp/vecadd_ex && cd /tmp/vecadd_ex
cat > ex1.py <<'PY'
import torch, triton, triton.language as tl

@triton.jit
def add_masked(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n
    tl.store(out_ptr + offs,
             tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

@triton.jit
def add_nomask(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs))   # 无 mask

n, BS = 1000, 1024
torch.manual_seed(0)
x = torch.randn(n, device="cuda"); y = torch.randn(n, device="cuda")
print(f"n={n} BLOCK_SIZE={BS} → grid={triton.cdiv(n,BS)}，越界 {BS-n} 个元素 = {(BS-n)*4} 字节")
for name, k in (("有 mask", add_masked), ("无 mask", add_nomask)):
    out = torch.empty_like(x)
    k[(triton.cdiv(n,BS),)](x, y, out, n, BLOCK_SIZE=BS)
    print(f"  {name}: 和参考值完全相同 = {torch.equal(out, x+y)}")
PY
$PY ex1.py
```

实测：

```
n=1000 BLOCK_SIZE=1024 → grid=1，越界 24 个元素 = 96 字节
  有 mask: 和参考值完全相同 = True
  无 mask: 和参考值完全相同 = True     ← 没报错，结果还是对的
```

**没有任何报错，`out` 的 1000 个元素完全正确。** 这就是最危险的形态：
`_shared.py` 的 `selftest()` 只比对 `out` 和 `x+y`，所以**自测会全部 PASS**，
你根本不会发现自己写了越界代码。

### 1.3 步骤二：让损坏变成看得见的

上面之所以没事，是因为越界写的 96 字节正好落在 torch 给这个张量留的 padding 里。
想看到真实后果，就让「别人的数据」紧挨在后面 —— 用同一块 storage 的两个 view：

```bash
cat > ex1b.py <<'PY'
import torch, triton, triton.language as tl

@triton.jit
def add_nomask(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs))

n, BS = 1000, 1024
torch.manual_seed(0)
x = torch.randn(n, device="cuda"); y = torch.randn(n, device="cuda")

buf = torch.full((1024,), 7.0, device="cuda")     # 前 1000 个当 out，后 24 个是"别人的数据"
out, victim = buf[:n], buf[n:]
print(f"  写之前 victim = {victim.tolist()[:6]} ...")
add_nomask[(1,)](x, y, out, n, BLOCK_SIZE=BS)
torch.cuda.synchronize()
print(f"  写之后 victim = {[round(v,4) for v in victim.tolist()[:6]]} ...")
print(f"  被踩掉的元素数 = {(victim != 7.0).sum().item()} / {victim.numel()}")
print(f"  out 自己对不对 = {torch.equal(out, x + y)}")
PY
$PY ex1b.py
```

```
  写之前 victim = [7.0, 7.0, 7.0, 7.0, 7.0, 7.0] ...
  写之后 victim = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ...
  被踩掉的元素数 = 24 / 24
  out 自己对不对 = True        ← 注意：out 是对的！
```

**24 个邻居元素被静默改写成了 0，而 `out` 自己完全正确。**
这正是这类 bug 难查的原因：报错的地方（别人的张量突然变成 0）
和出错的地方（你的 kernel 少写了一个 `mask=`）隔得很远。

> 为什么被写成 `0.0` 而不是随机值？因为越界**读** `x`/`y` 时也读到了各自 buffer
> 之外，那片内存刚好是 0，`0 + 0 = 0`。别指望这个规律 —— 越界读到什么是不确定的。

### 1.4 步骤三：`compute-sanitizer` —— 按题面提示做，它抓不到

前两步的脚本都不适合直接喂给 memcheck：`ex1.py` 跑了两个 kernel 不好归因，
而 `ex1b.py` 里 `out = buf[:1000]` 是 1024 元素 buffer 的 view ——
**写 1024 个元素对那块分配根本不越界**，结论会变味。

所以单独写一个最小脚本，`out` 就老老实实开 1000 个元素。
带一个 `mask` 参数，后面 §1.5、§1.6 都复用它：

```bash
cat > ex1c.py <<'PY'
import sys
import torch, triton, triton.language as tl

@triton.jit
def add_nomask(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs))

@triton.jit
def add_masked(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n
    tl.store(out_ptr + offs,
             tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

kernel = add_masked if "mask" in sys.argv else add_nomask    # 加 mask 参数 = 对照组
n, BS = 1000, 1024
x = torch.randn(n, device="cuda"); y = torch.randn(n, device="cuda")
out = torch.empty(n, device="cuda")
kernel[(1,)](x, y, out, n, BLOCK_SIZE=BS)
torch.cuda.synchronize()
print(f"launch 完成（{kernel.__name__}），out 正确 = {torch.equal(out, x + y)}")
PY
```

> 记住 `tl.store` 在第 **7** 行 —— 下一步 sanitizer 会把行号报出来。

按题面提示跑：

```bash
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --target-processes all \
    env CUDA_VISIBLE_DEVICES=7 $PY ex1c.py
```

```
========= COMPUTE-SANITIZER
launch 完成（add_nomask），out 正确 = True
========= ERROR SUMMARY: 0 errors
```

**0 errors。** 不是 sanitizer 不行，是**它看到的边界不是你以为的边界**：

```
你以为的分配                     实际的分配
┌──────────────┐                ┌───────────────────────────────┐
│ x  4000 B    │                │ torch 向驱动 cudaMalloc 的一整段 │
├──────────────┤                │   (通常 ≥ 2 MiB)               │
│ y  4000 B    │   ← 越界！      │  ┌────┬────┬─────┬──────────┐  │
├──────────────┤                │  │ x  │ y  │ out │  空闲    │  │
│ out 4000 B   │                │  └────┴────┴─────┴──────────┘  │
└──────────────┘                └───────────────────────────────┘
                                 越界 96 B 仍在这一段里 → 合法访问
```

torch 的 caching allocator 一次性 `cudaMalloc` 一大段，再自己切给各个张量。
memcheck 只知道驱动层面那一大段的边界，**段内怎么切它一无所知**。

### 1.5 步骤四：让 sanitizer 真的抓到

关掉缓存分配器，让每个张量走独立的 `cudaMalloc`：

```bash
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --target-processes all \
    env CUDA_VISIBLE_DEVICES=7 PYTORCH_NO_CUDA_MEMORY_CACHING=1 $PY ex1c.py 2>&1 | tee mc.log
```

```
========= Invalid __global__ read of size 16 bytes
=========     at add_nomask+0xc0 in /tmp/vecadd_ex/ex1c.py:7
=========     by thread (122,0,0) in block (0,0,0)
=========     Address 0x7757e4a00fa0 is out of bounds
=========     and is 1 bytes after the nearest allocation at 0x7757e4a00000 of size 4,000 bytes
========= Invalid __global__ read of size 16 bytes
=========     at add_nomask+0xc0 in /tmp/vecadd_ex/ex1c.py:7
=========     by thread (123,0,0) in block (0,0,0)
=========     Address 0x7757e4a00fb0 is out of bounds
=========     and is 17 bytes after the nearest allocation at 0x7757e4a00000 of size 4,000 bytes
...
========= ERROR SUMMARY: 10 errors
```

三个细节值得看：

1. **`of size 4,000 bytes`** —— 现在边界正好是 `1000 × 4`，和你脑子里的一致了。
   地址 `...fa0` = 十进制 4000，正是第 1000 个元素的位置。
2. **`in ex1c.py:7`** —— sanitizer 直接报到 **Python 源码行号**，正是那句
   `tl.store(...)`。Triton 把行号编进了 cubin，所以不用去读 PTX。
3. **`read of size 16 bytes`** —— 一次读 16 字节，说明编译器把访存**向量化**成了
   `ld.global.v4.b32`（一条指令 4 个 float）。这条信息在第 2 题会再用到。

`ERROR SUMMARY: 10 errors` 这个数字要拆开看，**别直接当成"越界了 10 次"**：

```bash
grep -oE "Invalid __global__ (read|write) of size [0-9]+ bytes" mc.log | sort | uniq -c
#       6 Invalid __global__ read of size 16 bytes
```

- **6 条**是真正的越界读，`6 × 16 B = 96 B` —— **正好等于越界的 24 个元素**，一条不多一条不少。
- 剩下 **4 条**是越界之后 CUDA context 已经坏掉，后续
  `cudaDeviceSynchronize` / `cudaGetLastError` / `cudaFree` 接连报
  `cudaErrorLaunchFailure (error 719)`，属于**同一个 bug 的余震**。

> 那 x 和 y 各越界 24 个元素，为什么只报了 96 字节而不是 192？
> 因为 `cudaMalloc` 把它们摆在哪儿不固定 —— 另一个 buffer 后面恰好是合法映射，
> 就不触发报告。**所以条数会随运行波动（实测 6~10），报不报才是关键，报几条不是。**

还有一点要有心理准备：越界会让 kernel 被强杀，所以脚本末尾的
`torch.cuda.synchronize()` 一定会抛
`torch.AcceleratorError: CUDA error: unspecified launch failure`。
**这是预期结果，不是另一个 bug** —— 它恰好也说明 §1.2 里那次"静默正确"有多侥幸。

### 1.6 步骤五：`TRITON_INTERPRET=1` 能替代吗？不能

还是用 §1.4 那个 `ex1c.py`，不加参数就是无 mask 版：

```bash
# 套一层 timeout：解释器模式极慢，而且这次注定要崩，别让它挂住终端
CUDA_VISIBLE_DEVICES=7 TRITON_INTERPRET=1 timeout 300 $PY ex1c.py
```

```
double free or corruption (out)
timeout: the monitored command dumped core
[2]    926298 IOT instruction  CUDA_VISIBLE_DEVICES=7 TRITON_INTERPRET=1 timeout 300 $PY ex1c.py
```

⚠️ **这句报错每次都不一样，别对着字面去搜。** 同一个脚本连跑 5 次的实测：

```
第 1 次: malloc(): unaligned tcache chunk detected
第 2 次: launch 完成（add_nomask），out 正确 = True     ← 先打印了正确结果！
         corrupted size vs. prev_size                    然后在退出时才崩
第 3 次: double free or corruption (out)
第 4 次: malloc(): unaligned tcache chunk detected
第 5 次: malloc(): unaligned tcache chunk detected
```

四种不同的 glibc 报错，**而且没有一种提到 Triton、kernel 或你的源码**。
最坏的是第 2 次：**它先把正确答案打印出来了**，等到进程退出、glibc 去
整理堆的时候才发现结构已经被踩坏 —— 崩溃点离出错点隔了整个程序。

稳定的只有退出方式：

```bash
echo $?      # → 134，即 128 + 6 = SIGABRT，zsh 显示成 "IOT instruction"
```

对照组 —— **同一个脚本，加个 `mask` 参数把 mask 切回来**：

```bash
CUDA_VISIBLE_DEVICES=7 TRITON_INTERPRET=1 timeout 300 $PY ex1c.py mask
echo $?      # → 0
```

```
launch 完成（add_masked），out 正确 = True
```

有 mask 稳定退 0，无 mask 稳定 SIGABRT，**变量只有 mask 一个**，
所以崩溃确实是它引起的 —— 但这个结论是靠**跑对照组**得出的，
不是报错信息告诉你的。

原因：解释器模式把访存搬到 host 上用 numpy 做，于是越界踩坏的是
**进程自己的堆**，而不是显存。你拿到的是 glibc 事后发现堆结构不一致时的
抱怨，**既不告诉你是哪一行，也不保证崩在出错的那一刻**。

> 所以分工是：`TRITON_INTERPRET=1` 用来查**逻辑**错误（下标算错、mask 写反，
> 可以在 kernel 里 `print`）；查**越界**要用 memcheck + `PYTORCH_NO_CUDA_MEMORY_CACHING=1`。
> 拿 interpret 模式查越界，只会得到一个更难看的崩溃。

### 1.7 第 1 题结论

| 手段 | 能发现这个 bug 吗 | 备注 |
|---|---|---|
| 直接跑 + 比对 `out` | ❌ | 结果完全正确，自测全 PASS |
| 邻居张量被踩 | ⚠️ | 能看到，但要故意构造内存布局 |
| `compute-sanitizer memcheck` | ❌ | 被 torch 缓存分配器挡住，0 errors |
| `memcheck` + `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | ✅ | **唯一可靠的办法**，还能报到 .py 行号 |
| `TRITON_INTERPRET=1` | ⚠️ | 会 SIGABRT，但崩在 host 堆上：报错每次不同、无定位信息，甚至可能先打印出正确结果 |

> **一句话：越界不会自己暴露，`mask=` 一律要写。**
> 这也意味着 `../README.md` §5 里「用 compute-sanitizer 验」需要补一个前提 ——
> 不加 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 的话，验了也是白验。

---

## 2. 第 2 题：改成 `out = x * a + y`，需要改 kernel 签名吗？

**结论：需要。** 但「怎么改」有三条路，代价完全不同；而且这一题真正的收获是
改完之后**正确性检查会挂**，原因和 `a` 无关。

### 2.1 先试试不改签名 —— 用 Python 全局变量

最偷懒的想法：在模块里写个 `A_GLOBAL = 3.0`，kernel 里直接引用。

```python
A_GLOBAL = 3.0

@triton.jit
def axpy_global(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n
    tl.store(out_ptr + offs,
             tl.load(x_ptr + offs, mask=m) * A_GLOBAL + tl.load(y_ptr + offs, mask=m), mask=m)
```

直接报错，而且这条报错信息写得非常好：

```
triton.compiler.errors.CompilationError: at 4:61:
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) * A_GLOBAL + ...
                                                             ^
NameError("Cannot access global variable A_GLOBAL from within @jit'ed function.
Triton kernels can only access global variables that are instanstiated as constexpr
(`x = triton.language.constexpr(42)`). Note that this is different from annotating a
variable as constexpr (`x: triton.language.constexpr = 42`), which is not supported.
Alternatively, set the envvar TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1, but we do not
promise to support this forever.")
```

**Triton 直接禁止读普通 Python 全局变量。** 理由：kernel 是被编译并缓存的，
而全局变量随时可能被改 —— 一旦缓存命中，你改了变量却还在跑旧代码，就成了
最难查的一类 bug。Triton 选择在编译期就把这条路堵死。

### 2.2 按提示包成 `tl.constexpr` —— 能跑，但改值会报错

```python
A_CE = tl.constexpr(3.0)        # 注意是「赋值成 constexpr 对象」，不是类型标注
```

第一次跑没问题。但把 `A_CE` 重新绑成 `tl.constexpr(5.0)` 再 launch：

```
RuntimeError: Global variable A_CE has changed since we compiled this kernel,
              from constexpr[3.0] to constexpr[5.0]
```

**Triton 记住了编译时全局变量的值，发现变了就直接拒绝运行。**
所以 2.1 里担心的「静默用旧代码」不会发生 —— 但代价是这个方案**只能用于
从头到尾不变的常量**（比如一个物理系数）。`a` 要能变，这条路就不通。

### 2.3 方案 B：加一个运行期参数（推荐）

签名加一个普通参数就行，位置放在 `n` 之后、`BLOCK_SIZE` 之前：

```python
@triton.jit
def axpy_arg(x_ptr, y_ptr, out_ptr, n, a, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n
    tl.store(out_ptr + offs,
             tl.load(x_ptr + offs, mask=m) * a + tl.load(y_ptr + offs, mask=m), mask=m)
```

host 侧：

```python
axpy_arg[grid](x, y, out, n, a, BLOCK_SIZE=block_size)
```

实测三个不同的 `a`：

```
  a=3.0 编译次数=1
  a=5.0 编译次数=1
  a=7.0 编译次数=1
```

**一次编译，所有取值复用。** 这是绝大多数情况下的正确选择。

### 2.4 方案 C：声明成 `tl.constexpr`

```python
def axpy_constexpr(x_ptr, y_ptr, out_ptr, n, a: tl.constexpr, BLOCK_SIZE: tl.constexpr):
```

```
  a=3.0 编译次数=1
  a=5.0 编译次数=2
  a=7.0 编译次数=3
```

**每个不同的 `a` 都要重新编译一次。** 换来的是 `a` 变成立即数，编译器可以做常量
折叠（`a=1.0` 时乘法能被整个消掉，`a=2.0` 能变成加法）。

> 想自己数编译次数，注意 `JITFunction.device_caches[dev]` 是个 5 元组
> `(dict, dict, GPUTarget, CUDABackend, function)`，**只有第 0 项是 kernel 缓存**：
> ```python
> ncomp = lambda k: sum(len(v[0]) for v in k.device_caches.values())
> ```
> 直接 `len(device_caches[dev])` 会恒等于 5，白忙。

取舍很清楚：

| | 编译次数 | 适用场景 |
|---|---|---|
| 方案 B 运行期参数 | 1 | **默认选这个**。`a` 会变、取值多 |
| 方案 C `tl.constexpr` | 每个取值一次 | 取值只有两三种，且常量折叠能省掉真实计算 |
| 方案 A2 全局 constexpr | 1（改了就报错） | 全程不变的常量 |

### 2.5 ⚠️ 顺手踩到的坑：传整数会多编译一次

同一个方案 B 的 kernel，喂不同类型/取值的 `a`：

```
  a=2    (int  ) 编译次数=2      ← int 和 float 是两套签名
  a=2.0  (float) 编译次数=2      ← 复用了前面 3.0/5.0/7.0 那份
  a=1    (int  ) 编译次数=3      ← ★ 整数 1 被单独特化
  a=0    (int  ) 编译次数=4      ← ★ 整数 0 也被单独特化
  a=1.0  (float) 编译次数=4      ← float 1.0 不特化
```

两条规律：

1. `a=2` 和 `a=2.0` 走**不同的编译产物**（`int32` vs `fp32`）。想避免意外多编译一份，
   host 侧显式 `float(a)`。
2. **整数参数取 0 或 1 时，Triton 会按「值」额外特化一份**（这样它能把 `*1` 消掉、
   把 `*0` 折叠成常量）。float 没有这个待遇。所以一个整数参数最多会生成
   「0 / 1 / 其它」三份代码。

### 2.6 ⚠️⚠️ 真正的坑：改完之后正确性检查会挂

按方案 B 改完，直接套用仓库自带的自测入口：

```python
from _shared import CASES
from common import check
...
ok &= check(out, x * a + y, f"n={n:<8}")
```

**四个用例全挂：**

```
[用 common.check 的默认容差 rtol=0 atol=0]
  [FAIL] n=1024    : max_abs_err = 9.53674e-07
  [FAIL] n=1000    : max_abs_err = 9.53674e-07
  [FAIL] n=98765   : max_abs_err = 9.53674e-07
  [FAIL] n=1048576 : max_abs_err = 9.53674e-07
```

注意 `common.py` 里的签名是 `check(got, ref, name, rtol=0, atol=0)` ——
**默认容差是 0，也就是要求逐位相等**（`max_abs_err` 只是顺便打印出来的信息，
不参与判定）。原版 `x + y` 能全 PASS（`max_abs_err = 0`），换成 `x * a + y`
就一个都过不了。

先别怀疑 kernel。看 PTX：

```python
c = axpy_arg.warmup(x, y, out, n, 3.0, BLOCK_SIZE=1024, grid=(1,))
ptx = c.asm["ptx"]
```

```
                 fma.rn.f32   mul.f32   add.f32
axpy x*a+y            8          0         0        ← 融合成一条 FMA
add  x+y              0          0         8        ← 对照：纯加法
```

**Triton 把 `x * a + y` 编译成了 8 条 `fma.rn.f32`，一条指令完成乘加。**
FMA 只在最后舍入一次；torch 的 `x * a + y` 是两个 kernel，乘完先舍入成 fp32
再去加。两者差 1 个 ulp 左右 —— **Triton 这边其实更准。**

那为什么 `allclose` 会挂？查一下是哪些元素：

```
不满足容差的元素数 = 41 / 98765
绝对误差最大的元素 i=691:  triton=8.1673402786e+00  torch=8.1673393250e+00
    |diff| = 9.537e-07   |ref| = 8.167e+00          ← 相对误差才 1.2e-7，这个能过
相对误差最大的元素 j=244:  triton=3.2529234886e-05  torch=3.2544136047e-05
    |ref[j]| = 3.254e-05                            ← 发生了抵消，接近 0
这些越界容差的元素里，|ref| 的中位数 = 1.138e-03
全体 |ref| 的中位数              = 2.134e+00
```

`torch.allclose` 的判据是 `|a-b| <= atol + rtol*|b|`，默认 `atol=1e-8, rtol=1e-5`。
`x*3+y` 里有少数元素发生了**抵消**（结果接近 0），它们的 `|b|` 只有 1e-3 量级，
于是容差塌缩到 ~1e-8，而 FMA 带来的 1e-7 级差异就超了。**41 个元素里没有一个是算错的，
全是抵消放大了相对误差。**

给个合适的绝对容差就全过了：

```python
ok &= check(out, x * a + y, f"n={n:<8}", atol=1e-5)
```

```
[改成 atol=1e-5]
  [PASS] n=1024    : max_abs_err = 9.53674e-07
  [PASS] n=1000    : max_abs_err = 9.53674e-07
  [PASS] n=98765   : max_abs_err = 9.53674e-07
  [PASS] n=1048576 : max_abs_err = 9.53674e-07
  → 全部通过
```

> **这是个通用陷阱，不止这道题。** 只要你的 kernel 里有乘加，就会被融合成 FMA，
> 于是**逐位对齐 torch 的结果是不可能的**，必须按算子的数值特性给容差。
> 原版 `x + y` 能用 `rtol=0, atol=0` 通过，纯粹因为单次加法没有中间舍入 ——
> **这是 `01_vector_add` 独有的运气，别把它当常态。** `03_matmul` 那边 `atol`
> 要随 K 放大（见 [`../README.md` §5](../README.md) 第 10 条），是同一件事的延续。

### 2.7 附加发现：`n % 16` 决定了能不能向量化

第 1 题里 sanitizer 报的是 `read of size 16 bytes`（向量化），可 `n=98765` 时
我 dump PTX 看到的却是 16 条标量 `ld.global.b32`。扫一遍就清楚了：

```
         n  n%16  有 mask                        无 mask
  16777216     0  {'ld.global.v4.b32': 4}       {'ld.global.v4.b32': 4}
     98765    13  {'ld.global.b32': 16}         {'ld.global.v4.b32': 4}
     98768     0  {'ld.global.v4.b32': 4}       {'ld.global.v4.b32': 4}
      1000     8  {'ld.global.b32': 16}         {'ld.global.v4.b32': 4}
      1024     0  {'ld.global.v4.b32': 4}       {'ld.global.v4.b32': 4}
```

规律：**`mask` + `n` 不是 16 的倍数 → 丢掉向量化。**
Triton 会把整数参数按「是否能被 16 整除」做特化；`n % 16 == 0` 时它能证明
mask 的边界落在 16 元素对齐处，于是敢发 128-bit 访存，否则只能退回标量。
（`n=1000` 虽然能被 8 整除，但 `1000 % 16 = 8`，照样不行。）

那这个损失值多少钱？**做一个干净的 A/B**：同一个 `n = 2²⁴ − 3`（`n%16=13`），
buffer 开满 `2²⁴` 所以无 mask 版越界也不出 buffer —— 两版搬的有效数据完全一样,
**唯一区别就是向量化与否，合并访存完全不变**：

```
  有 mask（标量  ld.global.b32 ×16）   149.5 us   1346.6 GB/s (86.6%)
  无 mask（向量化 ld.global.v4.b32 ×4）  150.0 us   1342.1 GB/s (86.3%)
```

**向量化本身值 0%**（标量版还快了 0.3%，纯噪声）。

这个结果有点分量：`README.md` §3.2 说 v1 那个反面教材「同时破坏了向量化和合并访存，
没法拆开单独归因」。上面这个实验把向量化单独拆出来了 —— 它几乎不值钱，
**所以 v1 那 5.8 倍的差距几乎全部来自合并访存**。

> 别推广过头：这里向量化不值钱，是因为 kernel 已经打满 DRAM 带宽（87%），
> 少发几条指令也没地方可省。在 compute-bound 的 kernel 上结论会反过来 ——
> 对照 [`../../cuda/03_matmul/README.md`](../../cuda/03_matmul/README.md) §3.2，
> 那边 v2→v3 的 1.52x **全部**来自减少指令条数。

### 2.8 第 2 题结论

- **要改签名**，加一个运行期参数 `a` 是默认答案（一次编译、任意取值）。
- 不改签名的两条路都堵着：普通全局变量被编译器直接拒绝，`tl.constexpr` 全局变量
  改值时会报 `Global variable ... has changed`。
- 改完**一定会遇到 `allclose` 失败**，原因是 FMA 融合 + 抵消，不是 kernel 错。
  按数值特性给容差，别追求逐位相等。
- 顺带记住：整数参数在 0/1 上会被额外特化；`n` 不是 16 的倍数会丢向量化，
  但在这个 memory-bound kernel 上不心疼。

---

## 3. 第 3 题：`BLOCK_SIZE = 100` 能编译吗？为什么？

### 3.1 直接跑

```python
add_kernel[(triton.cdiv(n, 100),)](x, y, out, n, BLOCK_SIZE=100)
```

```
BLOCK_SIZE= 128 → 编译并运行成功，正确 = True
BLOCK_SIZE= 100 → CompilationError
      at 2:43:
      def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
          offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                                                 ^
      arange's range must be a power of 2
```

**不能编译。** 报错精确指向 `tl.arange`，而不是 `BLOCK_SIZE` 本身 ——
这是个重要线索：**约束来自 `tl.arange` 要求长度是 2 的幂**，
`BLOCK_SIZE` 只是碰巧被当成了它的长度。

### 3.2 为什么必须是 2 的幂

容易误以为「大概要是 warp 大小 32 的倍数吧」。实测否掉这个猜想：

| `BLOCK_SIZE` | 是 2 的幂 | 是 32 的倍数 | 结果 |
|---|---|---|---|
| 1 | ✅ | ❌ | OK |
| 2 | ✅ | ❌ | OK |
| 32 | ✅ | ✅ | OK |
| 64 | ✅ | ✅ | OK |
| **96** | ❌ | **✅** | **arange's range must be a power of 2** |
| 100 | ❌ | ❌ | arange's range must be a power of 2 |
| 128 | ✅ | ✅ | OK |
| **192** | ❌ | **✅** | **arange's range must be a power of 2** |
| 1000 | ❌ | ❌ | arange's range must be a power of 2 |
| 1024 | ✅ | ✅ | OK |

**判据就是 2 的幂**：`1` 和 `2` 比一个 warp 还小却能过，`96` 和 `192` 是 32 的
整数倍却过不了。

原因在 Triton 的编译模型：一个 tile 要被**层层二分**地摊到 lane → warp → CTA 上
（这套映射就是 ttgir 里的 layout）。每一层都是「对半分」，所以只有 2 的幂能被
整齐分完。非 2 的幂会让某些层出现残缺的分块，编译器与其生成一堆特例，
不如直接要求你自己补齐 —— 反正补齐的办法就是 mask，而你本来就要写 mask。

> 换句话说：**这不是「不支持」，是「让你用已有的机制表达」。**
> Triton 里 tile 形状是 2 的幂、边界用 mask 处理，是贯穿全部练习的统一模式。

### 3.3 修法 A：向上取整到 2 的幂

```python
BLOCK_SIZE = triton.next_power_of_2(100)     # → 128
```

每个 program 处理 128 个元素。这是**正确答案**：`mask = offsets < n_elements`
本来就已经在兜边界，tile 变大不影响正确性。

### 3.4 修法 B：如果非要「每个 program 正好 100 个」

有时分块大小是业务约束（比如一行 100 列）。那就让 **tile 是 128，但只放 100 个有效 lane**，
多出来的 28 条用 mask 关掉 —— 需要**两个 mask 条件**：

```python
@triton.jit
def tile_padded(x_ptr, y_ptr, out_ptr, n,
                CHUNK: tl.constexpr,        # 每个 program 真正负责多少（可以非 2 的幂）
                BLOCK_SIZE: tl.constexpr):  # tile 大小，必须是 2 的幂，≥ CHUNK
    lane = tl.arange(0, BLOCK_SIZE)                 # 0..127
    offs = tl.program_id(0) * CHUNK + lane          # 步长是 CHUNK=100，不是 128
    m = (lane < CHUNK) & (offs < n)                 # ① 关掉多余 lane ② 关掉尾部越界
    tl.store(out_ptr + offs,
             tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)
```

launch 时 `grid = (triton.cdiv(n, CHUNK),)`，注意分母是 `CHUNK`。

> 漏掉 `lane < CHUNK` 就错了：`offs` 的步长是 100，而 lane 走到 127，
> 相邻 program 会重叠覆盖 28 个元素。这个 bug 不会报错，
> 结果在重叠处「碰巧也对」（重复写了同样的值），但换成 `out += ...` 之类
> 非幂等的操作就会错 —— 又是一个只能靠推理发现的坑。

### 3.5 两种修法的代价

`n = 16,777,216`，实测：

```
  修法 A  每 program 128        151.8 us    1326.3 GB/s (85.3%)   1.00x
  修法 A  每 program 1024       148.8 us    1352.6 GB/s (87.0%)   1.02x
  修法 B  每 program 100        164.5 us    1223.6 GB/s (78.7%)   0.92x   ← 慢 8%
  修法 B  每 program 1000       150.7 us    1336.2 GB/s (85.9%)   1.01x
```

两种修法都**正确**（都和 `x+y` 逐位相等），但修法 B 在 `CHUNK=100` 时慢 **8%**。

代价和**浪费的 lane 比例**直接对应：

| | tile | 有效 | 浪费 | 带宽 |
|---|---|---|---|---|
| B, CHUNK=100 | 128 | 100 | **21.9%** | 78.7% |
| B, CHUNK=1000 | 1024 | 1000 | **2.3%** | 85.9% |

`CHUNK=100` 时每 128 条 lane 里有 28 条在空转，而且每个 program 的 100 元素段
（400 字节）不再和 128 字节的 cache line 对齐，段与段之间错位。
`CHUNK=1000` 只浪费 2.3%，就基本没影响了。

### 3.6 第 3 题结论

- **不能编译**，报错是 `arange's range must be a power of 2`。
- 约束是**严格的 2 的幂**，不是 32 的倍数（`96`、`192` 都过不了）。
  根因是 tile → warp → lane 的 layout 靠逐层二分建立。
- 标准修法是 `triton.next_power_of_2()` 向上取整，靠已有的 `mask` 兜边界，**零代价**。
- 非要保住非 2 的幂的分块语义，就 tile 补齐 + 双 mask，代价 ≈ 浪费的 lane 比例
  （100/128 → 慢 8%；1000/1024 → 几乎免费）。

---

## 4. 三道题串起来的一条主线

三道题看着散，其实都在说同一件事：**Triton 里你能控制的是「访存模式」和「编译期常量」，
而正确性的兜底机制只有一个 —— `mask`。**

| | 题面 | 背后的机制 |
|---|---|---|
| 1 | 少写 mask | mask 是唯一的边界机制，漏了不会报错，工具链默认也查不出来 |
| 2 | 加一个标量参数 | 参数是运行期还是编译期，决定编译几份代码；乘加会融合成 FMA |
| 3 | tile 不是 2 的幂 | tile 形状受 layout 约束，「补齐 + mask」是统一解法 |

再叠上第 2.7 节那个隔离实验，`01_vector_add` 这个练习的完整结论是：

```
向量化          →  0%      （已经打满带宽，少发指令没用）
tile 大小       →  ±2%     （README §3.1）
循环展开        →  +0.8%   （README §3.3）
autotune        →  ±0%     （README §3.4）
lane 浪费 22%   →  −8%     （本文 §3.5）
合并访存写坏     →  −83%    ⭐（README §3.2，现在可以确认这 83% 几乎全是合并访存）
```

**memory-bound 算子上，唯一值钱的事是别把合并访存写坏。** 其它都是噪声。

---

## 5. 自测清单

做完可以自己核一遍：

- [ ] 第 1 题：不加 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 时，memcheck 报几个 error？（应该是 0）
- [ ] 第 1 题：加上之后，报错信息里的 `nearest allocation ... of size` 是多少？（应该是 4,000）
- [ ] 第 1 题：报错里的 `read of size N bytes`，N 是几？为什么不是 4？
- [ ] 第 1 题：`ERROR SUMMARY` 的条数里，有几条是真越界、几条是余震？真越界那几条
      乘以 16 字节，对得上 24 个越界元素吗？
- [ ] 第 2 题：`ncomp` 数一下方案 B 和方案 C 各编译了几次
- [ ] 第 2 题：传 `a=1`（整数）会不会多编译一份？
- [ ] 第 2 题：`common.check()` 的默认容差是多少？为什么 `x+y` 能过而 `x*a+y` 不能？
- [ ] 第 2 题：把 `allclose(out, x*a+y)` 的失败元素捞出来，看它们的 `|ref|` 有多小
- [ ] 第 3 题：`BLOCK_SIZE=96` 能过吗？（32 的倍数但不是 2 的幂）
- [ ] 第 3 题：修法 B 里故意删掉 `lane < CHUNK`，结果还对吗？换成非幂等操作呢？

清理临时文件：

```bash
rm -rf /tmp/vecadd_ex
```

---

## 6. 我的笔记

<!-- 自己复现时写在这里 -->

- [ ] 我机器上的基线带宽是多少？和 87% 差多少？
- [ ] 第 1 题的错误条数我这里是几条？（实测 `ERROR SUMMARY: 10` = 6 条真越界 + 4 条余震，会波动）
- [ ] 第 1 题 `TRITON_INTERPRET=1` 连跑 5 次，我这里出现了几种不同的 glibc 报错？
      （实测 4 种，退出码恒为 134）
- [ ] 第 2.7 节的隔离实验，我这里向量化值多少钱？
