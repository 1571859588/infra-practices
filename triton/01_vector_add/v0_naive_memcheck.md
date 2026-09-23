# 读懂 `compute-sanitizer` 的 memcheck 日志

[`v0_naive.md` §1.5](v0_naive.md) 的延伸篇。那里只说了「加上
`PYTORCH_NO_CUDA_MEMORY_CACHING=1` 就能报出越界」，这里回答接下来的问题：

- 那个 500 多行的 `mc.log` 该怎么读？哪些行有用、哪些可以跳过？
- `thread (122,0,0)`、`+0xc0`、`1 bytes after` 这些数字是从哪来的？
- 为什么 x 和 y 都越界了，却只报了 **6 条**、而且只报了 **x**？
- memcheck 报 0 errors，能说明没有越界吗？（**不能**，§7 有反例）

每个数字都要能从源码 → 编译产物 → 硬件执行一路推回来，所以这篇会涉及 Triton
的 layout 和 A100 的 SASS。所有输出都是实测的（A100-SXM4-40GB / triton 3.5.0 /
torch 2.9.0+cu128），命令在 zsh 和 bash 下都验证过。

---

## 0. 准备

还是在 `/tmp` 下另开目录，别在仓库里跑：

```bash
mkdir -p /tmp/vecadd_ex/memcheck && cd /tmp/vecadd_ex/memcheck
PY=/mnt/public/nyt1/docqa/restored_envs/cpp/bin/python
export CUDA_VISIBLE_DEVICES=7
export TRITON_CACHE_DIR=$PWD/tcache          # ① 见下方说明

# ② 用函数而不是 MS="compute-sanitizer ..." —— zsh 默认不对 $MS 分词，会报 no such file
ms() { /usr/local/cuda/bin/compute-sanitizer --tool memcheck --target-processes all "$@"; }

# ③ 把一份日志压成 4 行摘要，后面每个实验都用它
summ() {
  echo "  device 越界 : $(grep -c 'Invalid __global__' "$1") 条"
  echo "  出错 PC     : $(grep -oE 'at [a-z_]+\+0x[0-9a-f]+' "$1" | sort | uniq -c | tr -s ' ' | paste -sd';')"
  echo "  最近分配    : $(grep -oE 'allocation at 0x[0-9a-f]+' "$1" | sort | uniq -c | tr -s ' ' | paste -sd';')"
  echo "  $(grep -m1 'ERROR SUMMARY' "$1" | sed 's/^=* //')"
}
```

> ① **为什么要单独设 `TRITON_CACHE_DIR`**：memcheck 报出的源文件路径是**编译时**
> 写进 cubin 的。Triton 默认缓存在 `~/.triton/cache`，如果你以前在别的目录编译过
> 一模一样的 kernel，缓存命中后报告里会出现**那个旧目录的路径**，很让人困惑。
> 给每次实验一个新的缓存目录就没这个问题，顺便也方便 §5 去找 launcher。

实验脚本只有一个，用参数切换各种情况：

```bash
cat > probe.py <<'PY'
# 用法: probe.py [--bs 1024] [--n 1000] [--pad-x] [--victim] [--dump]
import argparse, torch, triton, triton.language as tl

@triton.jit
def add_nomask(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs))

p = argparse.ArgumentParser()
p.add_argument("--bs", type=int, default=1024)
p.add_argument("--n", type=int, default=1000)
p.add_argument("--pad-x", action="store_true", help="x 多开到 BLOCK_SIZE 个元素，让 x 的读不越界")
p.add_argument("--victim", action="store_true", help="在 out 后面再分配一块，看它有没有被踩")
p.add_argument("--dump", action="store_true", help="只编译：打印 layout，导出 cubin 供 cuobjdump")
a = p.parse_args()

x = torch.randn(max(a.n, a.bs) if a.pad_x else a.n, device="cuda")
y = torch.randn(a.n, device="cuda")
out = torch.empty(a.n, device="cuda")
victim = torch.full((a.n,), 7.0, device="cuda") if a.victim else None

if a.dump:
    c = add_nomask.warmup(x, y, out, a.n, BLOCK_SIZE=a.bs, grid=(1,))
    print("num_warps =", c.metadata.num_warps)
    print([l for l in c.asm["ttgir"].splitlines() if l.startswith("#blocked")])
    open(f"k_bs{a.bs}.cubin", "wb").write(c.asm["cubin"])
    raise SystemExit

print(f"PTR x={x.data_ptr():#x} y={y.data_ptr():#x} out={out.data_ptr():#x}"
      + (f" victim={victim.data_ptr():#x}" if a.victim else ""), flush=True)
add_nomask[(triton.cdiv(a.n, a.bs),)](x, y, out, a.n, BLOCK_SIZE=a.bs)
torch.cuda.synchronize()
msg = f"out 正确 = {torch.equal(out, x[:a.n] + y)}"
if a.victim:
    msg += f"   victim 被改写 {(victim != 7.0).sum().item()} / {a.n}"
print(msg)
PY
```

它和 `v0_naive.md` 里的 `ex1c.py` 是同一个 kernel，`tl.store` 也在**第 7 行**。
多了一行 `PTR ...`，把三个 buffer 的真实地址打出来 —— 读日志时要拿它对照。

---

## 1. 先看骨架：533 行里只有 13 行是「事件」

```bash
ms env PYTORCH_NO_CUDA_MEMORY_CACHING=1 $PY probe.py > A.log 2>&1
wc -l < A.log                          # → 533
grep -nE "^========= [A-Z]" A.log      # 只看顶格的事件行
```

```
1:========= COMPUTE-SANITIZER
3:========= Invalid __global__ read of size 16 bytes          ┐
60:========= Invalid __global__ read of size 16 bytes         │
117:========= Invalid __global__ read of size 16 bytes        │ 6 条 device 端越界
174:========= Invalid __global__ read of size 16 bytes        │ （每条 57 行，
231:========= Invalid __global__ read of size 16 bytes        │   其中 50 行是 host 调用栈）
288:========= Invalid __global__ read of size 16 bytes        ┘
345:========= Program hit cudaErrorLaunchFailure (error 719) ... cudaDeviceSynchronize.  ┐
384:========= Program hit cudaErrorLaunchFailure (error 719) ... cudaGetLastError.       │ 4 条 host 端
437:========= Program hit cudaErrorLaunchFailure (error 719) ... cudaFree.               │ API 报错（余震）
472:========= Program hit cudaErrorLaunchFailure (error 719) ... cudaGetLastError.       ┘
531:========= Error: process didn't terminate successfully
532:========= Target application returned an error
533:========= ERROR SUMMARY: 10 errors
```

**读 memcheck 日志的第一步永远是这条 grep。** 它告诉你三件事：

| 部分 | 含义 | 本例 |
|---|---|---|
| `Invalid __global__ ...` | **真正的 bug**：GPU 上某条指令访问了非法地址 | 6 条 |
| `Program hit ... on CUDA API call to ...` | host 端 API 返回错误，**通常是前者的连锁反应** | 4 条 |
| `ERROR SUMMARY` | 上面两类的**总和** | 10 = 6 + 4 |

所以 `ERROR SUMMARY: 10 errors` 不是「越界了 10 次」。这个数在本例里是**稳定的**
（连跑 3 次都是 6 + 4 = 10）。

嫌长可以关掉 host 调用栈，533 行 → 85 行：

```bash
ms --show-backtrace no env PYTORCH_NO_CUDA_MEMORY_CACHING=1 $PY probe.py > B.log 2>&1
wc -l < B.log                          # → 85
```

> 第一次读日志时**别关**，§5 会用到调用栈。弄明白之后日常用 `--show-backtrace no`。

---

## 2. 逐字段拆一条 device 错误

```
========= Invalid __global__ read of size 16 bytes                        ← (a)
=========     at add_nomask+0xc0 in /tmp/vecadd_ex/memcheck/probe.py:7    ← (b)
=========     by thread (122,0,0) in block (0,0,0)                        ← (c)
=========     Address 0x7d750ca00fa0 is out of bounds                     ← (d)
=========     and is 1 bytes after the nearest allocation at 0x7d750ca00000 of size 4,000 bytes   ← (e)
=========     Saved host backtrace up to driver entry point at kernel launch time               ← (f)
=========     Host Frame:cuLaunchKernelEx [0x39fc2f]
...
```

| | 字段 | 含义 | 本例怎么对上 | 详见 |
|---|---|---|---|---|
| (a) | `__global__` | 出错的**地址空间**。还可能是 `__shared__` / `__local__` | x/y/out 都在显存（global） | |
| | `read` | 读还是写（`write`） | `tl.load(x_ptr + offs)` | |
| | `of size 16 bytes` | **单条指令**访问的宽度 | `LDG.E.128`，一次 4 个 float | §4 |
| (b) | `add_nomask` | kernel 名 = 你的 `@triton.jit` 函数名 | | |
| | `+0xc0` | 出错指令在 **SASS** 里的偏移（PC） | 第 2 轮读 x 的那条 LDG | §4 |
| | `probe.py:7` | 这条 SASS 对应的**源码行** | `tl.store(...)` 那行 | |
| (c) | `thread (122,0,0)` | `threadIdx`。**不是元素下标** | 122 号线程读元素 1000~1003 | §3 |
| | `block (0,0,0)` | `blockIdx` = Triton 的 `program_id` | grid 只有 1 个 program | |
| (d) | `Address ...fa0` | 出错的**虚拟地址** | x 起点 + 4000 B = 元素 1000 | |
| (e) | `nearest allocation` | 离这个地址**最近**的一块合法分配 | 就是 x | §2.1 |
| (f) | host backtrace | **launch 时刻**的 CPU 调用栈 | 不是出错时刻！ | §5 |

(b) 里的行号为什么是 `tl.store` 那一行而不是 `tl.load`？因为三个访存写在同一行源码里。
想精确到「哪个 load」，得靠 `+0xc0` 去查 SASS（§4）。**写 kernel 时把 load 拆到
不同行，能让这种定位省一步。**

### 2.1 (e) 的陷阱：`before` 的那几条也是 x 的越界

把 6 条错误的线程和位置描述并排：

```bash
grep -A4 "Invalid __global__" A.log | grep -E "thread|nearest" | paste - - \
  | sed -E 's/=+ +//g; s/ in block \(0,0,0\)//; s/ of size 4,000 bytes//'
```

```
by thread (122,0,0)   and is 1 bytes after the nearest allocation at 0x7d750ca00000   ← x
by thread (123,0,0)   and is 17 bytes after the nearest allocation at 0x7d750ca00000
by thread (124,0,0)   and is 33 bytes after the nearest allocation at 0x7d750ca00000
by thread (125,0,0)   and is 48 bytes before the nearest allocation at 0x7d750ca01000  ← y？
by thread (126,0,0)   and is 32 bytes before the nearest allocation at 0x7d750ca01000
by thread (127,0,0)   and is 16 bytes before the nearest allocation at 0x7d750ca01000
```

对照 `PTR x=0x7d750ca00000 y=0x7d750ca01000 out=0x7d750ca02000`，后三条的
「最近分配」变成了 y —— 但**这不是 y 越界**。画出来：

```
0x…0000                           0x…0fa0              0x…1000
│◄──────────── x（4000 B）──────────►│◄─── 96 B 间隙 ───►│◄──── y ────
                                    │ fa0 fb0 fc0 │ fd0 fe0 ff0 │
                                    │ 122 123 124 │ 125 126 127 │  ← 线程
                                    │ 离 x 更近    │ 离 y 更近    │
```

6 个地址**全在 x 和 y 之间那 96 字节的间隙里**。memcheck 只是报「离哪块最近」，
过了中点就改口说 y。判断到底是谁越界，要看**地址落在谁的尾巴后面**，
而不是看它写的是 `after` 还是 `before`。

两个算术细节：

- `1 bytes after` 而不是 `0 bytes`：memcheck 从**最后一个合法字节**（`0x…0f9f`）算起。
  `0xfb0 − 0xf9f = 0x11 = 17` ✓
- 分配间隔是 `0x1000 = 4096 B`，而 x 只有 4000 B，所以间隙恰好 96 B ——
  **正好等于越界的 24 个元素**。本例越界完全落在间隙里、碰不到 y，是个巧合，§7 会打破它。

---

## 3. 从 layout 推出 `thread (122 ~ 127)`

为什么恰好是这 6 个线程？这取决于 Triton 把 1024 个元素怎么分给线程。

```bash
$PY probe.py --dump
```

```
num_warps = 4
['#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>']
```

这行 layout 就是答案。逐项翻译：

| 字段 | 值 | 含义 |
|---|---|---|
| `sizePerThread` | 4 | 每个线程一次拿**连续 4 个**元素 → 可以用 128-bit 访存 |
| `threadsPerWarp` | 32 | A100 的 warp 是 32 个线程（硬件决定） |
| `warpsPerCTA` | 4 | 一个 program（= 一个 CUDA block）有 4 个 warp = **128 个线程** |

一「轮」能覆盖 `4 × 32 × 4 = 512` 个元素，而 tile 是 1024 个 → **要走 2 轮**。
Triton 的 blocked layout 是循环铺开的，所以线程 `t` 负责的元素是：

```
第 0 轮：  4t ~ 4t+3              （元素   0 ~  511）
第 1 轮：  512 + 4t ~ 512 + 4t+3  （元素 512 ~ 1023）
```

`n = 1000`，越界的是元素 1000 ~ 1023，全在第 1 轮：

| 线程 t | 第 1 轮的元素 | 越界？ |
|---|---|---|
| 0 ~ 121 | 512 ~ 999 | 否 |
| **122** | **1000 ~ 1003** | **是** ← 地址 `x + 4000 = …fa0` |
| 123 | 1004 ~ 1007 | 是 |
| … | … | 是 |
| **127** | **1020 ~ 1023** | **是** |

**6 个线程 × 每个 16 B = 96 B = 24 个元素**，和日志一条不差。

> 这里有个容易想岔的地方：A100 硬件上执行的最小单位是 warp（32 线程），线程 122~127
> 属于 3 号 warp（96~127）的第 26~31 号 lane。同一条 LDG 指令，这个 warp 里
> 前 26 个 lane 合法、后 6 个非法 —— memcheck 是**按 lane 逐个检查**的，所以只报这 6 个。

---

## 4. 从 SASS 推出 `+0xc0`

`--dump` 顺便导出了 cubin，用 `cuobjdump` 反汇编（A100 = sm_80）：

```bash
/usr/local/cuda/bin/cuobjdump -sass k_bs1024.cubin | grep -E '/\*0[0-9a-f]{3}\*/' | grep -v NOP \
  | sed -E 's/ +\/\* 0x[0-9a-f]+ \*\/$//; s/^ +//; s/  +/ /g'
```

```
/*0010*/ S2R R0, SR_TID.X ;                        R0 = threadIdx.x                    = t
/*0030*/ S2R R3, SR_CTAID.X ;                      R3 = blockIdx.x                     = pid
/*0040*/ SHF.L.U32 R0, R0, 0x2, RZ ;               R0 = t << 2                         = 4t
/*0050*/ LOP3.LUT R2, R0, 0x1fc, RZ, 0xc0, !PT ;   R2 = 4t & 511                       ← 第 0 轮下标
/*0060*/ HFMA2.MMA R0, -RZ, RZ, 0, 2.38e-07 ;      R0 = 4（sizeof(float) 的位模式，编译器的小把戏）
/*0070*/ LEA R3, R3, R2, 0xa ;                     R3 = (pid << 10) + R2               = pid*1024 + 4t
/*0080*/ IMAD.WIDE R20, R3, R0, c[0x0][0x160] ;    R20 = x_ptr   + R3*4                ← c[0x0][0x160] 是第 1 个参数
/*0090*/ IMAD.WIDE R22, R3, R0, c[0x0][0x168] ;    R22 = y_ptr   + R3*4                ← 0x168 第 2 个
/*00a0*/ LDG.E.128 R12, [R20.64] ;                 读 x 第 0 轮
/*00b0*/ LDG.E.128 R16, [R22.64] ;                 读 y 第 0 轮
/*00c0*/ LDG.E.128 R4,  [R20.64+0x800] ;           读 x 第 1 轮   ★ +0x800 = 2048 B = 512 个元素
/*00d0*/ LDG.E.128 R8,  [R22.64+0x800] ;           读 y 第 1 轮
/*00e0*/ IMAD.WIDE R2, R3, R0, c[0x0][0x170] ;     R2 = out_ptr + R3*4                 ← 0x170 第 3 个
/*00f0*/ FADD ...                                  8 条 FADD
/*0150*/ STG.E.128 [R2.64], R12 ;                  写 out 第 0 轮
/*0180*/ STG.E.128 [R2.64+0x800], R4 ;             写 out 第 1 轮
/*0190*/ EXIT ;
```

（右侧注释是我加的。）对照日志：

- **`+0xc0`** 正是「读 x 第 1 轮」—— 和 §3 推出的「越界全在第 1 轮」吻合
- **`LDG.E.128`** 一次 128 bit = 16 B —— 就是日志里的 `read of size 16 bytes`
- 第 1 轮不需要重新算地址，直接用 `[R20.64+0x800]` 的立即数偏移，这也是 §3 里
  「循环铺开」在硬件上的样子

> sm_80 上 kernel 参数从常量区 `c[0x0][0x160]` 开始按 8 字节排，所以
> `0x160 / 0x168 / 0x170` 依次是 `x_ptr / y_ptr / out_ptr`。认出这三个偏移，
> 就能在任何 Triton kernel 的 SASS 里分清哪条访存对应哪个参数。

---

## 5. 调用链：日志里每一段分别来自哪一层

```
 Python                         ex: add_nomask[(1,)](x, y, out, n, BLOCK_SIZE=1024)
   │  JITFunction.run()
   │    ├─ 缓存未命中 → 编译 ttir → ttgir(§3 的 layout) → llir → ptx → cubin(§4 的 SASS)
   │    └─ 所有产物存进 $TRITON_CACHE_DIR/<hash>/
   ▼
 __triton_launcher.cpython-312-x86_64-linux-gnu.so : launch()      ← Triton 按参数签名生成的 C 启动器
   ▼
 libcuda.so : cuLaunchKernelEx()                                    ← 驱动入口，memcheck 在这里记下 host 栈
   ▼  ········································ 异步：CPU 立刻返回，GPU 开始跑 ···
 GPU 执行 SASS → 0xc0 的 LDG 越界 → memcheck 报告 (a)~(e) → kernel 被终止
   ▼  ········································ CPU 这边还不知道 ··················
 torch.cuda.synchronize() → cudaDeviceSynchronize()  → 错误在这里才冒出来   ← 余震 [1]
                          → cudaGetLastError()                                ← 余震 [2]
 进程退出，张量析构      → cudaFree()                                         ← 余震 [3]
                          → cudaGetLastError()                                ← 余震 [4]
```

### 5.1 device 错误的 host 栈：launch 时刻，不是出错时刻

日志原话是 `Saved host backtrace up to driver entry point at kernel launch time`。
过滤掉 CPython 解释器内部的帧：

```bash
sed -n '3,59p' A.log | grep "Host Frame" | grep -v "python-3.12"
sed -n '3,59p' A.log | grep -A1 "Host Frame:launch" | tail -1
```

```
=========     Host Frame:cuLaunchKernelEx [0x39fc2f]
=========     Host Frame:launch [0x1e0d]
=========     Host Frame:_PyEval_EvalFrameDefault in Python/bytecodes.c:3263 [0x125e32]
=========     Host Frame:_PyEval_EvalFrameDefault in Python/bytecodes.c:3263 [0x125e32]
=========     Host Frame:_PyEval_EvalFrameDefault in Python/bytecodes.c:3263 [0x125e32]
=========     Host Frame: [0x29d8f]
=========     Host Frame:__libc_start_main [0x29e3f]
=========     Host Frame: [0x1c5a08]
=========                in /tmp/vecadd_ex/memcheck/tcache/BCCJJ57LR5ZSLQJXFEHLTTSDTF63EPSEK3BBCX2JMDXTHNQ75DIA/__triton_launcher.cpython-312-x86_64-linux-gnu.so
```

最后一行是第二条命令的输出：`launch` 这一帧来自哪个文件。

从上往下正好是上图的三层：驱动 → Triton launcher → Python 解释器。

**这段栈对 Triton 几乎没用**：Python 那层只有 `_PyEval_EvalFrameDefault` 这种
解释器函数，**看不到是你 .py 的哪一行发起的 launch**。真正有用的定位信息是
device 侧的 (b) `probe.py:7`。只有一种情况值得看它：程序里 launch 了很多个 kernel，
你想确认出错的是**哪一次 launch** —— 这时看 launcher 的 hash 目录能区分不同签名的 kernel。

### 5.2 四条余震各自对应哪句 Python

每条 `Program hit` 下面也有 host 栈，找其中 torch 的帧：

```bash
awk '/^========= Program hit/{n++; ev=$0; sub(/.*API call to /,"",ev); print "\n["n"] "ev; k=0}
     n && /Host Frame:/ && /c10|at::|torch/ && k<3 {s=$0; sub(/.*Host Frame:/,"   ",s); sub(/ in \/.*$/,"",s); print s; k++}' A.log
```

```
[1] cudaDeviceSynchronize.
   c10::cuda::device_synchronize() [0x580e6]

[2] cudaGetLastError.
   c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) [0x57bdc]
   c10::cuda::device_synchronize() [0x58106]

[3] cudaFree.
   c10::cuda::CUDACachingAllocator::Native::uncached_delete(void*) [0x18073]
   c10::StorageImpl::~StorageImpl() [0x4827be]
   c10::TensorImpl::~TensorImpl() [0x7ad68]

[4] cudaGetLastError.
   c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) [0x57bdc]
   c10::StorageImpl::~StorageImpl() [0x4827be]
   c10::TensorImpl::~TensorImpl() [0x7ad68]
```

- [1][2] 是 `torch.cuda.synchronize()` 和 torch 紧跟着的错误检查 ——
  **CUDA 错误是异步的，要到下一次同步才会暴露**，这就是 Python 那边抛
  `unspecified launch failure` 的地方
- [3][4] 是进程退出时张量析构去释放显存，context 已经坏了，释放也失败
- [3] 里的 **`uncached_delete`** 顺带证明了 `PYTORCH_NO_CUDA_MEMORY_CACHING=1`
  确实生效：走的是「不缓存、直接 cudaFree」的路径

所以余震的条数取决于**出错之后程序又调了多少次 CUDA API**，跟 bug 本身无关。

---

## 6. 为什么只报了 x，没报 y 和 out？

§3 的推理对 y 同样成立 —— y 也是 1000 个元素、也被读了 1024 个。那 y 的 6 条呢？
out 的写也越界了，怎么一条 `write` 都没有？

**假设：kernel 在第一条出错的访存指令处就被终止了，后面的指令根本没执行。**

验证方法：让 x 不越界（`--pad-x` 把 x 开到 1024 个元素），看报告会不会**换成 y**：

```bash
ms --show-backtrace no env PYTORCH_NO_CUDA_MEMORY_CACHING=1 $PY probe.py --pad-x > C.log 2>&1
grep ^PTR C.log; summ C.log
```

| | `PTR` | 出错 PC | 最近分配 |
|---|---|---|---|
| 原始（A.log） | x=`…00000` y=`…01000` out=`…02000` | `6 at +0xc0`（读 x 第 1 轮） | x / y 之间 |
| `--pad-x`（C.log） | x=`…00000` y=`…01000` out=`…02000` | **`6 at +0xd0`（读 y 第 1 轮）** | **y / out 之间** |

x 不越界了，报告就**顺延到下一条 LDG**（`0xd0`），地址也移到了 y 的尾巴后面。
假设成立：

> **memcheck 只报「第一条出错的指令」上所有越界的 lane，然后 kernel 就被杀了。**
> 所以日志里的错误**不是完整的越界清单**，只是第一现场。修掉一处再跑，
> 可能会冒出下一处。

### 6.1 「第一条」是按 SASS 顺序，不是按源码顺序

源码里 `tl.load(x_ptr ...)` 写在 `tl.load(y_ptr ...)` 前面，那是不是总先报 x？**不一定。**
把 tile 开到 2048，看编译器怎么排指令：

```bash
$PY probe.py --dump --bs 2048
/usr/local/cuda/bin/cuobjdump -sass k_bs2048.cubin | grep -E 'LDG|STG' \
  | sed -E 's/ +\/\* 0x[0-9a-f]+ \*\/$//; s/^ +//; s/  +/ /g'
```

```
/*00a0*/ LDG.E.128 R4,  [R24.64] ;           x 第 0 轮     （R24 = x，R28 = y）
/*00b0*/ LDG.E.128 R8,  [R28.64] ;           y 第 0 轮
/*00c0*/ LDG.E.128 R16, [R24.64+0x800] ;     x 第 1 轮
/*00d0*/ LDG.E.128 R20, [R28.64+0x800] ;     y 第 1 轮
/*00e0*/ LDG.E.128 R12, [R28.64+0x1000] ;    y 第 2 轮   ★ y 排到了 x 前面
/*0140*/ LDG.E.128 R8,  [R24.64+0x1000] ;    x 第 2 轮
/*0180*/ LDG.E.128 R20, [R24.64+0x1800] ;    x 第 3 轮
/*0190*/ LDG.E.128 R24, [R28.64+0x1800] ;    y 第 3 轮
...
```

第 2 轮编译器先读 y 再读 x（寄存器分配的结果）。§7.2 的实验里越界恰好从第 2 轮开始，
于是 128 条错误**全部来自 `+0xe0`、全部是 y** —— x 明明越界得一样多，一条都没报。

> **哪个 buffer 先被报，由编译器的指令调度决定。** 看到「只有 y 越界」时，
> 别急着只查 y 的下标。

---

## 7. memcheck 的盲区：0 errors 不等于没越界

§2.1 说过，本例越界的 96 B 恰好落在分配之间的 96 B 间隙里。memcheck 能报，
**完全是因为这个间隙存在**。间隙不够大、或者根本没有，会怎样？

### 7.1 越界远大于间隙：只报了冰山一角

`--bs 2048`，同样 `n = 1000` → 越界 1048 个元素 = 4192 B，远超 96 B 间隙：

```bash
ms --show-backtrace no env PYTORCH_NO_CUDA_MEMORY_CACHING=1 $PY probe.py --bs 2048 > D.log 2>&1
grep ^PTR D.log; summ D.log
```

```
PTR x=0x77e4e2a00000 y=0x77e4e2a01000 out=0x77e4e2a02000
  device 越界 : 6 条
  出错 PC     :  6 at add_nomask+0xc0
  最近分配    :  3 allocation at 0x77e4e2a00000; 3 allocation at 0x77e4e2a01000
  ERROR SUMMARY: 10 errors
```

**和 BS=1024 时一模一样的 6 条。** 越界的 1048 个元素里：

| 元素 | 落在哪 | 被报告？ |
|---|---|---|
| 1000 ~ 1023 | x 和 y 之间的间隙 | ✅ 就是这 6 条 |
| 1024 ~ 2023 | **y 的合法分配里** | ❌ 地址合法，memcheck 无从判断 |
| 2024 ~ 2047 | y 后面的间隙 | ❌ 在第 3 轮，kernel 早就被杀了（§6） |

> **报了 6 条 ≠ 只越界了 24 个元素。** memcheck 告诉你「有越界」，不告诉你「越界多少」。

### 7.2 越界整段落进邻居：0 errors

`--n 1024`：每个 buffer 恰好 4096 B，和分配间隔一样大 → **buffer 之间没有间隙**。
再加 `--victim` 在 out 后面放一块「别人的数据」：

```bash
ms --show-backtrace no env PYTORCH_NO_CUDA_MEMORY_CACHING=1 $PY probe.py --n 1024 --bs 2048 --victim > E.log 2>&1
grep -E "^PTR|out 正确" E.log; summ E.log
```

```
PTR x=0x7b5256a00000 y=0x7b5256a01000 out=0x7b5256a02000 victim=0x7b5256a03000
out 正确 = True   victim 被改写 1024 / 1024
  device 越界 : 0 条
  ERROR SUMMARY: 0 errors
```

**关了缓存分配器，memcheck 照样 0 errors；而 victim 的 1024 个元素全部被改写。**
x 越界读到 y、y 越界读到 out、out 越界写到 victim —— 每个地址都属于某块合法分配。

### 7.3 修法：`--padding`

compute-sanitizer 可以在每个 `cudaMalloc` 后面**垫一块不可访问的区域**：

```bash
/usr/local/cuda/bin/compute-sanitizer --help 2>&1 | grep -A1 -- "--padding"
#   --padding arg (=0)    Size in bytes for padding buffer to add after each allocation.

ms --show-backtrace no --padding 8192 --print-limit 1000 \
   env PYTORCH_NO_CUDA_MEMORY_CACHING=1 $PY probe.py --n 1024 --bs 2048 --victim > F.log 2>&1
grep ^PTR F.log; summ F.log
```

```
PTR x=0x746c26a00000 y=0x746c26a03000 out=0x746c26a06000 victim=0x746c26a09000
  device 越界 : 128 条
  出错 PC     :  128 at add_nomask+0xe0
  最近分配    :  128 allocation at 0x746c26a03000
  ERROR SUMMARY: 132 errors
```

- 分配间隔从 `0x1000` 变成了 `0x3000` = 4096 + 8192，**padding 生效**
- **0 → 128 条**：第 2 轮整整 512 个元素全部落进 padding，128 个线程全报
- `+0xe0`、全是 y —— 正是 §6.1 说的「第 2 轮编译器先读 y」
- 加 `--print-limit 1000` 是因为默认只打印前 100 条（摘要行会提示
  `32 errors were not printed`）

**`--padding` 必须配合关缓存才有用**，否则它只能垫在 torch 那一整大段后面：

```bash
ms --show-backtrace no --padding 8192 $PY probe.py > G.log 2>&1      # 没有 PYTORCH_NO_CUDA_MEMORY_CACHING
summ G.log
#   device 越界 : 0 条
#   ERROR SUMMARY: 0 errors
```

---

## 8. 小结：推荐命令 + 读日志的顺序

**查 Triton kernel 越界的推荐姿势：**

```bash
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --target-processes all \
    --padding 8192 --show-backtrace no \
    env PYTORCH_NO_CUDA_MEMORY_CACHING=1 TRITON_CACHE_DIR=$PWD/tcache \
    python your_script.py 2>&1 | tee mc.log
```

| 选项 | 缺了会怎样 | 出处 |
|---|---|---|
| `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | 越界藏在 torch 的大段里，**0 errors** | `v0_naive.md` §1.4 |
| `--padding 8192` | 越界落进邻居分配时，**0 errors** | §7.2 |
| `--show-backtrace no` | 日志多 6 倍，全是 CPython 内部帧 | §1、§5.1 |
| `TRITON_CACHE_DIR=...` | 报告里的源码路径可能是旧目录 | §0 |

**拿到日志后按这个顺序读：**

1. `grep -nE "^========= [A-Z]" mc.log` —— 分清 **device 越界** 和 **API 余震**，余震先忽略
2. 看第一条 device 错误的 **源码行**（`xxx.py:N`）和 **读/写、宽度**
3. 看 **PC**（`+0x..`），用 `cuobjdump -sass` 确认是哪个指针的哪一轮访存
4. 看 **地址落在谁的尾巴后面**（对照 buffer 地址），别被 `before/after` 带偏
5. 用 layout 反推 **哪些线程、哪些元素** 越界，和 mask 的逻辑对一遍
6. 修完**再跑一次** —— 日志只是第一现场，后面可能还有

以及最重要的一条：**memcheck 报 0 errors 不是正确性的证明。** 它只能抓到落进
「无主地址」的访问。`mask=` 该写还是要写。

---

## 9. 自测

- [ ] `ERROR SUMMARY: 10 errors` 里，哪几条是真的 bug？
- [ ] 为什么 `thread (125,0,0)` 报的是 `before the nearest allocation at <y>`，却不是 y 越界？
- [ ] 如果 `num_warps=8`，BS=1024 时 layout 会变成几轮？越界的是哪几个线程？
      （提示：先 `--dump` 看 layout；`probe.py` 里没有 `num_warps` 参数，要自己在 launch 处加）
- [ ] 把 `probe.py` 第 7 行拆成三行（两个 load、一个 store 各一行），报告里的行号会变成几？
- [ ] 不看 SASS，你能说出 `--bs 2048` 时第一条出错的会是 x 还是 y 吗？（不能 —— 为什么？）
- [ ] §7.2 那个 0 errors 的例子里，`out 正确 = True`。这说明了自测的什么问题？

清理：

```bash
rm -rf /tmp/vecadd_ex/memcheck
```
