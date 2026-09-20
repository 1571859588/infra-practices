"""v3：online softmax —— 分块规约，解决「一行装不进寄存器」

v1/v2 有个硬前提：`BLOCK_SIZE >= n_cols`，一整行必须装进寄存器。
n_cols = 1<<18 时 BLOCK_SIZE 就是 262144，编译直接卡死。

online softmax 把行切成固定大小的块，**只用 O(1) 的状态**扫过去：

    m = -inf            # 到目前为止见过的最大值
    d = 0               # 到目前为止的 sum(exp(x - m))
    for each block:
        m_new = max(m, max(block))
        d     = d * exp(m - m_new) + sum(exp(block - m_new))   # ★ 修正项
        m     = m_new

关键是那个 `exp(m - m_new)` **修正因子**：最大值一变，之前累加的 d 是按旧的
m 归一化的，要整体乘上 `exp(m_old - m_new)` 才能换算到新基准。
这一步就是 FlashAttention 里 rescale 的来源。

代价：**要读两遍 x**。第一遍算出 (m, d)，第二遍才能算 exp(x-m)/d 写出去。
所以访存量是 3MN（读两次 + 写一次）而不是 2MN，理论上限只有 v1 的 2/3。

**这是一次明确的取舍**：用 1.5 倍访存量，换「行宽不再受寄存器限制」。
n_cols 小的时候别用它（v1 更快），n_cols 大到 v1 编译不动时它是唯一选择。

跑法：
  python v3_online.py
"""

import torch
import triton
import triton.language as tl

from _shared import selftest

NAME = "v3 online（分块规约）"


@triton.jit
def online_softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,     # 块宽，和 n_cols 无关，可以固定成 1024/2048
):
    row_idx = tl.program_id(axis=0)
    in_row = in_ptr + row_idx * in_row_stride
    out_row = out_ptr + row_idx * out_row_stride

    # 循环携带的状态。用 [1] 形状而不是 python 标量，避免循环内外类型不一致。
    m = tl.zeros([1], dtype=tl.float32) - float("inf")
    d = tl.zeros([1], dtype=tl.float32)

    # ---- 第一遍：扫一遍求出 (max, sum)，状态只有两个标量 ----
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        # other=-inf：越界 lane 不会成为 max，且 exp(-inf)=0 不污染 sum
        blk = tl.load(in_row + offs, mask=mask, other=-float("inf"))

        m_new = tl.maximum(m, tl.max(blk, axis=0))
        # ★ 修正项：把按旧 m 归一化的 d 换算到新的 m_new 基准上
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(blk - m_new), axis=0)
        m = m_new

    # ---- 第二遍：再读一遍 x，算出结果写回 ----
    # 这一遍是 online 算法的代价。x 已经不在寄存器里了，必须重新从显存读
    # （L2 里可能还有，所以实际没有「多读一遍显存」那么贵，见 README §3.4）
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        blk = tl.load(in_row + offs, mask=mask, other=-float("inf"))
        tl.store(out_row + offs, tl.exp(blk - m) / d, mask=mask)


def softmax(x: torch.Tensor, block_size: int = 1024):
    """block_size 和 n_cols **无关** —— 这是和 v1 最本质的区别。

    n_cols 可以是任意大，寄存器占用固定。
    """
    assert x.dim() == 2 and x.is_cuda
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    online_softmax_kernel[(n_rows,)](
        out, x,
        x.stride(0), out.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out


if __name__ == "__main__":
    selftest(softmax, NAME)

    # v1 在这个规模上根本编译不出来，v3 可以
    print("\n[能力对比] 超宽行：n_cols = 1<<18 = 262144")
    x = torch.randn(64, 1 << 18, device="cuda")
    from _shared import bench, check, nbytes, report_bw
    got = softmax(x)
    check(got, torch.softmax(x, axis=1), "v3 n_cols=262144", rtol=1e-5, atol=1e-6)
    ms = bench(lambda: softmax(x), warmup=5, iters=20)
    report_bw("v3 n_cols=262144", ms, nbytes(64, 1 << 18))
    print("  → v1 在这个行宽上会把 BLOCK_SIZE 设成 262144，编译卡死。")
    print("    v3 的 BLOCK_SIZE 固定 1024，行宽再大也不影响寄存器占用。")

    print("\n[练习] 试试看：")
    print("  1. 去掉 `d * tl.exp(m - m_new)` 里的修正因子，什么输入下会算错？")
    print("     （提示：需要后面的块比前面的块有更大的最大值）")
    print("  2. block_size 取 256/1024/4096，对性能的影响（bench.py 有扫描）")
    print("  3. 进阶：把第二遍读 x 换成「第一遍时把块缓存在 shared memory」，")
    print("     n_cols 不太大时能省掉重读 —— 这就是 FlashAttention 的做法")
    print("  4. 把这个 kernel 改成算 attention 的 softmax(QK^T)V，")
    print("     在第二遍里直接乘 V 并累加 —— 你就写出 FlashAttention 前向了")
