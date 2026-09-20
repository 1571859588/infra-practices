"""v1：融合 Softmax —— 一个 program 处理一整行

学习目标：
  - 掌握 block 内规约：tl.max / tl.sum（CUDA 里要手写 shared memory + shuffle）
  - 理解**算子融合**为什么能提速：一次读写完成本来要好几轮的事
  - 学会用 mask + other= 处理非 2 次幂的行宽
  - 知道 num_warps 该怎么选

原理：`BLOCK_SIZE = next_power_of_2(n_cols)`，让**一整行整个装进寄存器**。
行在片上之后，减最大值、exp、求和、除法全部在寄存器里完成，
中间结果一次都不写回显存 —— 访存量从 v0 的 ~8MN 降到 2MN。

前提条件很硬：`BLOCK_SIZE >= n_cols`。行太宽装不下就得换 v3_online。

跑法：
  python v1_fused.py
"""

import torch
import triton
import triton.language as tl

from _shared import selftest

NAME = "v1 融合版（一 program 一行）"


@triton.jit
def softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """一个 program 处理一整行。

    前提：BLOCK_SIZE >= n_cols，即一行能整个装进寄存器。
    这是「融合」的关键 —— 整行都在片上，中间结果不用写回显存。
    """
    row_idx = tl.program_id(axis=0)

    # 定位到本行起点。用 stride 而不是 n_cols，这样非 contiguous 的输入也对。
    row_start = in_ptr + row_idx * in_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # other=-inf：越界的 lane 填负无穷。
    # 这样它们既不会成为 max，exp(-inf)=0 也不会污染 sum —— 一举两得。
    row = tl.load(row_start + col_offsets, mask=mask, other=-float("inf"))

    # ---- 数值稳定的 softmax ----
    # 先减最大值再 exp，否则大数会 exp 出 inf。
    # tl.max 是 block 内规约：CUDA 里要写 shared memory + __shfl_down_sync，
    # Triton 一行搞定，编译器自己生成规约代码。
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator

    out_row_start = out_ptr + row_idx * out_row_stride
    tl.store(out_row_start + col_offsets, softmax_out, mask=mask)


def pick_num_warps(block_size: int) -> int:
    """行越宽，给越多线程去并行做规约。一个 warp = 32 线程。

    这是个启发式，不是最优解 —— 想系统化就用 @triton.autotune。
    """
    if block_size >= 4096:
        return 16
    if block_size >= 2048:
        return 8
    return 4


def softmax(x: torch.Tensor):
    """host 侧封装。x 必须是 2D。"""
    assert x.dim() == 2 and x.is_cuda
    n_rows, n_cols = x.shape

    # BLOCK_SIZE 必须是 2 的幂，且 >= n_cols（一行要整个装下）
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    out = torch.empty_like(x)
    # grid = 行数：每行一个 program，彼此完全独立，不需要任何同步
    softmax_kernel[(n_rows,)](
        out, x,
        x.stride(0), out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=pick_num_warps(BLOCK_SIZE),
    )
    return out


if __name__ == "__main__":
    selftest(softmax, NAME)

    from _shared import kernel_info
    print("\n[编译产物] num_warps 对寄存器 / shared memory 的影响")
    out = torch.empty(4, 4096, device="cuda")
    xin = torch.randn(4, 4096, device="cuda")
    for nw in (4, 8, 16):
        c = softmax_kernel.warmup(
            out, xin, xin.stride(0), out.stride(0), 4096,
            BLOCK_SIZE=4096, num_warps=nw, grid=(1,),
        )
        kernel_info(c, f"BLOCK=4096 warps={nw:<2}")

    print("\n[练习] 试试看：")
    print("  1. 去掉 `- tl.max(...)`，用 x*100 跑，观察 nan 是怎么出现的")
    print("  2. 把 other=-inf 改成 other=0，哪种 shape 会算错？为什么？")
    print("  3. 把 num_warps 固定成 1，看性能掉多少 —— 规约的并行度没了")
    print("  4. 试一个 n_cols = 1<<18 的输入，会发生什么？（然后看 v3_online.py）")
