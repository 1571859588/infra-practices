"""v1：反面教材 —— 打散访存模式，看合并访存和向量化值多少钱

**这个变体故意写慢**，用来回答一个问题：
「Triton 自动做的向量化加载（128-bit / .v4.b32）到底有多重要？」

原理对比：

  v0（好）  program 0 拿 [0, 1023]，program 1 拿 [1024, 2047] …
            lane i 的地址 = base + 4i，连续
            → 编译器给每个线程分配连续的 4 个元素，发 `ld.global.v4.b32`
            → 一个 warp 的 32 条 lane 落在同一批 128B cache line 上（合并访存）

  v1（坏）  program 0 拿 [0, P, 2P, …]，program 1 拿 [1, 1+P, …]（P = program 数）
            lane i 的地址 = base + 4·i·P，间隔 P 个元素（这里 P=16384，即 64 KB）
            → 每个线程要的元素彼此不相邻，无法凑成 128-bit 一次读
            → 一个 warp 的 32 条 lane 落在 32 个不同的 cache line 上
            → 同样的数据量，DRAM 事务数放大到 ~32 倍

**注意这一个改动同时破坏了两件事**：向量化（per-thread 连续）和
合并访存（per-warp 连续）。在 Triton 里你没法只破坏其中一个 ——
向量化是编译器在「访存模式允许时」自动做的，你控制的是模式，不是指令。
所以这个实验测的是「访存模式」的总价值，比单纯的向量化更大。

bench.py 会把两个版本的 PTX 访存指令都打出来，能直接看到
`.v4.b32` 变成 `.b32`。

跑法：
  python v1_strided_bad.py
"""

import torch
import triton
import triton.language as tl

from _shared import selftest

NAME = "v1 跨步访存（反面教材）"


@triton.jit
def add_strided_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    n_programs,                 # 运行期变量：总共有多少个 program
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # ★ 唯一的改动：从「连续分段」改成「循环分配（cyclic）」。
    #   program pid 负责下标 pid, pid+P, pid+2P, ...
    #   所有 program 合起来仍然恰好覆盖 [0, n)，结果完全正确 —— 只是慢。
    offsets = pid + tl.arange(0, BLOCK_SIZE) * n_programs
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024):
    assert x.is_cuda and y.is_cuda and x.shape == y.shape
    assert x.is_contiguous() and y.is_contiguous()
    out = torch.empty_like(x)
    n = x.numel()

    # program 数和 v0 完全一样，这样对比里唯一的变量就是访存模式本身
    n_programs = triton.cdiv(n, block_size)
    add_strided_kernel[(n_programs,)](
        x, y, out, n, n_programs, BLOCK_SIZE=block_size
    )
    return out


if __name__ == "__main__":
    selftest(add, NAME)
    print("\n  → 结果是对的，但慢得多。正确性和性能是两件独立的事，")
    print("    「跑对了」不代表「写好了」—— 这是 GPU 编程最基本的一课。")
    print("\n[练习] 试试看：")
    print("  1. 用 ncu 对比 v0 和 v1 的 l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
    print("     （实际发出的 32B sector 数），看是不是差了一个数量级")
    print("  2. 把 n_programs 换成一个小值（比如 32），跨步变小，性能会回来多少？")
