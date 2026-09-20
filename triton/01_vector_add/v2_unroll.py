"""v2：一个 program 处理多个 tile（循环展开）—— 减少 program 数量

原理：v0 里 n=16M、BLOCK_SIZE=1024 会起 16384 个 program。每个 program
都要走一遍「算 pid → 算地址 → 发访存 → 退出」。program 太多时，
调度开销和重复的地址计算会占掉一部分时间。

做法：让每个 program 连着处理 UNROLL 个 tile，program 数降到 1/UNROLL。
用 `tl.static_range` 而不是 `range` —— static_range 是编译期完全展开的，
循环体会被复制 UNROLL 份，**多个 tile 的访存可以在指令级并行（ILP）**，
访存延迟互相重叠。这是 memory-bound kernel 唯一还有点空间的地方：
提高 memory level parallelism（同时在飞的访存请求数）。

注意地址仍然保持「每个 tile 内部连续」，所以向量化没丢 ——
和 v1 的区别就在这里：v2 只改了工作划分的粒度，没改访存模式。

跑法：
  python v2_unroll.py
"""

import torch
import triton
import triton.language as tl

from _shared import selftest

NAME = "v2 循环展开"


@triton.jit
def add_unroll_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    UNROLL: tl.constexpr,       # 每个 program 处理几个 tile（编译期常量）
):
    pid = tl.program_id(axis=0)
    # 本 program 负责的区间是 [base, base + BLOCK_SIZE*UNROLL)
    base = pid * BLOCK_SIZE * UNROLL

    # static_range：编译期展开，循环体被复制 UNROLL 份。
    # 换成普通 range 的话是运行期循环，拿不到 ILP。
    for i in tl.static_range(UNROLL):
        offsets = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024, unroll: int = 4):
    assert x.is_cuda and y.is_cuda and x.shape == y.shape
    assert x.is_contiguous() and y.is_contiguous()
    out = torch.empty_like(x)
    n = x.numel()

    # 每个 program 吃 block_size*unroll 个元素，所以 grid 要相应缩小
    grid = (triton.cdiv(n, block_size * unroll),)
    add_unroll_kernel[grid](
        x, y, out, n, BLOCK_SIZE=block_size, UNROLL=unroll
    )
    return out


if __name__ == "__main__":
    selftest(add, NAME)
    print("\n[练习] 试试看：")
    print("  1. UNROLL 取 1/2/4/8/16，找拐点 —— 太大会怎样？（提示：看寄存器数）")
    print("  2. 把 tl.static_range 换成普通 range，性能变化多少？")
    print("  3. 用 kernel_info() 看 UNROLL=1 和 UNROLL=8 的 n_regs 差别")
