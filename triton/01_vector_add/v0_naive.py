"""v0：最朴素的向量加法 —— Triton 的 "Hello World"

学习目标：
  - 掌握 Triton 最基础的三件套：program_id / arange / mask
  - 理解「以 block 为编程粒度」和 CUDA「以 thread 为粒度」的区别
  - 会用 grid lambda 计算需要多少个 program

原理：把长度 n 的数组切成 ceil(n/BLOCK_SIZE) 段，每段交给一个 program。
每个 program 内部是**向量化**思维：offsets 是长度 BLOCK_SIZE 的向量，
一条 tl.load 读一整段。

跑法：
  python v0_naive.py
"""

import torch
import triton
import triton.language as tl

from _shared import selftest

NAME = "v0 朴素版"


@triton.jit
def add_kernel(
    x_ptr,                      # *float32，输入 x
    y_ptr,                      # *float32，输入 y
    out_ptr,                    # *float32，输出
    n_elements,                 # int，元素总数（运行期变量）
    BLOCK_SIZE: tl.constexpr,   # int，每个 program 处理多少元素（编译期常量）
):
    # ① 我是第几个 program？
    #    对比 CUDA：这相当于 blockIdx.x。Triton 里没有 threadIdx ——
    #    block 内部怎么切给线程，是编译器的事，你不用管。
    pid = tl.program_id(axis=0)

    # ② 我负责哪一段下标？
    #    offsets 是一个长度为 BLOCK_SIZE 的**向量**，不是标量。
    #    连续的 lane 对应连续的地址 —— 这一点是后面能自动向量化的前提，
    #    v1_strided_bad.py 就是故意破坏它。
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # ③ 边界保护。
    #    n_elements 通常不被 BLOCK_SIZE 整除，最后一个 program 会越界。
    #    mask 为 False 的 lane 不读不写 —— 等价于 CUDA 里手写的 if (idx < n)，
    #    但不用自己写分支，也不会漏。
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024):
    """host 侧封装：分配输出、算 grid、launch。"""
    assert x.is_cuda and y.is_cuda and x.shape == y.shape
    assert x.is_contiguous() and y.is_contiguous()
    out = torch.empty_like(x)
    n = x.numel()

    # grid 是个 lambda，参数 meta 里能拿到 BLOCK_SIZE 等 constexpr。
    # cdiv = 向上取整除法，保证覆盖所有元素。
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)
    return out


if __name__ == "__main__":
    selftest(add, NAME)
    print("\n[练习] 试试看：")
    print("  1. 把 mask 去掉，用 n=1000 跑，会发生什么？（提示：加 compute-sanitizer）")
    print("  2. 改成 out = x * a + y（a 是 python float），需要改 kernel 签名吗？")
    print("  3. 把 BLOCK_SIZE 设成 100（非 2 的幂），能编译吗？为什么？")
