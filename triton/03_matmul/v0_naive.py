"""v0：最直白的分块矩阵乘 —— 「能跑」版本

学习目标：
  - 掌握 2D 分块：两个维度的 offsets、广播成 2D 指针矩阵
  - 学会 K 维循环 + 寄存器累加器（accumulator）
  - 用 tl.dot 调 Tensor Core

原理：C[M,N] = A[M,K] @ B[K,N]。把 C 切成 BLOCK_M × BLOCK_N 的小块，
每个 program 负责算一块。算一块需要 A 的 BLOCK_M 行和 B 的 BLOCK_N 列，
沿 K 维分批读进来，在寄存器里累加。

这一版故意写得朴素：
  - pid 到 (pid_m, pid_n) 用最直观的 **row-major** 映射
  - 分块取 64×64×32（第一次写通常都这么填）

v1 会在这两处分别做优化，并给出各自的贡献。

跑法：
  python v0_naive.py
"""

import torch
import triton
import triton.language as tl

from _shared import selftest

NAME = "v0 朴素版"


@triton.jit
def matmul_naive_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # ---- ① row-major 映射：最直观的写法 ----
    # 相邻的 pid 沿着 C 的同一行往右走。
    # 问题：一行走完要用到 B 的**全部**列块，B 被完整扫一遍；
    #       而 A 只用了 BLOCK_M 行。cache 的复用机会被浪费。
    #       v1 的 group-major 就是来修这个的。
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # ---- ② 本 program 负责的行/列下标 ----
    # `% M` / `% N` 让越界下标回绕到合法地址：地址计算不会跑飞，
    # 读到的垃圾值由后面的 mask / 尾块处理挡掉。
    # （只有写回时必须用真正的 mask，见 ⑤）
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # ---- ③ 构造 2D 指针矩阵 ----
    # [:, None] 和 [None, :] 是广播：把两个 1D 向量拼成 2D。
    # a_ptrs 的 shape 是 [BLOCK_M, BLOCK_K]，每个元素是一个地址。
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # ---- ④ K 维循环，累加器常驻寄存器 ----
    # 关键：accumulator 从头到尾待在寄存器里，K 循环期间**一次都不写显存**。
    # 用 fp32 而不是 fp16 累加，避免累加误差（Tensor Core 本身就是
    # fp16 乘 + fp32 累加，这里是顺着硬件来）。
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # K 不被 BLOCK_K 整除时尾块要 mask 掉，补 0 不影响求和
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)

        # tl.dot 编译成 Tensor Core 指令（sm_80 上是 HMMA）。
        # 这是整个 kernel 唯一真正做计算的地方。
        accumulator = tl.dot(a, b, accumulator)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)

    # ---- ⑤ 写回，必须用真正的边界 mask（不能再用 % 回绕）----
    # 读的时候回绕只是读到无关数据，写的时候回绕会**覆盖别人的结果**。
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=4):
    assert a.shape[1] == b.shape[0], "K 维不匹配"
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    matmul_naive_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return c


if __name__ == "__main__":
    selftest(matmul, NAME)
    print("\n[练习] 试试看：")
    print("  1. 把 accumulator 的 dtype 改成 tl.float16，看 max_abs_err 涨多少")
    print("  2. 把 ⑤ 的 c_mask 换成 `% M/% N` 回绕，用 M=300 跑 —— 哪里会错？")
    print("  3. 手动把 BLOCK_M/N/K 改成 128/256/64（num_warps 也要跟着调到 8），")
    print("     看性能变化，然后去 v1 看调好的版本")
