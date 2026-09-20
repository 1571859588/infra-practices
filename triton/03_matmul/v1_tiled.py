"""v1：调好分块 + group-major 排序

在 v0 基础上改两件事：

1. **分块尺寸**：64×64×32 → 128×256×64，`num_warps` 4 → 8。
   这是 compute-bound kernel 的第一优化项，4096² 上实测 1.47 倍差距。
   直觉：分块越大，每读进来的一块数据能做的乘加越多（算术强度更高），
   但寄存器和 shared memory 是有限的，开太大就装不下。

   ⚠️ **这套参数只在大矩阵上对**：实测 v1 在 1024²/2048² 上**打不过 v0**
   （45% vs 71%、49% vs 61% of cuBLAS），只有 4096² 才反超（88% vs 55%）。
   大分块要足够多的分块数才能填满 108 个 SM；1024² 用 128×256 只切出
   4×8=32 个 program，一大半 SM 是空的。见 README §3.2。

2. **group-major 排序**：改变 program 走查 C 的顺序，提高 L2 命中率。
   ⚠️ 实测在 A100 + 4096² 上**完全无效**，到 8192² 才有 +2.8%。
   原因和完整分析见 README §3.4 —— 这是本练习最有教育意义的一个负结果。

跑法：
  python v1_tiled.py
"""

import torch
import triton
import triton.language as tl

from _shared import selftest

NAME = "v1 调优分块"


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """C[M,N] = A[M,K] @ B[K,N]，一个 program 算 C 的一个 [BLOCK_M, BLOCK_N] 分块。"""

    # ---- ① group-major 排序（唯一和 v0 不同的逻辑）----
    # v0 的 row-major 顺序下，相邻 pid 沿着 C 的一行走，会把 B 的一整行块
    # 反复读进来；换成「先走完一个 GROUP_M 高的横条」，同一组 program 复用
    # 同样的 A 行块和 B 列块，理论上 L2 命中率更高。
    # 这是纯调度技巧，不改变算的东西，也不影响正确性。
    #
    # ⚠️ 但效果强烈依赖硬件和规模：A100 的 L2 有 40 MB，4096x4096 的 B 矩阵
    #    才 32 MB，整个装得下 —— 实测这个尺寸上 GROUP_M 毫无影响。
    #    到 8192（B = 128 MB）才看出 +2.8%。见 README §3.4。
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)   # 最后一组可能不满
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- ② 之后和 v0 完全一样 ----
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64,
           GROUP_M=8, num_warps=8, num_stages=3):
    assert a.shape[1] == b.shape[0], "K 维不匹配"
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # 1D grid：总共 ceil(M/BM) * ceil(N/BN) 个分块
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=num_warps, num_stages=num_stages,
    )
    return c


if __name__ == "__main__":
    selftest(matmul, NAME)
    print("\n[练习] 试试看：")
    print("  1. GROUP_M 取 1（等于退回 row-major），在 4096 和 8192 上分别对比")
    print("  2. 用 ncu 看 Tensor Core 利用率和 L2 命中率：")
    print("     sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active")
    print("     lts__t_sector_hit_rate.pct")
    print("     （Docker 方案见 ../../综合练习/vector_mul2/README.md §5）")
    print("  3. num_stages 取 2/3/4/5，观察 shared memory 占用和性能的权衡")
