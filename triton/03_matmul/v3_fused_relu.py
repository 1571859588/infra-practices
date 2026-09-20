"""v3：融合 epilogue —— `relu(A @ B + bias)`，Triton 写 matmul 的真正理由

**别指望在裸 matmul 上打赢 cuBLAS**（v0/v1 的实测是 60~88%）。cuBLAS 对每个
尺寸段都有手调 kernel 和启发式选择，这是 NVIDIA 投了十几年的东西。

那为什么还要写 Triton matmul？因为 **cuBLAS 只会给你一个 C = A@B**。
后面接的 bias、激活、量化、scale，每一个都是独立 kernel，
每一个都要把整个 C 矩阵完整读一遍写一遍。

  torch 写法（3 个 kernel）：
      C = A @ B              cuBLAS：写 C          (MN 写)
      C = C + bias           kernel 2：读 C 写 C   (MN 读 + MN 写)
      C = relu(C)            kernel 3：读 C 写 C   (MN 读 + MN 写)
                             → 额外 4MN 字节的访存，全是白搬的

  Triton 写法（1 个 kernel）：
      accumulator 还在寄存器里的时候，顺手 + bias、顺手 relu，再写出去
                             → 额外访存 = 0

原理和练习 02 的融合完全一样，只是这次融的是 matmul 的**尾巴（epilogue）**。
这也是 FlashAttention、量化 GEMM、LoRA 融合这些工作的共同套路。

⚠️ **测量陷阱（我第一版就踩了）**：如果直接拿
    「triton 融合版」 vs 「torch.matmul + bias + relu」
比，测到的是两件事的叠加 —— epilogue 的收益，**加上** 我的 matmul 打不过
cuBLAS 的亏损。小矩阵上后者远大于前者，结果融合版反而慢，结论完全跑偏。

要隔离 epilogue 的贡献，必须固定 matmul 实现不变：
    A. triton matmul + torch bias + torch relu    ← 同一个 matmul，不融合
    B. triton matmul 融合 bias + relu             ← 同一个 matmul，融合
A/B 的差才是 epilogue 融合的真实收益。下面两个基线都会打印出来。

跑法：
  python v3_fused_relu.py
"""

import torch
import triton
import triton.language as tl

from _shared import atol_for, bench, check, nflops, report_flops

NAME = "v3 融合 epilogue"


@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # ---- 前面和 v1 一模一样 ----
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

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

    # ================== ★ epilogue：这里是全部的不同 ==================
    # accumulator 此刻还在寄存器里，fp32 精度。直接在上面做后处理，
    # 一个字节都不用过显存。
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # bias 是 [N]，只读 BLOCK_N 个元素（相比 C 的 BLOCK_M*BLOCK_N 可以忽略）
    bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    accumulator = accumulator + bias[None, :].to(tl.float32)

    # relu：在 fp32 上做，避免先转 fp16 再比较带来的额外舍入
    accumulator = tl.maximum(accumulator, 0.0)
    # ==================================================================

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_relu(a, b, bias, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64,
                GROUP_M=8, num_warps=8, num_stages=3):
    """返回 relu(a @ b + bias)，bias 形状 [N]。"""
    assert a.shape[1] == b.shape[0], "K 维不匹配"
    M, K = a.shape
    _, N = b.shape
    assert bias.shape == (N,), f"bias 应为 [{N}]"
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    matmul_relu_kernel[grid](
        a, b, c, bias,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=num_warps, num_stages=num_stages,
    )
    return c


def torch_unfused(a, b, bias):
    """全 torch 写法：cuBLAS gemm + bias + relu，3 个 kernel。

    这是「实际项目里会写出来的代码」，但拿它当 epilogue 融合的对照组
    是错的 —— 它换掉的不只是 epilogue，还把 matmul 换成了 cuBLAS。
    """
    return torch.relu(torch.matmul(a, b) + bias)


def triton_unfused(a, b, bias, **kw):
    """★ 隔离 epilogue 的正确对照组：matmul 用和融合版**完全相同**的
    triton kernel，只是 bias 和 relu 走 torch 的独立 kernel。

    和 matmul_relu() 的唯一差别就是 epilogue 融没融。
    """
    import v1_tiled
    return torch.relu(v1_tiled.matmul(a, b, **kw) + bias)


def main():
    torch.manual_seed(0)
    print("=" * 70)
    print(f"03_matmul / {NAME}")
    print("=" * 70)

    print("\n[正确性]")
    ok = True
    for (m, n, k) in ((256, 256, 256), (300, 500, 177), (1024, 1024, 1024)):
        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        bias = torch.randn(n, device="cuda", dtype=torch.float16)
        ok &= check(matmul_relu(a, b, bias), torch_unfused(a, b, bias),
                    f"M,N,K = {m},{n},{k}".ljust(22),
                    rtol=1e-2, atol=atol_for(k))

    print("\n[性能] relu(A@B + bias)，方阵 fp16")
    print("  ★ 只有「triton 不融合」→「triton 融合」这一对是隔离了 epilogue 的；")
    print("    「torch 3 kernels」那行还叠加了 cuBLAS vs 我的 matmul 的差距。")
    for sz in (512, 1024, 2048, 4096):
        a = torch.randn((sz, sz), device="cuda", dtype=torch.float16)
        b = torch.randn((sz, sz), device="cuda", dtype=torch.float16)
        bias = torch.randn(sz, device="cuda", dtype=torch.float16)
        nf = nflops(sz, sz, sz)

        ms_t = bench(lambda: torch_unfused(a, b, bias), warmup=10, iters=50)
        ms_u = bench(lambda: triton_unfused(a, b, bias), warmup=10, iters=50)
        ms_f = bench(lambda: matmul_relu(a, b, bias), warmup=10, iters=50)
        report_flops(f"torch 3 kernels {sz}", ms_t, nf, extra="(cuBLAS，仅供参考)")
        report_flops(f"triton 不融合 {sz}", ms_u, nf)
        report_flops(f"triton 融合 {sz}", ms_f, nf,
                     extra=f"epilogue 融合带来 {ms_u / ms_f:.2f}x")

    print("\n  → epilogue 省掉的是 4MN 字节访存，和 K 无关；matmul 的计算量是")
    print("    2MNK。所以方阵越大，epilogue 占比越低、融合收益越小。")
    print("    收益最大的场景是「扁矩阵」：M 或 N 大而 K 小（推理里很常见）。")

    print("\n[扁矩阵] M=4096 N=4096 固定，K 变小 —— epilogue 占比升高")
    for k in (4096, 1024, 256, 64):
        a = torch.randn((4096, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, 4096), device="cuda", dtype=torch.float16)
        bias = torch.randn(4096, device="cuda", dtype=torch.float16)
        nf = nflops(4096, 4096, k)
        ms_u = bench(lambda: triton_unfused(a, b, bias), warmup=10, iters=50)
        ms_f = bench(lambda: matmul_relu(a, b, bias), warmup=10, iters=50)
        report_flops(f"K={k:<5} 不融合", ms_u, nf)
        report_flops(f"K={k:<5} 融合", ms_f, nf, extra=f"{ms_u / ms_f:.2f}x")

    print(f"\n{'全部通过' if ok else '有用例失败'}")
    print("\n[练习] 试试看：")
    print("  1. 把 relu 换成 gelu（tanh 近似），融合收益变化吗？为什么？")
    print("  2. 加一个 residual：relu(A@B + bias) + residual，")
    print("     residual 是 [M,N] —— 这次要多读 MN，收益还剩多少？")
    print("  3. 和 torch.compile(torch_unfused) 比一比，inductor 能融到什么程度？")
    print("  4. 把输出改成 int8（带 scale 的量化 epilogue），")
    print("     这时写出去的字节数减半 —— 融合收益会变大还是变小？")


if __name__ == "__main__":
    main()
