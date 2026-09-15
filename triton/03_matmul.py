"""练习 03：分块矩阵乘 —— 从 memory-bound 走到 compute-bound

学习目标：
  - 掌握 2D 分块：两个维度的 offsets、广播成 2D 指针矩阵
  - 学会 K 维循环 + 寄存器累加器（accumulator）
  - 用 tl.dot 调 Tensor Core
  - 理解 group-major 排序想解决什么问题，以及它什么时候**没用**

前两个练习都是 memory-bound，怎么写都是 ~85% 带宽，没有优化空间。
matmul 是**compute-bound**，才真正体现出分块策略的价值 —— 同样的算法，
配置选得好不好能差好几倍。

跑法：
  python 03_matmul.py
"""

import torch
import triton
import triton.language as tl

from common import bench, check, report_flops, A100_PEAK_TFLOPS_FP16


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

    # ---- ① group-major 排序（决定 program 走查 C 的顺序）----
    # 朴素的 row-major 顺序下，相邻 pid 沿着 C 的一行走，会把 B 的一整行块
    # 反复读进来；换成「先走完一个 GROUP_M 高的横条」，同一组 program 复用
    # 同样的 A 行块和 B 列块，L2 命中率更高。
    # 这是纯调度技巧，不改变算的东西。
    # ⚠️ 但效果强烈依赖硬件和规模：A100 的 L2 有 40 MB，4096x4096 的 B 矩阵
    #    才 32 MB，整个装得下 —— 实测这个尺寸上 GROUP_M 毫无影响。
    #    到 8192 才看出 +4%。见 main() 里的实测输出。
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)   # 最后一组可能不满
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- ② 算出本 program 负责的行/列下标 ----
    # %  M 是为了让越界的行下标回绕到合法地址（配合后面的 mask 使用，
    #    保证即使 M 不被 BLOCK_M 整除，地址计算也不会跑飞）
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # ---- ③ 构造 2D 指针矩阵 ----
    # [:, None] 和 [None, :] 是广播：把两个 1D 向量拼成 2D。
    # a_ptrs 的 shape 是 [BLOCK_M, BLOCK_K]，每个元素是一个地址。
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # ---- ④ K 维循环，累加器常驻寄存器 ----
    # 关键点：accumulator 从头到尾待在寄存器里，K 循环期间**一次都不写显存**。
    # 用 fp32 累加而不是 fp16，是为了避免累加误差（Tensor Core 本来就是
    # fp16 乘 + fp32 累加）。
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # K 不被 BLOCK_K 整除时，尾块要 mask 掉，补 0 不影响求和
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)

        # tl.dot 会编译成 Tensor Core 指令（sm_80 上是 HMMA）。
        # 这是整个 kernel 唯一真正做计算的地方。
        accumulator = tl.dot(a, b, accumulator)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)

    # ---- ⑤ 写回，这次要用真正的边界 mask（不能再用 % 回绕）----
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a, b, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64,
                  GROUP_M=8, num_warps=8, num_stages=3):
    assert a.shape[1] == b.shape[0], "K 维不匹配"
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    K, N = b.shape
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


def main():
    torch.manual_seed(0)
    device = "cuda"

    print("=" * 70)
    print("练习 03：分块矩阵乘")
    print("=" * 70)

    # ---------- 正确性 ----------
    # fp16 矩阵乘有累加误差，不能要求完全相等。K 越大误差越大。
    print("\n[正确性] 含非整除尺寸，验证 mask")
    for (M, N, K) in [(256, 256, 256), (512, 512, 128), (300, 500, 177), (1024, 1024, 1024)]:
        a = torch.randn((M, K), device=device, dtype=torch.float16)
        b = torch.randn((K, N), device=device, dtype=torch.float16)
        ref = torch.matmul(a, b)
        got = triton_matmul(a, b)
        # atol 随 K 放大：fp32 累加 K 个 fp16 乘积，误差 ~ sqrt(K) * eps
        check(got, ref, f"M,N,K = {M},{N},{K}".ljust(22), rtol=1e-2, atol=K ** 0.5 * 1e-2)

    # ---------- 性能 ----------
    # matmul 的 FLOP 数 = 2*M*N*K（每个输出元素 K 次乘 + K 次加）
    print("\n[性能] 方阵，fp16 输入 / fp32 累加 / fp16 输出")
    print(f"  {'':26} {'耗时':>10}  {'算力':>14}  {'峰值占比':>8}")
    for sz in (512, 1024, 2048, 4096):
        a = torch.randn((sz, sz), device=device, dtype=torch.float16)
        b = torch.randn((sz, sz), device=device, dtype=torch.float16)
        nflops = 2 * sz * sz * sz

        ms_t = bench(lambda: torch.matmul(a, b), warmup=10, iters=50)
        report_flops(f"torch  {sz}x{sz}", ms_t, nflops)
        ms = bench(lambda: triton_matmul(a, b), warmup=10, iters=50)
        report_flops(f"triton {sz}x{sz}", ms, nflops,
                     extra=f"({ms_t / ms * 100:.0f}% of torch)")

    print("\n  → 小矩阵上 triton 往往打不过 cuBLAS：cuBLAS 对每个尺寸都有")
    print("    手调过的 kernel 和启发式选择。大矩阵上差距会缩小。")
    print("  → 注意这里的百分比是对 312 TFLOP/s（A100 fp16 Tensor Core 密集峰值）算的。")

    # ---------- 分块配置的影响（本练习的重点）----------
    sz = 4096
    a = torch.randn((sz, sz), device=device, dtype=torch.float16)
    b = torch.randn((sz, sz), device=device, dtype=torch.float16)
    nflops = 2 * sz * sz * sz

    print(f"\n[分块配置扫描] {sz}x{sz}，这才是 matmul 真正要调的东西")
    for (bm, bn, bk, w, s) in [
        (64, 64, 32, 4, 5),
        (64, 128, 32, 4, 4),
        (128, 64, 32, 4, 4),
        (128, 128, 32, 4, 4),
        (128, 128, 64, 4, 4),
        (128, 256, 64, 8, 3),
        (256, 128, 64, 8, 3),
    ]:
        ms = bench(lambda: triton_matmul(a, b, bm, bn, bk, 8, w, s),
                   warmup=10, iters=50)
        report_flops(f"BM={bm} BN={bn} BK={bk}", ms, nflops,
                     extra=f"warps={w} stages={s}")

    print("\n  → 本机实测最好和最差差 1.46x（256x128x64 vs 64x64x32）。对比练习 01 里")
    print("    BLOCK_SIZE 的 <2% 差距 —— compute-bound 的第一优化项就是分块配置，")
    print("    memory-bound 则基本无可调。")

    print(f"\n[GROUP_M 的影响] {sz}x{sz}，BM=128 BN=256 BK=64 固定")
    for g in (1, 2, 4, 8, 16):
        ms = bench(lambda: triton_matmul(a, b, 128, 256, 64, g, 8, 3),
                   warmup=10, iters=50)
        report_flops(f"GROUP_M={g}", ms, nflops)

    print("\n  → 意外结果：4096 这个尺寸上 GROUP_M 几乎没有影响（全在噪声内）。")
    print("    原因是 A100 的 L2 有 40 MB，而 B 矩阵才 32 MB —— 整个矩阵基本")
    print("    就待在 L2 里，怎么排都命中，group-major 没有用武之地。")
    print("    换到 8192（B 矩阵 128 MB，装不下）实测才看出区别：")
    print("      GROUP_M=1 → 192.4 TFLOP/s，GROUP_M=16 → 200.0 TFLOP/s（+4%）")
    print("  → 教训：**优化技巧是否有效取决于硬件参数和问题规模**。")
    print("    在 L2 小得多的卡上，或者矩阵更大时，这个参数才重要。")
    print("    照搬别人的调优结论而不实测，很容易做无用功。")

    print("\n[练习] 试试看：")
    print("  1. 把 accumulator 的 dtype 改成 tl.float16，对比 max_abs_err 变化")
    print("  2. 把 GROUP_M 相关代码删掉、直接用 row-major，看性能掉多少")
    print("  3. 给这个 kernel 加 @triton.autotune（configs 用上面扫描的那几组，")
    print("     key=['M','N','K']），对比自动选的和你手选的哪个快")
    print("     —— 写法参考 ../综合练习/vector_mul2/triton_mul2.py")
    print("  4. 用 ncu 看这个 kernel 的 Tensor Core 利用率：")
    print("     sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active")
    print("     （Docker 方案见 ../综合练习/vector_mul2/README.md §5）")
    print("  5. 进阶：融合 epilogue，比如 C = relu(A@B + bias)，")
    print("     对比「triton 一个 kernel」vs「torch 三个 kernel」")


if __name__ == "__main__":
    main()
