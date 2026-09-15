"""练习 02：融合 Softmax —— Triton 真正的价值所在

学习目标：
  - 掌握 block 内规约：tl.max / tl.sum（CUDA 里要手写 shared memory + shuffle）
  - 理解**算子融合**为什么能提速：一次读写完成本来要好几轮的事
  - 学会用 mask + other= 处理非 2 次幂的行宽
  - 知道 num_warps 该怎么选

这是本目录最重要的一个练习：向量加法那种单算子，Triton 只能追平 torch；
只有融合多个算子时，Triton 才会**显著更快**。

跑法：
  python 02_fused_softmax.py
"""

import torch
import triton
import triton.language as tl

from common import bench, check, report_bw, kernel_info


@triton.jit
def softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """一个 program 处理**一整行**。

    前提：BLOCK_SIZE >= n_cols，即一行能整个装进寄存器。
    这是"融合"的关键 —— 整行都在片上，中间结果一次都不用写回显存。
    """
    row_idx = tl.program_id(axis=0)

    # 定位到本行起点
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


def triton_softmax(x: torch.Tensor):
    """host 侧封装。x 必须是 2D。"""
    assert x.dim() == 2 and x.is_cuda
    n_rows, n_cols = x.shape

    # BLOCK_SIZE 必须是 2 的幂，且 >= n_cols（一行要整个装下）
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # num_warps 启发式：行越宽，给越多线程去并行做规约。
    # 一个 warp = 32 线程，num_warps=8 就是 256 线程处理这一行。
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    out = torch.empty_like(x)
    # grid = 行数：每行一个 program，彼此完全独立，不需要任何同步
    softmax_kernel[(n_rows,)](
        out, x,
        x.stride(0), out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out


def naive_softmax(x: torch.Tensor):
    """用 torch 算子手工拼出 softmax —— 故意不用融合版，看看差距。

    每一步都是一个独立 kernel，每个 kernel 都要完整读一遍、写一遍显存：
      read  x            (MN)
      write z = x - max  (MN)   ← 中间结果落显存
      read  z            (MN)
      write num = exp(z) (MN)   ← 中间结果落显存
      ...
    这就是"没融合"的代价。
    """
    z = x - x.max(dim=1, keepdim=True)[0]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1, keepdim=True)
    return numerator / denominator


def main():
    torch.manual_seed(0)
    device = "cuda"

    print("=" * 70)
    print("练习 02：融合 Softmax")
    print("=" * 70)

    # ---------- 正确性 ----------
    print("\n[正确性] 含非 2 次幂列宽，验证 mask + other=-inf")
    for shape in [(4, 128), (128, 781), (1823, 781), (64, 1)]:
        x = torch.randn(*shape, device=device)
        ref = torch.softmax(x, axis=1)
        check(triton_softmax(x), ref, f"shape={str(shape):<12}", rtol=1e-5, atol=1e-6)

    # 极端值：不减最大值的话这里会 inf/nan
    print("\n[正确性] 数值稳定性（大数值输入）")
    x = torch.randn(64, 512, device=device) * 100
    ref = torch.softmax(x, axis=1)
    got = triton_softmax(x)
    check(got, ref, "x * 100      ", rtol=1e-5, atol=1e-6)
    print(f"  含 nan/inf: {bool(torch.isnan(got).any() or torch.isinf(got).any())}")

    # ---------- 性能 ----------
    M, N = 4096, 4096
    x = torch.randn(M, N, device=device)
    # 融合版只读一次、写一次 → 2 * M * N * 4 字节
    nbytes = 2 * M * N * 4

    print(f"\n[性能] shape = ({M}, {N})")
    ms = bench(lambda: naive_softmax(x))
    report_bw("torch 手工拼（未融合）", ms, nbytes)
    ms_torch = bench(lambda: torch.softmax(x, axis=1))
    report_bw("torch.softmax（已融合）", ms_torch, nbytes)
    ms_triton = bench(lambda: triton_softmax(x))
    report_bw("triton 融合版", ms_triton, nbytes)

    print(f"\n  → triton vs 手工拼算子：{bench(lambda: naive_softmax(x)) / ms_triton:.2f}x")
    print("    注意带宽是按「理想访存量」（读一次写一次）算的，所以未融合版会")
    print("    显得远低于峰值 —— 它实际搬的字节数是这个数的好几倍。")
    print("    torch.softmax 自己也是融合实现，所以和 triton 打平很正常。")

    # ---------- 行宽扫描 ----------
    print(f"\n[行宽扫描] M = {M} 固定，看 N 变化的影响")
    for N2 in (256, 1024, 4096, 16384):
        xx = torch.randn(M, N2, device=device)
        nb = 2 * M * N2 * 4
        ms = bench(lambda: triton_softmax(xx))
        bs = triton.next_power_of_2(N2)
        nw = 4 if bs < 2048 else (8 if bs < 4096 else 16)
        report_bw(f"triton N={N2:<6}", ms, nb, extra=f"BLOCK={bs}, warps={nw}")

    print("\n  → N=256 那行带宽只有 ~24%：总共才 4 MiB，GPU 还没忙起来就结束了，")
    print("    瓶颈是 launch 开销和 program 数量不足，不是访存。规模太小时")
    print("    带宽利用率这个指标本身就没意义。")
    print("  → 行越宽，一行要占的寄存器越多。N 大到一定程度（本例 >64K）")
    print("    一行装不进寄存器，这个「一 program 一行」的写法就失效了，")
    print("    必须改成 online softmax（分块规约），也就是 FlashAttention 的核心技巧。")

    # ---------- 编译产物 ----------
    print("\n[编译产物] 看 num_warps 对寄存器的影响")
    out = torch.empty(4, 4096, device=device)
    xin = torch.randn(4, 4096, device=device)
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
    print("  4. 进阶：实现 online softmax，让 N=1<<20 也能跑")


if __name__ == "__main__":
    main()
