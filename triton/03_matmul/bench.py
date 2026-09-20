"""bench.py —— 把 03_matmul 的所有变体放在一起对比

跑法：
  python bench.py
  python bench.py --quick     # 跳过参数扫描（autotune 首次仍需约 2s）
  python bench.py --big       # 额外跑 8192²（GROUP_M 在这个尺寸才有效果）
"""

import argparse

import torch

import v0_naive
import v1_tiled
import v2_autotune
import v3_fused_relu
from _shared import (CASES, SZ_BENCH, atol_for, bench, check, nflops,
                     rand_pair, report_flops)


VARIANTS = [
    (v0_naive.NAME,    v0_naive.matmul),
    (v1_tiled.NAME,    v1_tiled.matmul),
    (v2_autotune.NAME, v2_autotune.matmul),
]


def correctness():
    print("\n[正确性] 所有变体 × 所有用例（含 M/N/K 非整除）")
    all_ok = True
    for name, fn in VARIANTS:
        print(f"  {name}")
        for (m, n, k) in CASES:
            a, b = rand_pair(m, k, n)
            all_ok &= check(fn(a, b), torch.matmul(a, b),
                            f"  M,N,K = {m},{n},{k}".ljust(24),
                            rtol=1e-2, atol=atol_for(k))
    return all_ok


def head_to_head(sizes=(512, 1024, 2048, 4096)):
    print("\n[变体对比] 方阵 fp16，基线是 cuBLAS（torch.matmul）")
    for sz in sizes:
        a, b = rand_pair(sz, sz, sz)
        nf = nflops(sz, sz, sz)
        print(f"  --- {sz} x {sz} x {sz} ---")
        ms_t = bench(lambda: torch.matmul(a, b), warmup=10, iters=50)
        report_flops("cuBLAS", ms_t, nf)
        for name, fn in VARIANTS:
            ms = bench(lambda fn=fn: fn(a, b), warmup=10, iters=50)
            report_flops(name, ms, nf, extra=f"{ms_t / ms * 100:.0f}% of cuBLAS")


def sweep_blocks():
    sz = SZ_BENCH
    a, b = rand_pair(sz, sz, sz)
    nf = nflops(sz, sz, sz)

    print(f"\n[扫描] 分块配置 {sz}x{sz} —— compute-bound kernel 的第一优化项")
    for (bm, bn, bk, w, s) in [
        (64, 64, 32, 4, 5),
        (64, 128, 32, 4, 4),
        (128, 64, 32, 4, 4),
        (128, 128, 32, 4, 4),
        (128, 128, 64, 4, 4),
        (128, 256, 64, 8, 3),
        (256, 128, 64, 8, 3),
    ]:
        ms = bench(lambda: v1_tiled.matmul(a, b, bm, bn, bk, 8, w, s),
                   warmup=10, iters=50)
        report_flops(f"BM={bm} BN={bn} BK={bk}", ms, nf,
                     extra=f"warps={w} stages={s}")


def sweep_group_m(sizes=(4096,)):
    """group-major 的效果 —— 本练习最重要的一个负结果。"""
    for sz in sizes:
        a, b = rand_pair(sz, sz, sz)
        nf = nflops(sz, sz, sz)
        b_mb = sz * sz * 2 / 2**20
        print(f"\n[扫描] GROUP_M @ {sz}x{sz}（B 矩阵 {b_mb:.0f} MB，"
              f"A100 L2 = 40 MB）")
        for g in (1, 2, 4, 8, 16):
            ms = bench(lambda g=g: v1_tiled.matmul(a, b, 128, 256, 64, g, 8, 3),
                       warmup=10, iters=50)
            report_flops(f"GROUP_M={g}", ms, nf)
        if b_mb < 40:
            print("  → B 矩阵装得进 L2，怎么排都命中，预期无效果")
        else:
            print("  → B 矩阵装不进 L2，group-major 开始有意义")


def sweep_stages():
    sz = SZ_BENCH
    a, b = rand_pair(sz, sz, sz)
    nf = nflops(sz, sz, sz)
    print(f"\n[扫描] num_stages @ {sz}x{sz}（128x256x64, warps=8）")
    print("  num_stages = software pipelining 深度，用 shared memory 换延迟隐藏")
    for s in (2, 3, 4, 5):
        try:
            ms = bench(lambda s=s: v1_tiled.matmul(a, b, 128, 256, 64, 8, 8, s),
                       warmup=10, iters=50)
            report_flops(f"num_stages={s}", ms, nf)
        except Exception as e:                       # shared memory 可能不够
            print(f"  num_stages={s}: 失败 —— {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:80]}")


def epilogue():
    print("\n[epilogue 融合] relu(A@B + bias)")
    print("  ★ 只有「triton 不融合」→「triton 融合」隔离了 epilogue：两者用")
    print("    完全相同的 matmul kernel。「torch 3 kernels」还叠加了 cuBLAS 的优势。")
    for sz in (512, 1024, 2048, 4096):
        a, b = rand_pair(sz, sz, sz)
        bias = torch.randn(sz, device="cuda", dtype=torch.float16)
        nf = nflops(sz, sz, sz)
        ms_t = bench(lambda: v3_fused_relu.torch_unfused(a, b, bias),
                     warmup=10, iters=50)
        ms_u = bench(lambda: v3_fused_relu.triton_unfused(a, b, bias),
                     warmup=10, iters=50)
        ms_f = bench(lambda: v3_fused_relu.matmul_relu(a, b, bias),
                     warmup=10, iters=50)
        report_flops(f"torch 3 kernels {sz}", ms_t, nf, extra="(cuBLAS，仅供参考)")
        report_flops(f"triton 不融合 {sz}", ms_u, nf)
        report_flops(f"triton 融合 {sz}", ms_f, nf,
                     extra=f"epilogue {ms_u / ms_f:.2f}x")

    print("\n  epilogue 省的是 4MN 访存（和 K 无关），matmul 的算力是 2MNK。")
    print("  所以 K 越小，epilogue 占比越高、融合越值。扫一下 K：")
    for k in (4096, 1024, 256, 64):
        a, b = rand_pair(4096, k, 4096)
        bias = torch.randn(4096, device="cuda", dtype=torch.float16)
        nf = nflops(4096, 4096, k)
        ms_u = bench(lambda: v3_fused_relu.triton_unfused(a, b, bias),
                     warmup=10, iters=50)
        ms_f = bench(lambda: v3_fused_relu.matmul_relu(a, b, bias),
                     warmup=10, iters=50)
        report_flops(f"M=N=4096 K={k:<5} 不融合", ms_u, nf)
        report_flops(f"M=N=4096 K={k:<5} 融合", ms_f, nf, extra=f"{ms_u / ms_f:.2f}x")


def autotune_overhead(sz=512):
    """autotune 在小 kernel 上为什么反而慢 —— 量一下 host 侧派发开销。

    做法：先让 autotune 在这个尺寸上 tune 完，把它**自己选中的配置**读出来，
    再用 v1 写死同一套配置跑。两者在 GPU 上执行的是同一个 kernel，
    所以差值就是 autotune 每次调用的 python 侧开销。
    """
    print(f"\n[autotune 的 host 开销] {sz}² 上 autotune 比手写慢，验证一下原因")
    a, b = rand_pair(sz, sz, sz)
    nf = nflops(sz, sz, sz)

    v2_autotune.matmul(a, b)                      # 触发 tune，填充 cache
    torch.cuda.synchronize()
    cfg = next((c for k, c in v2_autotune.matmul_autotuned_kernel.cache.items()
                if k[:3] == (sz, sz, sz)), None)
    if cfg is None:                               # cache key 布局换了就跳过
        print("  (拿不到 autotune 选中的配置，跳过)")
        return
    kw = cfg.kwargs
    label = (f"{kw['BLOCK_M']}x{kw['BLOCK_N']}x{kw['BLOCK_K']} "
             f"w{cfg.num_warps} s{cfg.num_stages}")

    ms_direct = bench(lambda: v1_tiled.matmul(
        a, b, kw["BLOCK_M"], kw["BLOCK_N"], kw["BLOCK_K"], kw["GROUP_M"],
        cfg.num_warps, cfg.num_stages), warmup=10, iters=50)
    ms_auto = bench(lambda: v2_autotune.matmul(a, b), warmup=10, iters=50)

    report_flops(f"v1 写死 {label}", ms_direct, nf)
    report_flops("v2 autotune (同配置)", ms_auto, nf,
                 extra=f"+{(ms_auto - ms_direct) * 1e3:.1f} us")
    print("  → 差值 = autotune 每次调用的 python 侧开销（取 key、查 cache）。")
    print("    kernel 本身只有几十微秒时，这点固定开销占比就很显眼了。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过参数扫描")
    ap.add_argument("--big", action="store_true", help="额外跑 8192²")
    args = ap.parse_args()

    torch.manual_seed(0)
    print("=" * 70)
    print("03_matmul 变体对比")
    print("=" * 70)
    print("注意：v2 autotune 第一次调用要试完 12 个候选（热缓存约 2s，冷编译约 8s）。")

    ok = correctness()
    head_to_head()
    if not args.quick:
        sweep_blocks()
        sweep_group_m((4096, 8192) if args.big else (4096,))
        sweep_stages()
    epilogue()
    if not args.quick:
        autotune_overhead()

    print("\n[autotune 的选择]")
    for k, v in v2_autotune.chosen_configs().items():
        print(f"  {k} → {v}")

    print("\n" + "=" * 70)
    print(f"正确性：{'全部通过' if ok else '有失败'}")
    print("结论见 README.md §3")
    print("=" * 70)


if __name__ == "__main__":
    main()
