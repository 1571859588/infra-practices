"""bench.py —— 把 02_fused_softmax 的所有变体放在一起对比

跑法：
  python bench.py
  python bench.py --quick     # 跳过参数扫描
"""

import argparse

import torch
import triton

import v0_unfused_torch
import v1_fused
import v2_persistent
import v3_online
from _shared import (CASES, M_BENCH, N_BENCH, bench, check, nbytes,
                     report_bw)


VARIANTS = [
    (v0_unfused_torch.NAME, v0_unfused_torch.softmax),
    (v1_fused.NAME,         v1_fused.softmax),
    (v2_persistent.NAME,    v2_persistent.softmax),
    (v3_online.NAME,        v3_online.softmax),
]


def correctness():
    print("\n[正确性] 所有变体 × 所有用例")
    all_ok = True
    for name, fn in VARIANTS:
        print(f"  {name}")
        for shape in CASES:
            x = torch.randn(*shape, device="cuda")
            all_ok &= check(fn(x), torch.softmax(x, axis=1),
                            f"  shape={str(shape):<12}", rtol=1e-5, atol=1e-6)
        # 数值稳定性
        x = torch.randn(64, 512, device="cuda") * 100
        all_ok &= check(fn(x), torch.softmax(x, axis=1),
                        "  x*100 (稳定性) ", rtol=1e-5, atol=1e-6)
    return all_ok


def head_to_head():
    m, n = M_BENCH, N_BENCH
    nb = nbytes(m, n)
    x = torch.randn(m, n, device="cuda")

    print(f"\n[变体对比] shape = ({m}, {n})")
    print("  带宽按「理想访存量」2*M*N*4 折算，所以未融合版会显得远低于峰值 ——")
    print("  它实际搬的字节数是这个数的好几倍，这正是我们想看到的。")
    ms_torch = bench(lambda: torch.softmax(x, axis=1))
    report_bw("torch.softmax (已融合)", ms_torch, nb)
    results = {}
    for name, fn in VARIANTS:
        ms = bench(lambda fn=fn: fn(x))
        results[name] = ms
        report_bw(name, ms, nb, extra=f"{ms_torch / ms * 100:.0f}% of torch")

    unfused = results[v0_unfused_torch.NAME]
    fused = results[v1_fused.NAME]
    print(f"\n  ★ 融合 vs 未融合：{unfused / fused:.2f}x")
    print("    这是本目录最重要的一个数字 —— Triton 的价值不在单算子更快，")
    print("    而在能把多个算子融成一个。")


def sweeps():
    m = M_BENCH
    x4096 = torch.randn(m, N_BENCH, device="cuda")

    print(f"\n[扫描] 行宽 N 的影响（M = {m} 固定，v1 融合版）")
    for n2 in (256, 1024, 4096, 16384):
        xx = torch.randn(m, n2, device="cuda")
        ms = bench(lambda: v1_fused.softmax(xx))
        bs = triton.next_power_of_2(n2)
        report_bw(f"v1 N={n2:<6}", ms, nbytes(m, n2),
                  extra=f"BLOCK={bs}, warps={v1_fused.pick_num_warps(bs)}")

    print("\n[扫描] v2 的 occupancy（每 SM 想驻留几个 program）")
    for occ in (1, 2, 4, 8, 16):
        ms = bench(lambda occ=occ: v2_persistent.softmax(x4096, occupancy=occ))
        n_prog = min(m, v2_persistent.num_sms() * occ)
        report_bw(f"v2 occupancy={occ:<3}", ms, nbytes(m, N_BENCH),
                  extra=f"{n_prog} programs")

    print("\n[扫描] v2 的 num_stages（software pipelining 深度，occupancy=4）")
    for ns in (1, 2, 4, 8):
        ms = bench(lambda ns=ns: v2_persistent.softmax(x4096, num_stages=ns))
        report_bw(f"v2 num_stages={ns}", ms, nbytes(m, N_BENCH))

    print("\n[扫描] v3 的 block_size（和行宽解耦，所以可以随便调）")
    for bs in (256, 1024, 4096):
        ms = bench(lambda bs=bs: v3_online.softmax(x4096, block_size=bs))
        report_bw(f"v3 block_size={bs:<5}", ms, nbytes(m, N_BENCH))

    print("\n[扫描] 小规模下 persistent 的优势（N=1024，M 变化）")
    for m2 in (128, 1024, 8192, 65536):
        xx = torch.randn(m2, 1024, device="cuda")
        ms1 = bench(lambda: v1_fused.softmax(xx))
        ms2 = bench(lambda: v2_persistent.softmax(xx))
        report_bw(f"M={m2:<6} v1", ms1, nbytes(m2, 1024))
        report_bw(f"M={m2:<6} v2 persistent", ms2, nbytes(m2, 1024),
                  extra=f"{ms1 / ms2 * 100:.0f}% of v1")


def capability():
    """v1 装不下的行宽，只有 v3 能跑。"""
    print("\n[能力边界] n_cols = 1<<18 = 262144（v1 会编译卡死，只测 v3）")
    x = torch.randn(64, 1 << 18, device="cuda")
    check(v3_online.softmax(x), torch.softmax(x, axis=1),
          "v3 n_cols=262144", rtol=1e-5, atol=1e-6)
    ms = bench(lambda: v3_online.softmax(x), warmup=5, iters=20)
    report_bw("v3 n_cols=262144", ms, nbytes(64, 1 << 18))
    print("  → v1 的 BLOCK_SIZE 会被设成 262144，一个 tile 就要 1 MB 寄存器，")
    print("    编译器直接卡死。v3 的 BLOCK_SIZE 固定，行宽无上限。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过参数扫描")
    args = ap.parse_args()

    torch.manual_seed(0)
    print("=" * 70)
    print("02_fused_softmax 变体对比")
    print("=" * 70)

    ok = correctness()
    head_to_head()
    if not args.quick:
        sweeps()
    capability()

    print("\n" + "=" * 70)
    print(f"正确性：{'全部通过' if ok else '有失败'}")
    print("结论见 README.md §3")
    print("=" * 70)


if __name__ == "__main__":
    main()
