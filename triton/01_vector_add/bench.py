"""bench.py —— 把 01_vector_add 的所有变体放在一起对比

跑法：
  python bench.py
  python bench.py --quick     # 跳过参数扫描
"""

import argparse
import re

import torch
import triton

import v0_naive
import v1_strided_bad
import v2_unroll
import v3_autotune
from _shared import N_BENCH, CASES, bench, check, nbytes, report_bw


VARIANTS = [
    (v0_naive.NAME,       lambda x, y: v0_naive.add(x, y)),
    (v1_strided_bad.NAME, lambda x, y: v1_strided_bad.add(x, y)),
    (v2_unroll.NAME,      lambda x, y: v2_unroll.add(x, y)),
    (v3_autotune.NAME,    lambda x, y: v3_autotune.add(x, y)),
]


def correctness():
    print("\n[正确性] 所有变体 × 所有用例")
    all_ok = True
    for name, fn in VARIANTS:
        print(f"  {name}")
        for n in CASES:
            x = torch.randn(n, device="cuda")
            y = torch.randn(n, device="cuda")
            all_ok &= check(fn(x, y), x + y, f"  n={n:<8}")
    return all_ok


def head_to_head():
    n = N_BENCH
    nb = nbytes(n)
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")

    print(f"\n[变体对比] n = {n:,}（{n * 4 / 2**20:.0f} MiB per buffer）")
    ms_torch = bench(lambda: torch.add(x, y))
    report_bw("torch x + y (基线)", ms_torch, nb)
    for name, fn in VARIANTS:
        ms = bench(lambda fn=fn: fn(x, y))
        report_bw(name, ms, nb, extra=f"{ms_torch / ms * 100:.0f}% of torch")


def sweeps():
    n = N_BENCH
    nb = nbytes(n)
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")

    print("\n[扫描] v0 的 BLOCK_SIZE")
    for bs in (128, 256, 1024, 4096):
        ms = bench(lambda bs=bs: v0_naive.add(x, y, bs))
        report_bw(f"v0 BLOCK_SIZE={bs}", ms, nb)

    print("\n[扫描] v2 的 UNROLL（BLOCK_SIZE=1024 固定）")
    for u in (1, 2, 4, 8, 16):
        ms = bench(lambda u=u: v2_unroll.add(x, y, 1024, u))
        n_prog = triton.cdiv(n, 1024 * u)
        report_bw(f"v2 UNROLL={u}", ms, nb, extra=f"{n_prog} programs")

    # v1 的跨步被正确性绑死 = program 数 = cdiv(n, BLOCK_SIZE)，所以只能通过
    # 调 BLOCK_SIZE 间接调跨步。注意 BLOCK_SIZE 不能开太大 —— 它是一个 tile
    # 的元素数，会整个进寄存器，开到 65536 编译就会卡死。
    print("\n[扫描] v1 的跨步大小（跨步 = program 数 = cdiv(n, BLOCK_SIZE)）")
    for bs in (256, 1024, 4096):
        ms = bench(lambda bs=bs: v1_strided_bad.add(x, y, bs))
        stride_bytes = triton.cdiv(n, bs) * 4
        report_bw(f"v1 BLOCK={bs}", ms, nb,
                  extra=f"跨步 {triton.cdiv(n, bs)} 元素 = {stride_bytes / 1024:.0f} KB")


def ptx_vectorization():
    """对比 v0 / v1 的 PTX 访存指令宽度 —— 这就是"向量化"看得见的地方。"""
    print("\n[PTX] 访存指令宽度对比")
    n = N_BENCH
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)

    compiled = {
        "v0 连续": v0_naive.add_kernel.warmup(
            x, y, out, n, BLOCK_SIZE=1024, grid=(1,)),
        "v1 跨步": v1_strided_bad.add_strided_kernel.warmup(
            x, y, out, n, triton.cdiv(n, 1024), BLOCK_SIZE=1024, grid=(1,)),
    }

    for label, c in compiled.items():
        ptx = c.asm["ptx"]
        loads = [l.strip() for l in ptx.splitlines() if "ld.global" in l]
        widths = {}
        for l in loads:
            m = re.search(r"ld\.global\.(?:nc\.)?(v\d\.)?b(\d+)", l)
            if m:
                w = (4 if m.group(1) == "v4." else 2 if m.group(1) == "v2." else 1) * int(m.group(2))
                widths[w] = widths.get(w, 0) + 1
        desc = ", ".join(f"{w}-bit × {c2}" for w, c2 in sorted(widths.items(), reverse=True))
        print(f"  {label:<10} ld.global 共 {len(loads)} 条：{desc or '(解析失败)'}")
        if loads:
            print(f"    e.g. {loads[0][:90]}")

    print("\n  → v0 是 128-bit（.v4.b32，等价手写 float4），v1 退化成 32-bit。")
    print("    同样的数据量，指令数和 DRAM 事务数差一个量级 —— 这就是性能差的来源。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过参数扫描")
    args = ap.parse_args()

    torch.manual_seed(0)
    print("=" * 70)
    print("01_vector_add 变体对比")
    print("=" * 70)

    ok = correctness()
    head_to_head()
    if not args.quick:
        sweeps()
    ptx_vectorization()

    print("\n" + "=" * 70)
    print(f"正确性：{'全部通过' if ok else '有失败'}")
    print("结论见 README.md §3")
    print("=" * 70)


if __name__ == "__main__":
    main()
