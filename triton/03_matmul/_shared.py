"""03_matmul 共用代码。

注意 fp16 matmul 的**校验口径**：不能要求逐元素相等。
fp32 累加 K 个 fp16 乘积，误差量级 ~ `sqrt(K) * eps_fp16`，所以 atol 必须
随 K 放大。写死 `atol=1e-3` 的话 K=4096 时必然 FAIL —— 然后你会去怀疑
kernel 写错了，浪费半天。
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import (A100_PEAK_TFLOPS_FP16, bench, check,  # noqa: E402,F401
                    kernel_info, report_flops)

# 正确性用例：含 K 非整除、M/N 非整除（打 mask）
CASES = ((256, 256, 256), (512, 512, 128), (300, 500, 177), (1024, 1024, 1024))

# 参数扫描用的方阵尺寸
SZ_BENCH = 4096


def nflops(m: int, n: int, k: int) -> int:
    """每个输出元素 K 次乘 + K 次加。"""
    return 2 * m * n * k


def atol_for(k: int) -> float:
    """fp16 matmul 的合理 atol：误差随 K 以 sqrt(K) 增长。"""
    return k ** 0.5 * 1e-2


def rand_pair(m, k, n, device="cuda"):
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    b = torch.randn((k, n), device=device, dtype=torch.float16)
    return a, b


def selftest(fn, name):
    """fn: (a, b) -> c"""
    torch.manual_seed(0)
    print("=" * 70)
    print(f"03_matmul / {name}")
    print("=" * 70)

    print("\n[正确性] 含非整除尺寸，验证 mask")
    ok = True
    for (m, n, k) in CASES:
        a, b = rand_pair(m, k, n)
        ok &= check(fn(a, b), torch.matmul(a, b),
                    f"M,N,K = {m},{n},{k}".ljust(22),
                    rtol=1e-2, atol=atol_for(k))

    print(f"\n[性能] 方阵 fp16，峰值按 {A100_PEAK_TFLOPS_FP16} TFLOP/s 算")
    print(f"  {'':26} {'耗时':>10}  {'算力':>14}  {'峰值占比':>8}")
    for sz in (512, 1024, 2048, 4096):
        a, b = rand_pair(sz, sz, sz)
        nf = nflops(sz, sz, sz)
        ms_t = bench(lambda: torch.matmul(a, b), warmup=10, iters=50)
        report_flops(f"cuBLAS {sz}x{sz}", ms_t, nf)
        ms = bench(lambda: fn(a, b), warmup=10, iters=50)
        report_flops(f"{name} {sz}", ms, nf,
                     extra=f"({ms_t / ms * 100:.0f}% of cuBLAS)")

    print(f"\n{'全部通过' if ok else '有用例失败'}。完整变体对比见 bench.py")
    return ok
