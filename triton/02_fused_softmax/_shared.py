"""02_fused_softmax 共用代码。

统一了正确性用例和「访存量」的算法口径 —— 后者特别重要：

  本练习所有变体报的 GB/s 都是按**理想访存量** `2*M*N*4`（读一次 + 写一次）
  折算的，而不是各自实际搬的字节数。这样：
    - 融合版的数字 ≈ 真实带宽利用率
    - 未融合版会显得远低于峰值，因为它实际搬了好几倍的数据

  这正是我们想看到的对比：**同样的「有用工作量」，谁搬的数据少。**
  如果按各自实际访存量算，未融合版也能显得「带宽利用率很高」，
  就把问题掩盖掉了 —— 这是 benchmark 口径最容易骗人的地方。
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import bench, check, kernel_info, report_bw  # noqa: E402,F401

# 正确性用例：含非 2 次幂列宽（打 mask）、单列、大行数
CASES = ((4, 128), (128, 781), (1823, 781), (64, 1))

# 性能用例
M_BENCH, N_BENCH = 4096, 4096


def nbytes(m: int, n: int) -> int:
    """理想访存量：读一遍 x，写一遍 out。"""
    return 2 * m * n * 4


def selftest(fn, name, big_case=True):
    """跑正确性 + 单点性能。fn: (x_2d) -> out_2d"""
    torch.manual_seed(0)
    print("=" * 70)
    print(f"02_fused_softmax / {name}")
    print("=" * 70)

    print("\n[正确性] 含非 2 次幂列宽，验证 mask")
    ok = True
    for shape in CASES:
        x = torch.randn(*shape, device="cuda")
        ok &= check(fn(x), torch.softmax(x, axis=1),
                    f"shape={str(shape):<12}", rtol=1e-5, atol=1e-6)

    # 数值稳定性：不减最大值的实现会在这里出 inf/nan
    print("\n[正确性] 数值稳定性（输入放大 100 倍）")
    x = torch.randn(64, 512, device="cuda") * 100
    got = fn(x)
    ok &= check(got, torch.softmax(x, axis=1), "x * 100     ", rtol=1e-5, atol=1e-6)
    print(f"  含 nan/inf: {bool(torch.isnan(got).any() or torch.isinf(got).any())}")

    if big_case:
        m, n = M_BENCH, N_BENCH
        x = torch.randn(m, n, device="cuda")
        print(f"\n[性能] shape = ({m}, {n})")
        report_bw("torch.softmax (基线)",
                  bench(lambda: torch.softmax(x, axis=1)), nbytes(m, n))
        report_bw(name, bench(lambda: fn(x)), nbytes(m, n))

    print(f"\n{'全部通过' if ok else '有用例失败'}。完整变体对比见 bench.py")
    return ok
