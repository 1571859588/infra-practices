"""01_vector_add 共用代码。

做两件事：
  1. 把上一级目录（triton/）加进 sys.path，这样各变体能 `from common import bench`
  2. 提供统一的自测入口 selftest()，让每个 vN_*.py 都能单独跑

各变体只负责写 kernel 和 launcher，正确性用例和计时口径都在这里统一，
保证「不同变体的数字是可比的」—— 这是做 A/B 对比最容易翻车的地方。
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import bench, check, report_bw  # noqa: E402,F401  供各变体直接 import

# 正确性用例：故意含非 2 次幂、非整除长度，专门打 mask
CASES = (1024, 1000, 98765, 1 << 20)

# 性能用例：64 MiB per buffer，足够大到不受 launch 开销影响
N_BENCH = 1 << 24


def nbytes(n: int) -> int:
    """这个 kernel 必须搬的字节数：读 x + 读 y + 写 out，各 n 个 float32。"""
    return 3 * n * 4


def selftest(fn, name):
    """跑正确性 + 单点性能。每个变体的 __main__ 都调这个。

    fn: (x, y) -> out
    """
    torch.manual_seed(0)
    print("=" * 70)
    print(f"01_vector_add / {name}")
    print("=" * 70)

    print("\n[正确性]")
    ok = True
    for n in CASES:
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        ok &= check(fn(x, y), x + y, f"n={n:<8}")

    n = N_BENCH
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    print(f"\n[性能] n = {n:,}（每个 buffer {n * 4 / 2**20:.0f} MiB）")
    report_bw("torch x + y", bench(lambda: torch.add(x, y)), nbytes(n))
    report_bw(name, bench(lambda: fn(x, y)), nbytes(n))

    print(f"\n{'全部通过' if ok else '有用例失败'}。完整变体对比见 bench.py")
    return ok
