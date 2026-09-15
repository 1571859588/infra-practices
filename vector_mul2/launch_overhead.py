"""launch_overhead.py —— 测量三种方案的 CPU 端 kernel launch 开销。

为什么单独测这个？
    在 bench_all.py --sweep 里会看到：n 很小时（4K 元素），三种实现的耗时
    完全不随 n 变化，且 Triton 明显最慢。这不是 GPU 慢，是 **CPU 端 launch
    开销** 成了瓶颈 —— GPU kernel 本身只要 1~2 us，但 host 把它提交上去就
    要 4~18 us。

测法：连续 launch N 次且**不做同步**，量的就是 host 端提交一次的成本。
      只要 host 提交比 GPU 执行慢，队列就永远排不满，GPU 在空转。

运行：
    conda activate cpp
    CUDA_VISIBLE_DEVICES=3 python launch_overhead.py
"""

import time

import torch

from bench_all import load_cuda_lib, make_cuda_callable
from common import make_input
from torch_mul2 import torch_vector_mul2_out
from triton_mul2 import triton_vector_mul2


def measure(fn, warmup=200, iters=2000):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = (time.perf_counter() - t0) / iters * 1e6     # us
    torch.cuda.synchronize()                          # 清掉积压的队列再进下一轮
    return dt


def main():
    lib = load_cuda_lib()
    n = 4096                     # 故意取很小，让 GPU 侧耗时可以忽略
    x = make_input(n)
    y = torch.empty_like(x)

    impls = {
        "torch  (ATen)":   lambda: torch_vector_mul2_out(x, y),
        "triton (JIT)":    lambda: triton_vector_mul2(x, y, BLOCK_SIZE=1024),
        "cuda   (ctypes)": make_cuda_callable(lib, x, y, 256),
    }

    print(f"CPU 端 launch 开销（n={n}，不同步，纯 host 时间 / 次）")
    print("-" * 52)
    for name, fn in impls.items():
        print(f"  {name:<18} {measure(fn):6.2f} us/launch")
    print("-" * 52)
    print("""
解读：
  cuda(ctypes) 最低 —— 只是一次 C 函数调用 + cudaLaunchKernel。
  torch 居中    —— 多了 Python dispatch、TensorIterator 配置、类型分发。
  triton 最高   —— 每次调用要查 JIT 缓存、算 grid、组装参数、走 cuLaunchKernelEx。

结论：小张量（几十 KB 以下）上，瓶颈是 launch 开销而不是带宽。
      这种场景下要提速，靠的是 CUDA Graph / 算子融合 / 增大 batch，
      而不是优化 kernel 内部。
""")


if __name__ == "__main__":
    main()
