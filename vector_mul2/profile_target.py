"""profile_target.py —— 专门给 profiler 用的轻量入口。

和 bench_all.py 的区别：
  - 迭代次数很少（默认 20），避免 profile 产物过大 / ncu replay 过慢；
  - 每个实现包在独立的 NVTX range 里，nsys 时间线上能直接按名字过滤；
  - 支持 --only 只跑某一个实现，方便 ncu 单独对比。

用法：
    nsys profile -t cuda,nvtx -o reports/all python profile_target.py
    ncu  --nvtx --nvtx-include "impl::triton/" python profile_target.py --only triton
"""

import argparse
import ctypes

import torch

from bench_all import load_cuda_lib, make_cuda_callable
from common import make_input
from torch_mul2 import torch_vector_mul2_out
from triton_mul2 import triton_vector_mul2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1 << 24)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--only", choices=["torch", "triton", "cuda", "cuda_vec4"],
                    default=None)
    args = ap.parse_args()

    nvtx = torch.cuda.nvtx
    lib = load_cuda_lib()

    x = make_input(args.n)
    y = torch.empty_like(x)

    impls = {
        "torch":     lambda: torch_vector_mul2_out(x, y),
        "triton":    lambda: triton_vector_mul2(x, y, BLOCK_SIZE=1024),
        "cuda":      make_cuda_callable(lib, x, y, 256, vec4=False),
        "cuda_vec4": make_cuda_callable(lib, x, y, 256, vec4=True),
    }
    if args.only:
        impls = {args.only: impls[args.only]}

    # warmup 单独框起来：Triton 的 JIT 编译发生在这里，
    # 在 nsys 时间线上能清楚看到第一次 launch 前有一大段 CPU 空窗。
    nvtx.range_push("warmup")
    for fn in impls.values():
        for _ in range(5):
            fn()
    torch.cuda.synchronize()
    nvtx.range_pop()

    # 正式区段。torch.cuda.profiler.start/stop 配合 nsys 的 --capture-range=cudaProfilerApi
    # 可以只捕获这一段，进一步缩小报告体积。
    torch.cuda.profiler.start()
    for name, fn in impls.items():
        nvtx.range_push(f"impl::{name}")
        for i in range(args.iters):
            nvtx.range_push(f"iter{i}")
            fn()
            nvtx.range_pop()
        torch.cuda.synchronize()
        nvtx.range_pop()
    torch.cuda.profiler.stop()

    print(f"profiled: {list(impls)}  n={args.n} iters={args.iters}")


if __name__ == "__main__":
    main()
