"""bench_all.py —— 三种方案在同一进程、同一份数据上的统一对比。

为什么用 ctypes 加载 libmul2.so，而不是 torch.utils.cpp_extension？
  torch 是 cu128 编译的，本机系统 nvcc 是 12.4、conda 的是 13.3，
  用 cpp_extension 会触发 torch 的 CUDA 版本一致性检查，容易卡在环境问题上。
  ctypes + 纯 C 接口完全绕开这一层：libmul2.so 不含任何 torch 符号，
  只要 ABI 是标准 C 就能调，跨 CUDA 版本也没问题。
  代价是要自己传 data_ptr()、自己管 stream。

所有区段都打了 NVTX 标记（torch.cuda.nvtx），nsys 时间线上可以直接按名字定位。

运行：
    conda activate cpp
    CUDA_VISIBLE_DEVICES=3 python bench_all.py
    CUDA_VISIBLE_DEVICES=3 python bench_all.py --n 1048576     # 换规模
"""

import argparse
import ctypes
import os

import torch
import triton

from common import bench, check, make_input, report, A100_PEAK_BW_GBS
from torch_mul2 import torch_vector_mul2_out
from triton_mul2 import triton_vector_mul2

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "libmul2.so")


def load_cuda_lib():
    if not os.path.exists(_LIB_PATH):
        raise RuntimeError(f"{_LIB_PATH} 不存在，先在本目录执行 `make`")
    lib = ctypes.CDLL(_LIB_PATH)
    # 显式声明签名，否则 64 位指针会被 ctypes 按 int 截断
    for fn in (lib.launch_mul2, lib.launch_mul2_vec4):
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        fn.restype = None
    lib.mul2_last_error.restype = ctypes.c_int
    return lib


def make_cuda_callable(lib, x, y, block_size=256, vec4=False):
    """把 torch 张量的裸指针喂给 .so。

    关键：拿 torch 当前流的句柄传进去，让 kernel 跑在和 torch 同一条流上。
    否则 kernel 在默认流、torch 在自己的流，计时和依赖关系都会错。
    """
    fn = lib.launch_mul2_vec4 if vec4 else lib.launch_mul2
    xp, yp = ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr())
    n = ctypes.c_int(x.numel())
    bs = ctypes.c_int(block_size)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    return lambda: fn(xp, yp, n, bs, stream)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1 << 24, help="元素个数")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--sweep", action="store_true", help="扫多个数据规模")
    args = ap.parse_args()

    nvtx = torch.cuda.nvtx
    lib = load_cuda_lib()

    print("=" * 78)
    print(f"device  : {torch.cuda.get_device_name()}")
    print(f"torch   : {torch.__version__}   triton: {triton.__version__}")
    print(f"peak BW : {A100_PEAK_BW_GBS} GB/s (A100-SXM4-40GB HBM2e 理论值)")
    print("=" * 78)

    sizes = [1 << k for k in (12, 16, 20, 24, 26)] if args.sweep else [args.n]

    for n in sizes:
        x = make_input(n)
        y = torch.empty_like(x)
        mib = n * 4 / 2 ** 20
        print(f"\nn = {n:>12,}  ({mib:.2f} MiB / buffer)")

        variants = [
            ("torch  x*2",        lambda: torch_vector_mul2_out(x, y)),
            ("triton BLOCK=1024", lambda: triton_vector_mul2(x, y, BLOCK_SIZE=1024)),
            ("cuda   scalar",     make_cuda_callable(lib, x, y, 256, vec4=False)),
            ("cuda   vec4",       make_cuda_callable(lib, x, y, 256, vec4=True)),
        ]

        # 先做一遍正确性校验（每个 variant 之间把 y 清零，避免上一个的结果蒙混过关）
        print("  正确性:")
        for name, fn in variants:
            if "vec4" in name and n % 4 != 0:
                print(f"  [SKIP] {name}: n 不是 4 的倍数")
                continue
            y.zero_()
            fn()
            torch.cuda.synchronize()
            assert lib.mul2_last_error() == 0, f"{name} 触发 CUDA 错误"
            check(y, x, name)

        print("  性能:")
        for name, fn in variants:
            if "vec4" in name and n % 4 != 0:
                continue
            nvtx.range_push(f"bench::{name}")      # nsys 时间线上的分段标记
            ms, gbs = bench(fn, n, iters=args.iters)
            nvtx.range_pop()
            report(name, ms, gbs)

    print("\n说明：GB/s = 2 * n * 4 bytes / 耗时，即读一次 + 写一次的有效带宽。")
    print("      mul2 是纯 memory-bound 算子，能到 80%+ 峰值带宽就基本到顶了。")


if __name__ == "__main__":
    main()
