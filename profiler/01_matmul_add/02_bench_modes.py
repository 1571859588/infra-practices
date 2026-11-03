# -*- coding: utf-8 -*-
"""02_bench_modes.py —— 五种编译模式的稳态性能对比（对应 README §4.6）

`torch.compile(fn, mode=...)` 的 mode 决定编译策略：

    none                         不编译（eager 基线）
    default                      只做算子融合，不开 CUDA Graph
    reduce-overhead              融合 + CUDA Graph
    max-autotune                 自动调优 + CUDA Graph
    max-autotune-no-cudagraphs   只自动调优，不套图

用法：

    PY=/mnt/public/nyt1/docqa/restored_envs/cpp/bin/python
    export CUDA_VISIBLE_DEVICES=3

    $PY 02_bench_modes.py --size 64
    $PY 02_bench_modes.py --size 4096
    $PY 02_bench_modes.py --size 64 --modes none,max-autotune-no-cudagraphs

⚠️ 测量口径：**pipelined** —— 连续下发 iters 次，用 iters 组 CUDA event 记录
每次的耗时（**不逐次 sync**），最后一次统一同步。

为什么不用「每次 sync 后再计时」这种更直观的口径？因为实测这台机器上
「让 GPU 完全空闲再唤醒」有约 **2.5ms 的固定代价**：

    fn + sync   1 次  → 2492us
    fn + sync  10 次  → 2562us     ← 只多了 70us，说明 2.49ms 是固定开销
    fn + sync 100 次  → 3274us

它跟被测算子没关系，却会把结果抬高两个数量级（64×64 真实只要 ~10us）。
pipelined 口径下 GPU 不会彻底空闲，测出来的中位数稳定在 11.3us。
"""

import argparse
import statistics as st
import time

import torch

MODES = ["none", "default", "reduce-overhead",
         "max-autotune", "max-autotune-no-cudagraphs"]

DESC = {
    "none": "不编译，eager 基线",
    "default": "只做算子融合",
    "reduce-overhead": "融合 + CUDA Graph",
    "max-autotune": "自动调优 + CUDA Graph",
    "max-autotune-no-cudagraphs": "只自动调优，不套图",
}


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=64, help="矩阵尺寸 N")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--iters", type=int, default=100, help="每种模式测量次数")
    p.add_argument("--warm", type=int, default=10, help="预热次数（同时完成编译/autotune）")
    p.add_argument("--modes", default=",".join(MODES), help="要测的模式，逗号分隔")
    return p.parse_args()


def build(mode):
    """构建待测算子：none 走 eager，其余交给 torch.compile。"""
    def fn(x, w, b):
        return torch.add(torch.matmul(x, w), b)
    return fn if mode == "none" else torch.compile(fn, mode=mode)


def measure(mode, x, w, b, iters, warm):
    """返回 (编译耗时s, 端到端吞吐us/次, per-iteration 耗时us 列表)。"""
    fn = build(mode)

    t0 = time.perf_counter()
    fn(x, w, b)                      # 第一次调用触发编译 / autotune
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0

    for _ in range(warm):            # 预热：让 cudagraph capture 等一次性工作做完
        fn(x, w, b)
    torch.cuda.synchronize()

    # 端到端吞吐 = 整个循环的墙钟时间 / 次数。它包含 CPU 下发开销，
    # 所以能同时反映「CPU 派活」和「GPU 干活」两条路径谁更慢。
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
           for _ in range(iters)]
    t0 = time.perf_counter()
    for s, e in evs:
        s.record()
        fn(x, w, b)
        e.record()
    torch.cuda.synchronize()         # 全程只 sync 这一次
    throughput_us = (time.perf_counter() - t0) / iters * 1e6

    per_iter = sorted(s.elapsed_time(e) * 1000 for s, e in evs)   # ms -> us
    return compile_s, throughput_us, per_iter


def main():
    args = parse_arguments()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    x = torch.randn(args.size, args.size, device="cuda", dtype=dtype)
    w = torch.randn(args.size, args.size, device="cuda", dtype=dtype)
    b = torch.randn(args.size, args.size, device="cuda", dtype=dtype)

    print(f"矩阵 {args.size}×{args.size}  {args.dtype}  每种模式 {args.iters} 次")
    print()
    print(f"{'mode':<30}{'编译(s)':>9}{'吞吐(us/次)':>13}{'单步中位':>10}"
          f"{'单步p95':>10}{'relative':>10}   说明")
    print("-" * 110)

    base = None
    for mode in args.modes.split(","):
        mode = mode.strip()
        cs, thr, per = measure(mode, x, w, b, args.iters, args.warm)
        med = st.median(per)
        p95 = per[min(int(len(per) * 0.95), len(per) - 1)]
        if base is None:
            base = thr
        print(f"{mode:<30}{cs:>9.2f}{thr:>13.1f}{med:>10.1f}{p95:>10.1f}"
              f"{thr/base:>9.2f}x   {DESC[mode]}")

    print()
    print("口径说明：")
    print("  · 吞吐(us/次) = 整个循环墙钟 / 次数 —— **主指标**，含 CPU 下发开销，")
    print("    所以小矩阵下能暴露 torch.compile 的 Python 包装成本。")
    print("  · 单步中位/p95 来自每组 CUDA event，只反映 GPU 侧推进，不含 CPU 开销。")
    print("  · relative 以第一行（none）的吞吐为 1.00x 基准。")


if __name__ == "__main__":
    main()
