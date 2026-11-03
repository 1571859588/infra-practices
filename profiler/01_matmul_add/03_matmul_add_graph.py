# -*- coding: utf-8 -*-
"""03_matmul_add_graph.py —— 手动 CUDA Graph：把 compile 的融合 kernel 捕获成图

对应笔记 §4.7。对比三种方案在 profiler 下的单步 CPU / GPU 耗时：

    1. eager                      y = x @ w + b 原样跑
    2. compile（无图）            torch.compile(mode="max-autotune-no-cudagraphs")
                                  —— 拿到融合 kernel，但每次调用仍要走 Dynamo + launch
    3. compile + 手动 CUDA Graph  —— 把上面那个捕获成图，CPU 只发一次 cudaGraphLaunch

**为什么要手动？** 因为 `mode="reduce-overhead"` 把「编译」和「套图」耦合在一起，
自动生成的图管理路径带着不小的固定开销（见 §4.6：64×64 下 reduce-overhead 比
eager 还慢 2.2 倍）。手动的做法是：先用 `max-autotune-no-cudagraphs` 拿融合
kernel，再自己把它捕获成图。

**三个要点**（错一个就是错的数或者报错）：

1. **输入必须驻留静态 buffer** —— 图里记录的是「捕获那一刻的指针地址」，
   之后每次 `replay()` 都往那些地址读写。所以要先 `clone()` 一份专用副本。
2. **捕获只记录不执行** —— `with torch.cuda.graph(g)` 里 kernel 只是被记进图，
   并没有真跑，因此捕获完必须 `replay()` 一次，否则输出是未初始化的。
3. **预热和捕获必须在 profiler 之外完成** —— 编译、显存分配、图捕获都是
   准备工作，会污染 profile 数据（而且 capture 期间不允许 CUDA 同步）。

用法：

    PY=/mnt/public/nyt1/docqa/restored_envs/cpp/bin/python
    export CUDA_VISIBLE_DEVICES=3

    $PY 03_matmul_add_graph.py --size 64
    $PY 03_matmul_add_graph.py --size 4096
    $PY 03_matmul_add_graph.py --size 64 --scheme graph    # 只看一种
"""

import argparse
import json
import math
import os

import torch

SCHEMES = ["eager", "compile", "graph"]
DESC = {
    "eager": "原样执行（基线）",
    "compile": "compile 融合，不套图",
    "graph": "compile 融合 + 手动 CUDA Graph",
}


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=64, help="矩阵尺寸 N")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--steps", type=int, default=8, help="profiler 记录的步数 (active)")
    p.add_argument("--warm", type=int, default=10, help="profiler 之前的预热步数")
    p.add_argument("--scheme", choices=SCHEMES + ["all"], default="all")
    p.add_argument("--trace-dir", default="./traces", help="trace 输出目录")
    return p.parse_args()


def check_correctness(out, ref, n, tag):
    """跟 eager 的参考值比一下，确认融合/套图没把数算错。"""
    max_diff = (out.float() - ref.float()).abs().max().item()
    assert max_diff < 0.1 * n * math.sqrt(n), f"{tag} 结果不一致: max_diff={max_diff:.3f}"
    print(f"      [正确性] {tag} 最大偏差 {max_diff:.3f}")


def capture_cuda_graph(cf, xs, ws, bs, ref, n):
    """把编译后的函数捕获成 CUDA Graph，返回 (graph, 输出张量)。"""
    # 1. 预热：让 torch.compile 完成编译，否则编译过程会被连带捕获进图
    for _ in range(3):
        cf(xs, ws, bs)
    torch.cuda.synchronize()

    # 2. 捕获：kernel「只记录不执行」，此刻 y 还没值
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = cf(xs, ws, bs)
    torch.cuda.synchronize()

    # 3. replay：必须重放一次才有值，才能做正确性校验
    graph.replay()
    torch.cuda.synchronize()
    check_correctness(y, ref, n, "graph")
    return graph, y


def read_step_times(path):
    """从 Chrome trace 里量单步的 CPU / GPU 时间。

    为什么不直接用 key_averages 的 `ProfilerStep*`？
    因为 CUDA Graph replay 的 kernel 是 `cudaGraphLaunch` 一次性提交的，
    kineto 不会把它的 GPU 时间归到 CPU 侧那个 `ProfilerStep*` 行上
    （实测 `device_time_total` 是 0），只有 trace 里能看到真实的 kernel。

    CPU = `ProfilerStep#N` 事件本身的宽度（含 Python + dispatch + launch）
    GPU = `gpu_user_annotation` 窗口内 kernel 的执行时间之和（**不含** GPU 空等）

    注意 GPU 时间**不能**用 CPU 侧窗口去圈 kernel：CPU 连续下发、GPU 追着跑，
    kernel 的时间戳整体落在 CPU 窗口之外（实测 eager 第一步的 kernel 比
    `ProfilerStep#N` 起点晚 396us，跑到 CPU 窗口结束后 2.5ms 还没完）。
    `gpu_user_annotation` 本来就在 GPU 时间轴上，用它圈才对得上。
    """
    ev = json.load(open(path))["traceEvents"]
    steps = [e for e in ev if e.get("cat") == "user_annotation"
             and e["name"].startswith("ProfilerStep#")]
    ga = [e for e in ev if e.get("cat") == "gpu_user_annotation"]
    ker = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])

    cpu = sum(e["dur"] for e in steps) / len(steps) if steps else 0.0
    if not ga:
        return cpu, 0.0
    gpu = 0.0
    for g in ga:
        lo, hi = g["ts"] - 1, g["ts"] + g["dur"] + 1
        gpu += sum(k["dur"] for k in ker if lo <= k["ts"] <= hi)
    return cpu, gpu / len(ga)


def build_step(scheme, fn, x, w, b, ref, n):
    """按方案构建「每步执行什么」，返回 step 函数。

    所有准备工作（编译 / 图捕获）都在这里做完，**不进 profiler**。
    """
    if scheme == "eager":
        y = fn(x, w, b)
        torch.cuda.synchronize()
        check_correctness(y, ref, n, "eager")
        return lambda: fn(x, w, b)

    cf = torch.compile(fn, mode="max-autotune-no-cudagraphs")

    if scheme == "compile":
        y = cf(x, w, b)
        torch.cuda.synchronize()
        check_correctness(y, ref, n, "compile")
        return lambda: cf(x, w, b)

    # graph：输入必须驻留静态 buffer（图里记的是指针地址），所以 clone 一份专用副本
    xs, ws, bs = x.clone(), w.clone(), b.clone()
    graph, _ = capture_cuda_graph(cf, xs, ws, bs, ref, n)
    return graph.replay


def run_scheme(scheme, fn, x, w, b, ref, args, tag):
    print(f"  [{scheme}] {DESC[scheme]}")
    step = build_step(scheme, fn, x, w, b, ref, args.size)

    for _ in range(args.warm):          # 预热在 profiler 之外
        step()
    torch.cuda.synchronize()

    def step_tagged():
        with torch.profiler.record_function(f"{scheme}_{tuple(x.shape)}"):
            return step()

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=args.steps, repeat=1)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
    ) as prof:
        for _ in range(args.steps + 2):
            step_tagged()
            prof.step()
    torch.cuda.synchronize()

    path = os.path.join(args.trace_dir, f"{tag}_{scheme}.json")
    prof.export_chrome_trace(path)

    cpu, gpu = read_step_times(path)
    print(f"      CPU {cpu:8.1f}us/步    GPU {gpu:8.1f}us/步    trace -> {path}")
    return cpu, gpu


def main():
    args = parse_arguments()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    os.makedirs(args.trace_dir, exist_ok=True)

    x = torch.randn(args.size, args.size, device="cuda", dtype=dtype)
    w = torch.randn(args.size, args.size, device="cuda", dtype=dtype)
    b = torch.randn(args.size, args.size, device="cuda", dtype=dtype)
    ref = torch.add(torch.matmul(x, w), b)          # eager 参考值，用于正确性校验
    torch.cuda.synchronize()

    def fn(a, bb, c):
        return torch.add(torch.matmul(a, bb), c)

    tag = f"{args.size}_{args.dtype}_graph"
    schemes = SCHEMES if args.scheme == "all" else [args.scheme]

    print(f"矩阵 {args.size}×{args.size}  {args.dtype}   profiler active={args.steps} 步")
    print()
    rows = []
    for s in schemes:
        cpu, gpu = run_scheme(s, fn, x, w, b, ref, args, tag)
        rows.append((s, cpu, gpu))

    print()
    print(f"{'方案':<34}{'CPU(us/步)':>12}{'GPU(us/步)':>12}")
    print("-" * 58)
    for s, cpu, gpu in rows:
        print(f"{s + ' — ' + DESC[s]:<34}{cpu:>12.1f}{gpu:>12.1f}")
    print()
    print("（CPU = ProfilerStep#N 事件宽度 / 步数，含 Python 与 launch 开销；")
    print("  GPU = 该步窗口内 kernel 执行时间之和 / 步数，不含 GPU 空等。")
    print("  两者都包含 profiler 自身的开销，所以比无 profiler 时偏大。）")


if __name__ == "__main__":
    main()
