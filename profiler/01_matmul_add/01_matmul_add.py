# -*- coding: utf-8 -*-
"""01_matmul_add.py —— torch.profiler 最小可跑示例

复现的内容（对应 README.md §2.3 / §2.4）：

  1. torch.profiler.record_function   给一段代码打上自定义标签
  2. torch.profiler.profile           配置 activities（CPU / CUDA 分别收哪些事件）
  3. torch.profiler.schedule          wait / warmup / active / repeat 四段规则
  4. prof.export_chrome_trace         Chrome trace，拖进 perfetto 看时间线
  5. prof.key_averages().table        按 cuda_time_total 排序的耗时汇总

被剖析的算子是 ``y = x @ w + b``，注意它**不是**一个 kernel：
matmul 和 add 是两个独立的算子，这正是为什么要在 profiler 里把它们
当成一段「业务逻辑」整体看待。

用法：

    PY=/mnt/public/nyt1/docqa/restored_envs/cpp/bin/python
    export CUDA_VISIBLE_DEVICES=7          # 先挑一张空闲卡

    $PY 01_matmul_add.py                   # 默认 64x64 bf16，20 步
    $PY 01_matmul_add.py --size 1024 --steps 5
    $PY 01_matmul_add.py --dtype fp32
    $PY 01_matmul_add.py --mode COMPILE    # torch.compile 包装后再剖析
"""

import argparse
import os

import torch


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=64, help="矩阵尺寸 N")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--mode", choices=["COMPILE", "EAGER"], default="none",
                   help="包装模式：COMPILE 用 torch.compile，none/EAGER 表示 eager")
    p.add_argument("--steps", type=int, default=20, help="正式记录的步数 (active 阶段)")
    p.add_argument("--warm", type=int, default=10, help="剖析前预热步数，用于让 torch.compile 完成编译/图捕获")
    p.add_argument("--trace-dir", default="./traces", help="trace 输出目录")
    return p.parse_args()


def main():
    args = parse_arguments()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    x = torch.randn(args.size, args.size, device="cuda", dtype=dtype)
    w = torch.randn(args.size, args.size, device="cuda", dtype=dtype)
    b = torch.randn(args.size, args.size, device="cuda", dtype=dtype)

    # 定义乘加算子：y = x @ w + b
    def fn(x, w, b):
        return torch.add(torch.matmul(x, w), b)

    # 按模式包装算子
    if args.mode == "COMPILE":
        fn = torch.compile(fn)

    # 打标签：record_function。
    # 标签里带上 shape —— 这样 matmul_add_((64, 64)) 会直接出现在时间线上，
    # 不用去翻日志就知道当前 x 的形状。
    def step():
        with torch.profiler.record_function(f"matmul_add_{tuple(x.shape)}"):
            return fn(x, w, b)

    # 预热：让 torch 初始化 / 算子库加载 /（torch.compile 编译）都做完，
    # 避免这些一次性开销污染 profile 数据
    for _ in range(args.warm):
        step()
        torch.cuda.synchronize()

    os.makedirs(args.trace_dir, exist_ok=True)
    tag = f"{args.size}_{args.dtype}_{args.mode}"
    table_path = os.path.join(args.trace_dir, f"{tag}.txt")
    trace_path = os.path.join(args.trace_dir, f"{tag}.json")

    # schedule: 前 2 步不记录，随后连续记录 args.steps 步
    schedule = torch.profiler.schedule(
        wait=1, warmup=1, active=args.steps, repeat=1)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(args.steps + 2):
            step()
            prof.step()
    torch.cuda.synchronize()

    # Chrome trace 格式，拖进 https://ui.perfetto.dev 即可查看
    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace -> {trace_path}")

    # 汇总表：按 CUDA 时间排序的算子耗时 Top 15
    with open(table_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
