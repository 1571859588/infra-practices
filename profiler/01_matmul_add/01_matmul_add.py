# -*- coding: utf-8 -*-
"""01_matmul_add.py —— torch.profiler 最小可跑示例 + 五种编译模式

复现的内容（对应 README.md）：

  1. torch.profiler.record_function   给一段代码打上自定义标签      §2.3
  2. torch.profiler.profile           配置 activities（收哪些事件）  §2.4
  3. torch.profiler.schedule          wait / warmup / active / repeat §2.4
  4. prof.export_chrome_trace         Chrome trace，拖进 perfetto    §3.5
  5. prof.key_averages().table        按 cuda_time_total 排序的汇总   §3.5
  6. torch.compile(fn, mode=...)      五种编译模式，看融合 / 套图    §4.4 §4.5

被剖析的算子是 ``y = x @ w + b``，注意它**不是**一个 kernel：
eager 下 matmul 和 add 是两个独立的算子（两次 launch），
这正是为什么要在 profiler 里把它们当成一段「业务逻辑」整体看待。

用法：

    PY=/mnt/public/nyt1/docqa/restored_envs/cpp/bin/python
    export CUDA_VISIBLE_DEVICES=3          # 先挑一张空闲卡

    $PY 01_matmul_add.py                                  # 默认 64x64 bf16 eager
    $PY 01_matmul_add.py --size 4096
    $PY 01_matmul_add.py --size 64 --mode reduce-overhead  # 融合 + CUDA Graph
    $PY 01_matmul_add.py --size 64 --mode max-autotune-no-cudagraphs  # 只融合不套图

看五个模式的横向对比、以及稳态性能（100 次取中位数）用 ``02_bench_modes.py``。
"""

import argparse
import os

import torch

# torch.compile 的合法 mode（"none" 是我们自己的标记，表示不编译）
COMPILE_MODES = ["none", "default", "reduce-overhead",
                 "max-autotune", "max-autotune-no-cudagraphs"]

# 会套 CUDA Graph 的模式：输入会被复制到静态 buffer，输出也会被复用
CUDA_GRAPH_MODES = {"reduce-overhead", "max-autotune"}

# 早期版本的 --mode 只有 COMPILE / EAGER 两个开关。保留别名，
# 免得 run.sh 和 traces/1024_bf16_COMPILE.* 这些已对齐的产物失效。
MODE_ALIAS = {"COMPILE": "reduce-overhead", "EAGER": "none"}


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=64, help="矩阵尺寸 N")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--mode", choices=COMPILE_MODES + list(MODE_ALIAS), default="none",
                   help="编译模式，见 README §4.5（COMPILE/EAGER 是旧别名）")
    p.add_argument("--steps", type=int, default=20, help="正式记录的步数 (active 阶段)")
    p.add_argument("--warm", type=int, default=10, help="剖析前预热步数，用于让 torch.compile 完成编译/图捕获")
    p.add_argument("--trace-dir", default="./traces", help="trace 输出目录")
    return p.parse_args()


def build_fn(mode):
    """按 mode 构建要执行的算子：none 走 eager，其余交给 torch.compile。"""
    mode = MODE_ALIAS.get(mode, mode)      # COMPILE -> reduce-overhead

    def fn(x, w, b):
        return torch.add(torch.matmul(x, w), b)

    if mode == "none":
        return fn
    return torch.compile(fn, mode=mode)


def main():
    args = parse_arguments()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    x = torch.randn(args.size, args.size, device="cuda", dtype=dtype)
    w = torch.randn(args.size, args.size, device="cuda", dtype=dtype)
    b = torch.randn(args.size, args.size, device="cuda", dtype=dtype)

    # 按模式包装算子
    fn = build_fn(args.mode)

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
