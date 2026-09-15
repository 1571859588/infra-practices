"""common.py —— 三个实现共用的计时 / 校验工具。

计时的几个关键点（新手最容易踩的坑，都在这里统一处理掉）：

1. **必须 warmup**。第一次调用包含 CUDA context 初始化、module 加载、
   Triton 的 JIT 编译（第一次可能要几百毫秒）。不 warmup 测出来的全是编译时间。

2. **必须同步**。CUDA kernel launch 是异步的，`t0=time(); kernel(); t1=time()`
   测到的只是 launch 的 CPU 开销（几微秒），不是 kernel 真实耗时。

3. **用 CUDA Event 而不是 time.time()**。Event 记录在 GPU 流上，测的是 GPU
   时间线上的间隔，不受 host 端调度抖动影响。

4. **取多次的中位数**。GPU 有时钟频率波动（DVFS），单次结果不可靠。
"""

import torch

# mul2 是纯 memory-bound：读一次 x，写一次 y。
# 有效访存字节数 = 2 * n * sizeof(float)
BYTES_PER_ELEM = 2 * 4

# A100-SXM4-40GB 的 HBM2e 理论峰值带宽（GB/s），用于算带宽利用率
A100_PEAK_BW_GBS = 1555.0


def bench(fn, n_elements, warmup=25, iters=100, reps=5):
    """用 CUDA Event 给 fn 计时，返回 (中位数毫秒, 有效带宽 GB/s)。

    fn: 无参可调用对象，内部执行一次目标操作
    reps: 重复整组测量的次数，取中位数抗抖动
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    samples = []
    for _ in range(reps):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iters)

    samples.sort()
    ms = samples[len(samples) // 2]
    gbs = (BYTES_PER_ELEM * n_elements / 1e9) / (ms / 1e3)
    return ms, gbs


def check(y, x, name):
    """校验 y == x * 2，打印结果并返回是否通过。"""
    ref = x * 2
    ok = torch.allclose(y, ref, rtol=0, atol=0)   # 乘 2 是精确的浮点运算，可以要求完全相等
    max_err = (y - ref).abs().max().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: max_abs_err = {max_err:g}")
    return ok


def make_input(n, device="cuda", dtype=torch.float32, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(n, device=device, dtype=dtype, generator=g)


def report(name, ms, gbs, extra=""):
    util = gbs / A100_PEAK_BW_GBS * 100
    print(f"  {name:<22} {ms*1e3:8.1f} us  {gbs:8.1f} GB/s  "
          f"({util:5.1f}% of peak) {extra}")
