"""common.py —— 纯 Triton 练习共用的计时 / 校验工具。

和 `综合练习/vector_mul2/common.py` 的思路一致，但这里的 kernel 不都是
memory-bound（比如 matmul 是 compute-bound），所以计时函数同时返回耗时，
由调用方决定换算成 GB/s 还是 TFLOP/s。

计时的四条铁律（写 benchmark 必须遵守）：

1. **必须 warmup**。Triton 第一次调用要 JIT 编译（几百毫秒），
   不 warmup 测出来的全是编译时间。
2. **必须同步**。kernel launch 是异步的，不同步只测到 launch 的几微秒。
3. **用 CUDA Event 而不是 time.time()**。Event 记录在 GPU 流上，
   测的是 GPU 时间线，不受 host 调度抖动影响。
4. **取多次的中位数**。GPU 有 DVFS 频率波动，单次结果不可靠。
"""

import torch

# A100-SXM4-40GB 硬件峰值，用于算利用率
A100_PEAK_BW_GBS = 1555.0        # HBM2e 理论带宽
A100_PEAK_TFLOPS_FP16 = 312.0    # Tensor Core FP16（带稀疏是 624，这里用密集）
A100_PEAK_TFLOPS_TF32 = 156.0    # Tensor Core TF32


def bench(fn, warmup=25, iters=100, reps=5):
    """用 CUDA Event 给 fn 计时，返回中位数耗时（毫秒）。

    fn:   无参可调用对象，内部执行一次目标操作
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
    return samples[len(samples) // 2]


def gbs(nbytes, ms):
    """有效带宽 GB/s。nbytes 是这个 kernel 必须读+写的总字节数。"""
    return (nbytes / 1e9) / (ms / 1e3)


def tflops(nflops, ms):
    """有效算力 TFLOP/s。"""
    return (nflops / 1e12) / (ms / 1e3)


def report_bw(name, ms, nbytes, extra=""):
    bw = gbs(nbytes, ms)
    print(f"  {name:<26} {ms*1e3:8.1f} us  {bw:8.1f} GB/s  "
          f"({bw / A100_PEAK_BW_GBS * 100:5.1f}% of peak) {extra}")


def report_flops(name, ms, nflops, peak=A100_PEAK_TFLOPS_FP16, extra=""):
    tf = tflops(nflops, ms)
    print(f"  {name:<26} {ms*1e3:8.1f} us  {tf:8.1f} TFLOP/s  "
          f"({tf / peak * 100:5.1f}% of peak) {extra}")


def check(got, ref, name, rtol=0, atol=0):
    """对比 got 和 ref，打印结果并返回是否通过。"""
    ok = torch.allclose(got, ref, rtol=rtol, atol=atol)
    max_err = (got.float() - ref.float()).abs().max().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: max_abs_err = {max_err:g}")
    return ok


def kernel_info(compiled, name="kernel"):
    """打印 Triton 编译产物的关键信息：寄存器数 / shared memory / num_warps。

    注意：`n_regs` 只有在 kernel 真正被加载到设备上之后才有值，
    刚 warmup() 出来的对象上可能取不到 —— 所以用 getattr 兜一下。
    """
    md = compiled.metadata
    print(f"  {name}: n_regs = {getattr(compiled, 'n_regs', 'n/a')}, "
          f"shared = {md.shared} bytes, num_warps = {md.num_warps}, "
          f"num_stages = {md.num_stages}")
