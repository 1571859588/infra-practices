"""triton_mul2.py —— 方案二：Triton。

特点：
  - 用 Python 写 kernel，@triton.jit 负责编译到 PTX/SASS。
  - 粒度是 **block（program）** 而不是 thread：你写的是"第 pid 个 block 处理
    哪一段数据"，block 内部的线程划分、向量化、寄存器分配由编译器决定。
    这是它和 CUDA 最大的心智差异 —— 你看不到 threadIdx。
  - mask 机制取代了 CUDA 里的 `if (idx < n)`：load/store 带 mask，
    越界的 lane 自动不读不写。
  - 天然支持算子融合：把多个操作写在同一个 kernel 里，中间结果留在寄存器。

单独运行：
    conda activate cpp
    CUDA_VISIBLE_DEVICES=3 python triton_mul2.py
"""

import torch
import triton
import triton.language as tl

from common import bench, check, make_input, report


# ---------------------------------------------------------------------------
# 核函数。BLOCK_SIZE 标成 tl.constexpr —— 它是编译期常量，
# 每个不同的 BLOCK_SIZE 会触发一次独立的 JIT 编译，产生一份特化的代码。
# ---------------------------------------------------------------------------
@triton.jit
def triton_mul2_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                  # 当前是第几个 block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)   # 本 block 负责的下标向量
    mask = offsets < n_elements                  # 尾块越界保护
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * 2
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_vector_mul2(x: torch.Tensor, y: torch.Tensor = None,
                       BLOCK_SIZE: int = 1024) -> torch.Tensor:
    """host 端 wrapper：负责分配输出、算 grid、launch。"""
    if y is None:
        y = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and x.is_contiguous()
    n_elements = x.numel()
    # grid 可以是 tuple，也可以是一个吃 meta 字典的 lambda。
    # 用 lambda 的好处是配合 @triton.autotune 时 BLOCK_SIZE 会自动代入。
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    triton_mul2_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y


# ---------------------------------------------------------------------------
# autotune 版本：让 Triton 自己在候选配置里挑最快的。
# 第一次调用每个 key 值时会把所有 config 都跑一遍（有明显耗时），之后走缓存。
# num_warps 决定一个 block 用多少 warp（32 线程/warp）来处理 BLOCK_SIZE 个元素。
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in (256, 512, 1024, 2048, 4096)
        for nw in (2, 4, 8)
    ],
    key=["n_elements"],          # n_elements 变化时重新调优
)
@triton.jit
def triton_mul2_kernel_tuned(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(y_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) * 2, mask=mask)


def triton_vector_mul2_tuned(x: torch.Tensor, y: torch.Tensor = None):
    if y is None:
        y = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    triton_mul2_kernel_tuned[grid](x, y, n_elements)
    return y


def main():
    n = 1 << 24
    x = make_input(n)
    y = torch.empty_like(x)

    print(f"Triton 实现    n = {n} ({x.numel() * 4 / 2**20:.1f} MiB per buffer)")
    print(f"  triton {triton.__version__}, device = {torch.cuda.get_device_name()}")

    triton_vector_mul2(x, y)
    check(y, x, "triton_vector_mul2")

    # 扫不同 BLOCK_SIZE，看它对 memory-bound kernel 的影响
    for bs in (256, 512, 1024, 2048, 4096):
        ms, gbs = bench(lambda bs=bs: triton_vector_mul2(x, y, BLOCK_SIZE=bs), n)
        report(f"BLOCK_SIZE={bs}", ms, gbs)

    triton_vector_mul2_tuned(x, y)
    check(y, x, "triton_vector_mul2_tuned")
    ms, gbs = bench(lambda: triton_vector_mul2_tuned(x, y), n)
    report("autotuned", ms, gbs, f"best={triton_mul2_kernel_tuned.best_config}")

    # 看编译产物：Triton 把 kernel 缓存在 ~/.triton/cache，
    # 也可以通过 .asm 直接拿到 ttir / ttgir / llir / ptx / cubin
    # 注意：n_regs 只有在 kernel 真正被加载到设备后才有值，
    #       所以这里必须放在上面的 launch 之后（用 getattr 兜底）。
    k = triton_mul2_kernel.warmup(x, y, n, BLOCK_SIZE=1024, grid=(1,))
    print(f"\n  编译产物可用形式: {sorted(k.asm.keys())}")
    print(f"  PTX 行数 = {len(k.asm['ptx'].splitlines())}, "
          f"n_regs = {getattr(k, 'n_regs', 'n/a')}, "
          f"shared = {k.metadata.shared} bytes, "
          f"num_warps = {k.metadata.num_warps}")

    # 关键观察：Triton 自动做了 128-bit 向量化访存（v4.b32），
    # 和手写 CUDA 的 float4 版本等价 —— 这是它性能能追平 torch 的原因。
    import re
    mem = [l.strip() for l in k.asm["ptx"].splitlines()
           if re.search(r"\b(ld|st)\.global", l)]
    print("  PTX 访存指令:")
    for line in mem:
        print(f"    {line}")


if __name__ == "__main__":
    main()
