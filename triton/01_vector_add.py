"""练习 01：向量加法 —— Triton 的 "Hello World"

学习目标：
  - 掌握 Triton 最基础的三件套：program_id / arange / mask
  - 理解"以 block 为编程粒度"和 CUDA"以 thread 为粒度"的区别
  - 会用 grid lambda 计算需要多少个 program

跑法：
  python 01_vector_add.py
"""

import torch
import triton
import triton.language as tl

from common import bench, check, report_bw, kernel_info


@triton.jit
def add_kernel(
    x_ptr,                      # *float32，输入 x
    y_ptr,                      # *float32，输入 y
    out_ptr,                    # *float32，输出
    n_elements,                 # int，元素总数（运行期变量）
    BLOCK_SIZE: tl.constexpr,   # int，每个 program 处理多少元素（编译期常量）
):
    # ① 我是第几个 program？
    #    对比 CUDA：这相当于 blockIdx.x。Triton 里没有 threadIdx ——
    #    block 内部怎么切给线程，是编译器的事，你不用管。
    pid = tl.program_id(axis=0)

    # ② 我负责哪一段下标？
    #    offsets 是一个长度为 BLOCK_SIZE 的**向量**，不是标量。
    #    Triton 的所有运算都是这种 block 级的向量运算。
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # ③ 边界保护。
    #    n_elements 通常不被 BLOCK_SIZE 整除，最后一个 program 会越界。
    #    mask 为 False 的 lane 不读不写 —— 等价于 CUDA 里手写的 if (idx < n)，
    #    但不用自己写分支，也不会漏。
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024):
    """host 侧封装：分配输出、算 grid、launch。"""
    assert x.is_cuda and y.is_cuda and x.shape == y.shape
    assert x.is_contiguous() and y.is_contiguous()
    out = torch.empty_like(x)
    n = x.numel()

    # grid 是个 lambda，参数 meta 里能拿到 BLOCK_SIZE 等 constexpr。
    # cdiv = 向上取整除法，保证覆盖所有元素。
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)
    return out


def main():
    torch.manual_seed(0)
    device = "cuda"

    print("=" * 70)
    print("练习 01：向量加法")
    print("=" * 70)

    # ---------- 正确性 ----------
    # 故意用一个不被 1024 整除的长度，专门验证 mask 的作用
    print("\n[正确性] 含非整除长度，验证 mask")
    for n in (1024, 1000, 98765, 1 << 20):
        x = torch.randn(n, device=device)
        y = torch.randn(n, device=device)
        check(triton_add(x, y), x + y, f"n={n:<8}")

    # ---------- 性能 ----------
    # 读 x、读 y、写 out，共 3 次访存 → 3 * n * 4 字节
    n = 1 << 24
    nbytes = 3 * n * 4
    x = torch.randn(n, device=device)
    y = torch.randn(n, device=device)

    print(f"\n[性能] n = {n:,}（每个 buffer {n * 4 / 2**20:.0f} MiB）")
    ms = bench(lambda: torch.add(x, y))
    report_bw("torch x + y", ms, nbytes)
    for bs in (128, 256, 1024, 4096):
        ms = bench(lambda bs=bs: triton_add(x, y, bs))
        report_bw(f"triton BLOCK_SIZE={bs}", ms, nbytes)

    print("\n  → 向量加法是纯 memory-bound，BLOCK_SIZE 影响很小；")
    print("    只要 program 数量足够把 SM 填满、隐藏住访存延迟就行。")

    # ---------- 看编译产物 ----------
    print("\n[编译产物]")
    out = torch.empty_like(x)
    compiled = add_kernel.warmup(x, y, out, n, BLOCK_SIZE=1024, grid=(1,))
    kernel_info(compiled, "add_kernel BLOCK=1024")

    # Triton 会自动把连续访存向量化成 128-bit（.v4.b32），
    # 等价于手写 CUDA 的 float4 —— 这是它能追平 torch 的关键。
    import re
    ptx = compiled.asm["ptx"]
    vec = [l.strip() for l in ptx.splitlines() if re.search(r"\.global\.v4", l)]
    print(f"  PTX 里的 128-bit 访存指令（共 {len(vec)} 条）：")
    for line in vec[:4]:
        print(f"    {line}")
    if not vec:
        print("    （没找到 .v4 —— 说明这次没有自动向量化，检查是否 contiguous）")

    print("\n[练习] 试试看：")
    print("  1. 把 mask 去掉，用 n=1000 跑，会发生什么？（提示：加 compute-sanitizer）")
    print("  2. 改成 out = x * a + y（a 是 python float），需要改 kernel 签名吗？")
    print("  3. 把 BLOCK_SIZE 设成 100（非 2 的幂），能编译吗？为什么？")


if __name__ == "__main__":
    main()
