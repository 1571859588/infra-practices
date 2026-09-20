"""v2：persistent kernel —— 固定 program 数，每个 program 循环处理多行

v1 的 grid 是「行数」。M=4096 时起 4096 个 program，每个只干一行就退出。
A100 只有 108 个 SM，所以这 4096 个 program 是分批调度上去的，每批都要
重新走一遍 launch / 寄存器分配 / 退出的流程。

persistent kernel 的思路反过来：
  **program 数固定为「刚好能填满 GPU 的数量」，每个 program 用循环处理多行。**

  grid = NUM_PROGRAMS（≈ SM 数 × 每 SM 能驻留的 block 数）
  program pid 处理 row = pid, pid + P, pid + 2P, ...

好处：
  1. 省掉 program 反复创建/退出的调度开销
  2. `tl.range(..., num_stages=N)` 能让编译器对行循环做 **software pipelining**：
     第 i 行还在算的时候，第 i+1 行的 load 已经发出去了 —— 这是主要收益来源
  3. 寄存器里的常量（行宽、指针基址）只算一次，跨行复用

注意这里的跨步循环和 `01_vector_add/v1_strided_bad.py` 形式上很像，
但**不会破坏合并访存** —— 因为跨的是「行」，一行内部仍然是连续访问的。
跨步是否有害，取决于跨的是不是「最内层连续维度」。

跑法：
  python v2_persistent.py
"""

import torch
import triton
import triton.language as tl

from _shared import selftest

NAME = "v2 persistent"


@triton.jit
def persistent_softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    row_start = tl.program_id(axis=0)
    row_step = tl.num_programs(axis=0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # tl.range 的 num_stages 让编译器对这个循环做 software pipelining：
    # 提前若干轮把下一行的 load 发出去，用计算掩盖访存延迟。
    # 普通 range 拿不到这个。
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=NUM_STAGES):
        row = tl.load(in_ptr + row_idx * in_row_stride + col_offsets,
                      mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        out = numerator / tl.sum(numerator, axis=0)
        tl.store(out_ptr + row_idx * out_row_stride + col_offsets, out, mask=mask)


def num_sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def softmax(x: torch.Tensor, occupancy: int = 4, num_stages: int = 4):
    """occupancy = 每个 SM 想驻留几个 program。

    严谨做法是用 kernel 的 n_regs / shared 反算真实 occupancy 上限
    （Triton 官方 tutorial 02 就是这么做的），这里用一个固定值简化，
    因为在本练习的规模下它不是瓶颈 —— 见 README §3.3 的扫描。
    """
    assert x.dim() == 2 and x.is_cuda
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # program 数封顶在行数：行比「填满 GPU 需要的 program 数」还少时，
    # 起更多 program 没有意义（多出来的直接空转退出）。
    n_programs = min(n_rows, num_sms() * occupancy)

    out = torch.empty_like(x)
    persistent_softmax_kernel[(n_programs,)](
        out, x,
        x.stride(0), out.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_STAGES=num_stages,
        num_warps=16 if BLOCK_SIZE >= 4096 else (8 if BLOCK_SIZE >= 2048 else 4),
    )
    return out


if __name__ == "__main__":
    selftest(softmax, NAME)
    print(f"\n  本机 SM 数 = {num_sms()}，occupancy=4 → program 数 = {num_sms() * 4}")
    print("  对比 v1 在 M=4096 时的 4096 个 program。")
    print("\n[练习] 试试看：")
    print("  1. occupancy 取 1/2/4/8，找拐点（bench.py 里有这个扫描）")
    print("  2. num_stages 取 1（等于关掉 pipelining），性能掉多少？")
    print("  3. 把 tl.range 换成普通 range，对比一下 —— num_stages 参数会失效")
    print("  4. M 很小时（比如 M=64）persistent 还有意义吗？为什么？")
