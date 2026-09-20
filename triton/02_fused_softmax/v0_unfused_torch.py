"""v0：用 torch 算子手工拼出 softmax —— 「没融合」的代价

这个变体**一行 Triton 都没有**，它是对照组：展示如果不融合，
同样的数学要付多少访存代价。

softmax 的定义拆开是 4 步，每步都是一个独立的 torch 算子，
也就是一个独立的 CUDA kernel。**kernel 之间只能通过显存传递数据**，
所以每一步都要把整个矩阵读一遍、写一遍：

    读 x            (MN)
    写 z = x - max  (MN)   ← 中间结果落显存
    读 z            (MN)
    写 num = exp(z) (MN)   ← 中间结果落显存
    读 num          (MN)
    写 out          (MN)
    ... 外加 max / sum 两次规约各读一遍

理想情况下 softmax 只需要「读一次 x、写一次 out」= 2MN。
手工拼的版本实际搬了 8MN 左右 —— 4 倍的浪费，全是白搬的。

**这就是算子融合要解决的问题**：把多步计算合并进一个 kernel，
中间结果留在寄存器 / shared memory 里，一次都不写回显存。

跑法：
  python v0_unfused_torch.py
"""

import torch

from _shared import selftest

NAME = "v0 torch 手工拼（未融合）"


def softmax(x: torch.Tensor):
    """故意不用 torch.softmax —— 那个是融合实现，见 bench.py 里的对比。"""
    assert x.dim() == 2
    z = x - x.max(dim=1, keepdim=True)[0]      # kernel 1(max) + kernel 2(sub)
    numerator = torch.exp(z)                   # kernel 3
    denominator = numerator.sum(dim=1, keepdim=True)   # kernel 4
    return numerator / denominator             # kernel 5


if __name__ == "__main__":
    selftest(softmax, NAME)
    print("\n  → 这个版本的带宽数字很低，但注意：它不是「访存效率低」，")
    print("    而是「访存量本身多了 4 倍」。每个单独的 torch 算子其实都跑在")
    print("    ~85% 带宽上 —— 问题出在它们之间必须过一趟显存。")
    print("\n[练习] 试试看：")
    print("  1. 用 nsys 抓这个函数，数一下到底 launch 了几个 kernel")
    print("     nsys profile -t cuda -o rep --force-overwrite true python v0_unfused_torch.py")
    print("     nsys stats --report cuda_gpu_kern_sum rep.nsys-rep")
    print("  2. 用 torch.compile 包一下这个函数，它能自动融合到什么程度？")
