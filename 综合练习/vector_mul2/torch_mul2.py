"""torch_mul2.py —— 方案一：纯 PyTorch。

特点：
  - 写法最短，一行搞定，完全不用关心 GPU 硬件细节。
  - 底层走 ATen 的 elementwise kernel（TensorIterator + 向量化访存），
    是 NVIDIA/Meta 调好的通用实现，对简单算子已经接近带宽上限。
  - 代价：每个算子一次 kernel launch + 一次完整的读写显存。
    多个算子串起来时（比如 x*2+1 再 relu）无法融合，中间结果要来回读写显存，
    这就是 Triton / CUDA 存在的意义。

单独运行：
    conda activate cpp
    CUDA_VISIBLE_DEVICES=3 python torch_mul2.py
"""

import torch
from common import bench, check, make_input, report


def torch_vector_mul2(x: torch.Tensor) -> torch.Tensor:
    """最朴素的写法：分配新输出。"""
    return x * 2


def torch_vector_mul2_out(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """写入预分配的 y，和 Triton / CUDA 版本的接口对齐，
    这样三者比较时不含显存分配开销，是公平的对比。"""
    return torch.mul(x, 2, out=y)


def main():
    n = 1 << 24                      # 16.7M 元素 = 64 MiB / buffer
    x = make_input(n)
    y = torch.empty_like(x)

    print(f"PyTorch 实现   n = {n} ({x.numel() * 4 / 2**20:.1f} MiB per buffer)")
    print(f"  torch {torch.__version__}, device = {torch.cuda.get_device_name()}")

    y_ref = torch_vector_mul2(x)
    check(y_ref, x, "torch_vector_mul2")

    torch_vector_mul2_out(x, y)
    check(y, x, "torch_vector_mul2_out")

    ms, gbs = bench(lambda: torch_vector_mul2(x), n)
    report("x * 2 (alloc)", ms, gbs, "含输出张量分配")

    ms, gbs = bench(lambda: torch_vector_mul2_out(x, y), n)
    report("torch.mul(out=)", ms, gbs, "复用输出缓冲")


if __name__ == "__main__":
    main()
