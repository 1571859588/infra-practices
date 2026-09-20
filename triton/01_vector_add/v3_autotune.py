"""v3：用 @triton.autotune 让 Triton 自己选配置

原理：v0/v2 里的 BLOCK_SIZE、num_warps、UNROLL 都是手填的。手填的问题是
「最优值依赖 GPU 型号和问题规模」—— 换张卡、换个 n 就得重新扫。
@triton.autotune 把候选配置列出来，第一次遇到某个 key 组合时把所有配置
各跑一遍，记下最快的那个，之后同样的 key 直接复用。

三个要点：

1. `key=["n_elements"]`：**缓存的键**。n_elements 变了才重新 tune。
   key 选太细（比如把每个 shape 都当 key）会导致反复 tune；
   选太粗会让不同规模共用一个不合适的配置。

2. **第一次调用会很慢**（要试完所有 config），所以 benchmark 必须 warmup，
   否则测到的是 tuning 时间。

3. 用 `TRITON_PRINT_AUTOTUNING=1` 可以打印它最终选了哪个配置：
       TRITON_PRINT_AUTOTUNING=1 python v3_autotune.py

autotune 也可以直接调 `num_warps` / `num_stages`（作为 Config 的关键字参数），
这些不是 kernel 签名里的 constexpr，而是 launch 参数。

跑法：
  python v3_autotune.py
  TRITON_PRINT_AUTOTUNING=1 python v3_autotune.py
"""

import torch
import triton
import triton.language as tl

from _shared import selftest

NAME = "v3 autotune"


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256,  "UNROLL": 1}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024, "UNROLL": 1}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024, "UNROLL": 4}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024, "UNROLL": 8}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048, "UNROLL": 2}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096, "UNROLL": 1}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096, "UNROLL": 4}, num_warps=16),
    ],
    key=["n_elements"],     # n_elements 变了才重新 tune
)
@triton.jit
def add_autotuned_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base = pid * BLOCK_SIZE * UNROLL
    for i in tl.static_range(UNROLL):
        offsets = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda and x.shape == y.shape
    assert x.is_contiguous() and y.is_contiguous()
    out = torch.empty_like(x)
    n = x.numel()

    # grid 必须是 lambda：autotune 选中的 BLOCK_SIZE / UNROLL 会通过 meta 传进来，
    # 写成固定 tuple 的话就没法随配置变了。
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"] * meta["UNROLL"]),)
    add_autotuned_kernel[grid](x, y, out, n)
    return out


def best_config(n: int):
    """读出 autotune 给某个 n 选中的配置（tune 过之后才有）。"""
    cache = add_autotuned_kernel.cache
    # cache 的 key 结构随版本变化，这里只做尽力展示，取不到就返回 None
    for v in cache.values():
        return v
    return None


if __name__ == "__main__":
    selftest(add, NAME)

    cfg = best_config(1 << 24)
    print(f"\n  autotune 选中的配置：{cfg}")
    print("  （也可以用 TRITON_PRINT_AUTOTUNING=1 让 Triton 自己打印）")

    print("\n[练习] 试试看：")
    print("  1. 把 key 改成 []（空），再用两个不同的 n 调用，会发生什么？")
    print("  2. 往 configs 里加一个明显很差的配置（BLOCK_SIZE=32），")
    print("     确认 autotune 不会选它 —— 顺便体会「候选越多第一次越慢」")
    print("  3. 对比 v3 选中的配置和你在 bench.py 里手动扫出来的最优，一致吗？")
