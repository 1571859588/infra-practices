"""v2：用 @triton.autotune 系统化调分块

v1 的 128×256×64 / warps=8 / stages=3 是人肉扫出来的。问题是这套参数
**只对「A100 + 4096² fp16」这一个组合最优** —— 换尺寸、换卡都得重扫。

autotune 把候选列出来，按 `key=["M","N","K"]` 分别 tune 并缓存。
这是 matmul 这类 compute-bound kernel 上 autotune 真正有价值的场景
（对比 `../01_vector_add/v3_autotune.py`：那里所有配置都一样好，autotune 白跑）。

三个要注意的：

1. **第一次调用很慢** —— 12 个候选各编译 + 各跑一遍。这里其实有**两层缓存**，
   实测值差 4 倍，别搞混：
     - 冷编译缓存（`TRITON_CACHE_DIR` 为空）：**8.0 s**，12 个 kernel 从头编译
     - 热编译缓存（同机器第二次跑这个脚本）：**2.1 s**，编译产物从磁盘读出来，
       但 12 个候选**还是要各跑一遍**才能选出最快的 —— 这 2.1 s 就是 benchmark 本身
   两层缓存的区别：编译产物落磁盘、跨进程有效；autotune 选出来的配置只在
   `kernel.cache` 这个 python dict 里，**进程一退就没了**。
2. **`key` 的粒度**：写 `["M","N","K"]` 意味着每个新尺寸都要重 tune 一次。
   推理服务里 M 是 batch×seqlen，会频繁变 —— 这时候常见做法是把 M 分桶，
   或者干脆只按 `["N","K"]`（权重形状固定）tune。
3. `num_warps` / `num_stages` 是 `triton.Config` 的**关键字参数**，
   不是 kernel 签名里的 constexpr —— 它们是 launch 参数。

跑法：
  python v2_autotune.py
  TRITON_PRINT_AUTOTUNING=1 python v2_autotune.py     # 看每个尺寸选了啥
"""

import torch
import triton
import triton.language as tl

from _shared import selftest

NAME = "v2 autotune"


# 候选配置。取自 v1 的手动扫描结果 + 几个边界值。
# 真实项目里这个列表通常有 20~40 项，覆盖不同的 M/N/K 形状比例。
CONFIGS = [
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8},
                  num_warps=4, num_stages=5),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                  num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8},
                  num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                  num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                  num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
                  num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
                  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_M": 8},
                  num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
                  num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8},
                  num_warps=2, num_stages=5),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32,  "BLOCK_K": 32, "GROUP_M": 8},
                  num_warps=2, num_stages=5),
]


@triton.autotune(configs=CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_autotuned_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "K 维不匹配"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # grid 必须是 lambda：BLOCK_M / BLOCK_N 由 autotune 决定，通过 meta 传进来
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    matmul_autotuned_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def chosen_configs():
    """把 autotune 已经缓存的选择打印出来。"""
    out = {}
    for key, cfg in matmul_autotuned_kernel.cache.items():
        out[key] = str(cfg)
    return out


if __name__ == "__main__":
    import time
    from _shared import rand_pair

    # 先量一下「第一次调用」的代价 —— 这是 autotune 最容易被忽略的成本
    a, b = rand_pair(4096, 4096, 4096)
    torch.cuda.synchronize()
    t0 = time.time()
    matmul(a, b)
    torch.cuda.synchronize()
    print(f"[autotune 首次开销] {len(CONFIGS)} 个候选，"
          f"第一次调用耗时 {time.time() - t0:.1f} s")
    print("  → 实测：冷编译缓存 8.0s / 热编译缓存 2.1s（跑一次就热了）。")
    print("    编译产物跨进程缓存在磁盘上；autotune 选中的配置只在内存里，")
    print("    进程退出就丢，所以每次起进程都要重跑这 2.1s 的 benchmark。")
    print("  → 所以性能测试必须 warmup，否则测到的是这个开销。")
    print("    想看冷缓存：TRITON_CACHE_DIR=/tmp/xxx python v2_autotune.py\n")

    selftest(matmul, NAME)

    print("\n[autotune 的选择] 每个 (M,N,K) 一条")
    for k, v in chosen_configs().items():
        print(f"  {k} → {v}")

    print("\n[练习] 试试看：")
    print("  1. 对比 autotune 选的和 v1 手调的（128x256x64/w8/s3），一致吗？")
    print("  2. key 改成 [] （空），再用两个不同尺寸调用，会发生什么？")
    print("  3. key 改成 ['N','K']（只按权重形状 tune），")
    print("     模拟推理服务里 M=batch*seqlen 频繁变化的场景")
    print("  4. 把 CONFIGS 砍到 3 个，首次开销降多少？最优性能掉多少？")
