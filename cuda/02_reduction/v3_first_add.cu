// v3：load 的时候就先加一次 —— 砍掉一半的 block
//
// v2 已经没有 divergence 也没有 bank conflict 了。还剩什么问题？
//
// **一半的线程从第一轮起就在闲着。** blockDim=256 的 block，
// 第一轮 s=128，只有 128 个线程干活。那开 256 个线程读 256 个元素、
// 立刻就浪费一半，不如**让每个线程读 2 个元素、读的时候顺手加起来**：
//
//   v2:  256 线程 读 256 元素 → 树形归约 8 轮
//   v3:  256 线程 读 512 元素（每人 2 个，先加一次）→ 树形归约 8 轮
//
// 同样的线程数，处理了 2 倍的数据 → **block 数量减半**。
// 归约是 memory-bound 的（每个元素只做 1 次加法），
// block 少一半意味着：
//   - 少一半的 block 启动/调度开销
//   - 少一半的「读进 shared → 同步 → 树形归约」的固定成本
//   - 多趟归约的趟数也可能少一趟
//
// ⚠️ 关键是**两次读要保持合并访存**：
//     i        = blockIdx.x * (blockDim.x*2) + threadIdx.x
//     第二个是 i + blockDim.x    ← 不是 i+1！
// 写成 i*2 和 i*2+1 的话，warp 内地址间隔 8 字节，
// 就踩进 ../01_vector_add/README.md §3.3 那张表里了。

#include "../common.cuh"
#include "reduce_common.cuh"
#include "variants.cuh"

__global__ void reduce_first_add(const float* in, float* out, int n) {
    extern __shared__ float sdata[];

    unsigned tid = threadIdx.x;
    // 每个 block 负责 blockDim.x * 2 个元素
    unsigned i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // ★ 读两个，顺手加起来，再写进 shared
    // 两次读各自都是合并的：第一次 warp 读 [i, i+32)，
    // 第二次读 [i+blockDim, i+blockDim+32)，都是连续的。
    float v = (i < n) ? in[i] : 0.0f;
    if (i + blockDim.x < n) v += in[i + blockDim.x];
    sdata[tid] = v;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

void reduce_v3_first_add(const float* d_in, float* sa, float* sb,
                         float* d_result, int n, int block) {
    // ELEMS_MULT = 2：告诉驱动每个 block 吃 2*blockDim 个元素
    run_multipass<reduce_first_add, 2>(d_in, sa, sb, d_result, n, block);
}
