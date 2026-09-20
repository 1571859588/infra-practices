// v0：交错寻址 —— 教科书上的第一版，也是问题最多的一版
//
// 归约的基本思路：树形两两相加。block 内 256 个线程，
// 第 1 轮 128 对相加，第 2 轮 64 对，……，7 轮之后剩 1 个。
//
//   轮次 1:  [0+1] [2+3] [4+5] [6+7] ...
//   轮次 2:  [0+2]       [4+6]       ...
//   轮次 3:  [0+4]                   ...
//
// 「交错寻址」指的是：第 s 轮由 tid % (2s) == 0 的线程干活。
//
// ★ 这一版有**两个**性能问题，后面两版各修一个：
//   1. warp divergence（v1 修）
//   2. shared memory bank conflict（v2 修）
//
// 把它们拆开一个个修，才能看清每一步各自值多少 ——
// 这是整个 cuda/ 目录反复强调的方法：一次只改一个变量。

#include "../common.cuh"
#include "reduce_common.cuh"
#include "variants.cuh"

__global__ void reduce_interleaved(const float* in, float* out, int n) {
    // extern __shared__：大小在 launch 时通过第三个 <<<>>> 参数给。
    // shared memory 是 block 内共享的片上内存，
    // 延迟约为 global memory 的 1/20 —— 归约必须用它。
    extern __shared__ float sdata[];

    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    // ---- ① 从 global 读进 shared ----
    // 这一步是合并访存的（相邻 tid 读相邻地址），没问题。
    sdata[tid] = (i < n) ? in[i] : 0.0f;

    // 必须同步：下面要读别的线程写的 sdata。
    // 漏掉 __syncthreads() 是归约最经典的 bug，而且**经常碰巧算对**
    // （block 小于一个 warp 时 warp 内天然同步），到大 block 才暴雷。
    __syncthreads();

    // ---- ② 树形归约 ----
    for (unsigned s = 1; s < blockDim.x; s *= 2) {
        // ★ 问题 1：warp divergence
        // 一个 warp 里 32 个线程必须走同一条指令路径。
        // `tid % (2*s) == 0` 在 s=1 时让 tid 为偶数的线程干活 ——
        // 一个 warp 里一半干活一半空转，但**两批都要各走一遍**，
        // 硬件是串行执行两个分支的。等于一半的算力白扔。
        // s 越大越糟：s=32 时整个 warp 只有 1 个线程在干活。
        //
        // ★ 问题 2：bank conflict
        // shared memory 分 32 个 bank，地址每 4 字节轮一个 bank。
        // s=1 时干活的线程访问 sdata[0], sdata[2], sdata[4]...
        // 间隔 2 → 只用到一半的 bank，每个 bank 被 2 个线程同时访问
        // → 2-way conflict，访问被串行化成 2 次。s 越大冲突越严重。
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // ---- ③ 0 号线程写出这个 block 的部分和 ----
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

void reduce_v0_interleaved(const float* d_in, float* sa, float* sb,
                           float* d_result, int n, int block) {
    run_multipass<reduce_interleaved, 1>(d_in, sa, sb, d_result, n, block);
}
