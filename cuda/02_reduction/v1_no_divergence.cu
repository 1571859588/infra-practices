// v1：消除 warp divergence（但 bank conflict 还在）
//
// v0 的问题是「干活的线程散布在整个 block 里」：
//     s=1 时 tid = 0,2,4,6,... 在干活
// 每个 warp 都是一半干一半闲，32 个 warp 全都 divergent。
//
// 修法：**不改变谁和谁相加，只改变由哪个线程来做这次加法**。
// 让前面连续的线程去干活：
//     s=1 时 tid = 0,1,2,3,... 干活，但操作的是 sdata[0]+=sdata[1],
//                                              sdata[2]+=sdata[3], ...
//
// 索引换算：`int index = 2 * s * tid;`
//
// 效果：干活的线程变成**连续的一段**。前几个 warp 全员干活，
// 后面的 warp 全员空闲 —— 全员空闲的 warp 直接不被调度，不占算力。
// divergence 只剩最后一个边界 warp。
//
// ⚠️ 但 bank conflict **更严重了**，v2 才修。
// 这正是「一次只改一个变量」的价值：如果直接从 v0 跳到 v2，
// 你会以为 1.x 倍的提升全是消除 divergence 的功劳。

#include "../common.cuh"
#include "reduce_common.cuh"
#include "variants.cuh"

__global__ void reduce_no_divergence(const float* in, float* out, int n) {
    extern __shared__ float sdata[];

    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned s = 1; s < blockDim.x; s *= 2) {
        // ★ 唯一的改动：用 index 而不是 tid 来定位
        // 干活的线程从「tid 能被 2s 整除」变成「tid < blockDim/(2s)」，
        // 也就是连续的前一段线程。
        unsigned index = 2 * s * tid;

        if (index < blockDim.x) {
            // ⚠️ bank conflict 反而恶化了：
            // s=1 时线程 0,1,2,3 访问 sdata[0],[2],[4],[6] —— 2-way conflict
            // s=2 时线程 0,1,2,3 访问 sdata[0],[4],[8],[12] —— 4-way conflict
            // s=16 时是 32-way，整个 warp 的访问完全串行化。
            // v2 会把这个也解决掉。
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

void reduce_v1_no_divergence(const float* d_in, float* sa, float* sb,
                             float* d_result, int n, int block) {
    run_multipass<reduce_no_divergence, 1>(d_in, sa, sb, d_result, n, block);
}
