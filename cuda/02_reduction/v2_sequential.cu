// v2：顺序寻址 —— 同时消除 divergence 和 bank conflict
//
// 换一个完全不同的配对方式：不再是「相邻两个相加」，
// 而是**前半段和后半段相加**，每轮把有效范围砍半。
//
//   blockDim = 8:
//     s=4:  sdata[0]+=sdata[4]  sdata[1]+=sdata[5]  sdata[2]+=sdata[6]  sdata[3]+=sdata[7]
//     s=2:  sdata[0]+=sdata[2]  sdata[1]+=sdata[3]
//     s=1:  sdata[0]+=sdata[1]
//
// 注意循环方向反了：`for (s = blockDim/2; s > 0; s >>= 1)`。
//
// 为什么这样就同时解决了两个问题：
//
//   1. **无 divergence**：干活的是 tid < s，连续的前一段线程。
//      和 v1 一样好。
//
//   2. **无 bank conflict**：干活的线程访问 sdata[tid] 和 sdata[tid+s]，
//      两次访问**内部都是连续的**（tid 连续 → 地址连续 → 32 个不同 bank）。
//      这是关键差别 —— v1 里访问的是 sdata[2*s*tid]，带步长；
//      这里步长恒为 1。
//
// 这一版是整条优化链里**收益最大**的一步，见 README §3.1。

#include "../common.cuh"
#include "reduce_common.cuh"
#include "variants.cuh"

__global__ void reduce_sequential(const float* in, float* out, int n) {
    extern __shared__ float sdata[];

    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    // ★ 从大到小，每轮折半
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // 两次访问的下标都以 tid 为基、步长为 1
            // → warp 内 32 个线程落在 32 个不同 bank，零冲突
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

void reduce_v2_sequential(const float* d_in, float* sa, float* sb,
                          float* d_result, int n, int block) {
    run_multipass<reduce_sequential, 1>(d_in, sa, sb, d_result, n, block);
}
