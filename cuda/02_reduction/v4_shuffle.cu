// v4：warp shuffle —— 最后 32 个元素不走 shared memory
//
// v3 的树形归约到最后几轮，活着的线程已经不足一个 warp 了
// （s = 16, 8, 4, 2, 1 这五轮都在同一个 warp 内）。
// 这几轮里：
//   - `__syncthreads()` 是**多余的** —— 一个 warp 内本来就是同步执行的
//   - 走 shared memory 也是多余的 —— 数据可以直接在寄存器间传
//
// `__shfl_down_sync(mask, var, delta)`：让当前线程直接读到
// **同一个 warp 内** lane+delta 号线程的寄存器值。不经过任何内存。
//
//   v = __shfl_down_sync(0xffffffff, v, 16);   // lane i 拿到 lane i+16 的 v
//
// 五行 shuffle 就把一个 warp 的 32 个值归约成 1 个（在 lane 0 里）。
//
// ⚠️ 两个坑：
//
// 1. **必须用 `_sync` 版本并传对 mask**。老代码里的 `__shfl_down`
//    （无 _sync）在 Volta 之后已经不安全 —— Volta 引入了独立线程调度
//    (independent thread scheduling)，warp 内的线程**不再保证步调一致**。
//    `0xffffffff` 表示「这 32 个 lane 都参与」。
//
// 2. **老教程里的 `volatile float* vsmem` 写法在 Volta 之后是错的**，
//    原因同上。见 README §5。现在一律用 shuffle 或
//    cooperative_groups，不要再写 volatile 那套。

#include "../common.cuh"
#include "reduce_common.cuh"
#include "variants.cuh"

// 一个 warp 有 32 个 lane，写成编译期常量而不是 `warpSize`。
// ⚠️ 这不是风格问题，是 **1.19x 的性能问题**，见 README §3.5：
//    `warpSize` 是运行时特殊寄存器（SASS 里的 WARP_SZ），不是常量，
//    编译器没法展开这个循环 —— 实测 SASS 只有 2 条 SHFL（循环体），
//    换成字面量 32 之后是 10 条（两处调用各展开 5 条）。
constexpr int WARP = 32;

// 把一个 warp 内 32 个 lane 的值归约成 1 个，结果在 lane 0。
// __device__ + inline，会被完全内联展开成 5 条 SHFL 指令。
__device__ __forceinline__ float warp_reduce_sum(float v) {
    // 折半交换，5 轮（32 = 2^5）
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__global__ void reduce_shuffle(const float* in, float* out, int n) {
    extern __shared__ float sdata[];

    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // ---- ① 和 v3 一样：读两个先加一次 ----
    float v = (i < n) ? in[i] : 0.0f;
    if (i + blockDim.x < n) v += in[i + blockDim.x];

    // ---- ② 先在 warp 内部用 shuffle 归约 ----
    // 一步就把 blockDim 个值降到 blockDim/32 个，且完全不碰 shared memory
    v = warp_reduce_sum(v);

    // ---- ③ 每个 warp 的 lane 0 把结果写进 shared ----
    // shared 只需要 blockDim/32 个槽（256 线程 → 8 个），
    // 比 v3 的 256 个槽省 32 倍
    unsigned lane = tid % WARP;
    unsigned warp_id = tid / WARP;
    if (lane == 0) sdata[warp_id] = v;
    __syncthreads();

    // ---- ④ 用第一个 warp 把那几个部分和再归约一次 ----
    unsigned num_warps = blockDim.x / WARP;
    if (warp_id == 0) {
        v = (lane < num_warps) ? sdata[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) out[blockIdx.x] = v;
    }
    // 整个 kernel 只有 **1 次** __syncthreads()，
    // 而 v3 有 log2(blockDim) = 8 次。
}

void reduce_v4_shuffle(const float* d_in, float* sa, float* sb,
                       float* d_result, int n, int block) {
    run_multipass<reduce_shuffle, 2>(d_in, sa, sb, d_result, n, block);
}
