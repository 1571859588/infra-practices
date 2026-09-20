// v2：grid-stride loop —— 让 grid 大小和数据量解耦
//
// v0/v1 的 grid 是 ceil(n/block)，n 有多大就开多少线程。
// grid-stride 反过来：**固定开一批线程**，每个线程用一个循环处理
// 多个元素，步长是整个 grid 的线程总数。
//
//   for (int i = tid; i < n; i += gridDim.x * blockDim.x) ...
//
// 好处：
//   1. **grid 大小可控** —— 可以正好开满 SM（persistent kernel 的雏形），
//      和 ../../triton/02_fused_softmax/v2_persistent.py 是同一个思路
//   2. **任意 n 都能跑**，包括超过 grid 上限的情况
//   3. **线程复用**：循环体里的常量、寄存器初始化只做一次
//   4. 调试友好：`<<<1,1>>>` 就能串行跑一遍，验证逻辑
//
// ⚠️ 关键是**步长必须是 gridDim.x * blockDim.x**（整个 grid 的宽度），
// 不能是 blockDim.x。这样才能保证同一时刻 warp 内的 32 个线程
// 访问的还是连续地址 —— 合并访存不能丢。
// （对比 v3_strided_bad.cu：那里步长取错了层次，性能直接崩。）

#include "../common.cuh"
#include "variants.cuh"

__global__ void vector_add_grid_stride(const float* __restrict__ x,
                                       const float* __restrict__ y,
                                       float* __restrict__ out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;   // ★ 整个 grid 的线程总数

    // 每一轮里，相邻线程访问的还是相邻地址 —— 合并访存保住了。
    // 轮与轮之间跳一整个 grid 的宽度。
    for (int i = tid; i < n; i += stride) {
        out[i] = x[i] + y[i];
    }
}

void launch_v2_grid_stride(const float* x, const float* y, float* out, int n,
                           int block, int grid) {
    if (block == 0) block = 256;
    if (grid == 0) {
        // 默认：每个 SM 给 4 个 block（经验值，够填满流水线又不过量）。
        // 这是 grid-stride 的典型用法 —— grid 跟硬件走，不跟数据量走。
        grid = num_sms() * 4;
        int max_needed = (n + block - 1) / block;
        if (grid > max_needed) grid = max_needed;   // 数据太少就别开那么多
    }
    vector_add_grid_stride<<<grid, block>>>(x, y, out, n);
}
