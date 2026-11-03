// v4：float4 向量化 + grid-stride loop —— v1 和 v2 的结合
//
// 这是 README §6 练习 4 的答案：把 v2 的 grid-stride 换成按 float4 步进。
// v2 在 grid=108（每 SM 一个 block）时只有 ~71%（§3.4）—— 线程太少，
// 在途访存请求不够（MLP 不足，和 §3.1 里 v0 block=32 是同一个病）。
// 按 float4 步进等于每个线程每轮发 4 倍的字节量，
// 那么在 grid 开不满的情况下，v4 能不能追平满配的 v0？
//
// 两个变换叠在一起，各管各的：
//   grid-stride 管「grid 大小和数据量解耦」（v2 的卖点）
//   float4    管「每个线程在途的字节数 ×4」（v1 的卖点）
// 两者正交：合并访存都保住了 —— warp 内相邻线程地址连续，
// 只是每轮的单位从 4 字节变成了 16 字节。
//
// ⚠️ 同样有 v1 的两个前提：16 字节对齐（cudaMalloc 保证）、
// 尾巴单独处理（n4 之后剩下的 0~3 个 float）。

#include "../common.cuh"
#include "variants.cuh"

__global__ void vector_add_vec4_grid(const float* __restrict__ x,
                                     const float* __restrict__ y,
                                     float* __restrict__ out, int n4) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;   // ★ 整个 grid 的线程总数（v2 的规则）

    // 一轮里相邻线程搬相邻的 float4 —— 合并访存保住了；
    // 每个线程每轮在途 32 字节（读写共 48）—— v1 的效果。
    for (int i = tid; i < n4; i += stride) {
        const float4 a = reinterpret_cast<const float4*>(x)[i];
        const float4 b = reinterpret_cast<const float4*>(y)[i];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        reinterpret_cast<float4*>(out)[i] = c;
    }
}

// 尾巴 kernel。名字带 _v4 只是为了不跟 v1_vec4.cu 里的重名 ——
// 两个 .o 会链接进同一个 bench。
__global__ void vector_add_tail_v4(const float* x, const float* y,
                                   float* out, int start, int n) {
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}

void launch_v4_vec4_grid(const float* x, const float* y, float* out, int n,
                         int block, int grid) {
    if (block == 0) block = 256;
    if (grid == 0) {
        // 默认和 v2 一样：每 SM 4 个 block，跟硬件走不跟数据量走
        grid = num_sms() * 4;
        int n4 = n / 4;
        int max_needed = (n4 + block - 1) / block;
        if (grid > max_needed) grid = max_needed;
    }

    int n4 = n / 4;
    if (n4 > 0) {
        vector_add_vec4_grid<<<grid, block>>>(x, y, out, n4);
    }

    int tail_start = n4 * 4;
    if (n - tail_start > 0) {
        vector_add_tail_v4<<<1, 32>>>(x, y, out, tail_start, n);
    }
}
