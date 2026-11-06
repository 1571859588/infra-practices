// v2：grid-stride loop 的两个版本 —— 标量（v2b）和 float4（v2）
//
// 本文件做的是归因实验：§1 的表里 v2（float4+grid）比 v1（float4）
// 慢 6 个点。这 6 个点是谁贡献的？两种可能：
//   a) grid-stride **循环本身**有代价（每轮一次比较 + 跳转，
//      线程数少、每线程多轮，循环开销占比变大）；
//   b) float4 和循环**组合**才出问题（每轮只发 1 条 128 bit 指令，
//      迭代间隙里在途请求断流）。
// 拆开的办法：再加一个**纯标量**的 grid-stride（v2b），每轮 1 个
// 4 B load —— 如果 v2b 掉到和 v2 一样，就是 (a)；如果 v2b 贴着
// v0（82.6%），就是 (b)。结论见 README §3.2。
//
// grid 大小跟硬件走（每 SM 4 个 block），不跟数据量走。
// 01 §3.4 已经证明：满配下它不比 v0 快，但在 grid 开不满的场合
// （每 SM 1 个 block 时标量版只有 71.8%，float4 版有 82.6%），
// 向量化能把在途请求数补回来 —— 那边也有一个标量 vs float4 的
// 同款对照。

#include "../common.cuh"
#include "variants.cuh"

// v2b：纯标量 grid-stride —— 循环体和 v0 一模一样，只是套上了循环
__global__ void relu_grid_scalar(const float* __restrict__ x,
                                 float* __restrict__ out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;   // ★ 整个 grid 的线程总数

    for (int i = tid; i < n; i += stride) {
        out[i] = fmaxf(x[i], 0.0f);
    }
}

// v2：float4 grid-stride
__global__ void relu_vec4_grid(const float* __restrict__ x,
                               float* __restrict__ out, int n4) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;   // ★ 整个 grid 的线程总数

    for (int i = tid; i < n4; i += stride) {
        const float4 a = reinterpret_cast<const float4*>(x)[i];
        float4 c;
        c.x = fmaxf(a.x, 0.0f);
        c.y = fmaxf(a.y, 0.0f);
        c.z = fmaxf(a.z, 0.0f);
        c.w = fmaxf(a.w, 0.0f);
        reinterpret_cast<float4*>(out)[i] = c;
    }
}

// 尾巴 kernel。名字带 _v2 是为了不跟 v1_vec4.cu 里的重名。
__global__ void relu_tail_v2(const float* x, float* out, int start, int n) {
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(x[i], 0.0f);
}

void launch_v2_scalar_grid(const float* x, float* out, int n, int block,
                           int grid) {
    if (block == 0) block = 256;
    if (grid == 0) {
        grid = num_sms() * 4;
        int max_needed = (n + block - 1) / block;
        if (grid > max_needed) grid = max_needed;
    }
    relu_grid_scalar<<<grid, block>>>(x, out, n);
}

void launch_v2_vec4_grid(const float* x, float* out, int n, int block, int grid) {
    if (block == 0) block = 256;
    if (grid == 0) {
        grid = num_sms() * 4;
        int n4 = n / 4;
        int max_needed = (n4 + block - 1) / block;
        if (grid > max_needed) grid = max_needed;
    }

    int n4 = n / 4;
    if (n4 > 0) {
        relu_vec4_grid<<<grid, block>>>(x, out, n4);
    }

    int tail_start = n4 * 4;
    if (n - tail_start > 0) {
        relu_tail_v2<<<1, 32>>>(x, out, tail_start, n);
    }
}
