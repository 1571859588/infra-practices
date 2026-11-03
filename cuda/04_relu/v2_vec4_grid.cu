// v2：float4 + grid-stride loop —— 01 里 v4 的结构原样搬过来
//
// grid 大小跟硬件走（每 SM 4 个 block），不跟数据量走。
// 01 §3.4 已经证明：满配下它不比 v0 快，但在 grid 开不满的场合
// （每 SM 1 个 block 时 v0 式只有 71.8%，float4 版有 82.6%），
// 向量化能把在途请求数补回来。这里只是验证 ReLU 上同样的结构
// 给出同样的行为 —— 结论见 README §3.4。

#include "../common.cuh"
#include "variants.cuh"

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
