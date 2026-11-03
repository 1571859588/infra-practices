// v3：融合演示 —— 单独的 ReLU kernel 降到 ~85% 带宽就到头了，
//     但真正的优化是让这 8 字节/元素的访存**根本不发生**。
//
// 场景：out = relu(x + y)。两种写法：
//
//   两个 kernel：tmp = x + y（12 B/elem：读 x 读 y 写 tmp）
//                out = relu(tmp)（8 B/elem：读 tmp 写 out）
//                合计 20 B/elem，中间结果 tmp 在显存里白住了一趟。
//
//   融合 kernel：out = relu(x + y)（12 B/elem）
//                add 的结果在寄存器里直接进 relu，tmp 根本不落地。
//
// 预期：融合版快 ~20/12 = 1.67 倍。这正是 PyTorch 里
// nn.Sequential 里两个 module 的写法 vs torch.compile / Triton
// 融合的差别 —— 对照 ../../triton/03_matmul/v3_fused_relu.py，
// 同一个思想在 Triton 里只要在 kernel 里加一行 tl.maximum。
//
// ⚠️ 注意这是「ReLU 唯一有意义的优化」的根本原因：
// 单独的 ReLU 已经贴着带宽天花板了（84%+），再怎么改 kernel
// 写法也只有 ±1% 的噪声；但作为**别人的 epilogue**，
// 它的一次显存往返（读+写）可以整个消失。

#include "../common.cuh"
#include "variants.cuh"

__global__ void add_kernel(const float* __restrict__ x, const float* __restrict__ y,
                           float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}

// 复用 v0 的 kernel 会跨编译单元链接，这里干脆各自持有 ——
// 每个变体文件独立可读，比省几行代码重要。
__global__ void relu_kernel(const float* __restrict__ x, float* __restrict__ out,
                            int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(x[i], 0.0f);
}

__global__ void add_relu_fused(const float* __restrict__ x,
                               const float* __restrict__ y,
                               float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = x[i] + y[i];
        out[i] = fmaxf(v, 0.0f);
    }
}

void launch_add_relu_two(const float* x, const float* y, float* tmp, float* out,
                         int n, int block) {
    if (block == 0) block = 256;
    int grid = (n + block - 1) / block;
    add_kernel<<<grid, block>>>(x, y, tmp, n);
    relu_kernel<<<grid, block>>>(tmp, out, n);
}

void launch_add_relu_fused(const float* x, const float* y, float* out, int n,
                           int block) {
    if (block == 0) block = 256;
    int grid = (n + block - 1) / block;
    add_relu_fused<<<grid, block>>>(x, y, out, n);
}
