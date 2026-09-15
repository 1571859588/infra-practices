// cuda_mul2.cu —— 向量元素乘 2 的原生 CUDA 实现
//
// 编译成两个产物（见 Makefile）：
//   1. libmul2.so   共享库，给 bench_all.py 用 ctypes 调用
//   2. cuda_mul2    独立可执行文件（配合 main.cu），给 nsys/ncu 单独 profile 用
//
// 这里刻意保留最朴素的 grid-stride 之前的写法（一个线程处理一个元素），
// 和 README 里的 Triton / PyTorch 版本保持语义一致，便于对比。

#include <cuda_runtime.h>
#include <cstdio>

// ---------------------------------------------------------------------------
// 核函数：每个线程负责一个元素
//   idx = blockIdx.x * blockDim.x + threadIdx.x
//   边界检查 if (idx < n) 是必须的：n 通常不能被 blockDim 整除，
//   最后一个 block 会有多余线程，不检查就会越界写。
// ---------------------------------------------------------------------------
__global__ void cuda_mul2_kernel(const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        y[idx] = x[idx] * 2.0f;
    }
}

// ---------------------------------------------------------------------------
// 向量化版本：每个线程用 float4 一次处理 4 个元素。
// mul2 是纯访存瓶颈的算子，128-bit 的 load/store 能显著减少访存指令数，
// 更容易把 HBM 带宽吃满。README 的 ncu 分析部分会对比这两个版本。
// 要求 n % 4 == 0 且指针 16 字节对齐（cudaMalloc / torch 都满足）。
// ---------------------------------------------------------------------------
__global__ void cuda_mul2_kernel_vec4(const float4* __restrict__ x,
                                      float4* __restrict__ y,
                                      int n_vec4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec4) {
        float4 v = x[idx];
        v.x *= 2.0f; v.y *= 2.0f; v.z *= 2.0f; v.w *= 2.0f;
        y[idx] = v;
    }
}

// ---------------------------------------------------------------------------
// 【教学用】故意写错的版本：去掉了 if (idx < n) 边界检查。
// n = 1<<20 且 block = 256 时 n 恰好整除，不会越界，跑起来"看着是对的"；
// 但 n 不是 blockDim 倍数时最后一个 block 会越界写 —— 这种 bug 靠肉眼和
// 单元测试都很难发现，必须用 compute-sanitizer。README 里有复现步骤。
// ---------------------------------------------------------------------------
__global__ void cuda_mul2_kernel_buggy(const float* x, float* y, int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    y[idx] = x[idx] * 2.0f;          // 少了边界检查
}

// ---------------------------------------------------------------------------
// extern "C" 包装：给 Python ctypes 调用。
// 不依赖任何 torch 头文件，所以不受 torch 编译时 CUDA 版本的约束 ——
// 这是本例故意选择 ctypes 而非 torch.utils.cpp_extension 的原因。
// stream 传 0 表示默认流；从 Python 传 torch 当前流的句柄可以避免额外同步。
// ---------------------------------------------------------------------------
extern "C" {

void launch_mul2(const float* x, float* y, int n_elements,
                 int block_size, void* stream) {
    int grid = (n_elements + block_size - 1) / block_size;
    cuda_mul2_kernel<<<grid, block_size, 0, (cudaStream_t)stream>>>(x, y, n_elements);
}

void launch_mul2_vec4(const float* x, float* y, int n_elements,
                      int block_size, void* stream) {
    int n_vec4 = n_elements / 4;                     // 调用方保证 n % 4 == 0
    int grid = (n_vec4 + block_size - 1) / block_size;
    cuda_mul2_kernel_vec4<<<grid, block_size, 0, (cudaStream_t)stream>>>(
        (const float4*)x, (float4*)y, n_vec4);
}

void launch_mul2_buggy(const float* x, float* y, int n_elements,
                       int block_size, void* stream) {
    int grid = (n_elements + block_size - 1) / block_size;
    cuda_mul2_kernel_buggy<<<grid, block_size, 0, (cudaStream_t)stream>>>(x, y, n_elements);
}

// 返回最近一次 CUDA 错误码，0 表示无错误。ctypes 侧用它做检查。
int mul2_last_error() {
    return (int)cudaGetLastError();
}

}  // extern "C"
