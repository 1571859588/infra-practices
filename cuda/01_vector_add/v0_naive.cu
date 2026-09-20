// v0：最朴素的向量加 —— CUDA 的 "hello world"
//
// 学习目标：
//   - 理解 thread / block / grid 三级层次，以及全局线程号怎么算
//   - 理解为什么一定要有边界判断
//   - 建立「合并访存（coalescing）」的概念
//
// 对照 ../../triton/01_vector_add/v0_naive.py：
//   Triton 里一个 program 处理一整个 BLOCK_SIZE 的**向量**，
//   你写的是块与块之间的逻辑，块内的线程分配由编译器决定。
//   CUDA 里你要亲手写到**单个线程**的粒度 —— 这就是两者最根本的差别。

#include "../common.cuh"
#include "variants.cuh"

__global__ void vector_add_naive(const float* __restrict__ x,
                                 const float* __restrict__ y,
                                 float* __restrict__ out, int n) {
    // ---- ① 全局线程号 ----
    // blockIdx.x  : 当前 block 在 grid 里的编号
    // blockDim.x  : 每个 block 有多少线程
    // threadIdx.x : 当前线程在 block 内的编号
    // 三者组合出全局唯一的 i。这是 CUDA 最基本的一行。
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // ---- ② 边界判断，不能省 ----
    // grid 是按 ceil(n / block) 算的，最后一个 block 通常有多余线程。
    // 没有这个 if 就会越界读写 —— 而越界**不一定立刻崩**，
    // 经常表现为「结果偶尔不对」，极难查。
    // 用 compute-sanitizer 可以直接抓出来（见 README §2）。
    if (i < n) {
        // ---- ③ 合并访存 ----
        // 同一个 warp 里的 32 个线程，threadIdx.x 连续 → i 连续 →
        // 访问的地址连续。硬件把这 32 个 float（128 字节）合并成
        // **一次** memory transaction。
        //
        // 这是 GPU 访存的核心规则：warp 内地址越连续，transaction 越少。
        // v3_strided_bad.cu 会故意破坏它，看看代价有多大。
        out[i] = x[i] + y[i];
    }
}

void launch_v0_naive(const float* x, const float* y, float* out, int n,
                     int block) {
    if (block == 0) block = 256;
    int grid = (n + block - 1) / block;   // ceil division
    vector_add_naive<<<grid, block>>>(x, y, out, n);
}
