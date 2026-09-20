// v6：CUB —— 别自己写归约的那个理由
//
// CUB 是 NVIDIA 官方的 CUDA 模板库（随 CUDA Toolkit 一起装，
// 不需要额外依赖）。`cub::DeviceReduce::Sum` 是产品级实现：
// 针对每个架构调过参、处理了所有边界、数值上也更稳。
//
// 这一版存在的意义是**当基线**。前面 v0~v5 的意义在于理解
// 每一步优化在干什么，不是为了在生产里替代 CUB。
//
// CUB 的典型用法是「两段式」：
//   1. 先用 d_temp_storage = nullptr 调一次，问它要多少临时空间
//   2. 分配好，再调一次真正执行
// 这个 API 设计是为了让调用方掌控内存分配（CUB 自己从不 cudaMalloc）。
//
// 这里为了公平对比，把临时空间的分配放到计时**外面** ——
// 只测 kernel 本身。这也是各家 benchmark 的惯例。

#include <cub/cub.cuh>

#include "../common.cuh"
#include "variants.cuh"

// 临时空间只分配一次，之后复用。
// static 局部变量在这里够用；真实项目里该用 allocator 管起来。
static void* g_temp = nullptr;
static size_t g_temp_bytes = 0;

static int g_temp_for_n = -1;

void reduce_v6_cub(const float* d_in, float* /*sa*/, float* /*sb*/,
                   float* d_result, int n, int /*block*/) {
    // 「问大小 + 分配」只在 n 变化时做一次。
    // 放在计时路径里的话，即便它只是 host 侧计算，也会让 GPU 空等 ——
    // 我们的计时用的是 CUDA event，这段空等会被算进去。
    if (n != g_temp_for_n) {
        size_t need = 0;
        CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, need, d_in, d_result, n));
        if (need > g_temp_bytes) {
            if (g_temp) CUDA_CHECK(cudaFree(g_temp));
            CUDA_CHECK(cudaMalloc(&g_temp, need));
            g_temp_bytes = need;
        }
        g_temp_for_n = n;
    }
    size_t bytes = g_temp_bytes;   // CUB 会改写这个参数，传副本
    CUDA_CHECK(cub::DeviceReduce::Sum(g_temp, bytes, d_in, d_result, n));
}
