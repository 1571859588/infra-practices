// main.cu —— 独立的 CUDA 版 benchmark 驱动，给 nsys / ncu 单独 profile 用
//
// 用法：  ./cuda_mul2 [n_elements] [iters] [block_size] [buggy]
// 默认：  n = 1<<24 (16.7M floats = 64 MiB), iters = 100, block = 256, buggy = 0
//
// 第 4 个参数传 1 时会额外跑一次"去掉边界检查"的错误 kernel，
// 用来演示 compute-sanitizer 能抓到什么（见 README）。
//   ./cuda_mul2 1000 1 256 1                       # n 不被 256 整除 -> 越界
//   compute-sanitizer --tool memcheck ./cuda_mul2 1000 1 256 1
//
// 流程刻意分成几个清晰阶段，并打上 NVTX 标记，这样在 Nsight Systems 时间线上
// 一眼就能看出 H2D / warmup / 计时循环 / D2H 分别占了多久。

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

extern "C" void launch_mul2(const float*, float*, int, int, void*);
extern "C" void launch_mul2_vec4(const float*, float*, int, int, void*);
extern "C" void launch_mul2_buggy(const float*, float*, int, int, void*);

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

int main(int argc, char** argv) {
    int n     = (argc > 1) ? atoi(argv[1]) : (1 << 24);
    int iters = (argc > 2) ? atoi(argv[2]) : 100;
    int block = (argc > 3) ? atoi(argv[3]) : 256;
    int buggy = (argc > 4) ? atoi(argv[4]) : 0;

    size_t bytes = (size_t)n * sizeof(float);
    printf("n=%d (%.1f MiB per buffer), iters=%d, block=%d\n",
           n, bytes / 1048576.0, iters, block);

    // ---- host 数据 ----
    std::vector<float> h_x(n), h_y(n);
    for (int i = 0; i < n; ++i) h_x[i] = (float)(i % 1000) * 0.001f;

    // ---- device 分配 + H2D ----
    float *d_x = nullptr, *d_y = nullptr;
    nvtxRangePushA("alloc");
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    nvtxRangePop();

    nvtxRangePushA("H2D");
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    nvtxRangePop();

    // ---- warmup：第一次 kernel launch 包含 module load / JIT，必须排除 ----
    nvtxRangePushA("warmup");
    for (int i = 0; i < 10; ++i) launch_mul2(d_x, d_y, n, block, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    // ---- 计时：用 cudaEvent，测的是 GPU 上的时间，不含 host 开销 ----
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // 朴素版
    nvtxRangePushA("bench_scalar");
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) launch_mul2(d_x, d_y, n, block, nullptr);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms_scalar = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_scalar, t0, t1));
    ms_scalar /= iters;
    nvtxRangePop();

    // float4 向量化版
    nvtxRangePushA("bench_vec4");
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) launch_mul2_vec4(d_x, d_y, n, block, nullptr);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms_vec4 = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_vec4, t0, t1));
    ms_vec4 /= iters;
    nvtxRangePop();

    // ---- 可选：跑一次故意越界的 kernel（给 compute-sanitizer 演示用）----
    if (buggy) {
        nvtxRangePushA("buggy");
        printf("!! 运行无边界检查的 kernel（n=%d, block=%d, 尾块多出 %d 个线程）\n",
               n, block, (block - n % block) % block);
        launch_mul2_buggy(d_x, d_y, n, block, nullptr);
        cudaError_t e = cudaDeviceSynchronize();
        printf("!! cudaDeviceSynchronize -> %s\n", cudaGetErrorString(e));
        nvtxRangePop();
    }

    // ---- D2H + 正确性校验 ----
    nvtxRangePushA("D2H");
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    nvtxRangePop();

    double max_err = 0.0;
    for (int i = 0; i < n; ++i)
        max_err = fmax(max_err, fabs((double)h_y[i] - 2.0 * h_x[i]));

    // mul2 读一次写一次，有效访存量 = 2 * n * 4 bytes
    double gb = 2.0 * bytes / 1e9;
    printf("scalar : %8.4f ms   %8.1f GB/s\n", ms_scalar, gb / (ms_scalar / 1e3));
    printf("vec4   : %8.4f ms   %8.1f GB/s\n", ms_vec4,   gb / (ms_vec4   / 1e3));
    printf("max_abs_err = %g  -> %s\n", max_err, max_err < 1e-6 ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return max_err < 1e-6 ? 0 : 1;
}
