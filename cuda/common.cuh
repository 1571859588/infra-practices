// common.cuh —— cuda/ 下所有练习共用的脚手架
//
// 只放三类东西：错误检查、计时、结果报告。
// 刻意不做封装框架 —— 每个练习的 kernel 都应该能一眼看懂，
// 不需要先读懂一层抽象。

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// ① 错误检查
// ---------------------------------------------------------------------------
// CUDA 的 API 全都返回错误码而不抛异常，不查就会静默出错 ——
// 典型症状是「结果不对但程序不报错」，能查半天。
// 所有 cudaXxx 调用都该包一层。
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n  -> %s\n", __FILE__,    \
                    __LINE__, #call, cudaGetErrorString(err_));               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// kernel launch 是异步的，launch 本身的错误（比如 block 太大）要单独查；
// 而 kernel 执行期间的错误（比如越界访问）只能在同步之后才看得到。
#define CUDA_CHECK_KERNEL()                                                   \
    do {                                                                      \
        CUDA_CHECK(cudaGetLastError());                                       \
        CUDA_CHECK(cudaDeviceSynchronize());                                  \
    } while (0)

// ---------------------------------------------------------------------------
// ② 计时
// ---------------------------------------------------------------------------
// 用 CUDA Event 而不是 CPU 端的 chrono：event 记录在 GPU 的时间线上，
// 不受 CPU 端异步提交的影响。
//
// 三个必须做的事，少一个数据就不可信：
//   - warmup：第一次 launch 包含 module 载入、JIT、cache 冷启动
//   - 重复多次取中位数：GPU 有 DVFS，单次测量抖动很大
//   - 同步：不同步的话测到的是「提交完成」而不是「执行完成」
template <typename F>
float bench(F&& fn, int warmup = 10, int iters = 50) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));

    std::vector<float> ms(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(beg));
        fn();
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaEventElapsedTime(&ms[i], beg, end));
    }
    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));

    // 中位数比均值稳：偶尔被别的进程抢一下不会污染结果
    std::sort(ms.begin(), ms.end());
    return ms[iters / 2];
}

// ⚠️ `bench()` 自带的 warmup 只够热「这一个 kernel」（module 载入、
// cache 冷启动），**不够把 GPU 的时钟从 idle 拉上来**。
// 实测：进程刚起来时第一个被测的变体会慢 20~30%，后面的都正常 ——
// 于是「第一个变体」白白背了一口锅，整张对比表的第一行是错的。
//
// 所以在做任何计时之前，先用一个访存密集的 kernel 空转一下。
// 200 ms 左右足够 A100 把 SM/显存时钟升到稳态。
static __global__ void warmup_kernel(float* buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = buf[i] * 1.0001f + 1.0f;
}

inline void warmup_gpu(int ms = 200) {
    const int n = 1 << 22;   // 16 MB，装得进 L2，纯粹是为了让 SM 忙起来
    float* buf = nullptr;
    if (cudaMalloc(&buf, size_t(n) * sizeof(float)) != cudaSuccess) return;
    cudaMemset(buf, 0, size_t(n) * sizeof(float));

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventRecord(beg);
    float elapsed = 0.0f;
    while (elapsed < float(ms)) {
        for (int k = 0; k < 50; ++k)
            warmup_kernel<<<(n + 255) / 256, 256>>>(buf, n);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed, beg, end);
    }
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    cudaFree(buf);
}

// ---------------------------------------------------------------------------
// ③ 硬件参数与报告
// ---------------------------------------------------------------------------
// A100-SXM4-40GB 实测值（common.cuh 里 print_device_info() 会打印真实值）：
//   108 SM / 40 MB L2 / 164 KB shared per SM / 峰值带宽 1555.2 GB/s
inline double peak_bandwidth_gbs() {
    static double cached = 0.0;
    if (cached == 0.0) {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
        // memoryClockRate 单位 kHz，×2 是 DDR 的双沿传输
        cached = 2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1e6;
    }
    return cached;
}

inline int num_sms() {
    static int cached = 0;
    if (cached == 0) {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
        cached = p.multiProcessorCount;
    }
    return cached;
}

inline void print_device_info() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("设备: %s\n", p.name);
    printf("  SM 数量          : %d\n", p.multiProcessorCount);
    printf("  L2 cache         : %.0f MB\n", p.l2CacheSize / 1048576.0);
    printf("  shared / block   : %.0f KB (默认) / %.0f KB (opt-in 上限)\n",
           p.sharedMemPerBlock / 1024.0, p.sharedMemPerMultiprocessor / 1024.0);
    printf("  峰值显存带宽     : %.1f GB/s\n", peak_bandwidth_gbs());
    printf("  compute capability: %d.%d\n", p.major, p.minor);
}

// memory-bound kernel：报告达到的带宽和占峰值的比例。
// bytes 传「理想访存量」（该读的读一遍、该写的写一遍），
// 这样实现得差的版本会被正确地惩罚 —— 理由见
// ../triton/02_fused_softmax/_shared.py 里的长注释。
inline void report_bw(const char* name, float ms, double bytes,
                      const char* extra = "") {
    double gbs = bytes / (ms * 1e-3) / 1e9;
    printf("  %-30s %8.1f us  %8.1f GB/s  (%5.1f%% of peak) %s\n", name,
           ms * 1e3, gbs, 100.0 * gbs / peak_bandwidth_gbs(), extra);
}

// compute-bound kernel：报告算力。
// fp32 FMA 峰值：A100 是 19.5 TFLOP/s（非 Tensor Core）
inline void report_flops(const char* name, float ms, double flops,
                         double peak_tflops = 19.5, const char* extra = "") {
    double tf = flops / (ms * 1e-3) / 1e12;
    printf("  %-30s %8.1f us  %8.2f TFLOP/s  (%5.1f%% of peak) %s\n", name,
           ms * 1e3, tf, 100.0 * tf / peak_tflops, extra);
}

// ---------------------------------------------------------------------------
// ④ 正确性检查
// ---------------------------------------------------------------------------
// 每个变体都必须过这一关才谈性能。特别要测**非整除**的尺寸：
// 边界处理写错是 CUDA 最常见的 bug，而整除的尺寸测不出来。
inline bool check(const std::vector<float>& got, const std::vector<float>& ref,
                  const std::string& label, float atol = 1e-5f,
                  float rtol = 1e-5f) {
    if (got.size() != ref.size()) {
        printf("  [FAIL] %-24s 长度不一致 %zu vs %zu\n", label.c_str(),
               got.size(), ref.size());
        return false;
    }
    double max_abs = 0.0;
    size_t bad = 0, first_bad = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        double diff = std::fabs(double(got[i]) - double(ref[i]));
        double tol = atol + rtol * std::fabs(double(ref[i]));
        if (diff > max_abs) max_abs = diff;
        if (diff > tol || std::isnan(diff)) {
            if (bad == 0) first_bad = i;
            ++bad;
        }
    }
    if (bad == 0) {
        printf("  [PASS] %-24s max_abs_err = %.3g\n", label.c_str(), max_abs);
        return true;
    }
    printf("  [FAIL] %-24s %zu/%zu 个元素超差, 首个在 i=%zu "
           "(got %.6g vs ref %.6g), max_abs_err = %.3g\n",
           label.c_str(), bad, got.size(), first_bad, got[first_bad],
           ref[first_bad], max_abs);
    return false;
}
