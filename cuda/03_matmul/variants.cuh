// 各变体对外暴露的 launch 函数。
//
// 统一计算 C = A × B，全部 row-major、fp32：
//   A: M×K   B: K×N   C: M×N
//
// 除了 cuBLAS 那版，其余都不做任何 shape 假设 —— M/N/K 可以是任意值，
// 边界由 kernel 内部处理。**非整除的尺寸是必测项**（见 bench.cu 的 CASES）。

#pragma once

void matmul_v0_naive(const float* A, const float* B, float* C, int M, int N,
                     int K);
void matmul_v1_smem(const float* A, const float* B, float* C, int M, int N,
                    int K);
void matmul_v2_regtile(const float* A, const float* B, float* C, int M, int N,
                       int K);
void matmul_v3_vec4(const float* A, const float* B, float* C, int M, int N,
                    int K);
void matmul_v4_cublas(const float* A, const float* B, float* C, int M, int N,
                      int K);

// cuBLAS handle 的生命周期管理（创建一次，别在计时循环里反复建）
void cublas_init();
void cublas_destroy();

// ---------------------------------------------------------------------------
// 占用率（occupancy）自查
//
// 每个变体报告自己的资源用量和理论占用率。用的是 CUDA Occupancy API
// （cudaOccupancyMaxActiveBlocksPerMultiprocessor），**不需要 root**——
// 这一点很重要：ncu 读性能计数器要特权（见 cuda/README.md 的 docker 方案），
// 但占用率是纯静态计算，随便谁都能查。
//
// 为什么每个 .cu 各实现一份：kernel 符号在各自的文件里（v3 还在匿名 namespace
// 里），Occupancy API 要拿到函数指针，只能在定义它的 TU 内部调用。
struct OccInfo {
    const char* name;
    int threads_per_block;   // block 大小
    int regs_per_thread;     // 每线程寄存器
    int smem_per_block;      // 每 block shared memory（字节）
    int blocks_per_sm;       // 理论上每 SM 能同时驻留几个 block
    float occupancy;         // 活跃 warp / 最大 warp，0~1
};

OccInfo occ_v0_naive();
OccInfo occ_v1_smem();
OccInfo occ_v2_regtile();
OccInfo occ_v3_vec4();

// 各 .cu 用这个模板填 OccInfo，别重复写。
template <typename Kernel>
OccInfo make_occ(const char* name, Kernel k, int threads, int dyn_smem = 0) {
    cudaFuncAttributes attr{};
    cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(k));
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, reinterpret_cast<const void*>(k), threads, dyn_smem);

    // A100 (sm_80) 每 SM 最多 64 个 warp。用设备属性查，别写死。
    int max_tpm = 2048;
    cudaDeviceGetAttribute(&max_tpm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);

    OccInfo o;
    o.name = name;
    o.threads_per_block = threads;
    o.regs_per_thread = attr.numRegs;
    o.smem_per_block = int(attr.sharedSizeBytes) + dyn_smem;
    o.blocks_per_sm = blocks;
    o.occupancy = float(blocks * threads) / float(max_tpm);
    return o;
}
