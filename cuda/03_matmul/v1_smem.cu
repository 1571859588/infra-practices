// v1：shared memory 分块 —— 把算术强度从 0.25 提到 8
//
// v0 的问题是没有复用。观察：一个 BM×BN 的输出块，需要 A 的 BM×K
// 和 B 的 K×BN。如果把它们分成 K/BK 个 BM×BK 和 BK×BN 的小块，
// 每次搬一对小块进 shared memory，块内的每个元素就能被复用多次。
//
//   搬进 shared 的字节数：(BM×BK + BK×BN) × 4
//   用这些数据做的运算数：BM × BN × BK × 2
//   算术强度 = 2·BM·BN·BK / (4·BK·(BM+BN)) = BM·BN / (2·(BM+BN))
//
// BM = BN = 32 时是 **8 FLOP/byte**（v0 是 0.25，提了 32 倍）。
// 还没到拐点 12.5，所以这一版仍然是 memory-bound —— v2 会继续往上推。
//
// ★ 这就是"分块越大越好"的来源：算术强度 ∝ 块边长。
//   限制来自 shared memory 容量（48 KB/block 默认上限）和寄存器数。
//
// 代码结构（每个 k-tile 一轮）：
//   ① 协作搬运：block 里每个线程搬 A 和 B 各一个元素进 shared
//   ② __syncthreads()   ← 等所有人搬完
//   ③ 从 shared 里算 BK 次乘加
//   ④ __syncthreads()   ← 等所有人算完，才能覆盖 shared
//
// ⚠️ ④ 那次同步最容易漏。漏了的话，跑得快的线程会在别人还没算完时
//    就把下一个 tile 写进 shared，结果随机错。`make racecheck` 能查出来。

#include "../common.cuh"
#include "variants.cuh"

// 32×32 的输出块，每个线程负责 1 个输出元素 → 正好 1024 线程。
// BK 也取 32，让搬运逻辑最简单：每个线程各搬 A、B 一个元素。
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 32;

__global__ void matmul_smem(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tx = threadIdx.x;   // 0..BN-1，对应列
    int ty = threadIdx.y;   // 0..BM-1，对应行

    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx;

    float acc = 0.0f;

    for (int kt = 0; kt < K; kt += BK) {
        // ---- ① 协作搬运 ----
        // 越界的位置填 0，而不是 return —— **绝对不能提前 return**：
        // 后面有 __syncthreads()，少一个线程参与就是死锁（或 UB）。
        // 填 0 之后这些位置对 acc 的贡献是 0，结果自然正确。
        int a_col = kt + tx;
        int b_row = kt + ty;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();   // ②

        // ---- ③ 从 shared 里算 ----
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            // As[ty][k]：同一个 warp 内 ty 相同、k 相同 → 广播
            // Bs[k][tx]：tx 连续 → 落在 32 个不同 bank → 无冲突
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();   // ④ 别漏
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

void matmul_v1_smem(const float* A, const float* B, float* C, int M, int N,
                    int K) {
    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_smem<<<grid, block>>>(A, B, C, M, N, K);
}

OccInfo occ_v1_smem() { return make_occ("v1_smem", matmul_smem, BM * BN); }
