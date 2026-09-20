// v2：寄存器分块 —— 一个线程算 TM×TN 个输出
//
// v1 已经把算术强度提到 8，但还是 memory-bound（拐点 12.5）。
// 再往上推，块就得更大 —— 而 BM=BN=64 的话，v1 的"一线程一输出"
// 需要 4096 个线程，超过 block 上限 1024。
//
// 解法：**让一个线程算多个输出**。这样块可以变大，线程数不变。
//
//   BM×BN = 64×64 的输出块，128 个线程，每个线程算 TM×TN = 4×8 = 32 个。
//
// 这带来第二层复用，在**寄存器**里：
//   线程把 A 的一小列（TM 个）和 B 的一小行（TN 个）读进寄存器，
//   然后做 TM×TN 次乘加 —— TM+TN 次 shared 读换来 TM×TN 次运算。
//
//   算术强度（global → shared）= BM·BN / (2·(BM+BN)) = 64·64/(2·128) = **16**
//   已经越过 12.5 的拐点 → 终于进入 compute-bound 区间。
//
// 两层分块的分工，是整个 matmul 优化的核心结构：
//
//   global memory ──BM×BK 块──▶ shared memory ──TM×TN 块──▶ 寄存器
//        慢、大                    中、每 block 共享           快、每线程私有
//
// cuBLAS / CUTLASS 也是这个结构，只是多了 double buffering、
// 更精细的 swizzle、以及 Tensor Core。

#include "../common.cuh"
#include "variants.cuh"

constexpr int BM = 64;   // 一个 block 负责的输出行数
constexpr int BN = 64;   // 一个 block 负责的输出列数
constexpr int BK = 16;   // k 方向一次处理多少
constexpr int TM = 4;    // 一个线程负责的输出行数
constexpr int TN = 8;    // 一个线程负责的输出列数

// 线程排布：(BM/TM) × (BN/TN) = 16 × 8 = 128 个线程
constexpr int TROW = BM / TM;   // 16
constexpr int TCOL = BN / TN;   // 8
constexpr int NTHREAD = TROW * TCOL;

__global__ void matmul_regtile(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BK][BM];   // ★ 转置存放，见下面注释
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int t_row = tid / TCOL;   // 0..TROW-1
    int t_col = tid % TCOL;   // 0..TCOL-1

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // 每个线程的累加器：TM×TN 个，全在寄存器里。
    // 4×8 = 32 个 float = 32 个寄存器。A100 每线程上限 255，够用。
    float acc[TM][TN] = {};

    // 搬运的分工：BM×BK = 1024 个 A 元素，128 个线程 → 每人 8 个。
    //             BK×BN = 1024 个 B 元素，同样每人 8 个。
    constexpr int A_PER_THREAD = BM * BK / NTHREAD;   // 8
    constexpr int B_PER_THREAD = BK * BN / NTHREAD;   // 8

    for (int kt = 0; kt < K; kt += BK) {
        // ---- ① 搬 A：global 里是 row-major 的 BM×BK，存进 shared 时转置 ----
        // 为什么转置？因为计算阶段要读 A 的**一列**（固定 k，连续的 TM 行）。
        // 不转置的话 As[row][k] 沿 row 走，步长是 BK=16 个 float
        // → 16 × 4 = 64 字节步长 → 落进同 1 个 bank → 16 路 bank conflict。
        // 转置成 As[k][row] 之后沿 row 走步长为 1 → 无冲突。
#pragma unroll
        for (int i = 0; i < A_PER_THREAD; ++i) {
            int idx = tid + i * NTHREAD;      // 0..BM*BK-1
            int r = idx / BK;                 // 0..BM-1
            int c = idx % BK;                 // 0..BK-1
            int gr = block_row + r, gc = kt + c;
            As[c][r] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }

        // ---- ② 搬 B：不需要转置，计算时本来就是按行读 ----
#pragma unroll
        for (int i = 0; i < B_PER_THREAD; ++i) {
            int idx = tid + i * NTHREAD;
            int r = idx / BN;                 // 0..BK-1
            int c = idx % BN;                 // 0..BN-1
            int gr = kt + r, gc = block_col + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        __syncthreads();

        // ---- ③ 计算：外积累加 ----
        // 每一个 k，读 TM 个 A 值 + TN 个 B 值到寄存器，
        // 做 TM×TN 次 FMA。这就是"寄存器分块"省下来的东西：
        // 12 次 shared 读换 32 次 FMA，而不是 32 次读换 32 次 FMA。
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_reg[TM], b_reg[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = As[k][t_row * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b_reg[j] = Bs[k][t_col * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a_reg[i] * b_reg[j];
        }

        __syncthreads();
    }

    // ---- ④ 写回 ----
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gr = block_row + t_row * TM + i;
        if (gr >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gc = block_col + t_col * TN + j;
            if (gc < N) C[gr * N + gc] = acc[i][j];
        }
    }
}

void matmul_v2_regtile(const float* A, const float* B, float* C, int M, int N,
                       int K) {
    dim3 block(NTHREAD);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_regtile<<<grid, block>>>(A, B, C, M, N, K);
}

OccInfo occ_v2_regtile() {
    return make_occ("v2_regtile", matmul_regtile, NTHREAD);
}
