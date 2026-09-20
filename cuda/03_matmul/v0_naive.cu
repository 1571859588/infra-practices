// v0：朴素实现 —— 一个线程算一个输出元素
//
// 最直白的翻译：C[i][j] = sum_k A[i][k] * B[k][j]。
// 三重循环里的外两层交给 GPU 并行，最内层 k 循环留在线程里。
//
// ★ 这一版的问题不是"慢"，是**算术强度太低**。
//
// 算一个 C[i][j] 要：读 K 个 A 的元素 + K 个 B 的元素 = 8K 字节，
// 做 2K 次浮点运算（1 乘 1 加）。算术强度 = 2K / 8K = **0.25 FLOP/byte**。
//
// A100 的 roofline 拐点：19.5 TFLOP/s ÷ 1555 GB/s ≈ **12.5 FLOP/byte**。
// 0.25 远在拐点左边 → 完全是 memory-bound，算力用不上。
//
// 整个 matmul 优化的主线就一句话：**把算术强度提上去**。
// 手段是复用 —— 同一个 A[i][k] 被 N 个输出元素用到，
// 同一个 B[k][j] 被 M 个用到。v0 完全没复用，每次都重新从
// global memory 读（好在有 L1/L2 兜着，不然更惨）。
//
// 唯一做对的一件事见下面 `col` 的注释：**让相邻线程算相邻的列**。

#include "../common.cuh"
#include "variants.cuh"

__global__ void matmul_naive(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
    // ⚠️ 这两行的搭配是有讲究的：
    //   row 跟 y 走，col 跟 x 走。
    // 因为 threadIdx.x 是 warp 内连续变化的那一维，让它对应 col，
    // 一个 warp 的 32 个线程就在读 B 的同一行的连续 32 列 → 合并访存；
    // 写 C 也是连续的 32 列 → 合并访存。
    //
    // 反过来写（row 跟 x 走）功能完全正确，但访存全不合并，
    // 实测慢 10 倍以上。见 README §3.1。
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // 累加器放寄存器里。用 float 而不是先写回 C 再累加 ——
    // 后者每次迭代都要读写一次 global memory。
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
        //     ^^^^^^^^^^^^^^   同一个 warp 内 row 相同、k 相同
        //                      → 32 个线程读同一个地址 → 广播，很快
        //                      ^^^^^^^^^^^^^^ col 连续 → 合并访存
    }
    C[row * N + col] = acc;
}

void matmul_v0_naive(const float* A, const float* B, float* C, int M, int N,
                     int K) {
    // 16×16 = 256 线程/block。block 的 x 维对应 N（列），y 维对应 M（行）。
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_naive<<<grid, block>>>(A, B, C, M, N, K);
}

OccInfo occ_v0_naive() { return make_occ("v0_naive", matmul_naive, 16 * 16); }
