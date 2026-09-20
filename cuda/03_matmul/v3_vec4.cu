// v3：float4 向量化 + 更大的分块
//
// v2 已经越过 roofline 拐点，进了 compute-bound 区间。再往上要抠的是
// **指令数**，不是访存量：
//
//   ① 搬运用 float4：一条 LDG.E.128 顶 4 条 LDG.E。
//      global→shared 的搬运指令数直接除以 4。
//   ② 从 shared 读到寄存器也用 float4（需要 TN 是 4 的倍数且对齐）。
//   ③ 分块加大到 128×128，TM×TN = 8×8。
//      算术强度 = 128·128/(2·256) = **32 FLOP/byte**，
//      是拐点 12.5 的 2.6 倍 —— 访存彻底不是瓶颈了。
//
// 资源核算（这是能不能编译过的硬约束，写之前先算）：
//   shared memory: (BM×BK + BK×BN) × 4 = (128×8 + 8×128) × 4 = 8192 B = 8 KB
//   线程数       : (128/8) × (128/8) = 16 × 16 = 256
//   寄存器/线程  : acc 8×8 = 64 个，加 a_reg/b_reg 16 个和索引 —— 实测 **123**
//   → 123 按 8 对齐到 128，×256 线程 = 32768，SM 只有 65536 个寄存器，
//     所以**每 SM 只放得下 2 个 block** = 512 线程 / 2048 → occupancy 25%。
//     瓶颈是寄存器，不是 shared（8 KB × 2 = 16 KB，离 164 KB 远得很）。
//
// ★ occupancy 只有 25%，却是自写版本里最快的 —— 这不是矛盾。
//   bench 实测的占用率表（README §3.4）是**反向**相关的：
//     v0/v1 occupancy 100% → 2.42 / 5.30 TFLOP/s
//     v2/v3 occupancy  25% → 10.72 / 16.29 TFLOP/s
//   memory-bound kernel 靠 TLP（多线程）隐藏访存延迟，所以要高 occupancy；
//   compute-bound kernel 靠 ILP（每线程 64 个独立 FFMA）就够了，
//   寄存器换来的复用比多几个 warp 值钱得多。
//   这一点和前面两个 memory-bound 的练习（01/02）正好相反。
//
// ⚠️ float4 要求地址 16 字节对齐。K 和 N 不是 4 的倍数时会崩，
//    所以下面的 launch 函数会**检查再决定用哪条路径**，
//    不满足就退回 v2。这不是偷懒 —— 真实库（cuBLAS 也一样）
//    都是这么做的：对齐的快路径 + 通用的慢路径。

#include "../common.cuh"
#include "variants.cuh"

namespace {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int NTHREAD = (BM / TM) * (BN / TN);   // 16×16 = 256

// 搬运分工：
//   A 块 BM×BK = 128×8 = 1024 个 float = 256 个 float4，256 线程 → 每人 1 个
//   B 块 BK×BN = 8×128 = 1024 个 float = 256 个 float4，每人 1 个
// 正好整除，搬运代码不需要循环。

__global__ __launch_bounds__(NTHREAD) void matmul_vec4(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BK][BM];   // 转置存，理由同 v2
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int t_row = tid / (BN / TN);   // 0..15
    const int t_col = tid % (BN / TN);   // 0..15

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // A 的搬运坐标：每行 BK/4 = 2 个 float4 → 一个线程负责 (tid/2, tid%2)
    const int a_r = tid / (BK / 4);          // 0..127
    const int a_c = (tid % (BK / 4)) * 4;    // 0 或 4
    // B 的搬运坐标：每行 BN/4 = 32 个 float4
    const int b_r = tid / (BN / 4);          // 0..7
    const int b_c = (tid % (BN / 4)) * 4;    // 0,4,...,124

    float acc[TM][TN] = {};

    for (int kt = 0; kt < K; kt += BK) {
        // ---- ① 搬 A，一条指令读 4 个 ----
        {
            const float4 v = *reinterpret_cast<const float4*>(
                &A[(block_row + a_r) * K + kt + a_c]);
            // 转置写入：4 个连续的 k 落到 As 的 4 个不同行，同一列。
            // 这里是 4 次单独的 shared 写（没法向量化），
            // 但换来了计算阶段的无冲突读 —— 划算。
            As[a_c + 0][a_r] = v.x;
            As[a_c + 1][a_r] = v.y;
            As[a_c + 2][a_r] = v.z;
            As[a_c + 3][a_r] = v.w;
        }
        // ---- ② 搬 B，读写都能向量化 ----
        *reinterpret_cast<float4*>(&Bs[b_r][b_c]) =
            *reinterpret_cast<const float4*>(
                &B[(kt + b_r) * N + block_col + b_c]);

        __syncthreads();

        // ---- ③ 外积累加，shared→寄存器也用 float4 ----
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_reg[TM], b_reg[TN];
#pragma unroll
            for (int i = 0; i < TM; i += 4)
                *reinterpret_cast<float4*>(&a_reg[i]) =
                    *reinterpret_cast<const float4*>(&As[k][t_row * TM + i]);
#pragma unroll
            for (int j = 0; j < TN; j += 4)
                *reinterpret_cast<float4*>(&b_reg[j]) =
                    *reinterpret_cast<const float4*>(&Bs[k][t_col * TN + j]);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }

        __syncthreads();
    }

    // ---- ④ 写回，每行 TN=8 个连续元素 = 2 条 STG.E.128 ----
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = block_row + t_row * TM + i;
        const int gc = block_col + t_col * TN;
#pragma unroll
        for (int j = 0; j < TN; j += 4)
            *reinterpret_cast<float4*>(&C[gr * N + gc + j]) =
                *reinterpret_cast<const float4*>(&acc[i][j]);
    }
}

// 快路径能用的条件：
//   - M/N/K 分别是分块大小的整数倍（kernel 内部没有边界判断）
//   - N、K 是 4 的倍数（float4 对齐）
// row-major 下 A 的行首地址间隔 K 个 float，B 和 C 是 N 个 ——
// 只要 K、N 是 4 的倍数且基址 16 字节对齐（cudaMalloc 保证 256 B 对齐），
// 每一行的 float4 访问就都是对齐的。
bool fast_path_ok(int M, int N, int K) {
    return M % BM == 0 && N % BN == 0 && K % BK == 0 && N % 4 == 0 &&
           K % 4 == 0;
}

}   // namespace

OccInfo occ_v3_vec4() { return make_occ("v3_vec4", matmul_vec4, NTHREAD); }

void matmul_v3_vec4(const float* A, const float* B, float* C, int M, int N,
                    int K) {
    if (!fast_path_ok(M, N, K)) {
        // 退回通用版本。真实的库也是这么组织的：
        // 一堆针对特定 shape 的快 kernel + 一个万能的慢 kernel。
        matmul_v2_regtile(A, B, C, M, N, K);
        return;
    }
    dim3 block(NTHREAD);
    dim3 grid(N / BN, M / BM);
    matmul_vec4<<<grid, block>>>(A, B, C, M, N, K);
}
