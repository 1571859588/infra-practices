// bench.cu —— matmul 优化链的完整对比
//
// 跑法见 README.md §2。

#include <cstring>
#include <random>
#include <vector>

#include "../common.cuh"
#include "variants.cuh"

// 正确性用例 {M, N, K}。**故意包含非整除、非 4 倍数、极端长条**：
//   - 128/256 是所有分块大小的整数倍 → 走所有快路径
//   - 129×257×65 谁都不整除 → 测边界处理
//   - 1×4096×4096 和 4096×1×4096 → 退化成矩阵向量乘，暴露 grid 算错
//   - 100×100×3 → K 极小，测 k 循环只跑一轮的情况
static const int CASES[][3] = {
    {128, 128, 128},  {256, 512, 128},   {129, 257, 65},
    {1, 4096, 4096},  {4096, 1, 4096},   {100, 100, 3},
    {512, 512, 511},  {1000, 1000, 1000},
};
static const int NCASES = sizeof(CASES) / sizeof(CASES[0]);

static const int SZ_BENCH = 4096;   // 主测点：4096³

// matmul 的运算量：每个输出元素 K 次乘 + K 次加
static double nflops(long long M, long long N, long long K) {
    return 2.0 * double(M) * double(N) * double(K);
}

struct Variant {
    const char* name;
    void (*launch)(const float*, const float*, float*, int, int, int);
    bool slow;   // v0 在 4096³ 上要跑好几秒，扫描时跳过
};

static const Variant VARIANTS[] = {
    {"v0 朴素（一线程一元素）", matmul_v0_naive, true},
    {"v1 shared memory 分块", matmul_v1_smem, false},
    {"v2 寄存器分块 64x64", matmul_v2_regtile, false},
    {"v3 float4 + 128x128", matmul_v3_vec4, false},
    {"v4 cuBLAS（官方基线）", matmul_v4_cublas, false},
};
static const int NV = sizeof(VARIANTS) / sizeof(VARIANTS[0]);

// ---------------------------------------------------------------------------

struct Buffers {
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    std::vector<float> hA, hB;
    int M, N, K;

    Buffers(int M_, int N_, int K_) : M(M_), N(N_), K(K_) {
        hA.resize(size_t(M) * K);
        hB.resize(size_t(K) * N);
        std::mt19937 rng(12345);
        // 用 [-1,1] 而不是 [0,1)：全正数累加会让误差单调增长，
        // 掩盖不掉真正的 bug（符号错、索引错都可能还"差不多对"）。
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : hA) v = dist(rng);
        for (auto& v : hB) v = dist(rng);

        CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, size_t(M) * N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    ~Buffers() {
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }
    std::vector<float> read_c() const {
        std::vector<float> out(size_t(M) * N);
        CUDA_CHECK(cudaMemcpy(out.data(), dC, out.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        return out;
    }
};

// CPU 参考实现，double 累加。
// 只在小用例上跑 —— 4096³ 在 CPU 上要几分钟，没必要。
static std::vector<float> cpu_ref(const Buffers& b) {
    std::vector<float> C(size_t(b.M) * b.N);
    for (int i = 0; i < b.M; ++i)
        for (int j = 0; j < b.N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < b.K; ++k)
                acc += double(b.hA[size_t(i) * b.K + k]) *
                       double(b.hB[size_t(k) * b.N + j]);
            C[size_t(i) * b.N + j] = float(acc);
        }
    return C;
}

static bool correctness() {
    printf("\n[正确性] 所有变体 × 所有用例\n");
    printf("  用例含非整除尺寸、K=3 的极端情况、以及退化成矩阵向量乘的长条\n");
    printf("  参考值：CPU double 累加。容差按 K 放大（见 README §2）\n");
    bool all_ok = true;
    for (int c = 0; c < NCASES; ++c) {
        int M = CASES[c][0], N = CASES[c][1], K = CASES[c][2];
        Buffers buf(M, N, K);
        std::vector<float> ref = cpu_ref(buf);
        printf("  --- M=%d N=%d K=%d ---\n", M, N, K);
        for (int v = 0; v < NV; ++v) {
            // 先把 C 填成 NaN：kernel 如果根本没启动（grid 算成 0）
            // 或者漏写了某些位置，结果里会留下 NaN 而不是恰好的 0。
            CUDA_CHECK(cudaMemset(buf.dC, 0x7F, size_t(M) * N * sizeof(float)));
            VARIANTS[v].launch(buf.dA, buf.dB, buf.dC, M, N, K);
            CUDA_CHECK_KERNEL();
            std::vector<float> got = buf.read_c();
            // fp32 累加 K 次，误差按 sqrt(K) 量级增长（随机符号下）。
            // 固定 1e-5 在 K=4096 时会误报，所以容差跟着 K 走。
            float tol = 1e-5f * std::sqrt(float(K)) * 4.0f;
            all_ok &= check(got, ref, VARIANTS[v].name, tol, tol);
        }
    }
    return all_ok;
}

static void head_to_head() {
    int S = SZ_BENCH;
    printf("\n[变体对比] %d x %d x %d（fp32，非 Tensor Core）\n", S, S, S);
    printf("  峰值 19.5 TFLOP/s；访存 3 x %d^2 x 4 = %.0f MB，算力 %.1f GFLOP\n",
           S, 3.0 * S * S * 4 / 1048576.0, nflops(S, S, S) / 1e9);
    Buffers buf(S, S, S);
    double nf = nflops(S, S, S);

    float ms_all[NV] = {};
    for (int v = 0; v < NV; ++v) {
        // v0 在 4096³ 上很慢，少跑几次省时间
        int warm = VARIANTS[v].slow ? 2 : 5;
        int iters = VARIANTS[v].slow ? 5 : 20;
        float ms = bench(
            [&] { VARIANTS[v].launch(buf.dA, buf.dB, buf.dC, S, S, S); }, warm,
            iters);
        char extra[96] = "";
        if (v > 0)
            snprintf(extra, sizeof(extra), "较上一版 %.2fx", ms_all[v - 1] / ms);
        report_flops(VARIANTS[v].name, ms, nf, 19.5, extra);
        ms_all[v] = ms;
    }
    printf("  → v0 → v3 总共 %.1fx；v3 达到 cuBLAS 的 %.1f%%\n",
           ms_all[0] / ms_all[3], 100.0 * ms_all[4] / ms_all[3]);
}

// 占用率一览。放在 benchmark 前面打印，是因为它解释了后面的数字 ——
// 尤其是「v3 占用率最低却最快」这个反直觉的结果（见 README §3.4）。
static void show_occupancy() {
    printf("\n[占用率] 静态资源用量（CUDA Occupancy API，不需要 root）\n");
    // 表头用纯 ASCII：中文在终端占 2 列但 printf 按字节补齐，混排会错位
    printf("  %-18s %7s %6s %9s %7s %9s\n", "variant", "thr/blk", "regs",
           "smem/blk", "blk/SM", "occupancy");
    OccInfo os[] = {occ_v0_naive(), occ_v1_smem(), occ_v2_regtile(),
                    occ_v3_vec4()};
    for (auto& o : os)
        printf("  %-18s %7d %6d %7d B %7d %8.1f%%\n", o.name,
               o.threads_per_block, o.regs_per_thread, o.smem_per_block,
               o.blocks_per_sm, 100.0 * o.occupancy);
}

static void sweep_size() {
    printf("\n[扫描] 矩阵规模 —— 小矩阵上分块的优势还在吗？\n");
    for (int S : {256, 512, 1024, 2048, 4096}) {
        Buffers buf(S, S, S);
        double nf = nflops(S, S, S);
        printf("  --- %d^3 ---\n", S);
        for (int v = 1; v < NV; ++v) {   // 跳过 v0
            float ms = bench(
                [&] { VARIANTS[v].launch(buf.dA, buf.dB, buf.dC, S, S, S); }, 5,
                20);
            report_flops(VARIANTS[v].name, ms, nf);
        }
    }
}

// v3 的快路径只在整除时生效。这里测它退化时损失多少 ——
// 也就是「真实库为什么要维护一堆 shape 专用 kernel」的量化答案。
static void sweep_alignment() {
    printf("\n[扫描] v3 的快路径 vs 退化路径\n");
    printf("  v3 只在 M%%128==0 && N%%128==0 && K%%8==0 时走 float4 快路径，\n");
    printf("  否则退回 v2。看看差一个元素的代价有多大。\n");
    struct { int M, N, K; const char* note; } cs[] = {
        {4096, 4096, 4096, "全整除 → 快路径"},
        {4096, 4096, 4088, "K 少 8   → 仍是快路径"},
        {4096, 4096, 4095, "K 少 1   → 退化"},
        {4095, 4096, 4096, "M 少 1   → 退化"},
    };
    for (auto& c : cs) {
        Buffers buf(c.M, c.N, c.K);
        double nf = nflops(c.M, c.N, c.K);
        float ms = bench(
            [&] { matmul_v3_vec4(buf.dA, buf.dB, buf.dC, c.M, c.N, c.K); }, 5,
            20);
        char label[80];
        snprintf(label, sizeof(label), "%dx%dx%d", c.M, c.N, c.K);
        report_flops(label, ms, nf, 19.5, c.note);
    }
}

int main(int argc, char** argv) {
    bool quick = (argc > 1 && strcmp(argv[1], "--quick") == 0);
    bool check_only = (argc > 1 && strcmp(argv[1], "--check") == 0);

    printf("======================================================================\n");
    printf("cuda/03_matmul 变体对比\n");
    printf("======================================================================\n");
    print_device_info();
    cublas_init();

    bool ok = correctness();
    if (check_only) {
        printf("\n正确性：%s（--check 模式，跳过 benchmark）\n",
               ok ? "全部通过" : "有失败");
        cublas_destroy();
        return ok ? 0 : 1;
    }

    show_occupancy();

    printf("\n（预热 GPU 时钟中……）\n");
    warmup_gpu();

    head_to_head();
    if (!quick) {
        sweep_size();
        sweep_alignment();
    }

    printf("\n======================================================================\n");
    printf("正确性：%s\n", ok ? "全部通过" : "有失败");
    printf("结论见 README.md §3\n");
    printf("======================================================================\n");
    cublas_destroy();
    return ok ? 0 : 1;
}
