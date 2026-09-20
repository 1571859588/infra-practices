// bench.cu —— 把 01_vector_add 的所有变体放在一起对比
//
// 跑法见 README.md §2。

#include <cstring>
#include <string>
#include <vector>

#include "../common.cuh"
#include "variants.cuh"

// 正确性用例。**故意都不是 4 的倍数 / 不是 block 的倍数** ——
// 边界处理写错只有非整除尺寸才测得出来，整除的尺寸全是假的绿灯。
static const int CASES[] = {1024, 1000, 98765, 1 << 20, (1 << 20) + 3};
static const int N_BENCH = 1 << 24;   // 16.8M 个 float，每个数组 64 MB

// 理想访存量：读 x、读 y、写 out，各一遍
static double nbytes(long long n) { return 3.0 * n * 4.0; }

struct Variant {
    const char* name;
    void (*launch)(const float*, const float*, float*, int, int);
};

static void v2_wrap(const float* x, const float* y, float* o, int n, int b) {
    launch_v2_grid_stride(x, y, o, n, b, 0);
}
static void v3_wrap(const float* x, const float* y, float* o, int n, int b) {
    launch_v3_strided_bad(x, y, o, n, b, 0);   // per_thread 默认 64
}

static const Variant VARIANTS[] = {
    {"v0 朴素（合并访存）", launch_v0_naive},
    {"v1 float4 向量化", launch_v1_vec4},
    {"v2 grid-stride", v2_wrap},
    {"v3 未合并（反面教材）", v3_wrap},
};
static const int NV = sizeof(VARIANTS) / sizeof(VARIANTS[0]);

// ---------------------------------------------------------------------------

struct Buffers {
    float *dx = nullptr, *dy = nullptr, *dout = nullptr;
    std::vector<float> hx, hy, href;
    int n = 0;

    explicit Buffers(int n_) : hx(n_), hy(n_), href(n_), n(n_) {
        for (int i = 0; i < n; ++i) {
            hx[i] = float(i % 1000) * 0.001f;
            hy[i] = float((i * 7) % 997) * 0.002f;
            href[i] = hx[i] + hy[i];
        }
        CUDA_CHECK(cudaMalloc(&dx, size_t(n) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy, size_t(n) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dout, size_t(n) * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), size_t(n) * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dy, hy.data(), size_t(n) * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    ~Buffers() {
        cudaFree(dx);
        cudaFree(dy);
        cudaFree(dout);
    }
    std::vector<float> download() const {
        std::vector<float> out(n);
        CUDA_CHECK(cudaMemcpy(out.data(), dout, size_t(n) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        return out;
    }
};

static bool correctness() {
    printf("\n[正确性] 所有变体 × 所有用例（含非整除尺寸）\n");
    bool all_ok = true;
    for (int v = 0; v < NV; ++v) {
        printf("  %s\n", VARIANTS[v].name);
        for (int n : CASES) {
            Buffers buf(n);
            // 先把 out 填成垃圾值，避免「kernel 根本没写」也能通过
            CUDA_CHECK(cudaMemset(buf.dout, 0xFF, size_t(n) * sizeof(float)));
            VARIANTS[v].launch(buf.dx, buf.dy, buf.dout, n, 0);
            CUDA_CHECK_KERNEL();
            all_ok &= check(buf.download(), buf.href,
                            "  n = " + std::to_string(n), 1e-6f, 1e-6f);
        }
    }
    return all_ok;
}

static void head_to_head() {
    printf("\n[变体对比] n = %d（每个数组 %.0f MB，总访存 %.0f MB）\n", N_BENCH,
           N_BENCH * 4.0 / 1048576.0, nbytes(N_BENCH) / 1048576.0);
    Buffers buf(N_BENCH);
    double bytes = nbytes(N_BENCH);

    // cudaMemcpy D2D 当作「纯搬运」的参考点：它是这台机器上
    // 实际能达到的访存上限，比理论峰值更有参考价值。
    float ms_copy = bench([&] {
        CUDA_CHECK(cudaMemcpy(buf.dout, buf.dx, size_t(N_BENCH) * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    });
    report_bw("cudaMemcpy D2D (参考)", ms_copy, 2.0 * N_BENCH * 4.0,
              "<- 读+写 2N，不是 3N");

    for (int v = 0; v < NV; ++v) {
        const Variant& var = VARIANTS[v];
        float ms = bench([&] { var.launch(buf.dx, buf.dy, buf.dout, N_BENCH, 0); });
        report_bw(var.name, ms, bytes);
    }
}

static void sweep_block() {
    printf("\n[扫描] block 大小 —— memory-bound kernel 上它重要吗？\n");
    Buffers buf(N_BENCH);
    double bytes = nbytes(N_BENCH);
    for (int b : {32, 64, 128, 256, 512, 1024}) {
        char label[64];
        for (int v = 0; v < 2; ++v) {   // 只扫 v0 和 v1
            snprintf(label, sizeof(label), "%s block=%d",
                     v == 0 ? "v0" : "v1", b);
            float ms = bench([&] {
                VARIANTS[v].launch(buf.dx, buf.dy, buf.dout, N_BENCH, b);
            });
            report_bw(label, ms, bytes);
        }
    }
}

static void sweep_grid_stride() {
    printf("\n[扫描] v2 的 grid 大小 —— 开多少 block 才够填满 GPU？\n");
    printf("  （%d 个 SM；grid = SM 数 × 每 SM 的 block 数）\n", num_sms());
    Buffers buf(N_BENCH);
    double bytes = nbytes(N_BENCH);
    for (int per_sm : {1, 2, 4, 8, 16, 32}) {
        int grid = num_sms() * per_sm;
        char label[64];
        snprintf(label, sizeof(label), "grid=%d (%d/SM)", grid, per_sm);
        float ms = bench([&] {
            launch_v2_grid_stride(buf.dx, buf.dy, buf.dout, N_BENCH, 256, grid);
        });
        report_bw(label, ms, bytes);
    }
    // 对照：v0 的 grid 是数据量决定的
    printf("  v0 的 grid 是 %d（= n/256），对比上面\n", (N_BENCH + 255) / 256);
}

// 本练习最有意思的一个扫描：合并访存到底在哪一步失效？
static void sweep_coalescing() {
    printf("\n[扫描] v3 的 per_thread —— 合并访存从哪一步开始崩？\n");
    printf("  per_thread 就是 warp 内相邻线程的地址间距（× 4 字节）。\n");
    printf("  留意两条线：sector = 32 B（8 个 float），"
           "cache line = 128 B（32 个 float）。\n");
    printf("  猜猜是哪一条决定性能？（答案和直觉不一样）\n");
    Buffers buf(N_BENCH);
    double bytes = nbytes(N_BENCH);

    float ms0 = bench([&] { launch_v0_naive(buf.dx, buf.dy, buf.dout, N_BENCH, 0); });
    report_bw("v0 基准（完全合并）", ms0, bytes);

    for (int p : {1, 2, 4, 8, 16, 32, 64, 128, 512, 2048}) {
        char label[96];
        snprintf(label, sizeof(label), "per_thread=%-5d (间距 %d B)", p, p * 4);
        float ms = bench([&] {
            launch_v3_strided_bad(buf.dx, buf.dy, buf.dout, N_BENCH, 256, p);
        });
        char extra[64] = "";
        if (ms > ms0 * 1.1) snprintf(extra, sizeof(extra), "%.2fx 慢于 v0", ms / ms0);
        report_bw(label, ms, bytes, extra);
    }
    printf("  → 分水岭是 **32 字节的 sector**，不是 128 字节的 cache line：\n");
    printf("    间距 ≤ 16 B 无惩罚，到 32 B 就开始掉。\n");
    printf("    而且是渐变不是断崖 —— L1 还能吸收一部分复用。见 README §3.3。\n");
}

static void sweep_size() {
    printf("\n[扫描] 数据规模 —— 多大才能把带宽跑起来？\n");
    double bytes_l2 = 40.0 * 1048576.0;
    for (int shift : {12, 16, 18, 20, 22, 24, 26}) {
        int n = 1 << shift;
        Buffers buf(n);
        double bytes = nbytes(n);
        float ms0 = bench([&] { launch_v0_naive(buf.dx, buf.dy, buf.dout, n, 0); });
        char label[80];
        snprintf(label, sizeof(label), "n=2^%d (%.2f MB)", shift,
                 bytes / 1048576.0);
        report_bw(label, ms0, bytes,
                  bytes < bytes_l2 ? "<- 工作集装得进 L2" : "");
    }
}

int main(int argc, char** argv) {
    bool quick = (argc > 1 && strcmp(argv[1], "--quick") == 0);

    printf("======================================================================\n");
    printf("cuda/01_vector_add 变体对比\n");
    printf("======================================================================\n");
    print_device_info();

    bool ok = correctness();

    // 把 GPU 时钟拉到稳态再计时，否则第一个被测的变体会虚慢 20~30%。
    // 见 ../common.cuh 里 warmup_gpu 的注释、以及
    // ../02_reduction/README.md §5.6（那边因为这个坑写错过一整张表）。
    printf("\n（预热 GPU 时钟中……）\n");
    warmup_gpu();

    head_to_head();
    if (!quick) {
        sweep_block();
        sweep_grid_stride();
        sweep_coalescing();
        sweep_size();
    }

    printf("\n======================================================================\n");
    printf("正确性：%s\n", ok ? "全部通过" : "有失败");
    printf("结论见 README.md §3\n");
    printf("======================================================================\n");
    return ok ? 0 : 1;
}
