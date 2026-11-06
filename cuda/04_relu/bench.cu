// bench.cu —— 把 04_relu 的所有变体放在一起对比
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
static const int N_BENCH = 1 << 24;   // 16.8M 个 float，x 64 MB

// 理想访存量：读 x 一遍、写 out 一遍（in-place 也是这两遍）
static double nbytes(long long n) { return 2.0 * n * 4.0; }

struct Variant {
    const char* name;
    void (*launch)(const float*, float*, int, int);
};

static void v2b_wrap(const float* x, float* o, int n, int b) {
    launch_v2_scalar_grid(x, o, n, b, 0);
}
static void v2_wrap(const float* x, float* o, int n, int b) {
    launch_v2_vec4_grid(x, o, n, b, 0);
}

static const Variant VARIANTS[] = {
    {"v0 朴素（三元）", launch_v0_naive},
    {"v0 手写 if/else 分支", launch_v0_ifbranch},
    {"v1 float4 向量化", launch_v1_vec4},
    {"v2b grid-stride（标量）", v2b_wrap},
    {"v2 float4+grid-stride", v2_wrap},
};
static const int NV = sizeof(VARIANTS) / sizeof(VARIANTS[0]);

// ---------------------------------------------------------------------------

// 输入数据的符号分布。ReLU 的 warp divergence 理论上和数据符号有关
// （一半正一半负 = 「该分裂的时候分裂」），所以做成可配置的去实测它。
enum Pattern { ALL_POS = 0, ALL_NEG, HALF_MIXED, P90_NEG };

static void fill_pattern(std::vector<float>& v, int pattern) {
    for (size_t i = 0; i < v.size(); ++i) {
        float x = float(int(i % 2001) - 1000) / 1000.0f;   // [-1, 1) 均匀来回
        switch (pattern) {
            case ALL_POS:  v[i] = std::fabs(x) + 0.001f; break;
            case ALL_NEG:  v[i] = -std::fabs(x) - 0.001f; break;
            case HALF_MIXED: v[i] = x; break;
            case P90_NEG:  v[i] = (i % 10 == 0) ? x : -std::fabs(x) - 0.001f; break;
        }
    }
}

struct Buffers {
    float *dx = nullptr, *dy = nullptr, *dout = nullptr, *dtmp = nullptr;
    std::vector<float> hx, hy, href, href2;
    int n = 0;

    explicit Buffers(int n_, int pattern = HALF_MIXED)
        : hx(n_), hy(n_), href(n_), href2(n_), n(n_) {
        fill_pattern(hx, pattern);
        for (int i = 0; i < n; ++i) {
            hy[i] = float((i * 7) % 997) * 0.002f - 1.0f;   // 也含正负
            href[i] = hx[i] > 0.0f ? hx[i] : 0.0f;
            float v = hx[i] + hy[i];
            href2[i] = v > 0.0f ? v : 0.0f;
        }
        CUDA_CHECK(cudaMalloc(&dx, size_t(n) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy, size_t(n) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dout, size_t(n) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dtmp, size_t(n) * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), size_t(n) * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dy, hy.data(), size_t(n) * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    ~Buffers() {
        cudaFree(dx);
        cudaFree(dy);
        cudaFree(dout);
        cudaFree(dtmp);
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
            VARIANTS[v].launch(buf.dx, buf.dout, n, 0);
            CUDA_CHECK_KERNEL();
            all_ok &= check(buf.download(), buf.href,
                            "  n = " + std::to_string(n), 0.0f, 0.0f);
        }
    }
    return all_ok;
}

static void head_to_head() {
    printf("\n[变体对比] n = %d（x 64 MB，理想访存 %.0f MB）\n", N_BENCH,
           nbytes(N_BENCH) / 1048576.0);
    Buffers buf(N_BENCH);
    double bytes = nbytes(N_BENCH);

    // cudaMemcpy D2D 是比 01 里更好的参考点：它读一遍写一遍（8 B/elem），
    // 和 ReLU 的理想访存量**一模一样**，还一个数都不算。
    // 「纯搬运的上限」就是这个 kernel 的物理天花板。
    float ms_copy = bench([&] {
        CUDA_CHECK(cudaMemcpy(buf.dout, buf.dx, size_t(N_BENCH) * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    });
    report_bw("cudaMemcpy D2D (参考)", ms_copy, bytes,
              "<- 和 relu 完全相同的 8 B/elem");

    for (int v = 0; v < NV; ++v) {
        const Variant& var = VARIANTS[v];
        float ms = bench([&] { var.launch(buf.dx, buf.dout, N_BENCH, 0); });
        report_bw(var.name, ms, bytes);
    }
}

// 符号分布对 warp divergence 有影响吗？
// 理论上：一半正一半负时 warp 内就该分裂（一半走 if 一半走 else）。
// 实际上：编译器根本没生成分支（FMNMX 或谓词化 store），怎么分都一样。
static void sweep_sign() {
    printf("\n[扫描] 输入的符号分布 —— warp divergence 有代价吗？\n");
    printf("  （猜：一半正一半负时最慢，因为「该分裂的时候分裂」）\n");
    const char* names[] = {"全正", "全负", "正负各半", "90% 负"};
    for (int p : {ALL_POS, ALL_NEG, HALF_MIXED, P90_NEG}) {
        Buffers buf(N_BENCH, p);
        float ms = bench([&] { launch_v0_naive(buf.dx, buf.dout, N_BENCH, 0); });
        report_bw(names[p], ms, nbytes(N_BENCH));
    }
    printf("  → 四种分布差距在噪声内：两种写法都不产生分支。\n");
    printf("    三元 → FMNMX 一条；if/else → 谓词化的双 store。\n");
    printf("    没有 BRA 分裂就没有 warp divergence。看 SASS：make sass。见 README §3.1。\n");
}

// in-place 省的不是带宽（读一遍写一遍，8 B/elem 一分不少），
// 省的是**显存占用** —— 工作集从 2n 缩到 n，L2 装得下的临界点翻倍。
// A100 L2 = 40 MB：n=2^23 时 in-place 工作集 32 MB 装得下，
// out-of-place 要 64 MB 装不下 —— 同样的流量，一个吃 L2 一个吃 DRAM。
static void sweep_inplace_size() {
    printf("\n[扫描] in-place vs out-of-place × 数据规模\n");
    printf("  （两者流量都是 8 B/elem；差别在工作集：in-place 是 n×4，"
           "out-of-place 是 2n×4，L2 是 %.0f MB）\n", 40.0);
    double bytes_l2 = 40.0 * 1048576.0;
    for (int shift : {18, 20, 22, 23, 24, 25}) {
        int n = 1 << shift;
        Buffers buf(n);
        double bytes = nbytes(n);

        float ms_out = bench([&] {
            launch_v1_vec4(buf.dx, buf.dout, n, 0);
        });
        char label[80];
        snprintf(label, sizeof(label), "n=2^%-2d out-of-place", shift);
        report_bw(label, ms_out, bytes,
                  2.0 * n * 4 < bytes_l2 ? "<- 工作集装得进 L2" : "");

        // in-place：x 和 out 是同一个 buffer。relu 幂等，反复跑数据不坏。
        float ms_in = bench([&] {
            launch_v1_vec4(buf.dx, buf.dx, n, 0);
        });
        snprintf(label, sizeof(label), "n=2^%-2d in-place    ", shift);
        report_bw(label, ms_in, bytes,
                  1.0 * n * 4 < bytes_l2 ? "<- 工作集装得进 L2" : "");
    }
    printf("  → in-place 省的是**占用**不是流量：流量同样是 8 B/elem，\n");
    printf("    但工作集减半，L2 的临界规模翻倍。见 README §3.3。\n");
}

// v2 的 6 个点去哪了：标量 grid-stride（v2b）vs float4 grid-stride（v2），
// 扫 grid 大小。如果 v2b 贴着 v0（82.6%）→ 损失来自 float4+循环的组合；
// 如果 v2b 掉到 v2 附近（78%）→ 损失来自循环本身。
static void sweep_grid() {
    printf("\n[扫描] grid 大小：标量 grid-stride vs float4 grid-stride\n");
    printf("  （%d 个 SM；对照 v0=%.1f%%、v1=%.1f%%，见 §1 表）\n", num_sms(),
           100.0 * 1285.0 / 1555.2, 100.0 * 1310.7 / 1555.2);
    Buffers buf(N_BENCH);
    double bytes = nbytes(N_BENCH);
    for (int per_sm : {1, 2, 4, 8, 32}) {
        int grid = num_sms() * per_sm;
        char label[64];
        snprintf(label, sizeof(label), "v2b grid=%d (%d/SM)", grid, per_sm);
        float ms = bench([&] {
            launch_v2_scalar_grid(buf.dx, buf.dout, N_BENCH, 256, grid);
        });
        report_bw(label, ms, bytes);

        snprintf(label, sizeof(label), "v2  grid=%d (%d/SM)", grid, per_sm);
        ms = bench([&] {
            launch_v2_vec4_grid(buf.dx, buf.dout, N_BENCH, 256, grid);
        });
        report_bw(label, ms, bytes);
    }
    printf("  v0/v1 的 grid 是 %d（= n/256，由数据量决定）\n",
           (N_BENCH + 255) / 256);
}

// 本练习真正的重点：单独的 ReLU 没有优化空间，但融合能让它的
// 8 B/elem 访存整个消失 —— 变成别人 epilogue 里的一条 FMNMX。
static void fusion_demo() {
    printf("\n[对比] out = relu(x + y)：两个 kernel vs 融合 —— 真正的优化\n");
    Buffers buf(N_BENCH);
    double bytes_two = 20.0 * N_BENCH;          // add 12 B/elem + relu 8 B/elem
    double bytes_fused = 12.0 * N_BENCH;        // 读 x 读 y 写 out，一遍

    // 先验一次正确性：两种做法必须给出同一个结果
    launch_add_relu_two(buf.dx, buf.dy, buf.dtmp, buf.dout, N_BENCH, 0);
    CUDA_CHECK_KERNEL();
    bool ok = check(buf.download(), buf.href2, "  two-kernel", 1e-5f, 1e-5f);
    CUDA_CHECK(cudaMemset(buf.dout, 0xFF, size_t(N_BENCH) * sizeof(float)));
    launch_add_relu_fused(buf.dx, buf.dy, buf.dout, N_BENCH, 0);
    CUDA_CHECK_KERNEL();
    ok &= check(buf.download(), buf.href2, "  fused", 1e-5f, 1e-5f);

    float ms_two = bench([&] {
        launch_add_relu_two(buf.dx, buf.dy, buf.dtmp, buf.dout, N_BENCH, 0);
    });
    report_bw("两个 kernel (add → relu)", ms_two, bytes_two,
              "<- 20 B/elem，tmp 在显存白住一趟");

    float ms_fused = bench([&] {
        launch_add_relu_fused(buf.dx, buf.dy, buf.dout, N_BENCH, 0);
    });
    report_bw("融合 kernel (add+relu)", ms_fused, bytes_fused,
              "<- 12 B/elem，中间结果不落地");

    printf("  → 墙钟快 %.2fx。融合省的是**整趟** relu 访存，\n",
           ms_two / ms_fused);
    printf("    不是把 84%% 优化到 90%% 的事。见 README §3.4，\n");
    printf("    Triton 对照：../../triton/03_matmul/v3_fused_relu.py。\n");
}

int main(int argc, char** argv) {
    bool quick = (argc > 1 && strcmp(argv[1], "--quick") == 0);

    printf("======================================================================\n");
    printf("cuda/04_relu 变体对比\n");
    printf("======================================================================\n");
    print_device_info();

    bool ok = correctness();
    // 把 GPU 时钟拉到稳态再计时，否则第一个被测的变体会虚慢 20~30%。
    printf("\n（预热 GPU 时钟中……）\n");
    warmup_gpu();

    head_to_head();
    if (!quick) {
        sweep_sign();
        sweep_grid();
        sweep_inplace_size();
        fusion_demo();
    }

    printf("\n======================================================================\n");
    printf("正确性：%s\n", ok ? "全部通过" : "有失败");
    printf("结论见 README.md §3\n");
    printf("======================================================================\n");
    return ok ? 0 : 1;
}
