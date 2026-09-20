// bench.cu —— 归约优化链的完整对比
//
// 跑法见 README.md §2。

#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../common.cuh"
#include "variants.cuh"

// 正确性用例。**故意包含非 2 的幂、非 block 倍数的尺寸** ——
// 归约的边界处理（补 0）只有这些尺寸才测得出来。
static const int CASES[] = {256, 1000, 1024, 65537, 1 << 20, (1 << 20) + 7};
static const int N_BENCH = 1 << 26;   // 64M 个 float = 256 MB，远超 40 MB L2

// 归约的理想访存量：**只读一遍输入**。
// 部分和的中间读写不算 —— 那是实现的开销，正是要被优化掉的东西。
// （同样的口径理由见 ../../triton/02_fused_softmax/_shared.py）
static double nbytes(long long n) { return double(n) * 4.0; }

struct Variant {
    const char* name;
    void (*launch)(const float*, float*, float*, float*, int, int);
};

static const Variant VARIANTS[] = {
    {"v0 交错寻址（divergent）", reduce_v0_interleaved},
    {"v1 消除 divergence", reduce_v1_no_divergence},
    {"v2 顺序寻址（无 bank 冲突）", reduce_v2_sequential},
    {"v3 load 时先加一次", reduce_v3_first_add},
    {"v4 warp shuffle", reduce_v4_shuffle},
    {"v5 grid-stride + float4", reduce_v5_gridstride},
    {"v6 CUB（官方基线）", reduce_v6_cub},
};
static const int NV = sizeof(VARIANTS) / sizeof(VARIANTS[0]);

// ---------------------------------------------------------------------------

struct Buffers {
    float *d_in = nullptr, *sa = nullptr, *sb = nullptr, *d_result = nullptr;
    double ref = 0.0;
    int n = 0;

    explicit Buffers(int n_) : n(n_) {
        std::vector<float> h(n);
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        // 参考值用 double 在 CPU 上累加。
        // 不能用 float 累加当参考 —— 2^26 个数顺序累加，float 会
        // 严重丢精度（加到后面时 sum 已经很大，小数被吃掉），
        // 反而是 GPU 的树形归约更准。拿它当参考会得出错误结论。
        for (int i = 0; i < n; ++i) {
            h[i] = dist(rng);
            ref += double(h[i]);
        }
        CUDA_CHECK(cudaMalloc(&d_in, size_t(n) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sa, (size_t(n) / 32 + 64) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sb, (size_t(n) / 32 + 64) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h.data(), size_t(n) * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    ~Buffers() {
        cudaFree(d_in);
        cudaFree(sa);
        cudaFree(sb);
        cudaFree(d_result);
    }
    float result() const {
        float out = 0.0f;
        CUDA_CHECK(cudaMemcpy(&out, d_result, sizeof(float),
                              cudaMemcpyDeviceToHost));
        return out;
    }
};

static bool correctness() {
    printf("\n[正确性] 所有变体 × 所有用例（含非 2 的幂 / 非 block 倍数）\n");
    printf("  参考值用 double 在 CPU 上算（float 顺序累加精度不够当参考）\n");
    bool all_ok = true;
    for (int v = 0; v < NV; ++v) {
        printf("  %s\n", VARIANTS[v].name);
        for (int n : CASES) {
            Buffers buf(n);
            CUDA_CHECK(cudaMemset(buf.d_result, 0xFF, sizeof(float)));
            VARIANTS[v].launch(buf.d_in, buf.sa, buf.sb, buf.d_result, n, 256);
            CUDA_CHECK_KERNEL();
            double got = buf.result();
            double rel = std::fabs(got - buf.ref) / std::fabs(buf.ref);
            bool ok = rel < 1e-5;
            all_ok &= ok;
            printf("  %s n = %-9d got %.4f  ref %.4f  相对误差 %.2e\n",
                   ok ? "[PASS]" : "[FAIL]", n, got, buf.ref, rel);
        }
    }
    return all_ok;
}

static void head_to_head() {
    printf("\n[变体对比] n = %d（输入 %.0f MB，远大于 40 MB L2）\n", N_BENCH,
           nbytes(N_BENCH) / 1048576.0);
    printf("  带宽按「输入读一遍」算 —— 中间部分和的开销正是要被优化的东西\n");
    Buffers buf(N_BENCH);
    double bytes = nbytes(N_BENCH);

    float ms_prev = 0.0f;
    for (int v = 0; v < NV; ++v) {
        const Variant& var = VARIANTS[v];
        float ms = bench([&] {
            var.launch(buf.d_in, buf.sa, buf.sb, buf.d_result, N_BENCH, 256);
        });
        char extra[80] = "";
        if (v > 0 && v < NV - 1)
            snprintf(extra, sizeof(extra), "较上一版 %.2fx", ms_prev / ms);
        report_bw(var.name, ms, bytes, extra);
        ms_prev = ms;
    }
}

static void sweep_block() {
    printf("\n[扫描] block 大小 × 变体\n");
    Buffers buf(N_BENCH);
    double bytes = nbytes(N_BENCH);
    for (int b : {64, 128, 256, 512, 1024}) {
        printf("  --- block = %d ---\n", b);
        for (int v = 0; v < NV - 1; ++v) {   // CUB 自己管 block，跳过
            char label[80];
            snprintf(label, sizeof(label), "%.20s", VARIANTS[v].name);
            float ms = bench([&] {
                VARIANTS[v].launch(buf.d_in, buf.sa, buf.sb, buf.d_result,
                                   N_BENCH, b);
            });
            report_bw(label, ms, bytes);
        }
    }
}

static void sweep_size() {
    printf("\n[扫描] 数据规模 —— v4 / v5 / CUB\n");
    for (int shift : {16, 20, 22, 24, 26}) {
        int n = 1 << shift;
        Buffers buf(n);
        double bytes = nbytes(n);
        printf("  --- n = 2^%d (%.2f MB) ---\n", shift, bytes / 1048576.0);
        float ms4 = bench([&] {
            reduce_v4_shuffle(buf.d_in, buf.sa, buf.sb, buf.d_result, n, 256);
        });
        float ms5 = bench([&] {
            reduce_v5_gridstride(buf.d_in, buf.sa, buf.sb, buf.d_result, n, 256);
        });
        float ms6 = bench([&] {
            reduce_v6_cub(buf.d_in, buf.sa, buf.sb, buf.d_result, n, 256);
        });
        report_bw("v4 warp shuffle", ms4, bytes);
        report_bw("v5 grid-stride + float4", ms5, bytes);
        report_bw("v6 CUB", ms6, bytes,
                  ms5 <= ms6 ? "<- v5 追平/反超" : "<- CUB 仍领先");
    }
}

// v5 唯一的结构参数：每个 SM 开几个 block。
// 这个扫描顺带回答「grid 该开多大」这个每次写 grid-stride kernel 都要问的问题。
static void sweep_v5_grid() {
    int sms = num_sms();
    printf("\n[扫描] v5：每个 SM 开几个 block（共 %d 个 SM，block = 256）\n", sms);
    printf("  每线程处理的元素数 = n / (SM数 × 每SM块数 × 256 × 4)\n");
    Buffers buf(N_BENCH);
    double bytes = nbytes(N_BENCH);
    for (int bps : {1, 2, 4, 8, 16, 32, 64}) {
        long long threads = 1LL * sms * bps * 256;
        char label[64];
        snprintf(label, sizeof(label), "每SM %2d 块 (grid=%5d)", bps, sms * bps);
        char extra[64];
        snprintf(extra, sizeof(extra), "每线程 %lld 个元素",
                 (long long)N_BENCH / threads);
        float ms = bench([&] {
            reduce_v5_tuned(buf.d_in, buf.sa, buf.sb, buf.d_result, N_BENCH,
                            256, bps);
        });
        report_bw(label, ms, bytes, extra);
    }
}

int main(int argc, char** argv) {
    bool quick = (argc > 1 && strcmp(argv[1], "--quick") == 0);
    // --check：只跑正确性，不跑任何 benchmark。
    // 给 compute-sanitizer 用 —— sanitizer 下每条访存都要检查，
    // n = 2^26 的 benchmark 要跑几十分钟，而查 race / 越界
    // 用几百个元素的用例就够了。
    bool check_only = (argc > 1 && strcmp(argv[1], "--check") == 0);

    printf("======================================================================\n");
    printf("cuda/02_reduction 变体对比\n");
    printf("======================================================================\n");
    print_device_info();

    bool ok = correctness();
    if (check_only) {
        printf("\n正确性：%s（--check 模式，跳过 benchmark）\n",
               ok ? "全部通过" : "有失败");
        return ok ? 0 : 1;
    }
    // 把时钟拉到稳态再计时，否则第一个被测的变体会虚慢 20~30%。
    // 见 ../common.cuh 里 warmup_gpu 的注释和 README §2。
    printf("\n（预热 GPU 时钟中……）\n");
    warmup_gpu();

    head_to_head();
    if (!quick) {
        sweep_block();
        sweep_v5_grid();
        sweep_size();
    }

    printf("\n======================================================================\n");
    printf("正确性：%s\n", ok ? "全部通过" : "有失败");
    printf("结论见 README.md §3\n");
    printf("======================================================================\n");
    return ok ? 0 : 1;
}
