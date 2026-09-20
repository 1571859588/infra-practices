// v4：cuBLAS —— 基线
//
// 和 02_reduction 里的 CUB 一样，这一版的意义是**当尺子**。
// 自己写的 kernel 到底算不算快，只有和产品级实现比过才知道。
//
// ★ 唯一的坑：cuBLAS 是 **column-major**（Fortran 传统），
//   而我们所有数据都是 row-major。
//
// 不需要真的转置数据，用一个恒等式就行：
//
//   row-major 的 C(M×N) = A(M×K) × B(K×N)
//   等价于
//   col-major 的 Cᵀ(N×M) = Bᵀ(N×K) × Aᵀ(K×M)
//
// 而"row-major 的 X"在内存里和"col-major 的 Xᵀ"是**同一串字节**。
// 所以只要把 A、B 的位置对调着传进去，什么都不用动：
//
//   cublasSgemm(h, N_OP, N_OP, N, M, K, &one, B, N, A, K, &zero, C, N)
//                                 ^^^^^^        ^     ^
//                                 注意是 N,M,K   B 在前 A 在后
//
// 记不住就记结论：**row-major 调 cuBLAS，交换 A/B 并把 M/N 对调。**

#include <cublas_v2.h>

#include "../common.cuh"
#include "variants.cuh"

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t st_ = (call);                                          \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "[cuBLAS ERROR] %s:%d  %s -> %d\n", __FILE__,     \
                    __LINE__, #call, int(st_));                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

static cublasHandle_t g_handle = nullptr;

void cublas_init() {
    if (!g_handle) {
        CUBLAS_CHECK(cublasCreate(&g_handle));
        // 默认是 CUBLAS_TF32_TENSOR_OP_MATH 还是 DEFAULT_MATH，
        // 取决于 CUDA 版本和环境变量 —— 显式写死，保证对比公平。
        //
        // ★ 这一行很关键：A100 上 fp32 矩阵乘默认可能走 **TF32 Tensor Core**
        //   （峰值 156 TFLOP/s，是 fp32 CUDA core 19.5 的 8 倍），
        //   尾数只有 10 bit。拿它和我们的 fp32 kernel 比是不公平的 ——
        //   那根本不是同一种运算。所以默认强制真 fp32。
        //
        //   想看 TF32 有多快：`MATMUL_TF32=1 ./bench --quick`。
        //   实测（见 README §3.6）：18.23 → 122.13 TFLOP/s，**快 6.7x**，
        //   代价是 1000³ 上 max_abs_err 从 1.8e-05 涨到 1.5e-02，
        //   正确性用例直接判 FAIL —— 这正是它不该当默认基线的原因。
        const char* tf32 = getenv("MATMUL_TF32");
        bool use_tf32 = tf32 && tf32[0] == '1';
        CUBLAS_CHECK(cublasSetMathMode(
            g_handle,
            use_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));
        if (use_tf32)
            printf("  ⚠️  MATMUL_TF32=1：cuBLAS 走 TF32 Tensor Core，"
                   "不再是 fp32，正确性用例会以更大误差通过或失败\n");
    }
}

void cublas_destroy() {
    if (g_handle) {
        cublasDestroy(g_handle);
        g_handle = nullptr;
    }
}

void matmul_v4_cublas(const float* A, const float* B, float* C, int M, int N,
                      int K) {
    cublas_init();
    const float alpha = 1.0f, beta = 0.0f;
    // 见文件头的推导：交换 A/B，M/N 对调，leading dimension 用行宽
    CUBLAS_CHECK(cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &alpha, B, N, A, K, &beta, C, N));
}
