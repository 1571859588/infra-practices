// v1：float4 向量化 —— 一条指令搬 128 bit
//
// 和 01_vector_add 的 v1 完全同构：把指针转成 float4*，
// 编译器就发 ld.global.v4 / st.global.v4。
// ReLU 这边甚至更简单 —— 每个分量一个 fmaxf 就完了。
//
// 预期（01 的结论）：block=256 下和 v0 差距在噪声内；
// 真正的收益在小 block 上（MLP 不足时一个线程发 4 倍的字节量）。
// 01 §3.1 已经完整扫过这个现象，这里只在 README §3.4 复测关键点。
//
// in-place 说明：这个 kernel 也用于 in-place 实验（x 和 out 传同一个指针）。
// 安全性：每个线程只读写「自己的」那 4 个元素，线程间无重叠；
// 而且 relu 是幂等的 —— relu(relu(x)) == relu(x)，
// 所以 in-place 的 benchmark 反复跑不会把数据跑坏（别的算子没这待遇，
// in-place 测性能前先想清楚这点）。

#include "../common.cuh"
#include "variants.cuh"

__global__ void relu_vec4(const float* __restrict__ x, float* __restrict__ out,
                          int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        const float4 a = reinterpret_cast<const float4*>(x)[i];
        float4 c;
        c.x = fmaxf(a.x, 0.0f);
        c.y = fmaxf(a.y, 0.0f);
        c.z = fmaxf(a.z, 0.0f);
        c.w = fmaxf(a.w, 0.0f);
        reinterpret_cast<float4*>(out)[i] = c;
    }
}

__global__ void relu_tail(const float* x, float* out, int start, int n) {
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(x[i], 0.0f);
}

void launch_v1_vec4(const float* x, float* out, int n, int block) {
    if (block == 0) block = 256;

    int n4 = n / 4;
    if (n4 > 0) {
        relu_vec4<<<(n4 + block - 1) / block, block>>>(x, out, n4);
    }

    int tail_start = n4 * 4;
    if (n - tail_start > 0) {
        relu_tail<<<1, 32>>>(x, out, tail_start, n);
    }
}
