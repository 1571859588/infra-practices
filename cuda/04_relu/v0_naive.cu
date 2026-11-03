// v0：朴素 ReLU —— 一线程一元素
//
// ReLU 是深度学习里出现频率最高的算子之一（激活函数、剪枝、mask），
// 但它可能是全宇宙最「没有优化空间」的 kernel：
//   * 1 个比较 + 1 个选择，0 次乘加 —— 算术强度趋近于 0
//   * 访存：读 x 一遍、写 out 一遍，8 字节/元素
//     （比 01_vector_add 的 12 字节还少 —— 那边读两个写一个）
//   * 纯流式、无数据依赖、无复用 —— 连 shared memory 都没用武之地
// 所以它的性能 = 带宽上限，和 01 的结论是同一座天花板（~85%）。
//
// 本文件还带了第二个 kernel：relu_ifbranch，源码写成真正的 if/else。
// 问题：warp 是 32 线程一起走同一条指令流的（SIMT），如果一半线程
// 走 if、一半走 else，理论上该有 warp divergence 的惩罚 ——
// 而 ReLU 的输入恰好是一半正一半负，「该分裂的时候分裂」。
// 实测它和三元版**一样快**，而且 SASS 比预测的还有意思：
//   三元版   → 一条 FMNMX（max/select）搞定；
//   if/else 版 → 编译器没选 FMNMX，而是把两个 store **谓词化**了
//              （@!P0 STG 写 0 / STG 写 x），照样没有 BRA。
// 两种写法都没有产生真正的分支，divergence 无从谈起。
// 见 README §3.1 和 make sass。

#include "../common.cuh"
#include "variants.cuh"

__global__ void relu_naive(const float* __restrict__ x, float* __restrict__ out,
                          int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

// 「手写分支」版 —— 源码上有分支，SASS 上没有。
__global__ void relu_ifbranch(const float* __restrict__ x,
                              float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (x[i] > 0.0f) {
            out[i] = x[i];
        } else {
            out[i] = 0.0f;
        }
    }
}

void launch_v0_naive(const float* x, float* out, int n, int block) {
    if (block == 0) block = 256;
    relu_naive<<<(n + block - 1) / block, block>>>(x, out, n);
}

void launch_v0_ifbranch(const float* x, float* out, int n, int block) {
    if (block == 0) block = 256;
    relu_ifbranch<<<(n + block - 1) / block, block>>>(x, out, n);
}
