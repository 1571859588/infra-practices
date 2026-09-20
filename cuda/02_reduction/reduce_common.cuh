// reduce_common.cuh —— 归约练习共用的多趟驱动
//
// 归约的结构性难点：**一个 block 只能规约它自己那部分**。
// block 之间没有全局同步（`__syncthreads()` 只同步 block 内），
// 所以 n 个元素归约成 1 个值，必须分多趟：
//
//   2^24 个元素  --(65536 个 block，每块出 1 个部分和)-->  65536
//              --(256 个 block)-->  256
//              --(1 个 block)-->  1        ← 完成
//
// 每一趟都是同一个 kernel，只是输入变小。这里把这个循环抽出来，
// 各变体就只用关心「一个 block 怎么规约」这件事本身。
//
// （另一种做法是用 atomicAdd 一趟出结果，但浮点 atomicAdd 的
//   累加顺序不确定 → 结果不可复现。见 README §5。）

#pragma once

#include "../common.cuh"

// 每个变体的 kernel 都是这个签名：
//   in  : 输入数组，n 个元素
//   out : 输出数组，每个 block 写一个部分和到 out[blockIdx.x]
using ReduceKernel = void (*)(const float*, float*, int);

// ELEMS_MULT：一个 block 一趟吃掉多少元素，以 blockDim 为单位。
//   v0~v2 是 1（一线程一元素）
//   v3 之后是 2（load 的时候就先加一次，一线程两元素）
template <ReduceKernel Kernel, int ELEMS_MULT = 1>
void run_multipass(const float* d_in, float* scratch_a, float* scratch_b,
                   float* d_result, int n, int block) {
    const float* cur_in = d_in;
    float* cur_out = scratch_a;
    int cur_n = n;
    size_t smem = size_t(block) * sizeof(float);

    while (true) {
        int per_block = block * ELEMS_MULT;
        int grid = (cur_n + per_block - 1) / per_block;

        // 最后一趟直接写进最终结果，省一次拷贝
        if (grid == 1) cur_out = d_result;

        Kernel<<<grid, block, smem>>>(cur_in, cur_out, cur_n);

        if (grid == 1) break;

        // 乒乓：这一趟的输出是下一趟的输入
        cur_n = grid;
        cur_in = cur_out;
        cur_out = (cur_out == scratch_a) ? scratch_b : scratch_a;
    }
}
