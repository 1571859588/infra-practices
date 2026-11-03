// 各变体对外暴露的 launch 函数。bench.cu 通过它把所有变体串起来对比。
//
// 每个变体一个 .cu 文件，可以单独读；bench.cu 链接它们做统一对比。
// 这样既保证「一个文件讲一件事」，又避免每个变体各写一份 main。

#pragma once

#include <cstddef>

// 统一签名：out = relu(x)，n 个 float 元素。
// block 参数由调用方给，方便扫描；传 0 表示用该变体自己的默认值。
void launch_v0_naive(const float* x, float* out, int n, int block);
// v0 的双胞胎：同样的计算，但源码写成真正的 if/else 分支。
// 用来验证「三元运算符 vs if 分支」在 SASS 里是不是同一条指令。
void launch_v0_ifbranch(const float* x, float* out, int n, int block);
void launch_v1_vec4(const float* x, float* out, int n, int block);
void launch_v2_vec4_grid(const float* x, float* out, int n, int block, int grid);

// 融合演示（v3_fused.cu）：out = relu(x + y)。
// 两种做法：先 add 再 relu（两个 kernel）vs 一个融合 kernel。
void launch_add_relu_two(const float* x, const float* y, float* tmp, float* out,
                         int n, int block);
void launch_add_relu_fused(const float* x, const float* y, float* out, int n,
                           int block);
