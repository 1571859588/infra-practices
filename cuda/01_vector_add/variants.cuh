// 各变体对外暴露的 launch 函数。bench.cu 通过它把所有变体串起来对比。
//
// 每个变体一个 .cu 文件，可以单独读；bench.cu 链接它们做统一对比。
// 这样既保证「一个文件讲一件事」，又避免每个变体各写一份 main。

#pragma once

#include <cstddef>

// 统一签名：out = x + y，n 个 float 元素。
// block 参数由调用方给，方便扫描；传 0 表示用该变体自己的默认值。
void launch_v0_naive(const float* x, const float* y, float* out, int n,
                     int block);
void launch_v1_vec4(const float* x, const float* y, float* out, int n,
                    int block);
void launch_v2_grid_stride(const float* x, const float* y, float* out, int n,
                           int block, int grid);
// v3 多一个 per_thread：每个线程负责多少个**连续**元素。
// 它同时就是 warp 内相邻线程的地址间距（× 4 字节）。
// 传 0 用默认值 64。扫描这个参数能看出合并访存到底在哪一步失效。
void launch_v3_strided_bad(const float* x, const float* y, float* out, int n,
                           int block, int per_thread);
