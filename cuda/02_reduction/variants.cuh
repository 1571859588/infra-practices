// 各变体对外暴露的 launch 函数。
//
// 统一签名：把 d_in 的 n 个 float 求和，结果写到 d_result（device 上 1 个 float）。
// scratch_a / scratch_b 是中间部分和的乒乓缓冲，每个至少要
// ceil(n / block) 个 float —— bench.cu 里按 n/32 分配，足够。

#pragma once

void reduce_v0_interleaved(const float* d_in, float* sa, float* sb,
                           float* d_result, int n, int block);
void reduce_v1_no_divergence(const float* d_in, float* sa, float* sb,
                             float* d_result, int n, int block);
void reduce_v2_sequential(const float* d_in, float* sa, float* sb,
                          float* d_result, int n, int block);
void reduce_v3_first_add(const float* d_in, float* sa, float* sb,
                         float* d_result, int n, int block);
void reduce_v4_shuffle(const float* d_in, float* sa, float* sb,
                       float* d_result, int n, int block);
void reduce_v5_gridstride(const float* d_in, float* sa, float* sb,
                          float* d_result, int n, int block);
void reduce_v6_cub(const float* d_in, float* sa, float* sb, float* d_result,
                   int n, int block);

// v5 的可调版本：多一个「每个 SM 开几个 block」参数，给 bench 扫描用。
// reduce_v5_gridstride 就是 blocks_per_sm = 8 的它。
void reduce_v5_tuned(const float* d_in, float* sa, float* sb, float* d_result,
                     int n, int block, int blocks_per_sm);
