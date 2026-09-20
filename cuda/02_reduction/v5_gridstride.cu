// v5：grid-stride + float4 —— 追平 CUB 的那一步
//
// v4 已经把「一个 block 内部怎么归约」做到头了（无 divergence、
// 无 bank conflict、只有 1 次 __syncthreads、shuffle 全展开），
// 85.6%，离 CUB 的 90.1% 只差一点。v5 换的是**整体结构**：
//
//   v0~v4 的结构：每个 block 只吃 2*blockDim 个元素 → block 特别多
//                 （n=2^26, block=256 时是 65536 个 block）
//                 → 多趟归约、大量 block 启动、每个 block 只干一点活
//
//   v5 的结构：  固定开 SM 数 × 几个 block，每个线程用 grid-stride
//                 循环吃掉很多元素，**累加在寄存器里**
//                 → 只有两趟，几乎所有时间都花在顺序读输入上
//
// 换来的不只是那 2 个点的带宽，更重要的是**鲁棒性**：v5 对 block 大小
// 几乎不敏感（64~1024 都在 86~88%），而 v4 从 block=1024 的 62% 到
// block=256 的 85.6% 差了 24 个点。见 README §3.3。
//
// 两个来自前面练习的技巧叠加在这里：
//   - grid-stride loop        ← ../01_vector_add/v2_grid_stride.cu
//   - float4 向量化访存       ← ../01_vector_add/v1_vec4.cu
//
// 注意 ../01_vector_add/README.md §3.4 说过 grid-stride 在向量加上
// **没有收益**。为什么这里有？因为那里每个线程本来就只读 1 个元素、
// 没有任何可复用的状态；这里每个线程在寄存器里维护一个累加器，
// 吃的元素越多，「读进来 → 归约 → 写部分和」这套固定成本就摊得越薄。
// **同一个技巧在不同 kernel 上价值完全不同**，这正是要自己测的原因。

#include "../common.cuh"
#include "variants.cuh"

constexpr int WARP = 32;   // 不用 warpSize，理由见 v4_shuffle.cu 和 README §3.5

__device__ __forceinline__ float warp_reduce_sum_v5(float v) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// 把 v4 的 block 内归约抽出来复用
__device__ __forceinline__ float block_reduce_sum(float v, float* sdata) {
    unsigned tid = threadIdx.x;
    unsigned lane = tid % WARP;
    unsigned warp_id = tid / WARP;

    v = warp_reduce_sum_v5(v);
    if (lane == 0) sdata[warp_id] = v;
    __syncthreads();

    unsigned num_warps = blockDim.x / WARP;
    if (warp_id == 0) {
        v = (lane < num_warps) ? sdata[lane] : 0.0f;
        v = warp_reduce_sum_v5(v);
    }
    return v;   // 只有 tid==0 的返回值有意义
}

__global__ void reduce_gridstride(const float* __restrict__ in,
                                  float* __restrict__ out, int n) {
    extern __shared__ float sdata[];

    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = gridDim.x * blockDim.x;

    // ---- ① grid-stride 累加到寄存器，用 float4 一次读 4 个 ----
    float v = 0.0f;
    int n4 = n / 4;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    for (int i = gid; i < n4; i += stride) {
        float4 a = in4[i];
        // 4 个分量在寄存器里加完，只占 1 个累加器
        v += (a.x + a.y) + (a.z + a.w);
    }
    // 尾巴：n 不是 4 的倍数时剩下的 1~3 个，交给前几个线程
    for (int i = n4 * 4 + int(gid); i < n; i += stride) v += in[i];

    // ---- ② block 内归约 ----
    v = block_reduce_sum(v, sdata);
    if (threadIdx.x == 0) out[blockIdx.x] = v;
}

// blocks_per_sm 是 v5 唯一的结构参数：每个 SM 上放几个 block。
// 太小 → SM 闲着、访存并行度不够；太大 → 退化回 v4 那种「block 太多、
// 每个只干一点活」的结构。README §3.4 有扫描结果。
void reduce_v5_tuned(const float* d_in, float* sa, float* /*sb*/,
                     float* d_result, int n, int block, int blocks_per_sm) {
    if (block == 0) block = 256;
    if (blocks_per_sm <= 0) blocks_per_sm = 8;

    int grid = num_sms() * blocks_per_sm;
    int n4 = n / 4;
    if (grid > n4 / block + 1) grid = n4 / block + 1;
    if (grid < 1) grid = 1;

    size_t smem = size_t(block) * sizeof(float);

    // 第一趟：n 个元素 → grid 个部分和
    reduce_gridstride<<<grid, block, smem>>>(d_in, sa, n);

    // 第二趟：grid 个部分和 → 1 个。grid 最多几千，一个 block
    // 用 grid-stride 就能吃完，所以固定两趟，不需要循环。
    reduce_gridstride<<<1, block, smem>>>(sa, d_result, grid);
}

void reduce_v5_gridstride(const float* d_in, float* sa, float* sb,
                          float* d_result, int n, int block) {
    // 8 是扫出来的默认值，见 README §3.4
    reduce_v5_tuned(d_in, sa, sb, d_result, n, block, 8);
}
