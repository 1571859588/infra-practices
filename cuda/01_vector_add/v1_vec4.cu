// v1：float4 向量化访存 —— 一条指令搬 128 bit
//
// 这就是用户问到的「向量化加载」。在 CUDA 里它是**显式**的：
// 你把指针转成 float4*，编译器就会发 `ld.global.v4.f32`（128 bit 一条）。
//
// 对照 ../../triton/01_vector_add/：
//   Triton **没有**对应的开关。你只能保证访问模式是连续的，
//   然后由编译器自己决定要不要 vectorize（实测它会，PTX 里能看到
//   `ld.global.v4.b32`）。CUDA 把方向盘给了你，Triton 替你打了。
//
// 为什么向量化有用（在一个已经完全合并的 kernel 上）？
//   不是为了「搬得更多」—— v0 已经把带宽用满了 transaction 层面。
//   是为了**减少指令数和在途请求数**：同样搬 N 个 float，
//   float4 只要 1/4 的 load 指令、1/4 的地址计算、1/4 的循环开销。
//   访存延迟靠的是「同时有足够多的请求在飞」(MLP)，
//   每个线程管 4 个元素，等于用更少的线程就能填满访存流水线。
//
// ⚠️ 两个前提，少一个就是 bug：
//   1. 指针必须 16 字节对齐（cudaMalloc 保证 256 字节对齐，直接用没问题；
//      但如果是 `base + offset` 这种偏移过的指针就要自己保证）
//   2. n 不是 4 的倍数时，尾巴必须单独处理

#include "../common.cuh"
#include "variants.cuh"

__global__ void vector_add_vec4(const float* __restrict__ x,
                                const float* __restrict__ y,
                                float* __restrict__ out, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        // 指针重解释成 float4*，一次读 16 字节。
        // reinterpret_cast 不产生任何指令，纯粹是告诉编译器
        // 「按 128 bit 的粒度去读」。
        const float4 a = reinterpret_cast<const float4*>(x)[i];
        const float4 b = reinterpret_cast<const float4*>(y)[i];

        // 计算部分还是逐分量，float4 只是访存的打包单位，
        // 不是 SIMD 计算单元 —— GPU 的算力本来就来自线程并行，
        // 不是像 CPU 那样来自 SIMD 宽度。
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;

        reinterpret_cast<float4*>(out)[i] = c;
    }
}

// 尾巴：n 不整除 4 时剩下的 1~3 个元素。
// 单独开一个 kernel 处理是最省事的写法（多一次 launch，
// 但只有几个元素，开销可以忽略）。
__global__ void vector_add_tail(const float* x, const float* y, float* out,
                                int start, int n) {
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}

void launch_v1_vec4(const float* x, const float* y, float* out, int n,
                    int block) {
    if (block == 0) block = 256;

    int n4 = n / 4;                       // 能凑成 float4 的组数
    if (n4 > 0) {
        int grid = (n4 + block - 1) / block;
        vector_add_vec4<<<grid, block>>>(x, y, out, n4);
    }

    int tail_start = n4 * 4;
    int tail = n - tail_start;            // 0 ~ 3 个
    if (tail > 0) {
        vector_add_tail<<<1, 32>>>(x, y, out, tail_start, n);
    }
}
