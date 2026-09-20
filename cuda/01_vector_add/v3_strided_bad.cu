// v3：故意写坏的版本 —— 破坏合并访存，量化它的代价
//
// 这个变体**不是优化**，是对照组。它和 ../../triton/01_vector_add/
// v1_strided_bad.py 是同一个实验：把「合并访存」从一句口号
// 变成一个可测量的数字。
//
// 坏在哪：**让每个线程负责一段连续的数据**。
//
//   v0/v2（对）：thread 0,1,2,3 同时访问元素 0,1,2,3      ← warp 内地址连续
//   v3  （错）：thread 0 管 [0,P)，thread 1 管 [P,2P)…
//               同一时刻 warp 内 32 个线程访问 0, P, 2P … 31P  ← 全散开
//
// 这是 CPU 程序员写 CUDA 最常犯的错。在 CPU 上「每个核分一块连续数据」
// 是标准做法（cache line 局部性好）；GPU 要的**恰恰相反** ——
// 数据要在**线程之间交错**，因为 GPU 的访存局部性是以 warp 为单位、
// 跨线程看的，不是单线程看的。
//
// 注意别和 v2 的 grid-stride 搞混 —— 两者都有循环，形式很像，
// 区别在**步长的层次**：
//     v2: i += gridDim.x * blockDim.x   每轮内 warp 地址仍连续  ✅
//     v3: i += 1（线程内连续）          warp 地址散开          ❌
// 这正是 grid-stride loop 必须用「整个 grid 的宽度」当步长的原因。
//
// ⚠️ **但「散开」到什么程度才真的慢？这件事我一开始想当然了。**
//
// 第一版我把 per_thread 写死成 4，测出来和 v0 **一样快**（84.8% vs 84.3%），
// 完全没有预期的惩罚。原因是 per_thread=4 时，一个 warp 的 32 个线程
// 一共覆盖 32 × 4 × 4 = 512 字节，**仍然是连续的一整段**：
// 第 0 条 load 指令取的是 0,16,32,… 字节，看着散，但这些地址落在
// 同样那几条 cache line 上，紧接着第 1/2/3 条 load 就把它们用掉了。
// DRAM 侧搬的字节数一模一样 —— 在一个纯 DRAM-bound 的 kernel 上自然不慢。
//
// 那真正的失效点在哪？我本来以为是 cache line（128 字节），
// **实测是 32 字节 —— sector 的大小**：
//
//     间距  4 B → 84.3%      间距  32 B → 67.2%   ← 这里开始掉
//     间距  8 B → 84.8%      间距  64 B → 35.8%
//     间距 16 B → 84.3%      间距 128 B → 13.9%   ← 6.1x 慢
//
// 原因：NVIDIA 的 L1/L2 访存**以 32 字节的 sector 为最小单位**，
// 不是以 128 字节的 line 为单位。间距一旦 ≥ 32 B，warp 内 32 个线程
// 就落进 32 个不同的 sector，一条 load 指令要碰 32 个 sector 而不是 4 个。
//
// 而且注意它是**渐变不是断崖** —— 因为 L1 还能吸收一部分复用
// （同一个线程的下一次循环会用到同一个 sector）。间距越大，
// 一个 warp 的工作集越大，L1 越兜不住，掉得越狠。
//
// 所以这里把 per_thread 做成参数让 bench 扫一遍。
// 知道「32 字节 sector 才是那条线」比记住「不合并会慢」有用得多 ——
// 前者能让你在写代码的时候就算出来会不会踩。结论见 README §3.3。

#include "../common.cuh"
#include "variants.cuh"

__global__ void vector_add_uncoalesced(const float* __restrict__ x,
                                       const float* __restrict__ y,
                                       float* __restrict__ out, int n,
                                       int per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int begin = tid * per_thread;
    int end = min(begin + per_thread, n);
    for (int i = begin; i < end; ++i) {   // ★ 线程内连续 = warp 间散开
        out[i] = x[i] + y[i];
    }
}

void launch_v3_strided_bad(const float* x, const float* y, float* out, int n,
                           int block, int per_thread) {
    if (block == 0) block = 256;
    if (per_thread == 0) per_thread = 64;

    int n_chunks = (n + per_thread - 1) / per_thread;
    int grid = (n_chunks + block - 1) / block;
    vector_add_uncoalesced<<<grid, block>>>(x, y, out, n, per_thread);
}
