# common.mk —— cuda/ 下所有练习共用的编译设置
#
# 每个练习的 Makefile 开头 include 它。想改编译器 / 架构 / GPU，
# 在命令行覆盖即可：
#     make ARCH=sm_90
#     make run GPU=3
#     make NVCC=/path/to/other/nvcc

# ---- 工具链 ----
# 本机有两套 nvcc（详见 ../README.md）：
#   /usr/local/cuda/bin/nvcc   12.4  ← 默认用这个，和 ncu/nsys 同版本
#   conda env `cpp` 里的       13.3
# 之所以默认选系统的：ncu / nsys 只有系统 CUDA 里有，版本对齐省事。
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
# 这些工具默认不在 PATH 里（除非按 ../README.md §1 配过环境变量），
# 所以一律写全路径，Makefile 才能开箱即用。
CUOBJDUMP ?= $(CUDA_HOME)/bin/cuobjdump
NVDISASM  ?= $(CUDA_HOME)/bin/nvdisasm
NCU       ?= $(CUDA_HOME)/bin/ncu
NSYS      ?= $(CUDA_HOME)/bin/nsys
SANITIZER ?= $(CUDA_HOME)/bin/compute-sanitizer

# ---- 目标架构 ----
# A100 = sm_80。写死具体架构（而不是用 -arch=native 或 compute_XX）
# 可以让 nvcc 直接生成 SASS，跑起来没有 JIT 开销，也方便 cuobjdump 看汇编。
ARCH ?= sm_80

# -O3          : host 代码优化
# -lineinfo    : 保留行号，ncu 才能把指标对到源码行（开销可忽略，建议常开）
# --use_fast_math 故意**不开** —— 它会改变数值结果，
#                 在做正确性对比的练习里会掩盖真实误差
NVCCFLAGS ?= -arch=$(ARCH) -O3 -std=c++17 -lineinfo -Xcompiler -Wall

# ---- 运行环境 ----
# 这台机器 GPU 0~6 常被别人占着，7 一般空闲。
# 测性能前先 nvidia-smi 确认，被别人抢着跑的卡测出来的数没有意义。
GPU    ?= 7
RUNENV ?= CUDA_VISIBLE_DEVICES=$(GPU)
