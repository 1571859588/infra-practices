# Makefile —— 构建 CUDA 版本的两个产物
#
#   make            构建 libmul2.so + cuda_mul2
#   make run        构建并运行独立 benchmark
#   make clean      清理
#
# 默认使用系统 CUDA 12.4（/usr/local/cuda），因为只有它带 ncu / nsys。
# 想用 conda env cpp 的 nvcc 13.3：  make CUDA_HOME=$CONDA_PREFIX
# A100 对应 sm_80。

CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      ?= sm_80

# -lineinfo 让 ncu / nsight 能把 SASS 指令映射回 .cu 源码行，profile 必备，
# 且不像 -G 那样会关掉优化（-G 会让性能数据完全失真，切勿用于 profiling）。
NVCCFLAGS := -O3 -arch=$(ARCH) -lineinfo -std=c++17
INCLUDES  := -I$(CUDA_HOME)/include
LIBS      := -lnvToolsExt

TARGETS := libmul2.so cuda_mul2

all: $(TARGETS)

# 共享库：给 bench_all.py 的 ctypes 用。不含 main，不依赖 NVTX。
libmul2.so: cuda_mul2.cu
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -shared -o $@ $<

# 独立可执行文件：给 nsys / ncu 单独 profile 用
cuda_mul2: main.cu cuda_mul2.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ main.cu cuda_mul2.cu $(LIBS)

run: cuda_mul2
	./cuda_mul2

# 看寄存器 / shared memory 占用，不需要任何 profiling 权限
occupancy:
	$(NVCC) $(NVCCFLAGS) -Xptxas=-v -c cuda_mul2.cu -o /dev/null

# 反汇编成 SASS，用于确认 vec4 版本真的发出了 128-bit 的 LDG.E.128
sass: cuda_mul2
	$(CUDA_HOME)/bin/cuobjdump -sass cuda_mul2 | grep -E 'Function|LDG|STG'

clean:
	rm -f $(TARGETS) *.o
	rm -rf __pycache__ reports/*.sqlite

.PHONY: all run occupancy sass clean
