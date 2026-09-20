# 02_reduction
#
#   make / make run / make quick / make ptx / make sass / make sanitize / make clean
# 说明见 ../01_vector_add/Makefile 和 README.md §2

include ../common.mk

TARGET := bench
SRCS   := bench.cu v0_interleaved.cu v1_no_divergence.cu v2_sequential.cu \
          v3_first_add.cu v4_shuffle.cu v5_gridstride.cu v6_cub.cu
OBJS   := $(SRCS:.cu=.o)

# CUB 随 CUDA Toolkit 一起装，不需要额外的 -I 或 -l

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

%.o: %.cu variants.cuh reduce_common.cuh ../common.cuh
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

run: $(TARGET)
	$(RUNENV) ./$(TARGET)

quick: $(TARGET)
	$(RUNENV) ./$(TARGET) --quick

# 看每一版归约的 shared memory 访问和同步指令
# BAR.SYNC   = __syncthreads()
# SHFL       = __shfl_down_sync()
# LDS / STS  = shared memory 读 / 写
sass: $(TARGET)
	@for f in v0_interleaved v1_no_divergence v2_sequential v3_first_add v4_shuffle v5_gridstride; do \
		$(NVCC) $(NVCCFLAGS) -cubin -o $$f.cubin $$f.cu; \
		echo "=== $$f ==="; \
		$(CUOBJDUMP) -sass $$f.cubin \
			| grep -oE '(LDS|STS|LDG|STG|SHFL|BAR)[A-Z0-9.]*' \
			| sort | uniq -c; \
	done

# 寄存器 / shared memory 用量 —— 决定 occupancy
regs:
	@for f in v0_interleaved v1_no_divergence v2_sequential v3_first_add v4_shuffle v5_gridstride; do \
		echo "=== $$f ==="; \
		$(NVCC) $(NVCCFLAGS) -Xptxas -v -c -o /dev/null $$f.cu 2>&1 \
			| grep -E 'registers|smem' ; \
	done

ptx:
	@for f in v0_interleaved v1_no_divergence v2_sequential v3_first_add v4_shuffle v5_gridstride; do \
		$(NVCC) $(NVCCFLAGS) -ptx -o $$f.ptx $$f.cu; \
	done
	@echo "PTX 已导出，用编辑器打开 v*.ptx 对比"

sanitize: $(TARGET)
	$(RUNENV) $(SANITIZER) --tool memcheck ./$(TARGET) --check

# 竞争检测：归约最容易漏 __syncthreads()，racecheck 专门查这个
racecheck: $(TARGET)
	$(RUNENV) $(SANITIZER) --tool racecheck ./$(TARGET) --check

clean:
	rm -f $(TARGET) $(OBJS) *.ptx *.cubin *.ncu-rep *.nsys-rep *.sqlite

.PHONY: all run quick ptx sass regs sanitize racecheck clean
