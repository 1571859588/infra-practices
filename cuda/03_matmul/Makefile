# 03_matmul
#
#   make / make run / make quick / make ptx / make sass / make sanitize / make clean
# 说明见 ../01_vector_add/Makefile 和 README.md §2

include ../common.mk

TARGET := bench
SRCS   := bench.cu v0_naive.cu v1_smem.cu v2_regtile.cu v3_vec4.cu v4_cublas.cu
OBJS   := $(SRCS:.cu=.o)
VARIANTS := v0_naive v1_smem v2_regtile v3_vec4

# cuBLAS 需要显式链接（CUB 不用，那是纯头文件）
LDLIBS := -lcublas

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cu variants.cuh ../common.cuh
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

run: $(TARGET)
	$(RUNENV) ./$(TARGET)

quick: $(TARGET)
	$(RUNENV) ./$(TARGET) --quick

# matmul 最该看的是 FFMA 条数和访存指令的比例 —— 这直接反映算术强度。
#   FFMA          = fp32 乘加（真正干活的指令）
#   LDS / STS     = shared memory 读 / 写
#   LDG.E.128     = float4 全局读（v3 才有）
#   BAR.SYNC      = __syncthreads()
sass: $(TARGET)
	@for f in $(VARIANTS); do \
		$(NVCC) $(NVCCFLAGS) -cubin -o $$f.cubin $$f.cu; \
		echo "=== $$f ==="; \
		$(CUOBJDUMP) -sass $$f.cubin \
			| grep -oE '(FFMA|LDS|STS|LDG|STG|BAR)[A-Z0-9.]*' \
			| sort | uniq -c | sort -rn; \
	done

# 寄存器用量和 shared memory 用量 —— 决定 occupancy，v3 的关键约束
regs:
	@for f in $(VARIANTS); do \
		echo "=== $$f ==="; \
		$(NVCC) $(NVCCFLAGS) -Xptxas -v -c -o /dev/null $$f.cu 2>&1 \
			| grep -E 'Function properties|registers|smem' ; \
	done

ptx:
	@for f in $(VARIANTS); do \
		$(NVCC) $(NVCCFLAGS) -ptx -o $$f.ptx $$f.cu; \
	done
	@echo "PTX 已导出，用编辑器打开 v*.ptx 对比"

# 只跑正确性用例（最大 1000³），完整 benchmark 在 sanitizer 下要跑几十分钟
sanitize: $(TARGET)
	$(RUNENV) $(SANITIZER) --tool memcheck ./$(TARGET) --check

racecheck: $(TARGET)
	$(RUNENV) $(SANITIZER) --tool racecheck ./$(TARGET) --check

clean:
	rm -f $(TARGET) $(OBJS) *.ptx *.cubin *.ncu-rep *.nsys-rep *.sqlite

.PHONY: all run quick ptx sass regs sanitize racecheck clean
