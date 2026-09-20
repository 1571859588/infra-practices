# 01_vector_add
#
#   make            编译出 ./bench
#   make run        编译并运行
#   make ptx        导出 PTX，看编译器到底发了什么访存指令
#   make sass       导出 SASS（真正执行的机器码）
#   make sanitize   用 compute-sanitizer 查越界/竞争
#   make clean
#
# ARCH=sm_80 是 A100。换卡要改：V100=sm_70, A100=sm_80, H100=sm_90, RTX40xx=sm_89

include ../common.mk

TARGET  := bench
SRCS    := bench.cu v0_naive.cu v1_vec4.cu v2_grid_stride.cu v3_strided_bad.cu
OBJS    := $(SRCS:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

%.o: %.cu variants.cuh ../common.cuh
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

run: $(TARGET)
	$(RUNENV) ./$(TARGET)

quick: $(TARGET)
	$(RUNENV) ./$(TARGET) --quick

# 看向量化有没有生效：v1 的 PTX 里应该出现 ld.global.v4.f32
ptx:
	@for f in v0_naive v1_vec4 v2_grid_stride v3_strided_bad; do \
		$(NVCC) $(NVCCFLAGS) -ptx -o $$f.ptx $$f.cu; \
		echo "=== $$f.ptx 里的全局访存指令 ==="; \
		grep -oE '(ld|st)\.global[a-z0-9.]*' $$f.ptx | sort | uniq -c; \
	done

sass: $(TARGET)
	@for f in v0_naive v1_vec4 v2_grid_stride v3_strided_bad; do \
		$(NVCC) $(NVCCFLAGS) -cubin -o $$f.cubin $$f.cu; \
		echo "=== $$f SASS 里的访存指令 ==="; \
		$(CUOBJDUMP) -sass $$f.cubin | grep -oE '(LDG|STG)[A-Z0-9.]*' \
			| sort | uniq -c; \
	done

# 寄存器 / shared memory 用量 —— 决定 occupancy
regs:
	@for f in v0_naive v1_vec4 v2_grid_stride v3_strided_bad; do \
		echo "=== $$f ==="; \
		$(NVCC) $(NVCCFLAGS) -Xptxas -v -c -o /dev/null $$f.cu 2>&1 \
			| grep -E 'registers|smem' ; \
	done

sanitize: $(TARGET)
	$(RUNENV) $(SANITIZER) --tool memcheck ./$(TARGET) --quick

clean:
	rm -f $(TARGET) $(OBJS) *.ptx *.cubin *.ncu-rep *.nsys-rep *.sqlite

.PHONY: all run quick ptx sass regs sanitize clean
