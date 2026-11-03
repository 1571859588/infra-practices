# 04_relu
#
#   make            编译出 ./bench
#   make run        编译并运行
#   make ptx        导出 PTX，看编译器到底发了什么指令
#   make sass       导出 SASS（真正执行的机器码），看 FMNMX
#   make sanitize   用 compute-sanitizer 查越界/竞争
#   make clean
#
# ARCH=sm_80 是 A100。换卡要改：V100=sm_70, A100=sm_80, H100=sm_90, RTX40xx=sm_89

include ../common.mk

TARGET  := bench
SRCS    := bench.cu v0_naive.cu v1_vec4.cu v2_vec4_grid.cu v3_fused.cu
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

# 看分支去哪了：v0 两个版本里都不该有真正的分支指令，
# 计算应该变成 fmax / FMNMX（select）。
ptx:
	@for f in v0_naive v1_vec4 v2_vec4_grid v3_fused; do \
		$(NVCC) $(NVCCFLAGS) -ptx -o $$f.ptx $$f.cu; \
		echo "=== $$f.ptx 里的计算/访存指令 ==="; \
		grep -oE '(fmax|selp|setp|(ld|st)\.global[a-z0-9.]*)' $$f.ptx \
			| sort | uniq -c; \
	done

sass: $(TARGET)
	@for f in v0_naive v1_vec4 v2_vec4_grid v3_fused; do \
		$(NVCC) $(NVCCFLAGS) -cubin -o $$f.cubin $$f.cu; \
		echo "=== $$f SASS 里的计算/访存指令 ==="; \
		$(CUOBJDUMP) -sass $$f.cubin \
			| grep -oE '(FMNMX|FSETP|LDG|STG)[A-Z0-9.]*' | sort | uniq -c; \
	done

# 寄存器 / shared memory 用量 —— 决定 occupancy
regs:
	@for f in v0_naive v1_vec4 v2_vec4_grid v3_fused; do \
		echo "=== $$f ==="; \
		$(NVCC) $(NVCCFLAGS) -Xptxas -v -c -o /dev/null $$f.cu 2>&1 \
			| grep -E 'registers|smem' ; \
	done

sanitize: $(TARGET)
	$(RUNENV) $(SANITIZER) --tool memcheck ./$(TARGET) --quick

clean:
	rm -f $(TARGET) $(OBJS) *.ptx *.cubin *.ncu-rep *.nsys-rep *.sqlite

.PHONY: all run quick ptx sass regs sanitize clean
