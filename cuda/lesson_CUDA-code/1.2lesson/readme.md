```bash
cd /mnt/public/nyt1/infra/practices/cuda/lesson_CUDA-code/1.2lesson
nvcc -arch=sm_80 test.cu -o test
./test
```

nvcc 在 /usr/local/cuda-12.4/bin/nvcc（CUDA 12.4.131），已在 PATH 里，直接可用。
显卡是 A100-SXM4-40GB，计算能力 sm_80，所以加 -arch=sm_80。省略也能编（默认会按 nvcc 默认架构编 JIT），但显式指定更快、避免架构不匹配。
当前 GPU 显存已被别的进程占了 38537/40960 MiB，本项目这种小 kernel 还能跑；后续如果报 out of memory，换卡或等释放：nvidia-smi 看进程，CUDA_VISIBLE_DEVICES=<id> 指定空闲卡。

常用变体

```bash
nvcc -arch=sm_80 -O2 test.cu -o test   # 开优化
nvcc -arch=sm_80 -g -G test.cu -o test # 调试（cuda-gdb / cuda-memcheck）
nvcc -arch=sm_80 -Xptxas -v test.cu -o test  # 看寄存器/共享内存占用
```
注意
hello_from_gpu<<<4, 4>>>() 后面必须跟 cudaDeviceSynchronize()，否则主机端 main 直接返回、进程退出，GPU 的 printf 可能来不及刷出来；同时核函数启动是异步的，没同步也看不到运行时错误。


# -arch 和 -code 虚拟架构和真实架构

**虚拟架构 `compute_XX`（PTX 虚拟指令集）**，配合 `-code` 或 `-gencode` 用。

## 三个概念

| 选项 | 含义 | 产物 |
|---|---|---|
| `-arch=compute_80` | 虚拟架构（virtual arch），PTX 虚拟指令集 | 只生成 PTX 文本，运行时由驱动 JIT 编译 |
| `-code=sm_80` | 真实架构（real arch），GPU 硬件指令集 SASS | 生成 cubin（ELF） |
| `-arch=sm_80` | 上面两者的简写 | 同时嵌入 SASS + PTX（fatbin） |

关系：**PTX 是中间表示，SASS 才是 A100 真正执行的机器码**。`-arch` 指定"用什么虚拟指令集来编译"，`-code` 指定"最终翻译成哪款硬件的机器码"。

## 我在你这台机器上实测的结果

```bash
nvcc -arch=sm_80 test.cu -o t_arch && cuobjdump t_arch
```
```
Fatbin elf code:   arch = sm_80        ← SASS，A100 直接执行
Fatbin ptx code:   arch = sm_80        ← PTX 也一起嵌进去了（兜底用）
```

```bash
nvcc -arch=compute_80 test.cu -o t_ptx && cuobjdump t_ptx
```
```
Fatbin ptx code:   arch = sm_80        ← 只有 PTX，没有 SASS
cuobjdump -sass t_ptx  → 空
```

第二种情况程序照样能跑，但**首次加载时驱动会现场把 PTX 编成 SASS（JIT）**，启动慢一点；好处是向前兼容——将来换 H100/L40S，没有对应 cubin 也能靠这份 PTX 跑起来。

## 实际用法

```bash
# 1) 常规：只要在当前 A100 上跑（推荐日常用这个）
nvcc -arch=sm_80 test.cu -o test

# 2) 只产 PTX，把 JIT 交给运行机（做库/分发时用）
nvcc -arch=compute_80 test.cu -o test
# 等价写法
nvcc -arch=compute_80 -code=compute_80 test.cu -o test

# 3) 显式两段式：先编成 PTX，再汇编成 sm_80 的 SASS
nvcc -arch=compute_80 -code=sm_80 test.cu -o test

# 4) 既出 SASS 又留 PTX（等价 -arch=sm_80）
nvcc -arch=compute_80 -code=sm_80,compute_80 test.cu -o test

# 5) 一个二进制塞多个架构（fatbin，最通用的分发方式）
nvcc -gencode arch=compute_80,code=sm_80 \
      -gencode arch=compute_90,code=sm_90 \
      -gencode arch=compute_90,code=compute_90 \
      test.cu -o test
```
第 5 种在 A100/H100 上都能原生跑，在更新的卡上靠最后那条 PTX JIT 兜底。

## 怎么看生成的是什么

```bash
cuobjdump test          # 列出 fatbin 里有哪些 arch/ptx
cuobjdump -sass test    # 反汇编 SASS（能看寄存器、指令）
cuobjdump -ptx test     # 打印 PTX
nvcc -arch=sm_80 -Xptxas -v test.cu -o test   # 编译期打印寄存器/共享内存用量
```

## 注意

- `code=sm_XX` 必须 ≥ `arch=compute_XX`，且两者要"同代可兼容"（例如 `compute_80` 不能 `code=sm_70`，PTX 用了新特性旧硬件跑不了）。
- `-arch=compute_80` 单独用**不会**报架构不匹配，但如果目标机驱动太老，JIT 会失败并报 `CUDA error: no kernel image is available for execution`。你这台驱动是 580（CUDA 13），完全够。