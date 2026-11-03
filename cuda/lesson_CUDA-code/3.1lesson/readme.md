# 3.1 CUDA 矩阵加法运算程序

本目录把「一维数组求和」分别用 CPU 和 GPU 实现，并引入公共头文件 `../tools/common.cuh`。
本文重点解释三个问题：

1. `#include <stdio.h>` 与 `#include "../tools/common.cuh"` 有什么区别；
2. `#pragma once` 是什么；
3. `tools/common.cuh` 里用到了哪些 CUDA 运行时 API。

---

## 一、本目录文件

| 文件 | 说明 |
| :--- | :--- |
| `matrixSum1D_CPU.cu` | 纯 CPU 实现，作为对照基准 |
| `matrixSum1D_GPU.cu` | GPU 实现，512 个元素、每块 32 线程，刚好整除，无需边界判断 |
| `matrixSum1D_GPU_if_device.cu` | GPU 实现，513 个元素（故意不整除），演示 `if (id >= N) return;` 边界保护 + `__device__` 设备函数 |
| `setDevice.cu` | 最小示例：只查询 GPU 数量并选定设备，不依赖 `common.cuh` |

---

## 二、两种 `#include` 写法的区别

```c
#include <stdio.h>              // 尖括号
#include "../tools/common.cuh"  // 双引号
```

### 1. 区别只在「搜索头文件的位置」

以 GCC / Clang / nvcc 为例（标准里属于实现定义行为，但这几家行为一致）：

| 写法 | 搜索顺序 |
| :--- | :--- |
| `#include <xxx>` | ① `-I` 指定的路径 → ② 编译器内置系统 include 路径（`/usr/include`、`/usr/local/cuda-12.4/include` 等） |
| `#include "xxx"` | ① **写这条 include 的源文件自己所在的目录** → ② `-I` 路径 → ③ 系统 include 路径 |

所以 `"../tools/common.cuh"` 的含义是：

> 从**当前源文件所在目录**（即 `3.1lesson/`）出发，向上一级到 `lesson_CUDA-code/`，再进 `tools/` 找 `common.cuh`。

关键点：**相对路径的基准是源文件所在目录，不是你敲命令时的工作目录**。因此在仓库根目录执行
`nvcc lesson_CUDA-code/3.1lesson/matrixSum1D_GPU.cu -o gpu` 一样能找到头文件。

### 2. 跟 C/C++ 的规则有什么区别？

**搜索规则本身完全一样**，CUDA 没改这一条。真正的差别在下面几点：

| 项 | C/C++ | CUDA |
| :--- | :--- | :--- |
| 头文件扩展名 | `.h` / `.hpp` | 约定用 **`.cuh`**（CUDA header），纯粹是命名习惯，文件内容仍然是 C++ |
| 编译器 | `gcc` / `g++` | 必须 `nvcc`——`.cuh` 里可以写 `__global__` / `__device__` 等 CUDA 关键字，`g++` 不认识 |
| 前端处理 | — | `nvcc` 会把 `.cu` / `.cuh` 先过一遍 CUDA 前端（`cudafe`），把主机代码和设备代码拆开，主机部分再交给 `g++` |
| 隐式包含 | 无 | 用 `nvcc` 编译 `.cu` 时会**自动隐式包含 `cuda_runtime.h`** |

最后一条解释了本目录的一个现象：`tools/common.cuh` 里根本没写 `#include <cuda_runtime.h>`，`setDevice.cu` 里也只写了 `#include <stdio.h>`，但 `cudaError_t`、`cudaGetDeviceCount` 这些照样能用——因为 `nvcc` 已经帮你把 CUDA 运行时头文件塞进来了。

如果用 `g++` 去编译这些文件，就会报 `cudaError_t was not declared in this scope`。

### 3. 想不用相对路径怎么写

```bash
nvcc -arch=sm_80 -I../tools matrixSum1D_GPU.cu -o gpu
```

然后在源码里写 `#include "common.cuh"`（或 `#include <common.cuh>`）。`-I` 把目录加进搜索路径，文件移动时不必改每一处相对路径。

---

## 三、`#pragma once` 是什么

`tools/common.cuh` 第 1 行就是它：

```c
#pragma once
```

### 1. 作用

**防止同一个头文件在同一个编译单元（.cu 文件）里被展开多次**，从而避免「函数/变量重定义」编译错误。

举例：如果 `a.cuh` 里 include 了 `common.cuh`，而 `main.cu` 又同时 include 了 `a.cuh` 和 `common.cuh`，那么 `common.cuh` 的内容会被贴进来两次，`ErrorCheck` 和 `setGPU` 就成了重复定义。加了 `#pragma once` 后，编译器记住「这个文件我见过了」，第二次直接跳过。

### 2. 传统写法对比（include guard）

```c
#ifndef _COMMON_CUH_
#define _COMMON_CUH_
/* 头文件内容 */
#endif
```

|  | `#pragma once` | `#ifndef` 宏保护 |
| :--- | :--- | :--- |
| 是否 C++ 标准 | 否，属于编译器扩展（但 GCC/Clang/MSVC/nvcc 全部支持） | 是，纯预处理技巧，任何编译器都行 |
| 写法 | 一行，省事 | 三行，且宏名要自己保证不撞车 |
| 速度 | 编译器可以直接跳过不再读文件，通常略快 | 每次都要读完再由预处理器丢弃 |
| 失效场景 | 同一个文件通过不同路径（符号链接、硬链接、大小写不同的路径）被包含时，可能被认为是两个文件 | 不会，只看宏名 |

实践建议：新项目用 `#pragma once` 就行，简单不易错。

### 3. 一个容易混淆的边界

`#pragma once` 只管**单个编译单元内部**的重复展开，管不了**链接期**的重复符号。见第五节第 2 点。

---

## 四、`tools/common.cuh` 用到了哪些运行时 API

常见叫法：CUDA Runtime API（区别于更底层的 Driver API `cu*` 系列）。它们的共同特征是 `cuda` 前缀、返回 `cudaError_t`。

### 1. `common.cuh` 内部用到的

| API / 类型 | 作用 |
| :--- | :--- |
| `cudaError_t` | 错误码枚举类型 |
| `cudaSuccess` | 唯一的「成功」值（等于 0），凡是 `!= cudaSuccess` 就是出错 |
| `cudaGetErrorName(cudaError_t)` | 错误码 → 枚举名字符串，如 `"cudaErrorMemoryAllocation"` |
| `cudaGetErrorString(cudaError_t)` | 错误码 → 人类可读描述，如 `"out of memory"` |
| `cudaGetDeviceCount(int *count)` | 查询本机可用 CUDA 设备数量，写入 `count` |
| `cudaSetDevice(int dev)` | 指定**当前主机线程**后续使用编号为 `dev` 的 GPU（编号从 0 起） |

外加两个预处理器宏（不是 CUDA 的，是 C 标准的）：

| 宏 | 展开结果 |
| :--- | :--- |
| `__FILE__` | 当前源文件名的字符串字面量 |
| `__LINE__` | 当前行号的整数 |

`ErrorCheck(err, __FILE__, __LINE__)` 就是靠这两个宏，把「出错在哪一行」打印出来。

### 2. `ErrorCheck` 的行为（重要）

```c
cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)
{
    if (error_code != cudaSuccess)
        printf("CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line%d\r\n", ...);
    return error_code;
}
```

注意它**只是打印，不会 `exit`，也不抛异常**。所以：

- 写 `ErrorCheck(cudaMalloc(...), __FILE__, __LINE__);` 而不接返回值 → 错了也照往下跑；
- 正确用法是像 `setGPU()` 里那样**接住返回值再判断**：

```c
cudaError_t error = ErrorCheck(cudaGetDeviceCount(&iDeviceCount), __FILE__, __LINE__);
if (error != cudaSuccess) { ...; exit(-1); }
```

`common.cuh` 底部被 `/* */` 注释掉的那段 `setGPU()`，就是「不做错误检测」的对照版本，可以对比着看。

### 3. `setGPU()` 做了什么

1. `cudaGetDeviceCount` 查有几块卡，没有就打印 `No CUDA campatable GPU found!` 并 `exit(-1)`；
2. `cudaSetDevice(0)` 选定 0 号卡。

注意：`cudaSetDevice` 是**线程级**的，只影响调用它的那个主机线程；多线程程序里每个线程都要各自设置（或依赖默认的 0 号设备）。

### 4. 主程序里用到的运行时 API（虽然不在 `common.cuh` 中）

| API | 作用 |
| :--- | :--- |
| `cudaMalloc(void **devPtr, size_t size)` | 在**设备显存**上分配 `size` 字节 |
| `cudaMemset(void *devPtr, int value, size_t count)` | 设备内存按**字节**置值（不是按元素） |
| `cudaMemcpy(dst, src, count, kind)` | 拷贝数据，`kind` 为 `cudaMemcpyHostToDevice` / `cudaMemcpyDeviceToHost` / `cudaMemcpyDeviceToDevice` |
| `cudaFree(void *devPtr)` | 释放 `cudaMalloc` 分配的设备内存 |
| `cudaDeviceSynchronize(void)` | 阻塞主机线程，直到设备上此前排入的所有任务执行完毕 |
| `cudaDeviceReset(void)` | 销毁当前进程的 CUDA 上下文，释放其所有资源；之后再用 CUDA 会重新初始化 |

补充：这些调用在 `matrixSum1D_GPU.cu` 里**全都没有做错误检查**（本课程 3.2 节才引入），实际项目里应当逐个包上 `ErrorCheck`。

---

## 五、几个容易踩的坑

### 1. `cudaMemcpy` 是同步的，`<<<>>>` 是异步的

`matrixSum1D_GPU.cu` 里把 `cudaDeviceSynchronize()` 注释掉了，结果依然正确——因为紧随其后的 `cudaMemcpy(..., cudaMemcpyDeviceToHost)` 会隐式等待设备跑完。

但这只是巧合。核函数启动本身是异步的，一旦中间插入任何异步操作（流式拷贝、`cudaMemcpyAsync`、另一个核函数），就必须显式 `cudaDeviceSynchronize()`。更稳妥的写法是始终同步，像 `matrixSum1D_GPU_if_device.cu` 那样。

### 2. `common.cuh` 里写的是函数「定义」，不是「声明」

`ErrorCheck` 和 `setGPU` 都是带函数体的完整定义。靠 `#pragma once` 只能保证单个 `.cu` 里不重复，**如果两个 `.cu` 都 include 它、再链接成同一个可执行文件，就会报 multiple definition**。

本目录每个文件各自编成独立可执行程序，所以没暴露问题。要改成多文件工程，二选一：

- 给函数加 `inline`（C++ 的 `inline` 允许多个 TU 各持一份定义）；
- 或者拆成 `common.h`（只放声明）+ `common.cu`（放定义），一起交给 `nvcc` 编译链接。

### 3. 数组长度和线程数的整除关系

- `matrixSum1D_GPU.cu`：`512 / 32 = 16`，正好整除，所以没有线程越界；
- `matrixSum1D_GPU_if_device.cu`：故意用 513，此时 `513 / 32 = 16`（C 整数除法向下取整）会漏掉最后 1 个元素，必须：
  - grid 向上取整：`(iElemCount + block.x - 1) / block.x` → 17；
  - 核函数里加 `if (id >= N) return;` 挡掉多出来的线程。

这是 CUDA 编程里最基础的边界处理套路。

---

## 六、编译与运行

本机是 A100（计算能力 `sm_80`），CUDA 12.4：

```bash
cd /mnt/public/nyt1/infra/practices/cuda/lesson_CUDA-code/3.1lesson

nvcc -arch=sm_80 matrixSum1D_CPU.cu           -o cpu
nvcc -arch=sm_80 matrixSum1D_GPU.cu           -o gpu
nvcc -arch=sm_80 matrixSum1D_GPU_if_device.cu -o gpu_if
nvcc -arch=sm_80 setDevice.cu                 -o setdev

./cpu
./gpu
./gpu_if
./setdev
```

`-arch=sm_80` 要按实际显卡调整：V100 是 `sm_70`，RTX 3090 是 `sm_86`，H100 是 `sm_90`；省略 `-arch` 也能编，但可能编出与本机不匹配的架构，运行时报 `no kernel image is available for execution`。

预期输出（`./gpu`）类似：

```
The count of GPUs is 1.
set GPU 0 for computing.
idx= 1	matrix_A:21.50	matrix_B:16.40	result=37.90
...
```
