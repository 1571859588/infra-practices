# 本机 CUDA / 性能分析工具环境速查

> 记录日期：2026-09-15
> 机器：8 × NVIDIA A100-SXM4-40GB（compute capability **8.0**，即 `sm_80`）
> 驱动：580.159.03（driver 侧 CUDA 版本 13.0）

本文档列出 `ncu`、`nsys`、`c++`、`nvcc`、CUDA 在本机的**实际可用位置**、对应的 conda / uv 环境，以及验证过的运行方法。

---

## 仓库结构

```
practices/
├── README.md              ← 本文件：环境速查（工具在哪、怎么跑、权限怎么解决）
├── 综合练习/               ← 一个算子从写到 profile 的完整流程
│   └── vector_mul2/          y = x*2，PyTorch / Triton / CUDA 三种写法对比
│                             + nsys + ncu + compute-sanitizer 全流程实测
└── triton/                ← 纯 Triton 练习，只练编程模型本身
    ├── 01_vector_add.py      program_id / arange / mask
    ├── 02_fused_softmax.py   规约 + 算子融合（4.3× 提速，最有说服力的一个）
    └── 03_matmul.py          2D 分块 / tl.dot / Tensor Core / 分块调优
```

| 想干什么 | 去哪 |
|---|---|
| 查 `nvcc`/`ncu`/`nsys`/`c++` 在哪、怎么跑 | 本文件 |
| 解决 `ncu` 的 `ERR_NVGPUCTRPERM` | 本文件 [§6](#6-ncunsight-compute) |
| 学 profiler 怎么用、报告怎么读 | [`综合练习/vector_mul2/`](综合练习/vector_mul2/README.md) |
| 对比 PyTorch / Triton / CUDA 三种写法 | [`综合练习/vector_mul2/`](综合练习/vector_mul2/README.md) §2 |
| 练 Triton 本身（规约 / 融合 / 分块） | [`triton/`](triton/README.md) |

两个练习目录都是**实测数据 + 踩坑清单 + 留白笔记**的结构，可以直接照着复现。

---

## 0. TL;DR 速查表

| 工具 | 可用来源 | 版本 | 是否在 PATH | 状态 |
|---|---|---|---|---|
| `nvcc` | `/usr/local/cuda/bin/nvcc`（系统） | 12.4.131 | ❌ 需手动加 PATH | ✅ 编译运行已验证 |
| `nvcc` | conda env `cpp` / `nccl-test` | 13.3.73 | ✅ 激活后即有 | ✅ 编译运行已验证 |
| `nvcc` | conda env `material-agent` / `lammps-deepmd` | 12.9.86 | ✅ 激活后即有 | 可用 |
| `nsys` | `/usr/local/cuda/bin/nsys`（**仅系统**） | 2023.4.4 | ❌ 需手动加 PATH | ✅ 可用（CPU 采样需走 Docker） |
| `ncu` | `/usr/local/cuda/bin/ncu`（**仅系统**） | 2024.1.1 | ❌ 需手动加 PATH | ⚠️ 宿主机报 `ERR_NVGPUCTRPERM`；**用 Docker 可正常采集**（见 §6） |
| `ncu` | Docker `nvidia/cuda:12.4.0-devel-ubuntu22.04` | 2024.1.0 | 容器内自带 | ✅ **推荐方式**，实测可用 |
| `c++` / `g++` | `/usr/bin/c++`（系统） | GCC 12.3.0 | ✅ 默认 | ✅ 可用 |
| `c++` / `g++` | conda env（`x86_64-conda-linux-gnu-g++`） | GCC 13.4.0 | ✅ 激活后即有 | ✅ 可用 |
| CUDA runtime | `/usr/local/cuda` → `cuda-12.4` | 12.4 | — | ✅ |
| CUDA runtime | conda env `cpp` | 13.3 + NCCL 2.30.7 | — | ✅ |

**重要：`ncu` 和 `nsys` 在任何 conda 环境里都不存在**，只有系统的 `/usr/local/cuda-12.4` 里有。conda 的 `cuda-nvcc` 包只装了编译器，不含 Nsight 工具。

**`ncu` 的权限问题有解，不用等管理员重启** —— 用 Docker 映射 GPU 并加
`--cap-add=SYS_ADMIN`，容器内以 root 跑即可。详见 [§6](#6-ncunsight-compute)。

---

## 1. 一键环境设置

把下面几行加到 `~/.zshrc`（或每次开 shell 时手动执行），之后 `nvcc / ncu / nsys / compute-sanitizer / cuda-gdb` 全部直接可用：

```bash
export CUDA_HOME=/usr/local/cuda            # -> /usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

验证：

```bash
nvcc --version      # release 12.4, V12.4.131
ncu --version       # 2024.1.1.0
nsys --version      # 2023.4.4.54
```

> 注意：如果之后又 `conda activate cpp`，conda 会把自己的 `bin` 插到 PATH 最前面，`nvcc` 会变成 conda 里的 13.3 版本，而 `ncu` / `nsys` 仍然落到系统的 12.4。这个组合是可以工作的（见第 5 节）。

---

## 2. CUDA 工具链

### 2.1 系统 CUDA 12.4（推荐做 profiling 时用）

```
/usr/local/cuda -> /usr/local/cuda-12.4
├── bin/           nvcc, ncu, nsys, cuda-gdb, compute-sanitizer, nvdisasm, cuobjdump, ptxas, nvprof ...
├── nsight-compute-2024.1.1/
└── nsight-systems-2023.4.4/
```

这是本机**唯一**带 Nsight 全家桶的安装。

### 2.2 conda 环境里的 CUDA

`~/.condarc` 把 env 目录指到了两处共享盘：

```yaml
envs_dirs:
  - /mnt/public/conda/envs
  - /mnt/public/nyt1/docqa/restored_envs
pkgs_dirs:
  - /data/home/nieyuntao/conda/pkgs
auto_activate_base: false
```

含 `nvcc` 的环境：

| env 名 | 路径 | nvcc | conda g++ | Python | torch |
|---|---|---|---|---|---|
| `cpp` | `/mnt/public/nyt1/docqa/restored_envs/cpp` | **13.3.73** | 13.4.0 | 3.12.12 | 2.9.0+cu128 |
| `nccl-test` | `/mnt/public/nyt1/docqa/restored_envs/nccl-test` | **13.3.73** | 13.4.0 | 3.12.12 | 2.9.0+cu128 |
| `material-agent` | `/mnt/public/conda/envs/material-agent` | 12.9.86 | 13.4.0 | 3.11.15 | 2.11.0 (cu12.9) |
| `lammps-deepmd` | `/mnt/public/conda/envs/lammps-deepmd` | 12.9.86 | 13.4.0 | 3.11.15 | — |

`cpp` 环境是最完整的 CUDA C++ 开发环境：`cuda-nvcc 13.3` + `cuda-cudart-dev` + `nccl 2.30.7` + `gcc/gxx 13.4` + `cmake` + `ninja` + `cuda-python` + `cupy-cuda12x`。

仅有 conda 工具链（无 nvcc）的环境：`dist_train`、`base`（都带 `x86_64-conda-linux-gnu-g++`）。

---

## 3. `nvcc` —— 运行方法

### 3.1 用系统 nvcc 12.4（已验证）

```bash
export PATH=/usr/local/cuda/bin:$PATH
nvcc -arch=sm_80 -O3 -o vecadd vecadd.cu
./vecadd
```

`-arch=sm_80` 对应 A100。写通用一点可以用 `-gencode arch=compute_80,code=sm_80`。

### 3.2 用 conda env `cpp` 的 nvcc 13.3（已验证）

```bash
conda activate cpp
nvcc -arch=sm_80 -O3 -o vecadd vecadd.cu
./vecadd
```

要点：
- CUDA 13.x 的 `nvcc` **默认静态链接 cudart**，产物 `ldd` 里看不到 `libcudart.so`，因此**不需要设 `LD_LIBRARY_PATH` 也能跑**（已实测）。
- 驱动报 CUDA 13.0，toolkit 是 13.3，靠 CUDA **minor version compatibility** 正常工作，实测无问题。
- conda 环境自带 `cmake` / `ninja`，做 CMake 工程时：
  ```bash
  conda activate cpp
  cmake -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=80
  cmake --build build
  ```

### 3.3 指定宿主编译器

nvcc 需要一个 host C++ 编译器。默认取 PATH 里的 `c++`。若版本不兼容可显式指定：

```bash
nvcc -ccbin /usr/bin/g++ -arch=sm_80 x.cu -o x                       # 系统 GCC 12.3
nvcc -ccbin $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ -arch=sm_80 x.cu -o x   # conda GCC 13.4
```

---

## 4. `c++` / `g++` —— 运行方法

### 4.1 系统编译器（默认，开箱即用）

```bash
c++ --version     # (Ubuntu 12.3.0-1ubuntu1~22.04.3) 12.3.0
g++ -std=c++20 -O2 main.cpp -o main
```

### 4.2 conda 环境编译器（GCC 13.4）

conda 环境里 `c++` 这个名字**不一定存在**，真正的驱动器是带 triplet 前缀的：

```bash
conda activate cpp
x86_64-conda-linux-gnu-g++ --version    # conda-forge gcc 13.4.0
```

激活脚本（`$CONDA_PREFIX/etc/conda/activate.d/activate-gxx_linux-64.sh`）会自动设好 `CXX`、`CC`、`CXXFLAGS`、`LDFLAGS`，所以直接用变量更稳：

```bash
conda activate cpp
echo $CXX                      # 指向 conda 的 g++
$CXX -std=c++20 -O2 main.cpp -o main
```

CMake 会自动读取 `CC` / `CXX`，无需额外配置。

---

## 5. `nsys`（Nsight Systems）—— 可用 ✅

只有系统版本：`/usr/local/cuda/bin/nsys`，版本 `2023.4.4`。

### 5.1 基本用法（已验证）

```bash
export PATH=/usr/local/cuda/bin:$PATH

# profile 一个 CUDA 可执行文件
nsys profile -o report --force-overwrite true ./vecadd

# profile 一个 conda 环境里的 PyTorch 脚本（注意用 env 的绝对 python 路径最稳）
nsys profile -t cuda,nvtx,osrt -o torch_rep --force-overwrite true \
    /mnt/public/nyt1/docqa/restored_envs/cpp/bin/python train.py
```

### 5.2 看结果

```bash
# 命令行汇总（会先生成 .sqlite）
nsys stats --report cuda_gpu_kern_sum report.nsys-rep     # kernel 耗时排行
nsys stats --report cuda_api_sum      report.nsys-rep     # CUDA API 耗时
nsys stats --report cuda_gpu_mem_time_sum report.nsys-rep # memcpy/memset

# 列出所有可用报表
nsys stats --help-reports
```

实测输出示例：

```
 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):
 Time (%)  Total Time (ns)  Instances   Avg (ns)    ...  Name
    100.0        3,139,041          1  3,139,041.0  ...  add(const float *, const float *, float *, int)
```

`.nsys-rep` 可以下载到本地用 Nsight Systems GUI 打开（本机 `nsys-ui` 需要 X11/图形环境，一般不用）。

### 5.3 ⚠️ 已知限制：宿主机上 CPU 采样不可用（Docker 里可用）

`/proc/sys/kernel/perf_event_paranoid = 4`，且无 root，所以每次 profile 都会打印：

```
WARNING: CPU IP/backtrace sampling not supported, disabling.
WARNING: CPU context switch tracing not supported, disabling.
```

`nsys status --environment` 显示：

```
Root privilege: disabled
Linux Kernel Paranoid Level = 4
Linux perf_event_open syscall available: Fail
CPU Profiling Environment (process-tree): Fail
```

**影响**：CPU 侧火焰图 / 线程调度时间线拿不到。
**不影响**：CUDA kernel 时间线、CUDA API trace、memcpy、NVTX、NCCL trace —— 这些走的是 CUPTI，全部正常。

**如需 CPU 采样，用 Docker（推荐，无需管理员）**：容器内加 `--cap-add=SYS_ADMIN`
即可绕过 paranoid 限制。镜像里没有 nsys，把宿主机的挂进去：

```bash
docker run --rm --entrypoint bash --gpus '"device=0"' --cap-add=SYS_ADMIN \
  -v "$PWD":/work -w /work \
  -v /usr/local/cuda-12.4/nsight-systems-2023.4.4:/opt/nsys:ro \
  nvidia/cuda:12.4.0-devel-ubuntu22.04 -c \
  '/opt/nsys/bin/nsys profile -t cuda,nvtx,osrt --sample=cpu \
      --cpuctxsw=process-tree -o /work/rep --force-overwrite true ./app
   chown '"$(id -u):$(id -g)"' /work/rep.nsys-rep'
```

实测容器内 `nsys status --environment` 全部变成 `OK`：

```
Root privilege: enabled
Linux perf_event_open syscall available: OK
CPU Profiling Environment (process-tree): OK
CPU Profiling Environment (system-wide): OK
```

对比两份报告的 sqlite：容器内采的有 `COMPOSITE_EVENTS`(322)、
`SAMPLING_CALLCHAINS`(2267)、`SCHED_EVENTS`(202)，宿主机采的**这三张表根本不存在**。

**或者**请管理员放开（需 root，会影响全机）：

```bash
sudo sh -c 'echo 2 > /proc/sys/kernel/perf_event_paranoid'
# 永久生效：
# echo "kernel.perf_event_paranoid=2" | sudo tee /etc/sysctl.d/99-perf.conf
```

---

## 6. `ncu`（Nsight Compute）

### 6.1 宿主机上直接跑：失败

```bash
$ /usr/local/cuda/bin/ncu --set basic ./vecadd
==PROF== Connected to process 2541179
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
          Performance Counters on the target device 0.
==WARNING== No kernels were profiled.
```

**原因**：NVIDIA 驱动默认把 GPU 性能计数器限制为管理员（`NVreg_RestrictProfilingToAdminUsers=1`）。换 conda 环境、装 pip 版 ncu、换 CUDA 版本都没用 —— 这不是版本问题。

### 6.2 ✅ 解决：用 Docker（实测可用，不需要管理员、不需要重启）

**关键认知：驱动检查的不是"你是不是 root"，而是进程有没有 `CAP_SYS_ADMIN` 这个 capability。**
容器里以 root 运行 + `--cap-add=SYS_ADMIN` 就满足了。这是 NVIDIA 官方文档给出的容器内 profiling 方案，宿主机一行都不用改。

本机已具备条件：`docker 29.6.0`、当前用户在 `docker` 组、`nvidia-container-toolkit` 已装、`nvidia` runtime 已注册。

```bash
docker run --rm --entrypoint bash \
    --gpus '"device=0"' \            # 映射指定 GPU
    --cap-add=SYS_ADMIN \            # ★ 关键：解锁 GPU 性能计数器
    -v "$PWD":/work -w /work \
    nvidia/cuda:12.4.0-devel-ubuntu22.04 -c \
    'ncu --set full -o /work/prof --force-overwrite ./app
     chown '"$(id -u):$(id -g)"' /work/prof.ncu-rep'
```

**四个必须知道的坑**（都是实测踩出来的）：

| 坑 | 现象 | 正确做法 |
|---|---|---|
| 不加 `--cap-add=SYS_ADMIN` | 照样 `ERR_NVGPUCTRPERM` | 必加，这一条是全部的关键 |
| 加了 `--user $(id -u):$(id -g)` | `CapEff: 0000000000000000`，capability 全丢，ncu 仍失败 | **必须容器 root 跑**，结尾 `chown` 把产物属主改回来 |
| 以为镜像带 nsys | `nvidia/cuda:*-devel` **只有 ncu 和 nvcc，没有 nsys** | 把宿主机的挂进去：`-v /usr/local/cuda-12.4/nsight-systems-2023.4.4:/opt/nsys:ro` |
| `--gpus '"device=0"'` 引号写少一层 | docker 参数解析失败 | shell 一层 + docker 一层，两层都要 |

验证权限到位：

```bash
$ docker run --rm --cap-add=SYS_ADMIN ... -c 'grep CapEff /proc/self/status'
CapEff:	00000000a82425fb        # 非 0，含 CAP_SYS_ADMIN
```

**要在容器里跑 conda 环境的 PyTorch/Triton**，把 env 所在盘按原路径挂进去即可（路径一致，就能直接用绝对路径的 python）：

```bash
-v /mnt/public:/mnt/public:ro \
-e TRITON_CACHE_DIR=/tmp/triton_cache      # 只读挂载下 triton 要能写缓存
# 然后容器内直接：/mnt/public/nyt1/docqa/restored_envs/cpp/bin/python train.py
```

实测容器内该环境正常：torch 2.9.0+cu128、triton 3.5.0、`torch.cuda.is_available() = True`。

> 📁 完整可跑的脚本见
> [`综合练习/vector_mul2/profile/run_ncu_docker.sh`](综合练习/vector_mul2/profile/run_ncu_docker.sh)，
> 实测数据和逐项解读见 [`综合练习/vector_mul2/README.md` §5](综合练习/vector_mul2/README.md#5-ncu-分析用-docker-绕过权限限制-)。

### 6.3 其他方案（不如 Docker 方便，记录备查）

1. **管理员放开驱动限制**（一劳永逸，但要**重启整机**，共享服务器上基本不现实）
   ```bash
   sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia-profiling.conf'
   sudo update-initramfs -u
   sudo reboot
   ```

2. **`sudo` 跑 ncu**：本机 `sudo` 需要密码、`sudo -n` 不可用；若你有密码可行，但每次都要输。
   ```bash
   sudo -E /usr/local/cuda/bin/ncu --set full -o prof ./vecadd
   ```

3. **不用 ncu**：先用 `nsys` 定位热点 kernel，用 `cuobjdump -sass` 看访存指令宽度，用 `nvcc -Xptxas=-v` 看寄存器/spill，用 `compute-sanitizer` 查正确性 —— 这些都不需要性能计数器权限，能回答相当多的问题（见 `综合练习/vector_mul2/README.md` §7）。

### 6.4 常用命令

```bash
# 宿主机 PATH（仅用于 --import 回看报告，采集要走容器）
export PATH=/usr/local/cuda/bin:$PATH

ncu --set basic ./app                                   # 快速概览
ncu --set full -o prof --force-overwrite ./app          # 全量，生成 prof.ncu-rep
ncu -k add --launch-count 1 --set full ./app            # 只测名为 add 的 kernel
ncu -k add --launch-skip 10 --launch-count 2 ./app      # 跳过 warmup 的前 10 次 launch
ncu --nvtx --nvtx-include 'bench/' ./app                # 按 NVTX 区段筛（比数 launch 次数稳）
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed ./app   # 只取指定指标，不出报告文件
ncu --import prof.ncu-rep --page details                # 回看报告（不需要 GPU 权限）
ncu --import prof.ncu-rep --page details --csv          # 转 CSV 便于脚本处理
ncu --import a.ncu-rep --baseline b.ncu-rep --page details  # 两份报告做 diff
```

两个反复会踩的点：

- **对 Python/PyTorch 必须收窄范围**（`-k` / `--nvtx-include` / `--launch-count`）。ncu 是 replay 机制，会把同一个 kernel 重放几十次采全指标；不收窄的话 torch 初始化那几十个 kernel 都要 replay，几十分钟到几小时。
- **`--launch-count N` 之前先数清楚 warmup 发了几次 launch**，否则名额被 warmup 吃光，报告里根本没有你要的 kernel。用 `--launch-skip` 或干脆按 NVTX 区段筛。

---

## 7. uv 环境

uv 已安装：`/data/home/nieyuntao/.local/bin/uv`，版本 `0.11.32`。

`~/.zshrc` 里的配置：

```bash
export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE=copy
export UV_CACHE_DIR=/mnt/public/nyt1/.uv_cache
```

可用的本地 Python：3.11.15、3.10.20（uv 管理）、3.10.12（系统）；其余版本可联网下载（本机 pypi 可达）。

### uv 与 CUDA 的关系

**uv 环境不提供 `nvcc` / `ncu` / `nsys`**。uv 只管 Python 包。做法是：**uv 管 Python 依赖 + 系统 `/usr/local/cuda` 管工具链**。

```bash
cd /mnt/gfs/nyt1/infra/practices
uv init                                  # 或 uv venv --python 3.12
uv add torch --index https://download.pytorch.org/whl/cu128
uv add cupy-cuda12x numpy

# 编译 / profile 仍然走系统 CUDA
export PATH=/usr/local/cuda/bin:$PATH
nsys profile -t cuda,nvtx -o rep uv run python train.py
```

> 如果确实想要更新版本的 Nsight Compute CLI，可以 `uv pip install nvidia-nsight-compute-cu12`（pypi 可达）。但**这只解决版本问题，不解决第 6 节的权限问题** —— 装完照样报 `ERR_NVGPUCTRPERM`，因为缺的是 `CAP_SYS_ADMIN`，不是新版本。要采集就走 [§6.2](#62--解决用-docker实测可用不需要管理员不需要重启) 的 Docker 方案。

---

## 8. 其他可用的 CUDA 工具（系统 12.4，均无需特殊权限）

```bash
export PATH=/usr/local/cuda/bin:$PATH

compute-sanitizer --tool memcheck ./app     # 显存越界 / 非法访问（2024.1.1）
compute-sanitizer --tool racecheck ./app    # shared memory 竞争
compute-sanitizer --tool synccheck ./app    # __syncthreads 误用
cuda-gdb ./app                              # CUDA 调试器（12.4）
cuobjdump -sass ./app                       # 反汇编 SASS
nvdisasm -c kernel.cubin                    # cubin 反汇编
ptxas -arch=sm_80 -v kernel.ptx             # 看寄存器 / shared mem 占用
nvcc -arch=sm_80 -Xptxas=-v -c kernel.cu    # 编译时直接打印占用信息
```

`nvcc -Xptxas=-v` 在做 occupancy 调优时基本可以替代 ncu 的一部分静态信息。

---

## 9. 推荐组合

| 场景 | 推荐配置 |
|---|---|
| 写 / 编译 CUDA C++ | `conda activate cpp`（nvcc 13.3 + gcc 13.4 + cmake + ninja） |
| Profiling（时间线，只要 GPU 侧） | `export PATH=/usr/local/cuda/bin:$PATH` + `nsys`，宿主机直接跑 |
| Profiling（时间线 + CPU 火焰图） | Docker + `--cap-add=SYS_ADMIN`，nsys 从宿主机挂载（§5.3） |
| Profiling（kernel 微观） | **Docker + `--cap-add=SYS_ADMIN` + 容器 root**（§6.2） |
| 正确性排查 | `compute-sanitizer`（系统 12.4，无需特殊权限） |
| 纯 Python / PyTorch 实验 | uv 项目 + `torch cu128`，或现成 env `cpp` / `vllm` / `dist_train` |
| 多卡 / NCCL 测试 | `conda activate nccl-test`（nccl 2.30.7） |

---

## 10. 验证记录

以下均在 2026-09-15 于本机实测通过：

- ✅ `/usr/local/cuda/bin/nvcc -arch=sm_80` 编译 vecadd 并正确运行
- ✅ `cpp` env 的 `nvcc 13.3 -arch=sm_80` 编译并运行（无需 `LD_LIBRARY_PATH`，cudart 静态链接）
- ✅ `nsys profile` + `nsys stats --report cuda_gpu_kern_sum` 对原生 CUDA 程序
- ✅ `nsys profile -t cuda,nvtx` 对 `cpp` env 的 PyTorch matmul 脚本
- ❌ 宿主机 `ncu --set basic` → `ERR_NVGPUCTRPERM`（驱动权限限制）
- ⚠️ 宿主机 `nsys` CPU 采样 → 被 `perf_event_paranoid=4` 禁用，GPU 部分不受影响

Docker 方案（同日实测，见 §6.2）：

- ✅ `docker run --gpus '"device=3"' --cap-add=SYS_ADMIN` 容器内 `ncu --set full` **成功采集**，
  生成 `ncu_cuda.ncu-rep`（含 scalar / vec4 两个 kernel 的完整指标）
- ✅ 不加 `--cap-add=SYS_ADMIN` 时对照复现 `ERR_NVGPUCTRPERM` —— 确认是这一条起作用
- ❌ 加 `--user $(id -u):$(id -g)` 后 `CapEff: 0000000000000000`，ncu 仍失败 → **必须容器 root**
- ✅ 容器内 `nsys status --environment` 全 `OK`（`Root privilege: enabled`、
  `perf_event_open: OK`、`CPU Profiling (process-tree/system-wide): OK`、LBR `Available`）
- ✅ sqlite 交叉验证：容器内报告有 `COMPOSITE_EVENTS`(322) / `SAMPLING_CALLCHAINS`(2267) /
  `SCHED_EVENTS`(202)，宿主机报告这三张表不存在
- ✅ 容器内跑 conda env `cpp` 的 python：torch 2.9.0+cu128 / triton 3.5.0 / `cuda ok: True`
- ℹ️ `nvidia/cuda:12.4.0-devel-ubuntu22.04` 自带 `ncu 2024.1.0` 和 `nvcc`，**不带 nsys**
