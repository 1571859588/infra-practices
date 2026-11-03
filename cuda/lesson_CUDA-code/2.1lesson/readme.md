## 1. `g++ -std=c++20 hello.cpp` 没输出 —— 这是正常的

编译成功本来就**不打印任何东西**。而且你没给 `-o`，产物默认叫 `a.out`，编译完也不会自动运行。要看到结果：

```bash
g++ -std=c++20 hello.cpp -o hello_cpp && ./hello_cpp
```

顺带提醒：`hello.cpp` 里是 `printf("Hello world!");`，**末尾没有 `\n`**，所以即使跑起来了，输出会跟你的 zsh 提示符挤在同一行，看起来像"没输出"。

## 2. `nvcc error : 'g++' died due to signal 2` —— 是你 Ctrl+C 造成的

signal 2 = `SIGINT`。nvcc 是编译器**驱动**，它内部会另起 `g++` 进程去编主机端代码，你按 Ctrl+C 时 nvcc 收到信号、正在跑的子进程 g++ 也一起被杀了，于是报这条。**不是编译错误，是"被打断"**。说明当时确实卡住了。

## 3. 卡在哪：实测结果

我复现了几次：

| 命令 | 耗时 |
|---|---|
| `nvcc -arch=sm_80 hello.cu -o hello2`（默认 TMPDIR） | **2.0 s** |
| `g++ -std=c++20 hello.cpp -o ...` | 0.44 s |
| `TMPDIR=/dev/shm nvcc -arch=sm_80 hello.cu -o hello` | **28 分 20 秒**，CPU 占用 0% |

最后一行是关键：CPU 0% 说明**纯 I/O 等待，不是算力问题**。原因这台机器上 `/dev/shm` 已经被别人占了 **199G / 252G（79%）**，把 nvcc 临时目录放上去会跟别人抢共享内存，直接卡死。

另外两个不利因素：
- **根分区 `/` 100% 满**，只剩 3.8 G（`/tmp` 就在上面，nvcc 默认临时目录在 `/tmp`）
- 系统负载 `load average: 6.80`，不低

所以你那次多半是共享内存/网络盘抖动 + 冷启动（首次 nvcc 要从网络盘加载 `cicc`/`ptxas`/`fatbinary` 一串工具）。

## 建议用法

```bash
# 默认临时目录，别改
nvcc -arch=sm_80 hello.cu -o hello && ./hello
```

- **不要设 `TMPDIR=/dev/shm`**，实测会把 2 秒变 28 分钟。
- 想再保险一点，可以让中间产物和输出都走本地盘，编完拷回来：
  ```bash
  TMPDIR=/tmp nvcc -arch=sm_80 hello.cu -o /tmp/hello && cp /tmp/hello ./
  ```
- 卡住时想看它停在哪一步：加 `-v`（`nvcc -v -arch=sm_80 hello.cu -o hello`），会打印出它依次调用了哪些子工具。

## 补充

`hello.cu` 里其实**没有任何 CUDA 代码**（就一句 `printf`），用 `gcc` 编完全一样，nvcc 只是白跑一遍完整 CUDA 工具链（cicc → ptxas → fatbinary），所以天生比 g++ 慢几倍。目录里 `hello` 已经编好了，直接 `./hello` 就行，输出 `Hello World!`。

（我测试时产生的临时文件已清理干净。）