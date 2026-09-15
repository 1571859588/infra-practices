#!/usr/bin/env bash
# run_all.sh —— 一键跑通全部内容
#
#   bash run_all.sh            跑 benchmark（不含 profile）
#   bash run_all.sh --profile  额外跑 nsys / ncu / compute-sanitizer
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# conda env `cpp` 自带 torch 2.9.0+cu128 + triton 3.5.0，用绝对路径避免依赖 shell 是否激活
PYTHON=${PYTHON:-/mnt/public/nyt1/docqa/restored_envs/cpp/bin/python}
# 选一张空闲卡。用 nvidia-smi 看 memory.used 最小的那张。
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

hr() { printf '%*s\n' 78 '' | tr ' ' '='; }

hr; echo "0. 环境"; hr
echo "python : $PYTHON"
echo "GPU    : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total \
           --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES"

hr; echo "1. 编译 CUDA（libmul2.so + cuda_mul2）"; hr
make

hr; echo "2. PyTorch 实现"; hr
"$PYTHON" torch_mul2.py

hr; echo "3. Triton 实现"; hr
"$PYTHON" triton_mul2.py

hr; echo "4. 原生 CUDA 实现（独立可执行文件）"; hr
./cuda_mul2

hr; echo "5. 三方统一对比"; hr
"$PYTHON" bench_all.py

hr; echo "6. 规模扫描"; hr
"$PYTHON" bench_all.py --sweep

hr; echo "6b. CPU 端 launch 开销（解释小规模下为什么 Triton 最慢）"; hr
"$PYTHON" launch_overhead.py

if [[ "${1:-}" == "--profile" ]]; then
    # profile 这一段里好几个命令预期会返回非 0（ncu 权限失败、sanitizer 抓到 bug），
    # 所以关掉 -e，改为逐条容错，保证整段能跑完。
    set +e

    hr; echo "7. nsys 时间线"; hr
    bash profile/run_nsys.sh

    hr; echo "8. nsys 结果汇总"; hr
    for r in cuda_gpu_kern_sum cuda_api_sum nvtx_sum; do
        /usr/local/cuda/bin/nsys stats --report "$r" reports/all.nsys-rep 2>/dev/null \
            | grep -vE 'NOTICE|assumed|Consider|Processing|Generating'
    done
    /usr/local/cuda/bin/nsys stats --report cuda_gpu_mem_time_sum \
        reports/cuda_only.nsys-rep 2>/dev/null \
        | grep -vE 'NOTICE|assumed|Consider|Processing|Generating'

    hr; echo "9. ncu"; hr
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        echo "检测到可用的 docker，走容器方案绕过 ERR_NVGPUCTRPERM（见 README §5.2）"
        bash profile/run_ncu_docker.sh
    else
        echo "没有可用的 docker，直接在宿主机跑（预期失败：ERR_NVGPUCTRPERM）"
        bash profile/run_ncu.sh
    fi

    hr; echo "10. 不需要 profiling 权限的替代分析"; hr
    make occupancy
    make sass
    echo "--- compute-sanitizer: 正常版本（预期 0 errors）---"
    /usr/local/cuda/bin/compute-sanitizer --tool memcheck ./cuda_mul2 $((1<<20)) 5 256
    echo "--- compute-sanitizer: 故意越界的版本（预期报错）---"
    /usr/local/cuda/bin/compute-sanitizer --tool memcheck ./cuda_mul2 1000 1 256 1 2>&1 \
        | grep -E 'Invalid __global__|is out of bounds|cuda_mul2\.cu:|ERROR SUMMARY' \
        | head -8

    set -e
fi

hr; echo "完成。"; hr
