#!/usr/bin/env bash
# run_nsys.sh —— 用 Nsight Systems 抓三种实现的时间线
#
# 用法：  bash profile/run_nsys.sh
#
# 前置：本机只有系统 CUDA 12.4 带 nsys（conda 环境里没有），所以显式用绝对路径。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

NSYS=/usr/local/cuda/bin/nsys
PYTHON=${PYTHON:-/mnt/public/nyt1/docqa/restored_envs/cpp/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

mkdir -p reports

# --- 1) 三种实现放在一起，看相对耗时和 launch 间隙 ---
#   -t cuda,nvtx   只 trace CUDA runtime/driver + NVTX（osrt 会让报告大很多）
#   --cuda-memory-usage=true  记录 cudaMalloc/Free，看显存分配开销
"$NSYS" profile \
    -t cuda,nvtx \
    --cuda-memory-usage=true \
    --force-overwrite true \
    -o reports/all \
    "$PYTHON" profile_target.py --n $((1<<24)) --iters 20

# --- 2) 纯 CUDA 可执行文件，无 Python 干扰，时间线最干净 ---
"$NSYS" profile \
    -t cuda,nvtx \
    --force-overwrite true \
    -o reports/cuda_only \
    ./cuda_mul2 $((1<<24)) 100 256

echo
echo "=== 生成的报告 ==="
ls -lh reports/*.nsys-rep
echo
echo "命令行看结果："
echo "  $NSYS stats --report cuda_gpu_kern_sum reports/all.nsys-rep"
echo "  $NSYS stats --report cuda_api_sum      reports/all.nsys-rep"
echo "  $NSYS stats --report nvtx_sum          reports/all.nsys-rep"
