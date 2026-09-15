#!/usr/bin/env bash
# run_all.sh —— 按顺序跑完三个 Triton 练习
#
#   bash run_all.sh
#   PYTHON=/path/to/python CUDA_VISIBLE_DEVICES=0 bash run_all.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# conda env `cpp` 自带 torch 2.9.0+cu128 + triton 3.5.0，用绝对路径避免依赖是否激活
PYTHON=${PYTHON:-/mnt/public/nyt1/docqa/restored_envs/cpp/bin/python}
# 挑一张空闲卡：nvidia-smi 看 memory.used 最小的那张
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

hr() { printf '%*s\n' 78 '' | tr ' ' '='; }

hr; echo "0. 环境"; hr
echo "python : $PYTHON"
echo "GPU    : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
"$PYTHON" -c "import torch, triton; print(f'torch {torch.__version__} / triton {triton.__version__} / cuda ok: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=index,name,memory.used,memory.total \
           --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES"

for f in 01_vector_add.py 02_fused_softmax.py 03_matmul.py; do
    hr; echo "$f"; hr
    "$PYTHON" "$f"
done

hr; echo "完成。"; hr
