#!/usr/bin/env bash
# run_all.sh —— 按顺序跑完三个 Triton 练习的 bench.py
#
#   bash run_all.sh              # 完整跑（含参数扫描，约几分钟）
#   bash run_all.sh --quick      # 跳过参数扫描，只跑正确性 + 变体对比
#   PYTHON=/path/to/python CUDA_VISIBLE_DEVICES=0 bash run_all.sh
#
# 每个练习是一个目录，里面 v*.py 是各个变体，bench.py 把它们串起来对比。
# 单独跑某一个变体：cd 01_vector_add && python v0_naive.py
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# conda env `cpp` 自带 torch 2.9.0+cu128 + triton 3.5.0，用绝对路径避免依赖是否激活
PYTHON=${PYTHON:-/mnt/public/nyt1/docqa/restored_envs/cpp/bin/python}
# 挑一张空闲卡：nvidia-smi 看 memory.used 最小的那张。
# 默认 7，和 cuda/common.mk 的 GPU?=7 保持一致。
# ⚠️ 卡被别人占着的话测出来的带宽会低一半以上（实测 GPU3 被占时
#    向量加只有 607 GB/s / 39%，空闲时是 1300+ GB/s / 84%），
#    而且 autotune 还可能直接 CUDA OOM。
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}

hr() { printf '%*s\n' 78 '' | tr ' ' '='; }

hr; echo "0. 环境"; hr
echo "python : $PYTHON"
echo "GPU    : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
"$PYTHON" -c "import torch, triton; print(f'torch {torch.__version__} / triton {triton.__version__} / cuda ok: {torch.cuda.is_available()}')"
# 只是给人看的信息，nvidia-smi 不可用时不该中断整个脚本（set -e）
nvidia-smi --query-gpu=index,name,memory.used,memory.total \
           --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES" \
    || echo "（nvidia-smi 不可用，跳过；测性能前请自行确认卡是空的）"

# 透传 --quick 之类的参数给每个 bench.py
for d in 01_vector_add 02_fused_softmax 03_matmul; do
    hr; echo "$d/bench.py $*"; hr
    ( cd "$d" && "$PYTHON" bench.py "$@" )
done

hr; echo "完成。"; hr
