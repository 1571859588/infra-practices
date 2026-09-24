#!/usr/bin/env bash
# 一键复现 README.md §4 里的四组实测数据。
#
#   bash run.sh                 # 用默认 python（conda env cpp）和默认卡 7
#   PYTHON=/path/to/python GPU=3 bash run.sh
#
# 产物全部落在 ./traces/ 下：每组配置一个 .json（Chrome trace）
# 和一个 .txt（key_averages 汇总表）。
set -euo pipefail

cd "$(dirname "$0")"

PYTHON=${PYTHON:-/mnt/public/nyt1/docqa/restored_envs/cpp/bin/python}
GPU=${GPU:-3}
export CUDA_VISIBLE_DEVICES=$GPU

echo "python : $PYTHON"
echo "GPU    : $CUDA_VISIBLE_DEVICES ($($PYTHON -c 'import torch;print(torch.cuda.get_device_name(0))'))"

run() {
  echo
  echo "=== 01_matmul_add.py ${*:-（默认：64x64 bf16，记录 20 步）} ==="
  "$PYTHON" 01_matmul_add.py "$@" >/dev/null
}

run                                              # 64   bf16 eager   20 步（视频默认）
run --size 1024 --steps 10                       # 1024 bf16 eager   10 步
run --size 1024 --dtype fp32 --steps 10          # 1024 fp32 eager   10 步
run --size 1024 --steps 10 --mode COMPILE        # 1024 bf16 compile 10 步

echo
echo "产物："
ls -1 traces/
