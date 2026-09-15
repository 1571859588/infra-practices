#!/usr/bin/env bash
# run_ncu.sh —— 用 Nsight Compute 做 kernel 级微观分析
#
# ⚠️ 本机当前会失败：ERR_NVGPUCTRPERM
#    NVIDIA 驱动默认把 GPU 性能计数器限制为 root 可读。
#    需要管理员执行（需重启）：
#      echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
#        | sudo tee /etc/modprobe.d/nvidia-profiling.conf
#      sudo update-initramfs -u && sudo reboot
#    权限放开后本脚本无需任何修改即可工作。
#
# 用法：  bash profile/run_ncu.sh
set -uo pipefail          # 注意没有 -e：权限失败时我们要继续跑完并给出提示

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

NCU=/usr/local/cuda/bin/ncu
PYTHON=${PYTHON:-/mnt/public/nyt1/docqa/restored_envs/cpp/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

mkdir -p reports

echo "### 1) 纯 CUDA 可执行文件：对比 scalar vs vec4"
# --launch-count 1  每个 kernel 只测第一次，避免 replay 100 次
# --set full        采全部 section（慢但信息最全）；快速看用 --set basic
"$NCU" --set full \
       --kernel-name-base function \
       --launch-count 1 \
       -o reports/cuda_only --force-overwrite \
       ./cuda_mul2 $((1<<22)) 5 256

echo
echo "### 2) Python 侧：只测 Triton kernel（用 NVTX 精确圈定范围）"
# 对 Python 程序一定要用 -k / --nvtx-include 收窄范围，
# 否则 torch 初始化时的几十个 kernel 都会被 replay，能跑几十分钟。
"$NCU" --set basic \
       -k triton_mul2_kernel \
       --launch-count 1 \
       -o reports/triton --force-overwrite \
       "$PYTHON" profile_target.py --n $((1<<22)) --iters 3 --only triton

echo
echo "### 3) 三种实现放一起，只取关键访存指标（最快的一种用法）"
"$NCU" --metrics \
gpu__time_duration.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__occupancy_limit_registers,\
sm__warps_active.avg.pct_of_peak_sustained_active \
       --launch-count 1 \
       "$PYTHON" profile_target.py --n $((1<<22)) --iters 3

rc=$?
echo
if [ $rc -ne 0 ] || ! ls reports/*.ncu-rep >/dev/null 2>&1; then
    cat <<'EOF'
------------------------------------------------------------------
ncu 未能产出数据。若看到 ERR_NVGPUCTRPERM，就是驱动权限问题（见脚本顶部）。
在权限放开前，可用这些替代手段（都不需要性能计数器）：
  nsys profile ...                     # kernel 时间线、耗时、launch 开销
  make occupancy                       # nvcc -Xptxas=-v 看寄存器 / shared mem
  make sass                            # cuobjdump 确认访存指令宽度
  compute-sanitizer --tool memcheck    # 正确性
------------------------------------------------------------------
EOF
fi
