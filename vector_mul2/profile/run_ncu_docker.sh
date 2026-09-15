#!/usr/bin/env bash
# run_ncu_docker.sh —— 在 Docker 容器里跑 ncu / nsys，绕过宿主机的 ERR_NVGPUCTRPERM
#
# 原理：GPU 性能计数器的限制（驱动的 NVreg_RestrictProfilingToAdminUsers=1）
#       检查的是调用进程有没有 **CAP_SYS_ADMIN**。容器里以 root 运行并加上
#       --cap-add=SYS_ADMIN 就满足了 —— 不用动宿主机驱动，不用重启，不用 sudo 密码。
#       这是 NVIDIA 官方文档给出的容器内 profiling 方案。
#       附带好处：nsys 的 CPU 采样也一并解锁（宿主机上被 perf_event_paranoid=4 挡住）。
#
# 前提：当前用户在 docker 组里（`id` 能看到 docker），且装了 nvidia-container-toolkit。
#
# 用法：  bash profile/run_ncu_docker.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

IMAGE=${IMAGE:-nvidia/cuda:12.4.0-devel-ubuntu22.04}
GPU=${CUDA_VISIBLE_DEVICES:-3}
HOST_NSYS=/usr/local/cuda-12.4/nsight-systems-2023.4.4   # 镜像里没有 nsys，从宿主机挂进去
CONDA_ENV=/mnt/public/nyt1/docqa/restored_envs/cpp       # torch + triton 环境
PY=$CONDA_ENV/bin/python

mkdir -p reports

# 容器内必须是 root：加了 --user 非 root 之后 CapEff 会变成 0，ncu 照样报
# ERR_NVGPUCTRPERM（实测验证过）。所以产物是 root 属主，结尾 chown 回来。
UIDGID="$(id -u):$(id -g)"

# --entrypoint bash 是为了跳过镜像默认 entrypoint 打印的一大段 license banner
dock() {
    docker run --rm --entrypoint bash \
        --gpus "\"device=$GPU\"" \
        --cap-add=SYS_ADMIN \
        -v "$HERE":/work -w /work \
        -v "$HOST_NSYS":/opt/nsys:ro \
        -v /mnt/public:/mnt/public:ro \
        -e TRITON_CACHE_DIR=/tmp/triton_cache \
        "$IMAGE" -c "$1"
}

hr() { printf '%*s\n' 76 '' | tr ' ' '-'; }

hr; echo "0) 确认容器里权限到位（CapEff 非 0 + nsys 环境检查全 OK）"; hr
dock 'grep CapEff /proc/self/status; /opt/nsys/bin/nsys status --environment | head -12'

hr; echo "1) ncu：scalar vs vec4 全量报告（两个 kernel 在同一份报告里，给 GUI 做 Baseline 对比）"; hr
# --launch-skip 10：main.cu 的 warmup 会先发 10 次 scalar，必须跳过，
#                   否则 --launch-count 用完了也抓不到 vec4。
# iters 传 1，于是第 10、11 次 launch 正好是 bench_scalar 和 bench_vec4 各一次。
dock "ncu --set full --launch-skip 10 --launch-count 2 \
        -o /work/reports/ncu_cuda --force-overwrite \
        ./cuda_mul2 16777216 1 256 >/dev/null 2>&1
      chown $UIDGID /work/reports/ncu_cuda.ncu-rep
      echo '报告里的 kernel：'
      ncu --import /work/reports/ncu_cuda.ncu-rep --page details --csv 2>/dev/null \
        | awk -F'\",\"' '{print \$5}' | sort -u | tail -n +2"

hr; echo "2) ncu：Speed Of Light 对比"; hr
dock "ncu --import /work/reports/ncu_cuda.ncu-rep --page details 2>/dev/null \
      | grep -E 'cuda_mul2_kernel|DRAM Throughput|Compute \(SM\) Throughput|Duration|L2 Hit Rate|Achieved Occupancy'"

hr; echo "3) ncu：四个 kernel 的关键指标（torch / triton / cuda scalar / cuda vec4）"; hr
METRICS='gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed_op_global_ld.sum,smsp__inst_executed_op_global_st.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread'
for K in 'regex:vectorized_elementwise' 'regex:triton_mul2_kernel' 'regex:cuda_mul2_kernel$' 'regex:vec4'; do
    echo "--- kernel filter: $K"
    dock "ncu --metrics '$METRICS' -k '$K' --launch-count 1 \
            $PY profile_target.py --n 16777216 --iters 3 2>&1 \
          | grep -E '^[[:space:]]+(dram__|gpu__|sm__|smsp__|launch__)'"
done

hr; echo "4) nsys：带 CPU 采样（宿主机上这部分会被 paranoid=4 禁用）"; hr
dock "/opt/nsys/bin/nsys profile -t cuda,nvtx,osrt --sample=cpu --cpuctxsw=process-tree \
        -o /work/reports/in_docker --force-overwrite true \
        ./cuda_mul2 16777216 50 256 >/dev/null 2>&1
      chown $UIDGID /work/reports/in_docker.nsys-rep
      ls -l /work/reports/in_docker.nsys-rep"

hr; echo "完成。产物："; hr
ls -l reports/
cat <<EOF

下载到本地用 GUI 看：
  scp <user>@<server>:$HERE/reports/ncu_cuda.ncu-rep .     # Nsight Compute
  scp <user>@<server>:$HERE/reports/in_docker.nsys-rep .   # Nsight Systems
EOF
