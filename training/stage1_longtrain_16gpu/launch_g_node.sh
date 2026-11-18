#!/usr/bin/env bash
set -euo pipefail

print_only=0
if [[ ${1:-} == --print-command ]]; then
  print_only=1
  shift
fi
if [[ $# -lt 4 || ! $1 =~ ^[01]$ ]]; then
  echo 'Usage: launch_g_node.sh [--print-command] NODE_RANK MASTER_ADDR CONFIG OUTPUT [native trainer options]' >&2
  exit 2
fi
node_rank=$1
master_addr=$2
config=$3
output=$4
shift 4
bundle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
train_python=${TRAIN_PYTHON:-/tmp/openwam-runtime-20261031/env/bin/python}
master_port=${MASTER_PORT:-30681}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OPENWAM_MECKA_DECODE_MODE=indexed OPENWAM_VIDEO_INDEX_MODE=read
export OPENWAM_VIDEO_DECODE_THREADS=1
export OPENWAM_VIDEO_INDEX_CACHE=${OPENWAM_VIDEO_INDEX_CACHE:-/mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/stage1_dlc_efficiency_20261101/video_index}

command=("$train_python" -m torch.distributed.run --nnodes 2 --node_rank "$node_rank"
  --nproc_per_node 8 --master_addr "$master_addr" --master_port "$master_port"
  "$bundle_dir/entrypoint.py" train --arch G --config "$config"
  --local-batch 80 --accumulation 2 --output "$output" "$@")
if (( print_only )); then
  printf '%q ' "${command[@]}"
  printf '\n'
else
  exec "${command[@]}"
fi
