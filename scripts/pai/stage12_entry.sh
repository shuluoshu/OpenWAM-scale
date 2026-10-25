#!/usr/bin/env bash
# DLC 4 Pods x 8 local GPUs. CPU preparation must finish before training.
set -euo pipefail
mode="${1:-check}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in prepare|check|train) ;; *) echo 'Usage: stage12_entry.sh {prepare|check|train} [--resume [--stage2-resume PATH]]' >&2; exit 2;; esac
: "${OPENWAM_CODE_DIR:?Set the shared CPFS checkout path}"
: "${OPENWAM_PYTHON:?Set Python inside the DLC image or a verified shared environment}"
: "${OPENWAM_EXPECTED_COMMIT:?Pin the training commit}"
export STAGE12_CONFIG="${STAGE12_CONFIG:-configs/goal/stage12_query_history_posttrain_3fps_4x8gpu.yaml}"
cd "$OPENWAM_CODE_DIR"
[[ "$(git rev-parse HEAD)" == "$OPENWAM_EXPECTED_COMMIT" ]] || { echo 'Code commit mismatch' >&2; exit 1; }
git diff --quiet HEAD -- openwam configs/goal configs/model configs/dataloader configs/train.yaml scripts/train_stage12.py scripts/train_goal_epoch.py scripts/train.py scripts/data || { echo 'Training code/config has uncommitted changes' >&2; exit 1; }
export PATH="$(dirname "$OPENWAM_PYTHON"):$PATH"
export PYTHONPATH="$OPENWAM_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [[ "$mode" == train && -n "${STAGE12_PREPARATION_STATUS:-}" ]]; then
    : "${STAGE12_PREPARED_MARKER:?Set shared preparation marker}"
    "$OPENWAM_PYTHON" scripts/pai/preparation_gate.py --wait \
      --status "$STAGE12_PREPARATION_STATUS" --marker "$STAGE12_PREPARED_MARKER"
fi
"$OPENWAM_PYTHON" - "$mode" <<'PY'
import importlib, os, sys
from pathlib import Path
from openwam.train.stage12 import load_pipeline
from openwam.train.goal_curated import load_config
cfg=load_pipeline(os.environ['STAGE12_CONFIG'])
assert (cfg['world_size'],cfg.get('nnodes'),cfg.get('gpus_per_node'))==(32,4,8), 'This DLC entry requires 4 x 8 GPUs'
stage1=load_config(cfg['stage1_config'],source_yaml=cfg['source_yaml'])
assert float(stage1['sample_fps'])==3., 'This entry requires 3 FPS Stage1'
for name in ('torch','torchvision','transformers','diffusers','deepspeed','av','pyarrow','hydra','omegaconf','psutil'):
    mod=importlib.import_module(name)
    print(name, getattr(mod,'__version__','available'), flush=True)
for path in (cfg['stage2_baseline'], stage1['baseline_checkpoint'], '/mnt/data/cpfs/b5big', '/mnt/data/nas/B_zone_nas/jeffry/calibration_results'):
    if not Path(path).exists(): raise FileNotFoundError(path)
if sys.argv[1]=='train':
    import torch
    from openwam.train.stage12_distributed import node_environment
    node_environment(cfg)
    if torch.cuda.device_count()!=8: raise RuntimeError(f'Expected 8 local GPUs, found {torch.cuda.device_count()}')
    for i in range(8):
        p=torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {p.name}, {p.total_memory/2**30:.1f} GiB')
    if not (Path(cfg['output_dir'])/'prepared.json').is_file():
        raise RuntimeError('CPU preparation missing; run prepare before allocating the GPU job')
PY
if [[ "$mode" == prepare ]]; then
    # Derive selection only if absent; never silently regenerate a running job's data.
    source_yaml="$("$OPENWAM_PYTHON" -c 'import os; from openwam.train.stage12 import load_pipeline; print(load_pipeline(os.environ["STAGE12_CONFIG"])["source_yaml"])')"
    if [[ ! -f "$source_yaml" ]]; then
        "$OPENWAM_PYTHON" scripts/data/prepare_stage12_selection.py \
          --original-yaml /mnt/data/cpfs/b5/yeqingda/pretrain/launch/moravla/train_files/pretrain_v0_3_curated_3000h_posttrain_20261019.yaml \
          --output-yaml "$source_yaml"
    fi
fi
exec "$OPENWAM_PYTHON" scripts/train_stage12.py --config "$STAGE12_CONFIG" "--$mode" "$@"
