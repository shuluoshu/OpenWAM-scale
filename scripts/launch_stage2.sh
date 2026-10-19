#!/usr/bin/env bash
set -euo pipefail
preset="${1:-}"
mode="${2:-}"
if [[ "$preset" != sample && "$preset" != 3000h ]] || [[ "$mode" != --preflight && "$mode" != --train ]]; then
  echo 'Usage: scripts/launch_stage2.sh sample|3000h --preflight|--train [Hydra overrides...]' >&2
  exit 2
fi
shift 2
: "${STAGE1_GOAL_CHECKPOINT:?Set STAGE1_GOAL_CHECKPOINT to the Stage1 patch bundle}"
: "${STAGE2_BASELINE_CHECKPOINT:?Set STAGE2_BASELINE_CHECKPOINT to baseline weights/run directory}"
: "${STAGE2_OUTPUT_DIR:?Set STAGE2_OUTPUT_DIR to the new run output directory}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$preset" == sample ]]; then
  default_source="$repo_root/../../pretrain/launch/moravla/train_files/pretrain_v0_3_curated_smoke.yaml"
else
  default_source="$repo_root/../../pretrain/launch/moravla/train_files/pretrain_v0_3_curated_3000h.yaml"
fi
export STAGE2_SOURCE_YAML="${STAGE2_SOURCE_YAML:-$default_source}"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"
python_bin="${PYTHON:-python}"
common=(--config-name "goal/stage2_joint_${preset}" "goal.checkpoint=${STAGE1_GOAL_CHECKPOINT}" "training.finetune_ckpt_path=${STAGE2_BASELINE_CHECKPOINT}" "dataloader.source_yaml=${STAGE2_SOURCE_YAML}" "project.output_dir=${STAGE2_OUTPUT_DIR}")
# Both paths validate assets/source contracts before constructing any full model.
"$python_bin" scripts/train.py "${common[@]}" "$@" goal.preflight=true
if [[ "$mode" == --preflight ]]; then exit 0; fi
: "${STAGE2_STATS_REGISTRY:?Set STAGE2_STATS_REGISTRY to the immutable native normalization registry}"
: "${NPROC_PER_NODE:=1}"
if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then echo 'NPROC_PER_NODE must be a positive integer' >&2; exit 2; fi
exec "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" scripts/train.py "${common[@]}" "$@" goal.preflight=false
