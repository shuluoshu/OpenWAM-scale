# Stage 1: joint goal-image / TCP denoising

`scripts/train_goal_curated.py` trains the terminal-image latent and four ordered
TCP coordinates `(u_left, v_left, u_right, v_right)` in shared Wan blocks. The
image latent and TCP patch receive independent Gaussian noise at a shared noise
level and are jointly trained with flow matching. Clean observation and language
are conditions. TCP uses the patch output, not a separate MLP/heatmap head.
The VAE and text encoder are frozen; this entry trains the shared DiT. Stage 2
robot action sequences and future-video prediction are outside this entry.

## Data and sampling

| Preset | Source YAML | Default steps |
| --- | --- | --- |
| `configs/goal/stage1_joint_sample.yaml` | `pretrain_v0_3_curated_smoke.yaml` | 5,000 |
| `configs/goal/stage1_joint_3000h.yaml` | `pretrain_v0_3_curated_3000h.yaml` | 100,000 |

Both inherit `stage1_joint_base.yaml`. Source YAML paths are resolved relative to
the repository root; the default location is
`../../pretrain/launch/moravla/train_files/`. Use an absolute `--source-yaml`
when checking out the code elsewhere. `--data-root` overrides the YAML's dataset
root. No experiment output directory or generated manifest is required by this
entry. The YAML and source datasets are external inputs, not bundled files.

Sampling preserves positive selected `repo_weights`, then selects an episode
proportional to its selected frame count. Observation anchors stay within the
YAML's `[start, end)` ranges. For episodes with usable subtask intervals
intersecting these ranges, task and subtask examples each have probability 0.5;
episodes without usable subtasks provide task examples only. This is a sampling
mixture, not two simultaneous losses for every observation.

Task text comes from `meta/tasks.parquet` and per-frame `task_index`. Subtasks
use `meta/sub_tasks.parquet` plus per-frame `subtask_index` and optional
`subtask_unannotated`. Repeated disjoint occurrences are separate intervals.
An anchor must precede its target. The target is the original contiguous
instruction interval's `end_exclusive - 1`, even when it lies beyond the curated
observation range. A single-task episode therefore uses the episode's last frame.
These endpoints are weak labels; they are not certified successful completion.

Missing annotations and explicitly unannotated frames allow task-only sampling.
Conflicting annotation mappings/intervals fail with source information instead
of silently changing the supervision. No usable interval in the selected range
is recorded as a distinct reason in provenance. The current YAMLs cover 12
source families; the 3000h selection has 5,750 repositories and 190,458 ranges.
Unsupported families fail explicitly.

Images and geometry are read at the same original source frame. The current
image transform is resize to 320×256 without random cropping. Pixel-center
normalized UV stays FP32 while images use BF16. Missing/invalid calibration,
TCP or an out-of-frame projection masks that point's loss; image supervision
remains. Numerical geometry candidates retain `pending_review` provenance.

## Commands

Use the project's Python environment, including PyAV, PyArrow, OmegaConf,
PyTorch, safetensors, Pillow and the Wan dependencies.

Inspect real model-input images and TCP labels on CPU without loading weights:

```bash
python scripts/train_goal_curated.py --preflight \
  --config configs/goal/stage1_joint_sample.yaml \
  --output-dir outputs/stage1_sample_preflight
```

Exercise tiny real Wan blocks, backward/optimizer, checkpoint and history on
synthetic inputs, without source data or pretrained weights:

```bash
python scripts/train_goal_curated.py --smoke --steps 500 \
  --output-dir outputs/stage1_synthetic_smoke
```

Set `OPENWAM_BASELINE_CHECKPOINT` to an OpenWAM checkpoint containing compatible
`video_backbone.dit.*`, `video_backbone.vae.*` and
`video_backbone.text_encoder.*` tensors. The default is
`models/openwam_alpha_v0.1/checkpoint.safetensors`. Model shapes must match the
base YAML. The tokenizer defaults to the checkpoint parent's
`tokenizer/google/umt5-xxl` directory; use `--tokenizer-path` for another local
directory. Only `--train` starts real training:

```bash
python scripts/train_goal_curated.py --train --device cuda \
  --config configs/goal/stage1_joint_sample.yaml \
  --output-dir outputs/stage1_sample_run

python scripts/train_goal_curated.py --train --device cuda \
  --config configs/goal/stage1_joint_3000h.yaml \
  --output-dir outputs/stage1_3000h_run
```

`--steps` overrides the preset budget. Use a new output directory for each run.
The implementation supports one process and batch size 1; distributed training
and optimizer resume are not implemented. Full-model GPU throughput/memory for
the 3000h run has not been benchmarked. Its default step budget is a preset,
not a claim of one dataset epoch or completed training.

## Memory and diagnostics

Samples are decoded on demand. Episode metadata and rotating training assets
are bounded by `dataset.max_cached_episodes` (8 by default). Frozen text contexts
use a bounded CPU LRU (`text_cache_size: 32`); the text encoder defaults to CPU.
Only the current training sample or evaluation case moves to the model device.
Mecka video decoding uses the existing sequential alignment path, which can be
slow for late frames in long files. The metadata selection index stays in RAM.

Evaluation uses fixed training examples: at most one task and one subtask per
family, searched among the first eight selected episodes per family. This
bounded set diagnoses training and does not measure all sources or held-out
generalization. Available language levels in this set are visible in the report.

At step 0, every 500 steps and the final step:

- `eval_step_XXXXXX/` preserves generated goal images, TCP predictions and metrics.
- `timeline.html` appends the new step beside prior results and exact input GT.
- Reports include image MAE in 0–255 RGB units and valid TCP mean Euclidean
  pixel error, grouped by dataset family and task/subtask. Missing TCP errors
  are unavailable, never counted as zero.
- `inputs/manifest.json` and NPZ files retain source frames, masks and actual
  BF16 model-input bits. `progress.jsonl` records each sampled source and loss.
- After each positive-step evaluation, `patch_goal.safetensors` stores the
  latest Stage 1 expert with dependency identities; prior evaluation directories
  remain intact. Step 0 produces reports only.

The previous fixed-manifest diagnostic entry `scripts/train_goal_patch.py`
remains available with explicit `--manifest` and optional `--mixed-manifest`.
The streaming entry above is the supported path for the two curated YAMLs.
