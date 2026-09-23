# OpenWAM-α Canonical EEF Data Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a reproducible two-source V0_1 smoke fine-tune, real RoboDojo/RobotWin offline open-loop inference, and forward-input visualization around one strict canonical EEF20 LeRobot-v3 reader.

**Architecture:** A thin `CanonicalEEFLerobotDataset` subclasses the existing `LeRobotV3Reader`, validates already-materialized absolute EEF20 fields, normalizes action/state from a provenance-bound external artifact, and lets the base scatter them into OpenWAM's 80D space. Small YAML profiles select exact buckets and bounded episodes/windows; standalone scripts reuse the same reader for stats, open-loop generation, and offline HTML/PNG/JSON/NPZ evidence.

**Tech Stack:** Python 3, PyTorch, NumPy, pandas, PyArrow, Pillow, Hydra/OmegaConf, pytest, OpenWAM checkpoint loader and inference engine.

## Global Constraints

- Training root is `/mnt/data/cpfs/b5big/pretrain_data_release/pretrain_datasetsV0_1`; do not scan or train the full corpus.
- Smoke training uses only the confirmed InternData-A1 Genie-1 and AgiBotWorld A2D gripper buckets, at most two episodes and eight windows per source.
- Raw contract is absolute EEF20; normalize in raw space, then map `0:10 -> 0:10` and `10:20 -> 34:44` in the 80D OpenWAM head.
- A sample has 33 raw steps, 32 action targets, frames `0,4,...,32`, nine 384×320 multiview canvases, and a non-empty prompt.
- MoraVLA reference facts come from `launch/qwena1_nogen/pretrain.sh` and its `train_files/pretrain.yaml`: 36D, delta, head-camera frame, optional wrist frame, offsets `[-15,0,15]`, launch FPS 5, external stats.
- OpenWAM does not consume MoraVLA's 36D delta or separate 96D layouts; missing EEF, schema, or matching stats is fatal.
- Open-loop uses `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model`; no closed-loop simulator is started.
- RobotWin metadata/EEF comes from `/mnt/data/cpfs/b5/post_train_data/robotwin_traj_fullfield`; its video symlink points to the supplied `/mnt/data/cpfs/b5/post_train_data/robotwin_traj`.
- Never modify source data below `/mnt/data/cpfs`; generated stats and reports stay in the repository output/assets paths.
- Preserve user files `burn_gpu.py`, `nohup.out`, and `openwam_codebase_guide.html`.
- Continuously update `docs/2026-09-23-openwam-alpha-data-adaptation-session.md` with observable commands, outcomes, and conversation decisions.

---

### Task 1: Canonical EEF20 reader and registry

**Files:**
- Create: `openwam/dataloader/canonical_eef_lerobot.py`
- Modify: `openwam/dataloader/registry.py`
- Test: `tests/dataloader/test_canonical_eef_lerobot.py`

**Interfaces:**
- Consumes: `LeRobotV3Reader`, `apply_normalization`, `load_stats_file`, `load_stats_metadata`, and base-class unified mapping.
- Produces: `CanonicalEEFLerobotDataset.from_config(config, split)`, `gather_unified_eef(array)`, `denormalize_eef(array, stream)`, and optional `sample['trace']`.

- [ ] **Step 1: Write fixture-backed failing contract tests**

Create a two-episode LeRobot-v3 fixture with `observation.states.eef`, `actions.eef`, three video keys, task text, a stats artifact, and monkeypatched frame decode. Assert strict EEF20 schema, selected episodes, `max_windows`, prompt lookup, nine frames, `(32,80)` action/mask, `(1,80)` proprio/mask, exact active destinations, and opt-in trace fields. Add failure tests for missing EEF, invalid rot6d, non-finite values, and mismatched provenance.

- [ ] **Step 2: Verify the tests fail before implementation**

Run: `pytest -q tests/dataloader/test_canonical_eef_lerobot.py`

Expected: collection fails because `openwam.dataloader.canonical_eef_lerobot` does not exist.

- [ ] **Step 3: Implement the strict reader**

Implement class constants and hooks with this public contract:

```python
class CanonicalEEFLerobotDataset(LeRobotV3Reader):
    ACTION_DIM = 20
    NEEDED_COLS = ("observation.states.eef", "actions.eef", "task_index")

    def __init__(self, dataset_dir: str, *, state_key="observation.states.eef",
                 action_key="actions.eef", head_camera="observation.images.cam_high",
                 left_wrist_camera="observation.images.cam_left_wrist",
                 right_wrist_camera="observation.images.cam_right_wrist",
                 normalization_stats_path: str, episode_indices=None,
                 max_episodes=2, max_windows=8, return_trace=False, **kwargs): ...

    def gather_unified_eef(self, array: np.ndarray) -> np.ndarray: ...
    def denormalize_eef(self, array: np.ndarray, stream: str = "action") -> np.ndarray: ...
```

Validate `info.json` fields and widths before `super().__init__`; bind configurable camera names and parquet columns on the instance; filter `_eps_df` deterministically in `_filter_episodes`; cap `_n_total` after initialization; load separate `eef` and `eef_state` blocks and reject metadata whose `dataset_dir`, keys, episode list, schema digest, or width differs. `_action_20d` and `_proprio_20d` validate finite EEF20/rot6d then call `apply_normalization`. `return_trace` adds source/window indices and raw/normalized/unified arrays without altering ordinary training samples.

- [ ] **Step 4: Register and verify**

Add the import and `register_dataset("canonical_eef_lerobot")` call in `openwam/dataloader/registry.py`.

Run: `pytest -q tests/dataloader/test_canonical_eef_lerobot.py tests/dataloader/test_mixture_from_config.py`

Expected: all selected tests pass.

- [ ] **Step 5: Commit reader slice**

```bash
git add openwam/dataloader/canonical_eef_lerobot.py openwam/dataloader/registry.py tests/dataloader/test_canonical_eef_lerobot.py
git commit -m "feat: add canonical EEF LeRobot reader"
```

### Task 2: Bounded stats preparation

**Files:**
- Create: `scripts/data/prepare_openwam_eef_smoke.py`
- Test: `tests/test_prepare_openwam_eef_smoke.py`

**Interfaces:**
- Consumes: one explicit bucket path, state/action keys, selected episode indices, and seed.
- Produces: `.npy` payload `{eef, eef_state, metadata}` accepted by `CanonicalEEFLerobotDataset`.

- [ ] **Step 1: Write failing stats tests**

Test exact selection from `meta/episodes`, shard-local ranges, separate state/action aggregates, `count/mean/std/min/max/q01/q99`, rot6d identity pinning, deterministic schema SHA256, output path creation, and refusal of an empty episode selection.

- [ ] **Step 2: Verify red state**

Run: `pytest -q tests/test_prepare_openwam_eef_smoke.py`

Expected: import failure for the absent script module.

- [ ] **Step 3: Implement the bounded scanner and CLI**

Expose:

```python
def prepare_stats(bucket: Path, output: Path, *, episode_indices: list[int],
                  state_key: str, action_key: str, seed: int = 42) -> dict: ...
```

Read only shards/ranges addressed by selected episodes, verify physical episode indices, aggregate float64 then serialize float32 vectors, pin the canonical rot6d dimensions to identity, and include absolute bucket path, selected IDs, field keys, row counts, seed, schema digest, and generation timestamp. CLI accepts repeatable `--episode-index` and never writes under the bucket.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_prepare_openwam_eef_smoke.py tests/dataloader/test_canonical_eef_lerobot.py`

Expected: all pass.

- [ ] **Step 5: Commit stats slice**

```bash
git add scripts/data/prepare_openwam_eef_smoke.py tests/test_prepare_openwam_eef_smoke.py
git commit -m "feat: prepare bounded canonical EEF stats"
```

### Task 3: Dataset profiles and smoke-training YAML

**Files:**
- Create: `configs/dataloader/canonical_eef_interndata_smoke.yaml`
- Create: `configs/dataloader/canonical_eef_agibotworld_smoke.yaml`
- Create: `configs/dataloader/canonical_eef_robodojo_openloop.yaml`
- Create: `configs/dataloader/canonical_eef_robotwin_openloop.yaml`
- Create: `configs/openwam_alpha_v0_1_smoke.yaml`
- Test: `tests/test_openwam_alpha_data_configs.py`

**Interfaces:**
- Consumes: generated artifacts under `assets/openwam_alpha_data_examples/stats/`.
- Produces: Hydra-loadable two-source training config and four standalone reader profiles.

- [ ] **Step 1: Write config contract tests**

Compose `openwam_alpha_v0_1_smoke` through Hydra and assert foundation warm start, dual-system 80D model, two enabled canonical sources, two episodes/eight windows each, batch 1, two steps, bf16, gradient checkpointing, workers 0, saving disabled, seed 42, and both loss weights 1. Validate all profile paths and mappings.

- [ ] **Step 2: Verify red state**

Run: `pytest -q tests/test_openwam_alpha_data_configs.py`

Expected: missing-config failures.

- [ ] **Step 3: Add exact YAMLs**

The shared reader values are:

```yaml
type: canonical_eef_lerobot
num_frames: 33
video_stride: 4
window_stride: 1
height: 384
width: 320
multiview: true
normalize_mode: quantile
unify_action: true
unify_action_map: ["0-9", "34-43"]
episode_indices: [0, 1]
max_episodes: 2
max_windows: 8
return_trace: false
```

Use exact confirmed bucket/checkpoint paths. The top-level training YAML uses `defaults: [model: dual_system, _self_]`, a list-form `dataloader.type: mixture`, `max_steps: 2`, `num_epochs: null`, and `save_steps: 0`.

- [ ] **Step 4: Verify config composition**

Run: `pytest -q tests/test_openwam_alpha_data_configs.py`

Expected: all pass and printed composed config contains no unresolved interpolation.

- [ ] **Step 5: Commit configuration slice**

```bash
git add configs/dataloader/canonical_eef_*.yaml configs/openwam_alpha_v0_1_smoke.yaml tests/test_openwam_alpha_data_configs.py
git commit -m "feat: add OpenWAM alpha smoke profiles"
```

### Task 4: Offline open-loop inference

**Files:**
- Create: `scripts/open_loop_inference.py`
- Test: `tests/test_open_loop_inference.py`

**Interfaces:**
- Consumes: profile YAML, checkpoint directory, output directory, sample index, denoise steps, device, seed.
- Produces: `prediction.npz` and `manifest.json`; returns nonzero on any contract/model failure.

- [ ] **Step 1: Write failing orchestration tests**

Use fake reader/model/engine objects to assert the script forwards prompt, first frame, normalized `(1,80)` proprio, action horizon 33, video horizon 9, two denoise steps, `decode_video=false`, and 80D active mask. Assert saved arrays include normalized prediction `(32,80)`, gathered normalized EEF `(32,20)`, physical EEF `(32,20)`, GT equivalents, and masks.

- [ ] **Step 2: Verify red state**

Run: `pytest -q tests/test_open_loop_inference.py`

Expected: import failure for the absent script.

- [ ] **Step 3: Implement loader and runner**

Expose:

```python
def run_open_loop(profile_path: Path, checkpoint_dir: Path, output_dir: Path,
                  *, sample_index=0, device="cuda:0", denoise_steps=2,
                  seed=42, model_loader=load_from_checkpoint_dir) -> dict: ...
```

Build the registered reader from `OmegaConf.load`, require trace mode, load the self-contained checkpoint, create a minimal deploy config with compilation/cache off and video decode off, call `JointInferenceEngine.generate`, gather/invert outputs via the reader, validate shapes/finiteness, and atomically write NPZ/JSON evidence.

- [ ] **Step 4: Verify unit behavior**

Run: `pytest -q tests/test_open_loop_inference.py`

Expected: all pass without GPU or a 24 GB checkpoint.

- [ ] **Step 5: Commit inference slice**

```bash
git add scripts/open_loop_inference.py tests/test_open_loop_inference.py
git commit -m "feat: add offline OpenWAM open-loop inference"
```

### Task 5: Forward-input visualization and offline report

**Files:**
- Create: `scripts/visualize_model_inputs.py`
- Create: `tests/test_visualize_model_inputs.py`
- Create runtime/generated directory: `assets/openwam_alpha_data_examples/`

**Interfaces:**
- Consumes: one or more profile YAMLs and optional open-loop prediction NPZs.
- Produces per-profile PNG/JSON/NPZ plus `index.html` with the full transformation chain.

- [ ] **Step 1: Write failing report tests**

With a fake traced sample, assert a nine-frame contact sheet is emitted; JSON includes prompt, source, episode/window, tensor shapes, normalization provenance, and active-mask indices; NPZ includes raw EEF20, normalized EEF20, unified 80D, GT, and optional prediction; HTML is standalone and links only existing relative assets. Assert per-profile failure is reported and makes the overall exit nonzero after remaining profiles run.

- [ ] **Step 2: Verify red state**

Run: `pytest -q tests/test_visualize_model_inputs.py`

Expected: import failure for the absent script.

- [ ] **Step 3: Implement deterministic artifacts**

Expose:

```python
def render_profile(name: str, profile_path: Path, output_dir: Path,
                   *, sample_index=0, prediction_path: Path | None = None) -> dict: ...
def render_index(results: list[dict], output_dir: Path) -> Path: ...
```

Render PIL frames in a labeled 3×3 contact sheet, serialize numeric arrays without lossy string conversion, include concise scalar summaries and the pipeline labels `parquet -> EEF20 -> normalize -> scatter80 -> forward`, and use embedded CSS/JavaScript with inline SVG plots so the report opens from disk.

- [ ] **Step 4: Verify generated asset integrity**

Run: `pytest -q tests/test_visualize_model_inputs.py`

Expected: all pass.

- [ ] **Step 5: Commit visualization slice**

```bash
git add scripts/visualize_model_inputs.py tests/test_visualize_model_inputs.py assets/openwam_alpha_data_examples
git commit -m "feat: visualize OpenWAM model inputs"
```

### Task 6: Usage guide and MoraVLA comparison HTML

**Files:**
- Create: `assets/openwam_usage_docs/openwam-alpha-data-adaptation.md`
- Create: `docs/pretrain_vs_openwam_data_pipeline.html`
- Modify: `docs/2026-09-23-openwam-alpha-data-adaptation-session.md`
- Test: `tests/test_openwam_alpha_data_docs.py`

**Interfaces:**
- Consumes: checked-in parsers/configs and generated examples.
- Produces: exact commands, full YAML explanation, and an offline comparison report.

- [ ] **Step 1: Write documentation consistency tests**

Assert every referenced local file exists, all four profiles appear, documented CLI flags exist in parser source, original comparison HTML remains unchanged, and the comparison explicitly includes `qwena1_nogen`, 36D delta/head-camera, 5 FPS launch override, 80D OpenWAM, and the two scatter ranges.

- [ ] **Step 2: Verify red state**

Run: `pytest -q tests/test_openwam_alpha_data_docs.py`

Expected: missing documentation files.

- [ ] **Step 3: Write the guide and comparison page**

Document, in order: stats commands for each profile, smoke training command, two open-loop commands, visualization command, every YAML field, physical-vs-normalized outputs, adding a new compatible source, and failure diagnostics. Base the new HTML on the supplied page but add a verified section distinguishing MoraVLA's actual 36D launch path from other 96D layouts and OpenWAM's absolute EEF20/80D path.

- [ ] **Step 4: Verify docs and log the implementation**

Run: `pytest -q tests/test_openwam_alpha_data_docs.py`

Expected: all pass. Append commands/results and file inventory to the session log.

- [ ] **Step 5: Commit documentation slice**

```bash
git add assets/openwam_usage_docs/openwam-alpha-data-adaptation.md docs/pretrain_vs_openwam_data_pipeline.html docs/2026-09-23-openwam-alpha-data-adaptation-session.md tests/test_openwam_alpha_data_docs.py
git commit -m "docs: explain OpenWAM alpha EEF adaptation"
```

### Task 7: Real-data CPU acceptance

**Files:**
- Generate: `assets/openwam_alpha_data_examples/stats/*.npy`
- Generate: `assets/openwam_alpha_data_examples/{interndata,agibotworld,robodojo,robotwin}/*`
- Modify: `docs/2026-09-23-openwam-alpha-data-adaptation-session.md`

**Interfaces:**
- Consumes: four exact real LeRobot buckets.
- Produces: bounded stats, four sample artifact sets, and acceptance evidence.

- [ ] **Step 1: Generate stats for only episodes 0 and 1**

Run `python scripts/data/prepare_openwam_eef_smoke.py` four times with the exact profile bucket/key/output values.

Expected: each command reports two episodes, finite 20D action/state stats, and writes below `assets/openwam_alpha_data_examples/stats/`.

- [ ] **Step 2: Load one real sample per profile**

Run a small Python check through `build_dataset` for each profile and assert nine frames, action `(32,80)`, action mask `(32,80)`, proprio `(1,80)`, proprio mask `(1,80)`, non-empty prompt, finite active values, and active dimensions exactly `0:10,34:44`.

Expected: four profiles pass; raw `robotwin_traj` without the fullfield companion fails with a message naming the companion.

- [ ] **Step 3: Generate the four visualization examples**

Run: `python scripts/visualize_model_inputs.py --profile ...` with all four checked-in profiles and output root `assets/openwam_alpha_data_examples`.

Expected: four PNG/JSON/NPZ sets and a standalone `index.html`.

- [ ] **Step 4: Run the focused and regression suites**

Run: `pytest -q tests/dataloader/test_canonical_eef_lerobot.py tests/test_prepare_openwam_eef_smoke.py tests/test_open_loop_inference.py tests/test_visualize_model_inputs.py tests/test_openwam_alpha_data_configs.py tests/test_openwam_alpha_data_docs.py tests/dataloader/test_mixture_from_config.py`

Expected: all pass.

- [ ] **Step 5: Commit real-data evidence**

```bash
git add assets/openwam_alpha_data_examples docs/2026-09-23-openwam-alpha-data-adaptation-session.md
git commit -m "test: add canonical EEF sample evidence"
```

### Task 8: External GPU training and open-loop verification

**Files:**
- Generate runtime output: `outputs/openwam_alpha_v0_1_smoke/`
- Generate runtime output: `outputs/openwam_alpha_open_loop/{robodojo,robotwin}/`
- Modify: `docs/2026-09-23-openwam-alpha-data-adaptation-session.md`

**Interfaces:**
- Consumes: OpenWAM Python environment, H20 GPU, foundation checkpoint, checked-in YAMLs/scripts.
- Produces: two finite training steps and two model-generated open-loop artifacts.

- [ ] **Step 1: Diagnose the prior Torch import timeout systematically**

Run progressively timed probes for Python startup, `import torch`, dynamic-library loading, `torch.cuda.is_available()`, and a small CUDA allocation. Record wall times and the first failing boundary; do not change dependencies until the root cause is established.

Expected: either CUDA initializes successfully or the log contains reproducible command/error evidence and the smallest identified environmental blocker.

- [ ] **Step 2: Run two-step single-GPU smoke training**

Run:

```bash
NPROC_PER_NODE=1 torchrun --standalone --nproc_per_node=1 scripts/train.py --config-name openwam_alpha_v0_1_smoke
```

Expected: optimizer steps 1 and 2 complete; video, action, and total losses are finite; no checkpoint weights are saved because `save_steps=0`.

- [ ] **Step 3: Run RoboDojo open-loop**

Run `scripts/open_loop_inference.py` with the RoboDojo profile, foundation checkpoint, CUDA device, two denoise steps, and an explicit output directory.

Expected: `prediction.npz` contains finite `(32,80)` and `(32,20)` predictions and `manifest.json` records checkpoint/sample/timing.

- [ ] **Step 4: Run RobotWin open-loop**

Repeat with the RobotWin fullfield profile.

Expected: same shape/finiteness contract and evidence files, with the video source resolving through the supplied trajectory symlink.

- [ ] **Step 5: Final verification and commit**

Run `git diff --check`, the focused tests, and an HTML relative-link checker. Review `git diff --stat` and `git status --short` to confirm the three pre-existing user files are untouched. Append exact GPU commands, elapsed time, losses/output shapes, and any remaining limitation to the session log, then commit the log.
