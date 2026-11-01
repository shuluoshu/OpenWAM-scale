# Reserved DLC launch and Stage1 batch profiles

`scripts/pai/prepare_stage12_dlc.py` defaults to `startup_mode: hold`. Its DLC
`UserCommand` is exactly `sleep 36000000` (shell seconds). Scheduling the job does
not unpack the runtime, initialize a model, or start training. The reservation
continues after training exits and must be stopped separately when no longer used.

The renderer writes `request.json`, `dry_run.sh`, `submit.sh`, `launch_train.sh`
and `LAUNCH.md`. Pin `commit`, `code_dir`, `pipeline_config`, `prepared_marker`
and the runtime archive SHA in the site file. A hold request can be submitted
before CPU preparation finishes; explicit training cannot start without it.

After the reservation is running and CPU preparation passes, run the generated
`launch_train.sh` separately on each of the four pods, preserving the DLC
rank/rendezvous environment. It checks preparation, the pinned clean checkout,
runtime archive and topology. Set `expected_min_gpu_memory_gib` to 75 for the
80 GiB profile or 130 for the 140 GiB profile. The launcher prints actual GPU
names and rejects fewer than eight cards or insufficient memory before unpacking
the runtime. Resource-pool names do not establish the installed GPU model.

Each pod keeps a lifetime `flock` to reject duplicate launches. Logs go to
`<code_dir>/outputs/dlc_launch/` with hostname, UTC timestamp and PID. The launcher
returns the training child's exit status. `startup_mode: train` is an explicit
legacy opt-in to automatic training; new reservations should retain `hold`.

## Stage1 profiles

| Profile | Local batch | Global batch, 32 ranks | Readers/rank | Prefetch factor | FP32 optimizer |
|---|---:|---:|---:|---:|---|
| `stage1_visual_history_relative112_80g.yaml` | 128 | 4096 | 8 | 16 | CPU |
| `stage1_visual_history_relative112_140g.yaml` | 64 | 2048 | 8 | 8 | GPU |

Both profiles use GPU text encoding and eight simultaneous model bootstrap
workers per node. They are intended for the existing 144/168 CPU, 1600 GiB RAM,
eight-GPU nodes. The eight readers per rank use one Torch CPU thread each; video
decoders retain their existing two-thread limit. Do not copy this reader count
to substantially smaller CPU/RAM nodes without measuring it.

The loader dispatches individual samples to spawned CPU readers; the parent
assembles the true model batch. Worker outstanding requests are bounded to
`num_workers * prefetch_factor` individual samples. This is one future local
batch in these profiles, plus the batch held by the parent. Keep sufficient
shared memory for the resulting CPU tensors. Workers never receive the model
or perform CUDA/text inference. Reader caches and scratch paths are private.

The profiles preserve random finite order, relative112/32 offline statistics,
missing-image/label masks and the convergence gate. Resume identity includes the
batch/reader contract; changing a live run's batch is not a transparent resume.
Prefetched samples do not advance a checkpoint cursor. Partial population tails
contain only real samples; ranks without a real row contribute zero loss.

The learning rate and update/evaluation budgets are unchanged. Increasing the
global batch increases samples processed per optimizer update, including the
minimum 50,000-update convergence budget. Compare samples/second and convergence,
not only seconds/update or allocated GPU memory.

Capacity measurements on one H20-3e with explicit memory limits included real
RGB/history/goal tensors, backward and initialized FP32 AdamW. At a 75 GiB
allocator limit, CPU optimizer batch 128 reserved about 51 GiB; at a 132 GiB
limit, GPU optimizer batch 64 reserved about 101 GiB. These measurements exclude
distributed communication buffers and do not establish four-node throughput.
The larger card's batch 128 improved model throughput by only about 5% over 64,
so the profile retains 64. Check cluster peaks and the new `loader_wait_s`,
`batch_prepare_host_s`, `update_host_s`, batch/count and memory log fields after
explicit launch. Slow video decoding can still limit throughput.

The complete pipeline profiles are
`stage12_visual_history_relative112_80g_joint_4x8gpu.yaml` and
`stage12_visual_history_relative112_140g_none_4x8gpu.yaml`. Their Stage2 local
batches are 8 and 16 respectively, with four spawned readers per rank and the
existing native one-batch-per-worker prefetch. Stage2 keeps its existing
DeepSpeed CPU optimizer offload. These are validated starting configurations,
not a claim that Stage2 reaches the maximum possible throughput or memory use.
Each Stage2 sample retains 32 future actions with 112 relative-action dimensions.
The spawned dataset clears parent reader caches; `ByteLRU` serialization retains
its entry/byte limits so enabling workers does not fail at process startup.

Stage2 also recognizes datasets whose stationary-curation metadata explicitly
declares a full source video with a densely reindexed parquet timeline. Removed
episodes still occupy video frames in this format, so the reader uses validated
physical video segments, including their gaps. It checks integer frame bounds,
segment length, non-overlap and selected-row consistency. Undeclared sources
retain the cumulative-offset check and do not acquire a new `to_timestamp`
requirement. No dataset manifest, source payload or normalization value is edited.
