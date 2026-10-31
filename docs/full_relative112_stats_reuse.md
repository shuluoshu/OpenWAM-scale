# Reuse offline relative112 statistics in Stage2 / Stage12

The full selected-population statistics destination is
`/mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/curated_relative112_stats_20261031_exclude_missing_wipe_table`.
Use the exact immutable `registry-<sha256>.json` named by its `readiness.json`;
the filename is determined by the completed population, so it cannot be fixed in
advance. A missing readiness file or a status other than `complete` is not ready
for reuse. This document does not assert that the full scan has completed.

The missing `wipe_table/wipe_table_v1_test` source was excluded with explicit user
authorization. The new selection retains 5,702 repositories and 283,698 episodes.
Completed per-repository statistics can be reused only after proving the retained
paths, ranges, weights, metadata, reader policy and code are identical. The subset
finalizer preserves the original histogram bytes and records the old/new selection
identities; it does not replace failed data or publish an incomplete population.

The registry describes native raw52 targets scattered into unified112, with 32
future source rows relative to the fixed current anchor. TCP/head axes, joint
profiles, source masks and gripper filtering must match the native reader. The
offline population's count/min/max/moments and approximate-quantile policy are
recorded in the published artifact; training retains its existing quantile
normalization and clipping behavior.

## Resolve the completed artifact on CPFS

```bash
export STAGE2_STATS_REGISTRY="$(/root/miniconda3/envs/openwam/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
root = Path('/mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/curated_relative112_stats_20261031_exclude_missing_wipe_table')
ready = json.loads((root / 'readiness.json').read_text())
assert ready['status'] == 'complete', ready
path = Path(ready['stats_registry'])
raw = path.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
assert digest == ready['registry_sha256']
assert path.name == 'registry-' + digest + '.json'
print(path)
PY
)"
test -n "$STAGE2_STATS_REGISTRY" || exit 1
```

The selected YAML is exactly:
`/mnt/data/cpfs/b5/yeqingda/pretrain/launch/moravla/train_files/pretrain_v0_3_curated_3000h_posttrain_20261019_exclude_rcv1_rmv1_rmv2_20261030_exclude_missing_wipe_table_20261031.yaml`.
Changing its bytes, the effective data root, the reader policy or gripper filter
requires a matching new registry. If verified relocated sources require a path
alias JSON, retain that file on CPFS and use the same aliases during preparation
and training. An alias does not authorize excluding a missing source.

## Minimal configuration overrides

For an existing native Stage2 configuration:

```yaml
dataloader:
  source_yaml: /mnt/data/cpfs/b5/yeqingda/pretrain/launch/moravla/train_files/pretrain_v0_3_curated_3000h_posttrain_20261019_exclude_rcv1_rmv1_rmv2_20261030_exclude_missing_wipe_table_20261031.yaml
  normalization_registry: ${oc.env:STAGE2_STATS_REGISTRY}
```

Keep the audited `dataloader.gripper_state_filter` settings. If aliases are
needed, also set `dataloader.path_aliases` to the verified CPFS JSON path.

For an existing compatible Stage12 pipeline configuration:

```yaml
source_yaml: /mnt/data/cpfs/b5/yeqingda/pretrain/launch/moravla/train_files/pretrain_v0_3_curated_3000h_posttrain_20261019_exclude_rcv1_rmv1_rmv2_20261030_exclude_missing_wipe_table_20261031.yaml
stats_registry: ${oc.env:STAGE2_STATS_REGISTRY}
output_dir: /mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/stage12_full_relative112_reuse_20261031
```

Use a Stage1 configuration whose history/representation matches the chosen
Stage2 configuration. For aliases, set `dataset.path_aliases` in that Stage1
configuration; Stage12 forwards it to Stage2. The top-level `stats_registry`
field supplies the shared registry; `stage2_overrides` cannot replace
`dataloader.normalization_registry` or the shared source/history contract.
Other supported Stage2 overrides are composed before statistics validation,
including gripper filter settings.

With that pipeline saved at a concrete CPFS path, run CPU preparation and
validation from the code checkout:

```bash
/root/miniconda3/envs/openwam/bin/python scripts/train_stage12.py --config /absolute/cpfs/path/pipeline.yaml --prepare
/root/miniconda3/envs/openwam/bin/python scripts/train_stage12.py --config /absolute/cpfs/path/pipeline.yaml --check
```

An explicit registry is checked for immutable bytes and exact selection/reader
contract. A projected audit compares every selected source's info/schema,
calibration identity and episode identity/length digest, plus every profile's
representation contract, before expensive Stage1 index preparation starts.
Both action and state statistics must contain finite numeric raw52 vectors,
nonnegative integer counts, ordered extrema/quantiles and nonnegative standard
deviations. Declared joints require positive counts; valid inactive dimensions
may keep zero counts and finite filler values.
This path never calls the sampled statistics generator. `prepared.json` then
records the same absolute registry path, satisfying the existing DLC preparation
gate's registry/manifest equality check. When `stats_registry` is absent, the
existing bounded sampled preparation remains available.

The full metadata audit runs once during CPU `--prepare`; it is not added to
each training node. Normal native dataset construction still validates registry
bytes and the selection contract, and native episode loading checks source/schema
identity as data are used. Audit identity covers selected metadata, not a hash
of every source parquet byte. Keep source assets unchanged after preparation.

Persist the resolved registry path in the deployment configuration, or provide
the same environment variable on every node. A `readiness.json` for statistics
alone does not replace Stage12's complete `population.sqlite` and `prepared.json`.
Preparing these files does not submit a DLC job or start GPU training.
