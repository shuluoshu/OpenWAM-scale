# Subtask occurrence boundary repair

Adjacent source annotations can reuse a subtask ID (text identity). Preserve each explicit annotation boundary when deriving segments and terminal goal frames. Frame-label changes and unannotated masks still split runs. Gaps remain invalid; overlapping same-ID intervals are rejected as ambiguous. Task-only sources are unchanged.

The original failure was LightWheel/EgoProStandard-body/shard_0005 episode 123: [5181,5511) and [5511,5754) both use ID 503. They now yield goals 5510 and 5753.

Finite population version is now 3. Version 2 complete and partial SQLite populations must be rebuilt in a new path; do not resume the old preparation index. Original input data is unchanged.

The DSW diagnostic uses the selected posttrain YAML (with the previously authorized wipe_table exclusion), one sample per positive-weight source directory, without collapsing shards. One real Stage1 forward per source uses the production model/config, without backward or optimizer updates; results must distinguish data preparation failures from model failures. A one-sample-per-repo diagnostic is not a full-epoch or full-source validation.

## Per-directory DSW trial

Run from the code checkout with the OpenWAM Python environment. The derived selection retains all original selected sources except the previously authorized missing wipe_table source.

```bash
python scripts/validate_goal_all_repos.py --prepare \
  --config configs/goal/stage1_query_history_epoch_3fps.yaml \
  --source-yaml outputs/stage12_posttrain_selection/source.yaml \
  --output-dir outputs/dsw_all_repos_smoke_20261025
python scripts/validate_goal_all_repos.py --forward \
  --config configs/goal/stage1_query_history_epoch_3fps.yaml \
  --source-yaml outputs/stage12_posttrain_selection/source.yaml \
  --output-dir outputs/dsw_all_repos_smoke_20261025
```

`identity.json` records the code/config/source identity; `samples.jsonl` records one deterministic pair per directory (subtask preferred, task fallback, 3 FPS grid); `results.jsonl` records forward success, finite loss, masks, timing and CUDA peak memory. All failures are retained in `errors.jsonl` and counted in `summary.json`. Preparation failures remain failures in the overall trial; no shard is silently omitted. Each source uses the first selected episode with a legal 3 FPS pair; no-pair episodes are recorded, while metadata errors remain explicit failures. This is a diagnostic sequence of forward calls, not a resumable production run or Stage2 training.
