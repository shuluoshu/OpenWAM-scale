#!/usr/bin/env python3
"""Read-only audit of production shuffle coverage and early batch composition."""
from collections import Counter
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def audit(index, source_yaml, *, seed=42, batches=1000, world=32, verify_full=False):
    from openwam.train.epoch_order import EpochOrder, sampling_contract
    from openwam.dataloader.curated_goal import parse_curated_yaml
    from openwam.dataloader.goal_epoch import _selected_repos_digest
    from openwam.train.goal_epoch import sha256
    index = Path(index).resolve()
    with sqlite3.connect(index.as_uri() + '?mode=ro', uri=True) as db:
        metadata = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM metadata')}
        if not metadata.get('complete'): raise ValueError('population is not complete')
        if sha256(source_yaml) != metadata['source_yaml_sha256']: raise ValueError('source YAML identity mismatch')
        selection = parse_curated_yaml(source_yaml)
        rows = db.execute('SELECT sample_start,count,repo_id,level FROM runs ORDER BY sample_start').fetchall()
    starts = np.array([r[0] for r in rows], dtype=np.int64)
    total = metadata['total_samples']
    started = time.monotonic()
    order = EpochOrder(sampling_contract({'seed': seed}, total))
    initialization_s = time.monotonic() - started
    count = min(total, batches * world)
    ids = np.fromiter((order[i] for i in range(count)), dtype=np.int64, count=count)
    run_ids = np.searchsorted(starts, ids, side='right') - 1
    family = lambda run: selection['repos'][run[2]]['family']
    families = [family(rows[i]) for i in run_ids]
    levels = [rows[i][3] for i in run_ids]
    population_families = Counter()
    population_levels = Counter()
    for row in rows:
        population_families[family(row)] += row[1]
        population_levels[row[3]] += row[1]
    old_run_ids = np.searchsorted(starts, np.arange(count), side='right') - 1
    report = dict(total_samples=total, world_size=world, prefix_samples=count,
                  sampling=order.contract, sampling_sha256=order.sha256,
                  permutation_host_bytes=order.nbytes, permutation_initialization_s=initialization_s,
                  prefix_unique_samples=int(len(np.unique(ids))),
                  population_family_counts=dict(population_families), population_granularity_counts=dict(population_levels),
                  shuffled_family_counts=dict(Counter(families)), shuffled_granularity_counts=dict(Counter(levels)),
                  sequential_family_counts=dict(Counter(family(rows[i]) for i in old_run_ids)),
                  sequential_granularity_counts=dict(Counter(rows[i][3] for i in old_run_ids)),
                  distinct_families_per_batch=[len(set(families[i:i+world])) for i in range(0, count, world)],
                  batches_with_both_levels=sum(len(set(levels[i:i+world])) == 2 for i in range(0, count, world)),
                  first_batch=[dict(position=i, dataset_index=int(ids[i]), family=families[i],
                                    granularity=levels[i], repo=selection['repos'][rows[run_ids[i]][2]]['repo'])
                               for i in range(min(world, count))])
    class Selected:
        repos = selection['repos']
    report['selected_repos_sha256'] = _selected_repos_digest(Selected())
    if verify_full:
        # Audit-only scratch copy; training does not allocate this sorted array.
        sorted_ids = np.sort(order._order)
        for start in range(0, total, 1_000_000):
            end = min(total, start + 1_000_000)
            if not np.array_equal(sorted_ids[start:end], np.arange(start, end, dtype=sorted_ids.dtype)):
                raise AssertionError('permutation has missing or duplicate sample IDs')
        report['full_population_exactly_once'] = True
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', required=True)
    p.add_argument('--source-yaml', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batches', type=int, default=1000)
    p.add_argument('--world-size', type=int, default=32)
    p.add_argument('--verify-full', action='store_true')
    args = p.parse_args()
    if args.batches < 1 or args.world_size < 1: p.error('batch count and world size must be positive')
    report = audit(args.index, args.source_yaml, seed=args.seed, batches=args.batches,
                   world=args.world_size, verify_full=args.verify_full)
    from openwam.train.goal_epoch import atomic_json
    atomic_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('first_batch', 'distinct_families_per_batch')}, indent=2))


if __name__ == '__main__': main()
