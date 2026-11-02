#!/usr/bin/env python3
"""Inventory or prebuild exact Mecka ordinal indexes outside raw datasets.

Example:
  python scripts/data/build_mecka_video_index.py --curated-yaml selection.yaml \
      --cache-dir /shared/openwam/video-index --inventory-only --output inventory.json
  python scripts/data/build_mecka_video_index.py --video /data/video.mp4 \
      --cache-dir /shared/openwam/video-index --output built.json

With --curated-yaml, --limit N limits builds after the complete deduplicated
inventory is counted. Indexing is CPU only and decodes each selected video once.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def selected_videos(curated_yaml, *, data_root=None):
    import pyarrow.parquet as pq
    from openwam.dataloader.curated_goal import parse_curated_yaml
    camera = 'observation.images.egocentric'
    prefix = f'videos/{camera}/'
    by_path = {}
    for repo in parse_curated_yaml(curated_yaml, data_root=data_root)['repos']:
        if repo['family'] != 'mecka':
            continue
        root = Path(repo['path'])
        info = json.loads((root / 'meta/info.json').read_text())
        episodes = sorted({episode for episode, _, _ in repo['ranges']})
        table = pq.read_table(root / 'meta/episodes',
                              columns=['episode_index', prefix + 'chunk_index', prefix + 'file_index'],
                              filters=[('episode_index', 'in', episodes)])
        seen = set()
        for ep in table.to_pylist():
            number = int(ep['episode_index'])
            if number in seen:
                raise ValueError(f'duplicate episode metadata: {repo["repo"]}:{number}')
            seen.add(number)
            path = str((root / info['video_path'].format(
                video_key=camera, episode_index=number,
                chunk_index=int(ep[prefix + 'chunk_index']),
                file_index=int(ep[prefix + 'file_index']))).resolve())
            record = by_path.setdefault(path, dict(path=path, repo=repo['repo'], episodes=[]))
            record['episodes'].append(number)
        if seen != set(episodes):
            raise ValueError(f'missing selected episode metadata: {repo["repo"]}')
    for record in by_path.values():
        record['episodes'].sort()
    return sorted(by_path.values(), key=lambda record: record['path'])


def read_inventory(path):
    """Read a JSON list of paths/records or a previous builder inventory report."""
    document = json.loads(Path(path).read_text())
    records = document.get('inventory') if isinstance(document, dict) else document
    if not isinstance(records, list):
        raise ValueError('inventory must contain a list of video paths')
    result = []
    for item in records:
        record = dict(path=item) if isinstance(item, str) else item
        if not isinstance(record, dict) or not isinstance(record.get('path'), str):
            raise ValueError('inventory record requires a string path')
        result.append(dict(record, path=str(Path(record['path']).resolve()), episodes=record.get('episodes', [])))
    return result


def main():
    from openwam.dataloader.video_ordinal_index import build_video_index, index_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curated-yaml', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--video', type=Path, action='append', default=[])
    parser.add_argument('--inventory-json', type=Path)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=1,
                        help='Concurrent CPU builders, 1..16; decoder threads use OPENWAM_VIDEO_DECODE_THREADS (default 2)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.curated_yaml and not args.video and not args.inventory_json:
        parser.error('provide --curated-yaml, --inventory-json, or --video')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be between 1 and 16')
    records = selected_videos(args.curated_yaml, data_root=args.data_root) if args.curated_yaml else []
    if args.inventory_json:
        records.extend(read_inventory(args.inventory_json))
    records.extend(dict(path=str(video.resolve()), episodes=[]) for video in args.video)
    unique = {}
    for record in records:
        unique.setdefault(record['path'], record)
    records = list(unique.values())
    report = dict(curated_yaml=str(args.curated_yaml) if args.curated_yaml else None,
                  cache_dir=str(args.cache_dir.resolve()), workers=args.workers, selected_unique_videos=len(records),
                  selected_episodes=sum(len(record['episodes']) for record in records),
                  inventory=records, builds=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    save()
    print(json.dumps({key: report[key] for key in ('selected_unique_videos', 'selected_episodes')}), flush=True)
    if args.inventory_only:
        return
    def build_one(record):
        start = time.perf_counter()
        result = dict(path=record['path'])
        try:
            artifact = index_path(record['path'], cache_dir=args.cache_dir)
            result['artifact_existed'] = artifact.exists()
            index = build_video_index(record['path'], cache_dir=args.cache_dir)
            if index is None:
                result['status'] = 'busy'
            else:
                result.update(status='ready', frames=len(index['pts']), keyframes=len(index['keyframes']),
                              seek_safe=index['metadata']['seek_safe'], artifact=str(artifact),
                              index_bytes=artifact.stat().st_size,
                              source_sha256=index['metadata']['source_sha256'])
        except Exception as exc:
            result.update(status='error', error=f'{type(exc).__name__}: {exc}')
        result['seconds'] = time.perf_counter() - start
        return result
    failed = False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(build_one, records[:args.limit]):
            failed |= result['status'] != 'ready'
            report['builds'].append(result)
            save()
            print(json.dumps(result), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
