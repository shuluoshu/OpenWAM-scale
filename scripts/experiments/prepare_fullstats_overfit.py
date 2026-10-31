"""CPU-only three-sequence preparation under the unchanged full stats contract.

The full YAML/registry define normalization. ``selection.json`` is an explicit
preparation filter, never a replacement normalization population. Training
uses the native reader and native masks; FullSpanDataset is report-only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    from openwam.train.stage2_eval import _json_safe
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def verify_readiness(path):
    """Recheck immutable bytes both before and after CPU materialization."""
    ready = json.loads(Path(path).read_text())
    if ready.get('status') != 'complete':
        raise ValueError('statistics readiness is not complete')
    if (ready.get('horizon'), ready.get('dimensions'), ready.get('raw_dimensions')) != (32, 112, 52):
        raise ValueError('statistics must use horizon32 / unified112 / raw52')
    for key, digest_key, label in [('stats_registry', 'registry_sha256', 'registry'),
                                    ('source_yaml', 'source_sha256', 'source YAML')]:
        if sha(ready[key]) != ready[digest_key]:
            raise ValueError(f'{label} changed after readiness publication')
    registry = json.loads(Path(ready['stats_registry']).read_text())
    if registry.get('preparation', {}).get('complete') is not True:
        raise ValueError('full registry preparation is not complete')
    return ready


def restrict_preparation(dataset, sequences):
    """Narrow copied repos after full-contract construction; retain full selection."""
    originals = {r['repo']: r for r in dataset.selection['repos']}
    selected = {}; seen = set()
    for row in sequences:
        identity = row['repo'], row['episode']
        if identity in seen:
            raise ValueError('duplicate selected episode')
        seen.add(identity)
        repo = originals.get(row['repo'])
        if repo is None:
            raise ValueError(f"unknown selected repository: {row['repo']}")
        if row['family'] != repo['family']:
            raise ValueError('selected family does not match full source')
        episode, start, end = row['episode'], row['start'], row['end_exclusive']
        if any(type(v) is not int for v in (episode, start, end)) or not 0 <= start < end:
            raise ValueError('selection requires integral nonempty half-open ranges')
        if row['scope'] not in ('task', 'subtask'):
            raise ValueError('selection scope must be task/subtask')
        # A single native selected range must contain the full instruction. Never
        # join adjacent or disjoint ranges: that would change native window bounds.
        if not any(e == episode and a <= start < end <= b for e, a, b in repo['ranges']):
            raise ValueError(f'{identity}: requested range outside full selection')
        target = selected.setdefault(row['repo'], {**repo, 'ranges': []})
        target['ranges'].append((episode, start, end))
    if not selected:
        raise ValueError('empty preparation selection')
    dataset.repos = list(selected.values())
    dataset._weights = np.array([r['weight'] for r in dataset.repos], float)
    dataset._weights /= dataset._weights.sum()
    return dict(normalization_selection_unchanged=True,
                full_repository_count=len(originals), selected_repository_count=len(selected),
                selected_episode_count=len(seen), sequences=copy.deepcopy(sequences),
                policy='copied repository ranges restricted only after full registry contract validation')


def select_training_descriptors(dataset, sequences):
    """Every valid stride10 display anchor, including any supervised short tail."""
    from scripts.experiments.fullspan_samples import anchor_frames
    repos = {r['repo']: r for r in dataset.repos}
    descriptors = []; excluded = []; display_count = 0; by_family = {}
    for row in sequences:
        repo = repos[row['repo']]; episode = row['episode']; level = row['scope']
        metadata = dataset._episode(repo, episode)
        instructions = metadata['subtasks' if level == 'subtask' else 'tasks']
        matching = [i for i in instructions if i['start'] == row['start']
                    and i['end_exclusive'] == row['end_exclusive'] and i.get('valid', True)]
        if len(matching) != 1:
            raise ValueError(f"selection must identify one complete native {level}: {row}")
        ranges = [r for r in repo['ranges'] if r[0] == episode]
        options = dataset._options(metadata, ranges, level)
        family = dict(display=0, trained=0, short_tail=0, report_only=0)
        for anchor in anchor_frames(row['start'], row['end_exclusive']):
            display_count += 1; family['display'] += 1
            base = dict(repo=row['repo'], family=row['family'], episode=episode,
                        anchor=anchor, level=level)
            reason = None; option = None
            if anchor == row['end_exclusive'] - 1:
                reason = 'terminal'
            else:
                choices = [o for o in options if o['start'] <= anchor < o['end_exclusive']
                           and o['instruction'] == matching[0]]
                if len(choices) > 1:
                    raise ValueError(f'overlapping native anchor options: {base}')
                if not choices:
                    reason = 'native_anchor_filter'
                else:
                    option = choices[0]
                    _, _, mask, _, _, _ = dataset.raw_window(metadata, option, anchor)
                    if not np.asarray(mask).any(): reason = 'no_action_supervision'
            if reason:
                excluded.append(dict(base, reason=reason)); family['report_only'] += 1
            else:
                descriptors.append(dict(base, option=copy.deepcopy(option)))
                family['trained'] += 1
                family['short_tail'] += int(option['last_source'] - anchor < 32)
        if family['trained'] == 0:
            raise ValueError(f"selected family has no trainable display anchors: {row['family']}")
        by_family[row['family']] = family
    return descriptors, dict(mode='all native-valid stride10 display anchors; short tails included',
        stride_source_frames=10, display_anchor_count=display_count,
        trained_anchor_count=len(descriptors), selected_cases=len(descriptors),
        available_cases=len(descriptors), families=sorted(by_family), by_family=by_family,
        report_only=excluded, terminal_policy='report only; never trained')


def normalization_floor(raw, mask, stats):
    """Native quantile normalization roundtrip on masked physical raw values."""
    from openwam.dataloader.native_stage2_stats import normalize
    raw = np.asarray(raw, np.float32); mask = np.asarray(mask, bool)
    if raw.shape != mask.shape or raw.ndim != 2 or raw.shape[1] != 52:
        raise ValueError('raw normalization comparison requires [T,52] values and masks')
    normalized = normalize(raw, mask, stats, source='fullstats_overfit_preflight')
    lo = np.asarray(stats['q01'], np.float32)
    span = np.maximum(np.asarray(stats['q99'], np.float32) - lo, 1e-6)
    restored = lo + (normalized + 1) * span / 2
    restored[:, [9, 19]] = normalized[:, [9, 19]]
    unbounded = 2 * (raw - lo) / span - 1
    clipped = (np.abs(unbounded) > 1) & mask
    clipped[:, [9, 19]] = ((raw[:, [9, 19]] < 0) | (raw[:, [9, 19]] > 1)) & mask[:, [9, 19]]
    delta = np.abs(restored - raw)
    def summary(valid, errors):
        v = errors[valid]
        return dict(count=int(v.size), mean=float(v.mean()) if v.size else None,
                    max=float(v.max()) if v.size else None)
    aggregate = summary(mask, delta)
    per_dim = []
    for dim in range(52):
        part = summary(mask[:, dim], delta[:, dim])
        per_dim.append(dict(raw_dimension=dim, valid_scalars=part['count'],
            clipped_scalars=int(clipped[:, dim].sum()),
            raw_roundtrip_mae=part['mean'], raw_roundtrip_max=part['max']))
    tcp = {}
    for name, start in [('local_left', 0), ('local_right', 10), ('head_left', 34), ('head_right', 43)]:
        valid = mask[:, start:start+3].all(-1)
        error = np.linalg.norm(restored[:, start:start+3] - raw[:, start:start+3], axis=-1)
        tcp[name] = summary(valid, error)
    count = aggregate['count']
    return dict(valid_scalars=count, clipped_scalars=int(clipped.sum()),
        clipped_fraction=float(clipped.sum()/count) if count else None,
        raw_roundtrip_mae=aggregate['mean'], raw_roundtrip_max=aggregate['max'],
        by_raw_dimension=per_dim, tcp_position_roundtrip_m=tcp)


def validate_masked_loss(sample):
    """Exercise actual native loss on CPU without constructing a model."""
    from types import SimpleNamespace
    from openwam.model.architectures.base import BaseWAMArchitecture
    action = torch.as_tensor(sample['action']); mask = torch.as_tensor(sample['action_mask'])
    if action.shape != (32, 112) or mask.shape != action.shape or mask.dtype != torch.bool:
        raise ValueError('action and boolean mask must have shape [32,112]')
    if not mask.any() or not torch.isfinite(action[mask]).all():
        raise ValueError('empty or nonfinite valid action supervision')
    target = action.unsqueeze(0).float()
    # Large finite invalid errors verify they do not enter the denominator/loss.
    noise = target + torch.where(mask.unsqueeze(0), 1., 100.)
    loss = BaseWAMArchitecture._compute_action_loss(None, noise, target, torch.zeros(1),
        SimpleNamespace(training_weight=lambda ids: torch.ones_like(ids)),
        {'action_is_pad': ~mask.unsqueeze(0)}, 'cpu')
    if not torch.isclose(loss, torch.tensor(1.), atol=1e-6):
        raise ValueError('native masked loss did not normalize over valid elements')
    return dict(valid_elements=int(mask.sum()), valid_steps=int(mask.any(-1).sum()),
                masked_unit_error_loss=float(loss), shape=[32, 112])


def bind_config(config_path, ready):
    cfg = yaml.safe_load(Path(config_path).read_text())
    loader = cfg['dataloader']
    if Path(loader['source_yaml']).resolve() != Path(ready['source_yaml']).resolve():
        raise ValueError('comparison config must retain full normalization source YAML')
    current = loader.get('normalization_registry')
    if current is not None and Path(current).resolve() != Path(ready['stats_registry']).resolve():
        raise ValueError('refusing to replace a previously bound normalization registry')
    if current is None:
        loader['normalization_registry'] = ready['stats_registry']
        temporary = Path(config_path).with_suffix('.yaml.tmp')
        temporary.write_text(yaml.safe_dump(cfg, sort_keys=False))
        temporary.replace(config_path)
    return cfg


def prepare(run, readiness_path, selection_path):
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import read_registry, lookup_stats
    from openwam.train.stage2_eval import NativeFixedCases, _validate_sample
    from openwam.dataloader.utils.video_color import validate_stage2_color
    from scripts.experiments.fullspan_samples import (FullSpanDataset,
        build_inventory, materialize, normalize_terminal_timestamps)
    run = Path(run).resolve(); selection_path = Path(selection_path).resolve()
    ready = verify_readiness(readiness_path); selection_digest = sha(selection_path)
    selection = json.loads(selection_path.read_text()); sequences = selection['sequences']
    if len(sequences) != 3 or len({s['family'] for s in sequences}) != 3:
        raise ValueError('requires exactly three sequences, one from each family')
    config_path = run/'comparison_config.yaml'; cfg = bind_config(config_path, ready)
    config_digest = sha(config_path)
    output = run/'data'; output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise FileExistsError('immutable prepared data already exists')
    dataset = CuratedStage2Dataset.from_config(cfg['dataloader'])
    if dataset.registry.get('preparation', {}).get('complete') is not True:
        raise ValueError('full registry preparation is not complete')
    full_identity = dataset.normalization_identity()
    provenance = restrict_preparation(dataset, sequences)
    provenance.update(selection_path=str(selection_path), selection_sha256=selection_digest,
                      normalization_identity=full_identity)
    # Timestamp roundoff correction is decoding-only and is recorded; all native
    # options, pose masks, gripper masks, action and video supervision remain intact.
    for row in sequences:
        repo = next(r for r in dataset.repos if r['repo'] == row['repo'])
        normalize_terminal_timestamps(dataset._episode(repo, row['episode']))
    descriptors, coverage = select_training_descriptors(dataset, sequences)
    prior = Path(selection['prior_run']).resolve()
    old_manifest = json.loads((prior/'data/manifest.json').read_text())
    old_registry_path = Path(old_manifest['normalization_registry'])
    if sha(old_registry_path) != old_manifest['normalization_sha256']:
        raise ValueError('prior normalization registry changed')
    # Read under its OWN immutable contract, only for comparison of the same raw
    # windows. Never rebind its contract or feed old normalized tensors to training.
    old_contract = json.loads(old_registry_path.read_text())['contract']
    old_registry = read_registry(old_registry_path, old_contract)
    cases = NativeFixedCases(dataset, run_dir=output, seed=42,
                             max_cases=max(128, len(descriptors)), max_search_episodes=1)
    cases.descriptors = descriptors; cases.coverage = coverage; cases.publish_descriptors()
    samples = []; rows = []; masked_checks = []
    for index, descriptor in enumerate(descriptors):
        sample = cases[index]
        if 'visual_availability' in sample['trace']:
            raise ValueError('report-only visual restoration entered training sample')
        _validate_sample(sample, observation_mode='multiview'); validate_stage2_color(sample)
        masked_checks.append(validate_masked_loss(sample))
        path = output/f'case_{index:03d}.pt'; torch.save(sample, path)
        samples.append(sample)
        rows.append(dict(index=index, family=descriptor['family'], repo=descriptor['repo'],
            episode=descriptor['episode'], anchor=descriptor['anchor'], granularity=descriptor['level'],
            prompt=sample['prompt'], path=str(path), sha256=sha(path),
            valid_action_elements=int(sample['action_mask'].sum()), report_target=sample['report_target']))
        print(json.dumps(dict(event='training_case_ready', index=index, total=len(descriptors),
                              family=descriptor['family'], anchor=descriptor['anchor'])), flush=True)
    torch.save(samples, output/'samples.pt'); del samples
    eval_indices = []
    for family in sorted({r['family'] for r in rows}):
        choices = [r['index'] for r in rows if r['family'] == family]
        eval_indices.append(choices[len(choices)//2])
    eval_indices.extend(r['index'] for r in rows if r['family'] == 'abc_130k'
                        and r['anchor'] == 70 and r['index'] not in eval_indices)

    report_output = run/'fullspan/data'
    report_output.mkdir(parents=True, exist_ok=True)
    report_dataset = FullSpanDataset.from_config(cfg['dataloader'])
    restrict_preparation(report_dataset, sequences)
    report = build_inventory(report_dataset, report_output, config_path)
    expected = {(r['repo'], r['episode'], r['start'], r['end_exclusive'], r['scope']) for r in sequences}
    actual = {(r['repo'], r['episode'], r['start'], r['end_exclusive'], r['scope']) for r in report['sequences']}
    if actual != expected:
        raise ValueError('fullspan inventory selected a different complete instruction')
    report['preparation_filter'] = provenance
    report['visual_validity_policy'] = ('decoded_source_images_independent_of_tracking_v1; '
        'original supervision masks retained in sample trace; action/proprio/FK unchanged; report only')
    materialize(report_dataset, report, report_output)
    training_lookup = {(r['repo'], r['episode'], r['anchor']): r['index'] for r in rows}
    excluded_lookup = {(r['repo'], r['episode'], r['anchor']): r['reason'] for r in coverage['report_only']}
    preflight_rows = []
    report_sequences = {r['sequence_id']: r for r in report['sequences']}
    for record in report['anchors']:
        seq = report_sequences[record['sequence_id']]
        repo = next(r for r in dataset.repos if r['repo'] == seq['repo'])
        metadata = dataset._episode(repo, seq['episode'])
        sample = torch.load(record['sample_path'], map_location='cpu', weights_only=False)
        raw = np.asarray(sample['trace']['raw_action']); mask = np.asarray(sample['trace']['raw_action_mask'], bool)
        key = seq['repo'], seq['episode'], record['anchor']
        index = training_lookup.get(key)
        # Raw masks are identical between report and native training; only decoded
        # visual availability is restored for report generation.
        if index is not None:
            trained = torch.load(rows[index]['path'], map_location='cpu', weights_only=False)
            if not np.array_equal(raw, trained['trace']['raw_action']) or not np.array_equal(mask, trained['trace']['raw_action_mask']):
                raise ValueError('training/report raw windows disagree')
            del trained
        preflight_rows.append(dict(family=seq['family'], repo=seq['repo'], episode=seq['episode'],
            anchor=record['anchor'], granularity=seq['scope'], trainable=index is not None,
            training_index=index, exclusion_reason=excluded_lookup.get(key),
            raw_window_sha256=hashlib.sha256(raw.tobytes()+mask.tobytes()).hexdigest(),
            old=normalization_floor(raw, mask, lookup_stats(old_registry, repo, metadata)['action']),
            new=normalization_floor(raw, mask, lookup_stats(dataset.registry, repo, metadata)['action'])))
    if verify_readiness(readiness_path) != ready:
        raise ValueError('statistics readiness changed during materialization')
    if sha(selection_path) != selection_digest or sha(config_path) != config_digest:
        raise ValueError('preparation selection/config changed during materialization')
    if dataset.normalization_identity() != full_identity:
        raise ValueError('full normalization identity changed during preparation')
    preflight = dict(schema='openwam.fullstats_overfit.preflight.v1', status='complete',
        old_registry=dict(path=str(old_registry_path), sha256=sha(old_registry_path)),
        new_registry=dict(path=dataset.registry_path, sha256=ready['registry_sha256']),
        comparison_policy='same raw native windows and masks; quantile clipping unchanged; gripper absolute clipped [0,1]',
        physical_floor_note='Raw-coordinate TCP displacement roundtrip floor in metres; not independently calibrated world trajectory error.',
        coverage=coverage, masked_loss_checks=masked_checks, cases=preflight_rows)
    dump(output/'preflight.json', preflight)
    manifest = dict(schema='openwam.z_action.fixed_cpu.v1', status='complete',
        source_yaml=dataset.selection['source_yaml'], source_sha256=dataset.source_yaml_sha256,
        normalization_registry=dataset.registry_path, normalization_sha256=ready['registry_sha256'],
        normalization_horizon=32, action_horizon=32, video_stride=4, video_frames=9,
        coverage=coverage, preparation_filter=provenance, cases=rows, evaluation_indices=eval_indices,
        samples_path=str(output/'samples.pt'), samples_sha256=sha(output/'samples.pt'),
        fullspan_manifest=str(report_output/'manifest.json'), preflight=str(output/'preflight.json'),
        comparison_config=str(config_path), comparison_config_sha256=config_digest)
    dump(output/'manifest.json', manifest)
    print(json.dumps(dict(status='complete', samples=len(rows), evaluation_indices=eval_indices,
                          display_anchors=report['anchor_count'], manifest=str(output/'manifest.json'))), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stats-readiness', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args(); torch.set_num_threads(2)
    import scripts.data.prepare_subtask_goals as decoder
    from scripts.experiments.fullspan_samples import DecodedFrameCache
    decoder.decode_frames = DecodedFrameCache(decoder.decode_frames)
    prepare(args.run, args.stats_readiness, args.selection or args.run/'selection.json')


if __name__ == '__main__':
    main()
