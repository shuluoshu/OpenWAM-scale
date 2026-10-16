"""Execute isolated Mora/OpenWAM GT code on identical source anchors.

This is report evidence, not a complete Mora loader or registry hydration.
Mora keeps canonical actions.eef endpoints and source action target timing.
OpenWAM keeps configured physical TCP and next-state targets. Metres/radians.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import traceback

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

OUT = Path(__file__).resolve().parent
REPORT = OUT.parent
ROOT = REPORT.parents[1]
MORA = (ROOT / '../../pretrain').resolve()
sys.path[:0] = [str(REPORT), str(ROOT)]
from prepare import build_clip_reader, write
from openwam.dataloader.utils.mora_relative import _rotation
from openwam.dataloader.utils.normalization import NORM_EPS, apply_normalization
sys.path.insert(0, str(MORA))
from src.data.modules.transforms.action_space import (
    ActionSpaceComposer, compute_eef_delta_action, transform_eef_world_to_camera,
)


def matrix_blocks(values, starts=(0, 10)):
    """Convert EEF rot6d blocks into matrices without changing physical endpoints."""
    values = np.asarray(values)
    matrices = np.broadcast_to(np.eye(4), (*values.shape[:-1], 2, 4, 4)).copy()
    valid = np.zeros((*values.shape[:-1], 2), bool)
    for side, start in enumerate(starts):
        if values.shape[-1] < start + 9:
            continue
        block = values[..., start:start + 9]
        matrices[..., side, :3, 3] = block[..., :3]
        matrices[..., side, :3, :3], good = _rotation(block[..., 3:9])
        valid[..., side] = good & np.isfinite(block).all(-1)
    return matrices, valid


def masked_error(actual, expected, valid):
    """Return maximum element error on valid poses; reject numerical failures."""
    if not valid.any():
        return None
    error = float(np.max(np.abs(actual[valid] - expected[valid])))
    if error > 5e-5:
        raise AssertionError(f'Physical pose roundtrip error {error} exceeds 5e-5')
    return error


def camera_valid(camera):
    rotation = camera[:3, :3]
    return bool(np.isfinite(camera).all() and np.allclose(camera[3], [0, 0, 0, 1], atol=1e-5)
                and np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4)
                and np.isclose(np.linalg.det(rotation), 1, atol=1e-4))


def source_rows(record):
    """Read original action/state rows; preserve source metadata and timing."""
    names = pq.ParquetFile(record['source_parquet']).schema_arrow.names
    wanted = [key for key in names if key in {
        'frame_index', 'timestamp', 'actions.eef', 'observation.states.eef',
        'actions.eef.mask', 'observation.states.eef.mask',
        'observation.states.camera.world_to_camera', 'observation.states.camera.intrinsic',
    } or 'target_frame' in key or 'target_valid' in key or key.endswith('canonical.frame_valid')]
    start = record['start']
    table = ds.dataset(record['source_parquet']).to_table(columns=wanted, filter=(
        (ds.field('episode_index') == record['episode']) & (ds.field('frame_index') >= start)
        & (ds.field('frame_index') < start + 53))).sort_by([('frame_index', 'ascending')])
    np.testing.assert_array_equal(table['frame_index'].to_numpy()[:50], np.arange(start, start + 50))
    return {key: np.asarray(table[key].to_pylist()) for key in table.column_names}


def process(record, stats_root=None):
    """Save per-clip physical targets, executed masks, timing, and verified roundtrips."""
    destination = OUT / record['id']
    destination.mkdir(exist_ok=True, parents=True)
    clip = REPORT / 'clips' / record['id']
    config = OmegaConf.load(clip / 'data.yaml')
    reader = build_clip_reader(config)
    representation = reader.representation
    table = pq.read_table(clip / 'dataset/data/chunk-000/file-000.parquet')
    columns = {key: np.asarray(table[key].to_pylist())[:33] for key in representation.columns}
    columns['_rows'] = np.arange(33)
    raw, mask, state, state_mask = representation.encode_window(columns)
    absolute, absolute_mask = representation._absolute_values(columns)
    ow_absolute, _ = matrix_blocks(absolute)
    ow_local, _ = matrix_blocks(raw)
    ow_head, _ = matrix_blocks(raw, (34, 43))
    ow_local_mask, ow_head_mask = mask[:, [0, 10]], mask[:, [34, 43]]
    model = np.load(clip / 'model_inputs.npz')
    normalized = np.where(mask, apply_normalization(raw, reader._normalization_stats, 'quantile'), 0)
    saved_difference = float(np.max(np.abs(normalized - model['action'][:, representation.scatter])))
    saved_raw = np.concatenate([model['raw_action_eef20'], model['raw_action_joint14'], model['raw_action_head18']], axis=-1)
    raw_difference = float(np.max(np.abs(saved_raw - raw)))
    np.testing.assert_allclose(raw, saved_raw, atol=5e-7, rtol=0)
    saved_normalized = np.where(mask, apply_normalization(saved_raw, reader._normalization_stats, 'quantile'), 0)
    np.testing.assert_array_equal(saved_normalized, model['action'][:, representation.scatter])
    normalized = model['action'][:, representation.scatter]
    np.testing.assert_array_equal(mask, model['action_mask'][:, representation.scatter])
    stats = reader._normalization_stats
    scale = np.maximum(stats['q99'] - stats['q01'], NORM_EPS)
    recovered = (normalized + 1) * 0.5 * scale + stats['q01']
    recovered[~mask] = 0
    ow_model_local, _ = matrix_blocks(recovered)
    ow_model_head, _ = matrix_blocks(recovered, (34, 43))
    original = dict(np.load(clip / 'source.npz'))
    source = source_rows(record)
    camera = np.asarray(source.get('observation.states.camera.world_to_camera', np.full((1, 16), np.nan))[0]).reshape(4, 4)
    rigid_camera = camera_valid(camera)
    head_trusted = rigid_camera and bool(config.get('head_camera_available', True)) and 'audit.unavailable' not in config.head_world_to_camera_key
    intrinsic_key = next((key for key in [
        f'observation.states.cameras.{record["head"].split(".")[-1]}.intrinsic',
        'observation.states.camera.intrinsic', 'head_camera_intrinsics'] if key in original), None)
    intrinsic = np.full((3, 3), np.nan)
    if intrinsic_key:
        values = original[intrinsic_key][0]
        intrinsic = np.asarray(values).reshape(3, 3) if len(values) == 9 else np.array([
            [values[0], 0, values[2]], [0, values[1], values[3]], [0, 0, 1]])
    arrays = dict(ow_local=ow_local, ow_head=ow_head, ow_tcp_anchor=ow_absolute[0],
                  ow_tcp_world=ow_absolute[1:], ow_local_mask=ow_local_mask, ow_head_mask=ow_head_mask,
                  ow_anchor_mask=absolute_mask[0, [0, 10]], ow_raw_action52=raw,
                  ow_action_mask52=mask, ow_state52=state, ow_state_mask52=state_mask,
                  ow_model_normalized112=model['action'], ow_model_action_mask112=model['action_mask'],
                  ow_model_physical52=recovered, ow_model_local=ow_model_local, ow_model_head=ow_model_head,
                  ow_target_source_frame_indices=np.arange(record['start'] + 1, record['start'] + 33),
                  world_to_camera=camera, intrinsic=intrinsic,
                  anchor_source_frame_index=np.asarray(record['start']))
    ow_roundtrip = masked_error(ow_absolute[0][None] @ ow_local, ow_absolute[1:], ow_local_mask)
    head_roundtrip = None
    if rigid_camera and ow_head_mask.any():
        anchor_camera = camera[None] @ ow_absolute[0]
        restored = ow_head.copy()
        restored[..., :3, 3] += anchor_camera[None, :, :3, 3]
        restored[..., :3, :3] = ow_head[..., :3, :3] @ anchor_camera[None, :, :3, :3]
        head_roundtrip = masked_error(restored, camera[None, None] @ ow_absolute[1:], ow_head_mask)
    evidence = dict(id=record['id'], repo=record['repo'], status='ok', anchor_source_frame=record['start'],
        units={'translation': 'metres', 'rotation': 'rotation matrix / source rot6d'},
        mora_anchor_sampling=dict(forced_same_anchor=True, natural_global_stride2_eligible=(record['start'] % 2 == 0), note='15fps sampling at 30fps would normally use global even frame ordinals; odd source starts are explicitly forced for this paired evidence'),
        execution_scope='actual isolated representation/transform/composer calls; no training, no full Mora loader hydration',
        ow=dict(endpoint='explicit configured physical TCP', tcp_reason=record['tcp_reason'],
            local_valid_pose_steps=int(ow_local_mask.sum()), head_valid_pose_steps=int(ow_head_mask.sum()),
            action_shape=[32, 112], saved_raw_match_atol=5e-7, saved_raw_max_abs_difference=raw_difference, saved_raw_normalization_exact=True, saved_model_max_abs_difference=saved_difference, local_roundtrip_max_abs=ow_roundtrip,
            head_roundtrip_max_abs=head_roundtrip, configuration=OmegaConf.to_container(config),
            clipped_valid_cells=int(((np.abs(recovered - raw) > 1e-6) & mask).sum()),
            clipped_xyz_max_m=max([float(np.max(np.abs(recovered[:, index:index+3] - raw[:, index:index+3])[mask[:, index:index+3]]))
                                  for index in (0,10,34,43) if mask[:, index:index+3].any()] or [0.])),
        calibration=dict(numerically_rigid=rigid_camera, trusted_for_projection=head_trusted,
                         intrinsic_key=intrinsic_key, reason='source audit strict camera policy; ABC rejected/not QA-scored is not trusted'),
        source_endpoint_note='Mora inherits canonical EEF exactly; no assumption that EEF equals wrist, flange, or physical TCP.',
        source_masks_note='geometric masks below are report-side finite/rot6d/source-mask checks; actual composer availability saved separately')
    for key, value in source.items():
        arrays['source_' + key.replace('.', '_')] = value
    if not {'actions.eef', 'observation.states.eef'} <= source.keys():
        evidence['mora'] = dict(status='unavailable', reason='Original source lacks actions.eef or observation.states.eef; no fabricated EEF')
    else:
        current = np.asarray(source['observation.states.eef'][0], np.float32)
        targets = np.asarray(source['actions.eef'][:50], np.float32)
        current_tensor, target_tensor = torch.from_numpy(current), torch.from_numpy(targets)
        local_raw = compute_eef_delta_action(current_tensor, target_tensor, delta_frame='local').numpy()
        mora_local, _ = matrix_blocks(local_raw)
        mora_anchor, anchor_valid = matrix_blocks(current)
        mora_world, target_valid = matrix_blocks(targets)
        geometric_mask = target_valid & anchor_valid
        mask_fields = {}
        for key in ('actions.eef.mask', 'observation.states.eef.mask'):
            if key in source:
                source_mask = source[key][:50] if key.startswith('actions') else source[key][0:1]
                for side in range(2):
                    if source_mask.shape[-1] >= side * 10 + 9:
                        geometric_mask[:, side] &= (source_mask[:, side*10:side*10+9] > 0).all(-1)
                mask_fields[key] = {'shape': list(source[key].shape), 'zero_cells': int((source[key] == 0).sum())}
        for key in ('action.eef_target_valid',):
            if key in source:
                geometric_mask &= source[key][:50, None].astype(bool)
        mora_head_raw = np.full_like(local_raw, np.nan)
        if rigid_camera:
            current_head = transform_eef_world_to_camera(current_tensor, torch.from_numpy(camera.astype(np.float32).reshape(16)))
            target_head = transform_eef_world_to_camera(target_tensor, torch.from_numpy(camera.astype(np.float32).reshape(16)))
            mora_head_raw = compute_eef_delta_action(current_head, target_head).numpy()
        mora_head, _ = matrix_blocks(mora_head_raw)
        mora_head_mask = geometric_mask & head_trusted
        mora_roundtrip = masked_error(mora_anchor[None] @ mora_local, mora_world, geometric_mask)
        mora_head_error = None
        if mora_head_mask.any():
            anchor_head = camera[None] @ mora_anchor
            restored = mora_head.copy()
            restored[..., :3, 3] += anchor_head[None, :, :3, 3]
            restored[..., :3, :3] = mora_head[..., :3, :3] @ anchor_head[None, :, :3, :3]
            mora_head_error = masked_error(restored, camera[None, None] @ mora_world, mora_head_mask)
        target_key = next((key for key in ('action.eef_target_frame_index', 'action_target_frame_index') if key in source), None)
        target_ids = source[target_key][:50].astype(np.int64) if target_key else np.full(50, -1, np.int64)
        shared, ow_indices, mora_indices = np.intersect1d(arrays['ow_target_source_frame_indices'], target_ids, return_indices=True)
        arrays.update(mora_local=mora_local, mora_head=mora_head, mora_eef_anchor=mora_anchor,
            mora_eef_world=mora_world, mora_local_mask=geometric_mask, mora_head_mask=mora_head_mask,
            mora_head_numeric_mask=geometric_mask & rigid_camera, mora_anchor_mask=anchor_valid,
            mora_raw_local20=local_raw, mora_raw_head20=mora_head_raw,
            mora_target_source_frame_indices=target_ids, mora_action_row_source_frame_indices=source['frame_index'][:50],
            shared_target_source_frame_indices=shared, shared_ow_indices=ow_indices, shared_mora_indices=mora_indices)
        shared_mask=ow_local_mask[ow_indices] & geometric_mask[mora_indices]
        separation=np.linalg.norm(ow_absolute[1:][ow_indices,:,:3,3] - mora_world[mora_indices,:,:3,3],axis=-1)
        arrays['paired_endpoint_separation_m']=separation
        arrays['paired_endpoint_mask']=shared_mask
        evidence['mora'] = dict(status='executed', endpoint='canonical actions.eef unchanged', action_rows=50,
            local_valid_pose_steps=int(geometric_mask.sum()), head_valid_pose_steps=int(mora_head_mask.sum()),
            numeric_head_available=rigid_camera, source_mask_fields=mask_fields,
            target_index_field=target_key, target_first=int(target_ids[0]) if target_key else None,
            target_last=int(target_ids[-1]) if target_key else None, action_first_row=int(source['frame_index'][0]),
            local_roundtrip_max_abs=mora_roundtrip, head_roundtrip_max_abs=mora_head_error,
            same_target_frame_count=len(shared), shared_target_ids=shared.tolist(),
            same_target_endpoint_separation_max_m=float(separation[shared_mask].max()) if shared_mask.any() else None,
            fair_pairing='Only source target-frame intersection is comparable in time; raw EEF and physical TCP remain different endpoints.')
        # Execute an EEF-only isolated 96D compose. This intentionally does not claim joint registry hydration.
        composer = ActionSpaceComposer(space={'profile': 'dual_eef_local_96d'},
            eef_mapping={'action':['actions.eef'], 'observation.state':['observation.states.eef']},
            supervision_targets=('head_eef','wrist_eef'))
        data = {'actions.eef':target_tensor, 'observation.states.eef':current_tensor,
                'source_repo_id':record['repo'], 'action_space':'eef_delta', 'action_space_supervision':'eef_only'}
        if rigid_camera:
            data['observation.states.camera.world_to_camera'] = torch.from_numpy(camera.astype(np.float32).reshape(16))
        try:
            composed = composer(data.copy())
            arrays.update(mora_composed_physical96=composed['action'].numpy(),
                          mora_composer_availability96=composed['action_availability_mask'].numpy(),
                          mora_composer_loss_mask96=composed['action_loss_mask'].numpy())
            for side, spans in enumerate(composer._dual_eef_spans()):
                local_span=spans['wrist_eef']
                np.testing.assert_allclose(composed['action'][:, local_span.start:local_span.stop].numpy(),
                    local_raw[:, side*10:side*10+9], atol=2e-6, equal_nan=True)
            evidence['mora']['composer'] = 'executed EEF-only physical 50x96; no stats, no joints; actual composer masks saved'
            if stats_root:
                width = targets.shape[-1]
                paths={'head_camera':stats_root/f'eef/head_delta_eef_{width}d/stats.json',
                       'wrist':stats_root/f'eef/local_delta_eef_{width}d/stats.json'}
                frame_stats = {frame:json.loads(path.read_text()) for frame,path in paths.items()}
                composer.frame_norm_stats=frame_stats
                normalized_composed=composer(data.copy())
                arrays['mora_composed_external_stats96']=normalized_composed['action'].numpy()
                arrays['mora_composed_external_state96']=normalized_composed['observation.state'].numpy()
                inverse_errors={}
                for label, frame, component, expected, valid in (
                    ('local', 'wrist', 'wrist_eef', mora_local, geometric_mask),
                    ('head', 'head_camera', 'head_eef', mora_head, geometric_mask & rigid_camera)):
                    decoded=np.zeros((50,20),np.float32)
                    action_std=np.asarray(frame_stats[frame]['actions.eef']['std'],np.float32)
                    for side, spans in enumerate(composer._dual_eef_spans()):
                        if side*10 >= width:
                            continue
                        span=spans[component]
                        decoded[:,side*10:side*10+9]=arrays['mora_composed_external_stats96'][:,span.start:span.stop]
                        decoded[:,side*10:side*10+3] *= action_std[side*10:side*10+3] + np.float32(1e-6)
                    decoded_pose,_=matrix_blocks(decoded)
                    if label=='head' and not rigid_camera:
                        decoded_pose[:]=np.nan
                    arrays['mora_model_'+label]=decoded_pose
                    inverse_errors[label]=masked_error(decoded_pose,expected,valid)
                evidence['mora']['external_stats']={'scope':'controlled explicit published EEF stats; NOT full registry training hydration',
                    'normalization':'actual composer action XYZ / (published std + 1e-6), no clipping; inverse uses identical float32 scale',
                    'inverse_roundtrip_max_abs':inverse_errors,
                    'files':{frame:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'eef_metadata':frame_stats[frame].get('_eef_stats_metadata'),
                        'filter_metadata':frame_stats[frame].get('_eef_stats_filter')} for frame,path in paths.items()}}
                for frame,path in paths.items():
                    (destination/f'gt_stats_{frame}.json').write_bytes(path.read_bytes())
        except Exception as error:
            evidence['mora']['composer_error']=f'{type(error).__name__}: {error}'
    np.savez_compressed(destination/'gt_trajectories.npz', **arrays)
    evidence['trajectory_npz']=f'comparison/{record["id"]}/gt_trajectories.npz'
    write(destination/'gt_evidence.json', evidence)
    return evidence


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ids', nargs='*', type=int)
    parser.add_argument('--stats-root', type=Path)
    args=parser.parse_args()
    results=[]
    for record in json.loads((REPORT/'manifest.json').read_text())['results']:
        if args.ids is not None and int(record['id'].split('_')[0]) not in args.ids:
            continue
        try:
            result=process(record,args.stats_root)
            print(record['id'],result['mora']['status'],result['ow']['local_valid_pose_steps'],flush=True)
        except Exception as error:
            result={'id':record['id'],'status':'failed','reason':f'{type(error).__name__}: {error}','traceback':traceback.format_exc()}
            print(result['id'],result['reason'],flush=True)
        results.append(result)
    passed=[item for item in results if item['status']=='ok']
    summary={'scope':'same source anchor, isolated actual GT code; not full training', 'results':results,
        'mora_configuration_limit':'current smoke YAML loader state_as_action and composer hand_rotation_unreliable_sources/action_token_excluded_sources are unsupported constructor arguments; no full training-loader execution is claimed',
        'stats_selection':'explicit --stats-root published tables by native EEF width; not automated registry hydration',
        'stats_root':str(args.stats_root) if args.stats_root else None,
        'code_sha256':{str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__),MORA/'src/data/modules/transforms/action_space.py',ROOT/'openwam/dataloader/utils/mora_relative.py')},
        'counts':{'clips':len(results),'failed':len(results)-len(passed),
            'mora_eef_executed':sum(item['mora']['status']=='executed' for item in passed),
            'mora_eef_unavailable':sum(item['mora']['status']=='unavailable' for item in passed),
            'mora_explicit_stats_composed':sum('external_stats' in item['mora'] for item in passed),
            'ow_tcp_clips':sum(item['ow']['local_valid_pose_steps']>0 for item in passed),
            'ow_trusted_head_clips':sum(item['ow']['head_valid_pose_steps']>0 for item in passed),
            'forced_odd_anchors':sum(not item['mora_anchor_sampling']['natural_global_stride2_eligible'] for item in passed)},
        'verification':{'all_saved_raw_normalization_exact':all(item['ow']['saved_raw_normalization_exact'] for item in passed),
            'max_ow_raw_difference_from_saved_float32':max([item['ow']['saved_raw_max_abs_difference'] for item in passed] or [0]),
            'max_mora_pose_roundtrip':max([item['mora'].get(key) or 0 for item in passed for key in ('local_roundtrip_max_abs','head_roundtrip_max_abs')] or [0]),
            'max_mora_inverse_normalization_roundtrip':max([value or 0 for item in passed for value in item['mora'].get('external_stats',{}).get('inverse_roundtrip_max_abs',{}).values()] or [0]),
            'max_ow_quantile_clip_translation_m':max([item['ow']['clipped_xyz_max_m'] for item in passed] or [0])}}
    write(OUT/('gt_results.json' if args.ids is None else 'gt_results_subset.json'),summary)
    if any(item['status']=='failed' for item in results):
        raise SystemExit(1)


if __name__=='__main__':
    main()
