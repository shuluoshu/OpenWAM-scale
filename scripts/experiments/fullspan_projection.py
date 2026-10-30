"""CPU geometry for 32 future actions on three *current* camera images.

Call ``prepare_projection_geometry`` once per source anchor, then reuse its
result for all checkpoints. Camera K/E, distortion and native image dimensions
belong to that anchor. No future camera, estimated transform, or FK substitute
enters the direct-action projection. Images are owned by the renderer: an image
without usable calibration must remain visible with an unavailable overlay.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from openwam.dataloader.native_cameras import resolve_native_cameras
from scripts.experiments import stage12_projection as stage12

VIEW_ROLES = ('head', 'left_wrist', 'right_wrist')
CHUNK_LENGTH = 32


def _source_end(sample):
    trace = sample['trace']; end = int(trace['anchor']) + CHUNK_LENGTH + 1
    if 'last_source' in trace:
        end = min(end, int(trace['last_source']) + 1)
    if 'instruction_end_exclusive' in sample:
        end = min(end, int(sample['instruction_end_exclusive']))
    return end


def _source_future_rows(sample):
    """One bounded read per traced shard for at most 32 future endpoint rows."""
    import pyarrow.parquet as pq
    trace = sample['trace']; start = int(trace['anchor']) + 1; end = _source_end(sample)
    paths = list(dict.fromkeys(trace.get('source_parquets') or [trace['source_parquet']]))
    if len(paths) > 16:
        raise ValueError('unexpected number of traced source shards')
    rows = {}
    if end <= start:
        return rows, paths
    endpoint_columns = {'frame_index', 'episode_index', 'canonical.frame_valid', 'source.frame_validity',
        'observation.states.hand.position', 'observation.states.hand.valid', 'observation.states.hand.confidence',
        'observation.states.finger.tip.position', 'observation.states.joints',
        'observation.states.eef', 'observation.states.eef.mask', 'states.robot_to_env_pose',
        'states.left_tcp_to_robot_pose', 'states.right_tcp_to_robot_pose',
        'raw.robot1.finger.eef_pose.alignment_valid', 'raw.robot2.finger.eef_pose.alignment_valid'}
    for path in paths:
        available = pq.ParquetFile(path).schema_arrow.names
        columns = [key for key in available if key in endpoint_columns]
        filters = [('frame_index', '>=', start), ('frame_index', '<', end)]
        if 'episode_index' in available:
            filters.append(('episode_index', '=', int(trace['episode'])))
        for row in pq.read_table(path, columns=columns, filters=filters).to_pylist():
            frame = int(row['frame_index'])
            if frame in rows:
                raise ValueError(f'ambiguous source endpoint frame {frame}')
            rows[frame] = row
    return rows, paths


def _prepare_source_gt(sample, info):
    """Independent report GT, never a replacement for a model/action mask."""
    from scripts.data.prepare_subtask_goals import recorded_tcp_transform
    from scripts.data.trajectory_projection import goal_source_tcp
    frames = int(sample['trace']['anchor']) + np.arange(1, CHUNK_LENGTH + 1)
    result = dict(world_xyz=np.full((CHUNK_LENGTH, 2, 3), np.nan), valid=np.zeros((CHUNK_LENGTH, 2), bool),
                  source_frames=frames, reason='', provenance=dict(kind='source_endpoint_annotation',
                  source_frames=frames.tolist(), end_exclusive=_source_end(sample), frame_reasons={}))
    try:
        family = sample['_eval']['family']
        transforms = recorded_tcp_transform(family, info)
        rows, paths = _source_future_rows(sample)
        result['provenance']['source_parquets'] = paths
        for index, frame in enumerate(frames):
            if frame >= _source_end(sample):
                result['provenance']['frame_reasons'][int(frame)] = 'outside task/episode interval'
                continue
            if int(frame) not in rows:
                result['provenance']['frame_reasons'][int(frame)] = 'source endpoint row unavailable'
                continue
            endpoint = goal_source_tcp(family, rows[int(frame)], tcp_from_eef=transforms)
            result['world_xyz'][index] = endpoint['tcp_world']
            result['valid'][index] = endpoint['tcp_valid']
            result['provenance']['endpoint'] = {key: value for key, value in endpoint.items()
                if key not in ('tcp_world', 'tcp_valid', 'missing_reasons')}
            if endpoint['missing_reasons']:
                result['provenance']['frame_reasons'][int(frame)] = '; '.join(endpoint['missing_reasons'])
        if not result['valid'].any():
            result['reason'] = 'source endpoint annotations unavailable or invalid'
    except (KeyError, ValueError, TypeError, OSError) as exc:
        result['reason'] = str(exc)
    return result


def _source_anchor_row(sample):
    """Read only camera columns of the one current source row, never an episode."""
    import pyarrow.parquet as pq
    trace = sample['trace']
    paths = list(dict.fromkeys(trace.get('source_parquets') or [trace['source_parquet']]))
    if len(paths) > 16:
        raise ValueError('unexpected number of traced source shards')
    rows = []
    for path in paths:
        available = pq.ParquetFile(path).schema_arrow.names
        columns = [key for key in available if key in ('frame_index', 'episode_index')
                   or ('camera' in key and not key.startswith('observation.images.'))]
        filters = [('frame_index', '=', int(trace['anchor']))]
        if 'episode_index' in available:
            filters.append(('episode_index', '=', int(trace['episode'])))
        rows.extend(pq.read_table(path, columns=columns, filters=filters).to_pylist())
    if len(rows) != 1:
        raise ValueError(f'expected one source anchor row, got {len(rows)}')
    return rows[0], paths


def _endpoint(sample, info):
    """Name the native training endpoint; never silently relabel it as contact."""
    family = sample['_eval']['family']
    result = dict(definition_id='missing_verified_tcp', label='native TCP mapping unavailable',
                  used_fields=[], transforms=[], missing_reasons=[],
                  contact_center_verified=False)
    try:
        from openwam.dataloader.native_stage2_profiles import resolve_native_profile
        profile = resolve_native_profile(family + '/' + Path(sample['trace']['source_repo']).name, info)
        rep = profile['representation']
        result['profile_id'] = profile['profile_id']
        if family in {'LightWheel', 'mecka', 'egodex', 'egoverse'}:
            result.update(definition_id='human_thumb_index_midpoint_v1', label='human thumb/index fingertip midpoint',
                          transforms=['native human fingertip midpoint'])
        elif rep.get('tcp_pose_keys'):
            result.update(definition_id='interndata_native_tcp_env_v1' if family == 'interndata_a1' else 'native_tcp_pose',
                          label='source-native TCP (physical contact center not independently verified)',
                          used_fields=list(rep['tcp_pose_keys'].values()) + [rep.get('tcp_reference_to_world_key')],
                          transforms=['native source-reference TCP transformed to world'])
        elif rep.get('tcp_from_eef'):
            result.update(definition_id='genrobot_finger_device_tcp_offset_v1' if family == 'Genrobot' else 'recorded_contact_tcp_restored_v1',
                          label='finger-device TCP; not human fingertip midpoint' if family == 'Genrobot' else 'recorded source TCP restored from EEF',
                          used_fields=[rep.get('state_key')], transforms=rep['tcp_from_eef'])
        else:
            result['missing_reasons'].append('missing verified native TCP action mapping')
    except (KeyError, ValueError, TypeError) as exc:
        result['missing_reasons'].append(str(exc))
    return result


def _wrist_camera(family, values, camera_key, info, wh, eligible):
    """Resolve this wrist only; generic camera K/E always denote the head."""
    cam = camera_key.rsplit('.', 1)[-1]
    # Genrobot's published wrist repair is a CAD-derived preview, explicitly
    # not a verified acquisition optical chain. Do not promote that candidate
    # or use goal_camera's hard-coded head camera2 double-sphere calibration.
    if family == 'Genrobot':
        return None, dict(missing_reasons=['Genrobot wrist optical chain is CAD-derived preview; verified native calibration unavailable'])
    own = {key: value for key, value in values.items()
           if key.startswith(f'observation.states.cameras.{cam}.')
           or key.startswith(f'observation.states.camera.calibration.{cam}.')}
    metadata = info.get('starvla_metadata', {})
    camera_metadata = metadata.get('coordinate_system', {}).get('cameras', {}).get(cam, {})
    candidate = metadata.get('camera_candidate_backfill', {})
    statuses = [camera_metadata.get('status')]
    if set(candidate.get('fields', {}).values()).intersection(own):
        statuses.append(candidate.get('status'))
    if 'visual_candidate_manual_selected' in statuses:
        return None, dict(missing_reasons=[
            f'{cam} calibration is visual_candidate_manual_selected; unverified optical chain'])
    for key, value in own.items():
        if key.endswith(('.valid', '.world_to_camera_valid')):
            valid = np.asarray(value, float).reshape(-1)
            if valid.size != 1 or not np.isfinite(valid).all() or valid[0] <= .5:
                return None, dict(missing_reasons=[f'{key} false or invalid'])
    return stage12.goal_camera(family, own, camera_key, info, wh, geometry_eligible=eligible)


def prepare_projection_geometry(sample):
    """Cache current cameras and the existing Stage2 native decode geometry.

    ``views`` has head/left_wrist/right_wrist entries with ``camera``,
    ``eligible``, ``image_available``, ``reason`` and native provenance. Missing
    calibration does not remove an image. ``source_gt`` caches independent
    source endpoint annotations for the next 32 rows; this report data does
    not modify action supervision or prediction availability. Returned arrays
    are CPU numpy arrays.
    """
    trace = sample['trace']; anchor = int(trace['anchor'])
    base = stage12.prepare_geometry([sample])[0]
    result = dict(anchor=anchor, base_geometry=base, endpoint={}, views={},
                  camera_policy='all 32 future world points projected into each fixed current-anchor camera')
    for role in VIEW_ROLES:
        result['views'][role] = dict(source_frame=anchor, camera_key=None, camera=None,
            eligible=False, image_available=False, reason='current image unavailable', provenance={})
    try:
        info = json.loads((Path(trace['source_repo']) / 'meta/info.json').read_text())
        family = sample['_eval']['family']
        cameras = resolve_native_cameras(family + '/' + Path(trace['source_repo']).name, info)
        result['endpoint'] = _endpoint(sample, info)
        result['source_gt'] = _prepare_source_gt(sample, info)
        qa = trace.get('calibration_eligibility', sample.get('report_target', {}).get('calibration_eligibility'))
        view_mask = sample.get('goal_view_valid')
        if view_mask is None and 'video_view_mask' in sample:
            view_mask = stage12._array(sample['video_view_mask'])[0]
        if view_mask is not None:
            view_mask = stage12._array(view_mask).astype(bool)
            if view_mask.shape != (3,):
                raise ValueError('current view mask must have shape [3]')
        # Establish image availability before reading calibration. A broken or
        # absent parquet row must not turn a decoded wrist image into black.
        for index, (role, key) in enumerate(zip(VIEW_ROLES, cameras)):
            view = result['views'][role]; view['camera_key'] = key
            alignment = trace.get('video_alignment', {}).get(key, {}).get('alignment', {})
            current = alignment.get(anchor, alignment.get(str(anchor), {}))
            view['image_available'] = key is not None and (bool(view_mask[index]) if view_mask is not None else bool(current))
        values, paths = _source_anchor_row(sample)
        for index, (role, key) in enumerate(zip(VIEW_ROLES, cameras)):
            view = result['views'][role]; view['camera_key'] = key
            if key is None:
                continue
            alignment = trace.get('video_alignment', {}).get(key, {}).get('alignment', {})
            current = alignment.get(anchor, alignment.get(str(anchor), {}))
            # Existence is separate from projection eligibility, including when
            # the alignment/calibration audit below rejects a visible image.
            view['image_available'] = bool(view_mask[index]) if view_mask is not None else bool(current)
            view['provenance'] = dict(video_alignment=current, source_parquets=paths, calibration_eligibility=qa)
            try:
                if not view['image_available']:
                    raise ValueError('current image unavailable or source view masked')
                if int(current.get('source_frame_index', -1)) != anchor:
                    raise ValueError('missing/mismatched current decoded source frame identity')
                wh = current.get('native_wh')
                if wh is None:
                    raise ValueError('missing current decoded dimensions')
                qa_key = 'strict_head_camera_eef_eligible' if index == 0 else 'strict_wrist_camera_eef_eligible'
                eligible = qa is None or bool(qa.get(qa_key, False))
                if index == 0:
                    if not eligible:
                        base = dict(base, camera=None, eligible=False, reason='source camera geometry QA rejected or unavailable')
                        result['base_geometry'] = base
                    if not base['eligible'] or base['camera'] is None:
                        raise ValueError(base.get('reason') or 'head camera unavailable')
                    if tuple(wh) != tuple(base['camera']['wh']):
                        raise ValueError('calibration/video dimensions mismatch')
                    view['camera'] = base['camera']
                    view['provenance']['camera'] = base.get('provenance', {}).get('camera', {})
                else:
                    camera, provenance = _wrist_camera(family, values, key, info, wh, eligible)
                    view['provenance']['camera'] = provenance
                    if camera is None:
                        raise ValueError('; '.join(provenance['missing_reasons']))
                    k, e, model, d, native_wh = camera
                    view['camera'] = dict(K=k, E=e, model=model, D=d, wh=tuple(native_wh))
                view.update(eligible=True, reason='')
            except (KeyError, ValueError, TypeError) as exc:
                view.update(camera=None, eligible=False, reason=str(exc))
    except (KeyError, ValueError, TypeError, OSError) as exc:
        for view in result['views'].values():
            view.update(camera=None, eligible=False, reason=str(exc))
    return result


def _pad(array, fill):
    array = np.asarray(array)
    out = np.full((CHUNK_LENGTH, *array.shape[1:]), fill, dtype=array.dtype)
    out[:len(array)] = array
    return out


def _merge_source_gt(source, annotation, sample):
    """Prefer consistent native GT; annotate fallback/conflict without changing predictions."""
    result = dict(source)
    for key in ('world_xyz', 'gt_world_xyz', 'valid', 'gt_valid', 'metric_valid', 'metric_gt_valid'):
        value = np.asarray(source[key])
        result[key] = _pad(value, False if value.dtype == bool else np.nan)
    native = result['gt_world_xyz'].copy(); native_valid = result['gt_valid'].copy()
    gt_source = np.full((CHUNK_LENGTH, 2), 'unavailable', dtype='<U32')
    gt_source[native_valid] = 'native_action_supervision'
    disagreement = np.zeros_like(native_valid)
    if annotation is not None:
        xyz = np.asarray(annotation['world_xyz']); valid = np.asarray(annotation['valid'], bool)
        if xyz.shape != (CHUNK_LENGTH, 2, 3) or valid.shape != (CHUNK_LENGTH, 2):
            raise ValueError('source GT must have shape [32,2,3] with [32,2] validity')
        valid = valid & np.isfinite(xyz).all(-1)
        disagreement = valid & native_valid & (np.linalg.norm(native - xyz, axis=-1) > 1e-4)
        fallback = valid & (~native_valid | disagreement)
        result['gt_world_xyz'][fallback] = xyz[fallback]
        result['gt_valid'] |= fallback
        gt_source[fallback] = 'source_endpoint_annotation'
        # Source-only annotations are visual GT, not native action supervision.
        # Disagreement must not silently change the endpoint used for metrics.
        result['metric_valid'] &= ~disagreement
        result['metric_gt_valid'] &= ~disagreement
    allowed = (int(sample['trace']['anchor']) + np.arange(1, CHUNK_LENGTH + 1)) < _source_end(sample)
    result['gt_valid'] &= allowed[:, None]
    result['metric_valid'] &= allowed[:, None]
    result['metric_gt_valid'] &= allowed[:, None]
    result['gt_world_xyz'][~result['gt_valid']] = np.nan
    gt_source[~result['gt_valid']] = 'unavailable'
    result.update(gt_source=gt_source, gt_native_disagreement=disagreement,
                  native_gt_world_xyz=native, native_gt_valid=native_valid)
    return result


def _project_branch(source, camera):
    pred = source['world_xyz']; gt = source['gt_world_xyz']
    pixels, uv, projected, inside = stage12._project(pred, source['valid'], camera)
    gt_pixels, gt_uv, gt_projected, gt_inside = stage12._project(gt, source['gt_valid'], camera)
    metric = source['metric_valid']
    pixel_valid = metric & projected & gt_inside
    eligible_gt = source['metric_gt_valid'] & gt_inside
    masks = dict(eligible_gt=eligible_gt, invalid_pred=eligible_gt & ~projected,
                 outside_pred=eligible_gt & projected & ~inside, scored=pixel_valid)
    arrays = dict(world_xyz=pred, gt_world_xyz=gt, valid=source['valid'], gt_valid=source['gt_valid'],
        metric_valid=metric, metric_gt_valid=source['metric_gt_valid'], pred_uv=uv * [319., 255.],
        gt_uv=gt_uv * [319., 255.], pred_uv_normalized=uv, gt_uv_normalized=gt_uv,
        uv_pixels=pixels, gt_uv_pixels=gt_pixels, pred_projectable=projected, pred_in_frame=inside,
        gt_projection_valid=gt_projected, gt_in_frame=gt_inside, pixel_error_valid=pixel_valid,
        pixel_error=np.where(pixel_valid, np.linalg.norm((uv - gt_uv) * [319., 255.], axis=-1), np.nan))
    result = {key: _pad(value, False if value.dtype == bool else np.nan) for key, value in arrays.items()}
    for key in ('gt_source', 'gt_native_disagreement', 'native_gt_world_xyz', 'native_gt_valid'):
        if key in source:
            result[key] = source[key].copy()
    result.update(label=source['label'], counts={key: int(value.sum()) for key, value in masks.items()})
    return result


def project_action_chunks(sample, normalized_pred, geometry):
    """Return [32,left/right,xy] tile pixels and independent masks per view.

    ``views[role]['direct']`` and ``['fk']`` are always separate. ``pred_uv``
    and ``gt_uv`` use a 320x256 tile with pixel-center resize; normalized and
    native coordinates are also retained. Offscreen predictions are not clipped.
    Short end-of-episode tails are NaN/false padded. Invalid predictions do not
    erase valid GT. ``image_available`` is independent of ``projection_available``.
    ``gt_source`` distinguishes native action supervision from source endpoint
    annotations. Source-only GT is visible but does not enable native metrics;
    native/source disagreements above 0.1 mm are explicitly excluded from them.
    """
    anchor = int(sample['trace']['anchor'])
    if int(geometry['anchor']) != anchor:
        raise ValueError('projection geometry anchor does not match sample anchor')
    prediction = stage12._array(normalized_pred)
    if prediction.shape != (CHUNK_LENGTH, 112):
        raise ValueError('normalized predictions must have shape [32,112]')
    decoded = stage12.trajectories(prediction, sample, geometry['base_geometry'])
    annotation = geometry.get('source_gt')
    for branch in ('direct', 'fk'):
        decoded[branch] = _merge_source_gt(decoded[branch], annotation, sample)
    result = dict(anchor=anchor, source_frames=anchor + np.arange(1, CHUNK_LENGTH + 1),
                  endpoint=geometry.get('endpoint', {}), views={},
                  source_gt_provenance=(annotation or {}).get('provenance', {}),
                  source_gt_reason=(annotation or {}).get('reason', ''),
                  gt_native_consistency_atol_m=1e-4,
                  camera_policy=geometry.get('camera_policy', 'fixed current-anchor cameras'),
                  direct_reason=decoded['reason'], fk_label=decoded['fk_label'])
    for role in VIEW_ROLES:
        view = geometry['views'][role]
        available = bool(view['eligible'] and view['image_available'] and view['camera'] is not None)
        camera = view['camera'] if available else None
        result['views'][role] = dict(image_available=bool(view['image_available']),
            projection_available=available, reason=view['reason'], camera_key=view.get('camera_key'),
            source_frame=anchor, provenance=view.get('provenance', {}),
            direct=_project_branch(decoded['direct'], camera), fk=_project_branch(decoded['fk'], camera))
    return result
