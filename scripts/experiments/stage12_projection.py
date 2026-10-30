"""CPU report geometry, using the observed camera and explicit source masks.

FK world coordinates use source-anchor registration; not independent whole-body
calibration. Only the current source row is read to register the arm bases.
No future pose, predicted trajectory, or per-frame fit enters that registration.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from openwam.model.fk_supervision import torch_fk
from scripts.data.prepare_goal_labels import goal_camera
from scripts.data.trajectory_projection import project_points, transform_points
from scripts.data.urdf_fk_audit import pose7_to_matrix


FK_LABEL = 'source-anchor registration; not independent whole-body calibration'
SCATTER = np.r_[0:10, 34:44, 80:94, 94:112]


def _array(value):
    return value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _anchor_row(sample):
    """Read selected scalar/geometry columns of one traced source row only."""
    import pyarrow.parquet as pq
    trace = sample['trace']; rows = []
    paths = list(dict.fromkeys(trace.get('source_parquets', [trace['source_parquet']])))
    if len(paths) > 16:
        raise ValueError('unexpected number of traced source shards')
    for path in paths:
        available = pq.ParquetFile(path).schema_arrow.names
        columns = [k for k in available if k in ('frame_index', 'episode_index', 'states.robot_to_env_pose')
                   or ('camera' in k and any(p in k for p in ('intrinsic', 'world_to_camera', 'valid', 'camera_info.K', 'camera_info.D')))
                   or (k.startswith('states.') and ('_ee_to_' in k or '_tcp_to_' in k))]
        filters = [('frame_index', '=', int(trace['anchor']))]
        if 'episode_index' in available:
            filters.append(('episode_index', '=', int(trace['episode'])))
        rows.extend(pq.read_table(path, columns=columns, filters=filters).to_pylist())
    if len(rows) != 1:
        raise ValueError(f'expected one source anchor row, got {len(rows)}')
    return rows[0], paths


def prepare_geometry(samples):
    """Return one geometry dict per sample; unavailable calibration stays masked.

    ``camera`` is {K,E,model,D,wh} in native source pixels. ``fk_anchor_world``
    contains two world-from-URDF-base matrices, or None without FK payload.
    Eligibility, failure reasons and exact source provenance accompany both.
    """
    output = []
    for sample in samples:
        trace = sample['trace']
        g = dict(camera=None, eligible=False, reason='', fk_anchor_world=None,
                 fk_eligible=np.zeros(2, bool), fk_reason=['no matched FK payload'] * 2,
                 provenance=dict(anchor=int(trace['anchor']), episode=int(trace.get('episode', -1)),
                                 camera_policy='fixed observed-frame camera and background', fk_label=FK_LABEL))
        try:
            values, paths = _anchor_row(sample)
            root = Path(trace['source_repo']); info = json.loads((root / 'meta/info.json').read_text())
            from openwam.dataloader.native_cameras import HEAD_CAMERAS
            family = sample['_eval']['family']
            head = HEAD_CAMERAS.get(family) or next(iter(trace['video_alignment']))
            alignment = trace.get('video_alignment', {}).get(head, {}).get('alignment', {})
            current = alignment.get(int(trace['anchor']), alignment.get(str(trace['anchor']), {}))
            wh = current.get('native_wh')
            if wh is None:
                raise ValueError('missing observed-frame decoded dimensions')
            qa = sample.get('report_target', {}).get('calibration_eligibility')
            eligible = qa is None or bool(qa.get('strict_head_camera_eef_eligible', False))
            camera, camera_provenance = goal_camera(family, values, head, info, wh, geometry_eligible=eligible)
            g['provenance'].update(source_parquets=paths, camera=camera_provenance, calibration_eligibility=qa)
            if camera is not None:
                k, e, model, d, wh = camera
                state = _array(trace['raw_state'])[0]
                state_valid = _array(trace['raw_state_mask'])[0].astype(bool)
                anchor_audit = {}
                for side, world_start, head_start in (('left', 0, 34), ('right', 10, 43)):
                    world_slots = slice(world_start, world_start + 3)
                    head_slots = slice(head_start, head_start + 3)
                    available = state_valid[world_slots].all() and state_valid[head_slots].all()
                    if not available:
                        anchor_audit[side] = dict(checked=False, reason='native anchor masked')
                        continue
                    expected_head = transform_points(e, state[world_slots])
                    error = np.abs(expected_head - state[head_slots])
                    anchor_audit[side] = dict(checked=True, max_error_m=float(error.max()), atol_m=1e-4)
                    if not np.isfinite(error).all() or not np.allclose(expected_head, state[head_slots], atol=1e-4, rtol=0):
                        g['provenance']['native_anchor_audit'] = anchor_audit
                        raise ValueError(f'native head anchor mismatch ({side}): camera world/head mapping exceeds 1e-4 m')
                g['provenance']['native_anchor_audit'] = anchor_audit
                g.update(camera=dict(K=k, E=e, model=model, D=d, wh=tuple(wh)), eligible=True)
            else:
                g['reason'] = '; '.join(camera_provenance['missing_reasons'])
            payload = sample.get('fk_supervision')
            if payload:
                mapping = np.full((2, 4, 4), np.nan)
                g['fk_anchor_world'] = mapping
                for arm in payload['arms']:
                    side_id = 0 if arm['slots'][0] == 80 else 1
                    side = ('left', 'right')[side_id]
                    try:
                        chain = arm['chain']
                        # This registration is audited only for the Genie1 native frames.
                        if (family != 'interndata_a1' or chain['base_link'] != 'arm_base_link'
                                or chain['target_link'] != f'arm_{side[0]}_end_link' or arm['endpoint'] != 'tcp'):
                            raise ValueError('FK world registration unsupported for this chain/endpoint')
                        if hashlib.sha256(Path(chain['urdf_path']).read_bytes()).hexdigest() != chain['urdf_sha256']:
                            raise ValueError('matched URDF hash changed')
                        def pose(key):
                            return pose7_to_matrix(np.asarray(values[key], float)[None])[0]
                        world_robot = pose('states.robot_to_env_pose')
                        robot_ee = pose(f'states.{side}_ee_to_robot_pose')
                        base_ee = pose(f'states.{side}_ee_to_{side}_armbase_pose')
                        base_tcp = pose(f'states.{side}_tcp_to_{side}_armbase_pose')
                        anchor_fk = torch_fk(torch.as_tensor(arm['q_anchor'], dtype=torch.float64), chain).numpy()
                        if not np.allclose(anchor_fk, base_tcp, atol=3e-6, rtol=0):
                            raise ValueError('native TCP and matched URDF endpoint disagree at anchor')
                        mapping[side_id] = world_robot @ robot_ee @ np.linalg.inv(base_ee)
                        if not np.isfinite(mapping[side_id]).all():
                            raise ValueError('nonfinite FK anchor mapping')
                        g['fk_eligible'][side_id] = True; g['fk_reason'][side_id] = ''
                        g['provenance'][f'fk_{side}'] = dict(urdf_path=chain['urdf_path'],
                            urdf_sha256=chain['urdf_sha256'], base_link=chain['base_link'], target_link=chain['target_link'],
                            endpoint_transform=chain['tool'], source_frame=int(trace['anchor']),
                            formula='T_env_robot(anchor) @ T_robot_EE(anchor) @ inv(T_armbase_EE(anchor)) @ FK(q)',
                            gt_key=arm['gt_pose_key'])
                    except (KeyError, ValueError, OSError) as exc:
                        g['fk_reason'][side_id] = str(exc)
        except (KeyError, ValueError, OSError) as exc:
            g['reason'] = str(exc)
        output.append(g)
    return output


def _project(xyz, valid, camera):
    n = xyz.shape[:-1]
    if camera is None:
        return np.full((*n, 2), np.nan), np.full((*n, 2), np.nan), np.zeros(n, bool), np.zeros(n, bool)
    pixels, projected = project_points(transform_points(camera['E'], xyz), camera['K'],
                                       model=camera['model'], distortion=camera['D'])
    projected &= valid
    pixels = np.where(projected[..., None], pixels, np.nan)
    width, height = camera['wh']
    scaled = (pixels + .5) * [320. / width, 256. / height] - .5
    inside = projected & (pixels[..., 0] >= 0) & (pixels[..., 0] <= width - 1) & (pixels[..., 1] >= 0) & (pixels[..., 1] <= height - 1)
    inside &= (scaled[..., 0] >= 0) & (scaled[..., 0] <= 319) & (scaled[..., 1] >= 0) & (scaled[..., 1] <= 255)
    # Keep outside coordinates for inspection; the independent mask controls drawing.
    return pixels, scaled / [319., 255.], projected, inside


def trajectories(predicted, sample, geometry):
    """Decode normalized [T,112] actions into direct and FK trajectories.

    Arrays use [time,left/right,xyz/uv]; time k means anchor+k+1. ``uv_pixels``
    are native pixels; ``pred_uv`` and ``gt_uv`` are 320x256 tile pixels. ``valid`` is
    physical availability; ``metric_valid`` additionally honors FK clipping
    exclusions. Pixel errors require in-frame GT and a projectable prediction;
    finite offscreen predictions remain scored. Drawing uses ``pred_in_frame``.
    This function never fits a coordinate transform to predictions/GT.
    """
    pred = _array(predicted).astype(float)
    raw = _array(sample['trace']['raw_action']).astype(float)
    if pred.ndim != 2 or pred.shape[1] != 112:
        raise ValueError('predictions must have shape [T,112]')
    n = min(len(pred), len(raw)); pred = pred[:n]; raw = raw[:n]
    state = _array(sample['trace']['raw_state'])[0]
    state_valid = _array(sample['trace']['raw_state_mask'])[0].astype(bool)
    raw_valid = _array(sample['trace']['raw_action_mask'])[:n].astype(bool)
    action_valid = _array(sample['action_mask'])[:n].astype(bool)
    stats = sample['_eval']['action_stats']; lo = np.asarray(stats['q01']); hi = np.asarray(stats['q99'])
    physical = lo + (pred[:, SCATTER] + 1.) * np.maximum(hi - lo, 1e-6) / 2.
    mask = raw_valid & action_valid[:, SCATTER] & (np.asarray(stats['count']) > 0)[None]
    camera = geometry['camera']; direct = np.full((n, 2, 3), np.nan); gt = np.full_like(direct, np.nan)
    valid = np.zeros((n, 2), bool); gt_valid = valid.copy()
    if camera is not None and geometry['eligible']:
        inverse = np.linalg.inv(camera['E'])
        for side, start in enumerate((34, 43)):
            slots = slice(start, start + 3)
            base_ok = state_valid[slots].all() and np.isfinite(state[slots]).all()
            gt_valid[:, side] = mask[:, slots].all(-1) & base_ok & np.isfinite(raw[:, slots]).all(-1)
            valid[:, side] = gt_valid[:, side] & np.isfinite(physical[:, slots]).all(-1)
            direct[:, side] = transform_points(inverse, state[slots] + physical[:, slots])
            gt[:, side] = transform_points(inverse, state[slots] + raw[:, slots])
    direct = np.where(valid[..., None], direct, np.nan); gt = np.where(gt_valid[..., None], gt, np.nan)
    fk = np.full_like(direct, np.nan); fk_valid = np.zeros_like(valid); fk_metric = np.zeros_like(valid)
    fk_gt_metric = np.zeros_like(valid)
    mapping = geometry.get('fk_anchor_world')
    for arm in (sample.get('fk_supervision') or {}).get('arms', []):
        side = 0 if arm['slots'][0] == 80 else 1
        if mapping is None or not geometry.get('fk_eligible', [False, False])[side]:
            continue
        slots = arm['slots']; q = np.asarray(arm['q_anchor']) + pred[:, slots] * arm['scale'] + arm['offset']
        good = action_valid[:, slots].all(-1) & np.isfinite(q).all(-1)
        if good.any():
            pose = torch_fk(torch.as_tensor(q[good], dtype=torch.float64), arm['chain']).numpy()
            fk[good, side] = (mapping[side] @ pose)[:, :3, 3]
        fk_valid[:, side] = good
        fk_gt_metric[:, side] = action_valid[:, slots].all(-1) & np.asarray(arm['valid'][:n], bool) & gt_valid[:, side]
        fk_metric[:, side] = good & fk_gt_metric[:, side]
    result = dict(source_frames=int(sample['trace']['anchor']) + np.arange(1, n + 1), fk_label=FK_LABEL,
                  reason=geometry.get('reason', ''), coverage=dict(camera=bool(geometry.get('eligible')),
                  fk=geometry.get('fk_eligible', np.zeros(2, bool)), direct_points=int(gt_valid.sum()),
                  fk_metric_points=int(fk_metric.sum())))
    if not result['reason'] and not gt_valid.any():
        result['reason'] = 'native head-frame action/state trajectory masked or unavailable'
    gt_pixels, gt_uv, gt_projected, gt_inside = _project(gt, gt_valid, camera)
    for name, xyz, available, metric, eligible in [
            ('direct', direct, valid, valid & gt_valid, gt_valid), ('fk', fk, fk_valid, fk_metric, fk_gt_metric)]:
        pixels, uv, projected, inside = _project(xyz, available, camera)
        pixel_valid = metric & projected & gt_inside
        eligible_gt = eligible & gt_inside
        count_masks = dict(eligible_gt=eligible_gt, invalid_pred=eligible_gt & ~projected,
                           outside_pred=eligible_gt & projected & ~inside, scored=pixel_valid)
        tile_pixels = uv * [319., 255.]; gt_tile_pixels = gt_uv * [319., 255.]
        result[name] = dict(world_xyz=xyz, gt_world_xyz=gt.copy(), pred_xyz=xyz, gt_xyz=gt.copy(),
            valid=available, gt_valid=gt_valid.copy(),
            metric_valid=metric, metric_gt_valid=eligible, uv_pixels=pixels, uv=uv, projection_valid=projected, in_frame=inside,
            pred_uv=tile_pixels, gt_uv=gt_tile_pixels, pred_projectable=projected, pred_in_frame=inside,
            gt_uv_pixels=gt_pixels.copy(), gt_uv_normalized=gt_uv.copy(), gt_projection_valid=gt_projected.copy(), gt_in_frame=gt_inside.copy(),
            position_error_m=np.where(metric, np.linalg.norm(xyz - gt, axis=-1), np.nan),
            pixel_error_valid=pixel_valid,
            counts={key:int(value.sum()) for key,value in count_masks.items()},
            counts_by_side={key:value.sum(0) for key,value in count_masks.items()},
            pixel_error=np.where(pixel_valid, np.linalg.norm(tile_pixels - gt_tile_pixels, axis=-1), np.nan),
            native_pixel_error=np.where(pixel_valid, np.linalg.norm(pixels - gt_pixels, axis=-1), np.nan),
            label=FK_LABEL if name == 'fk' else 'native head-frame TCP delta; fixed observed camera')
    return result


def _video_source_rows(sample, source_frames):
    """Bounded projection: at most nine unique rows and sixteen traced shards."""
    import pyarrow.parquet as pq
    trace=sample['trace']; paths=list(dict.fromkeys(trace.get('source_parquets') or [trace['source_parquet']]))
    frames=sorted(set(map(int,source_frames)))
    if len(paths)>16 or len(frames)>9: raise ValueError('unexpected video geometry read size')
    rows={}
    for path in paths:
        available=pq.ParquetFile(path).schema_arrow.names
        columns=[k for k in available if k in ('frame_index','episode_index')
                 or ('camera' in k and any(x in k for x in ('intrinsic','world_to_camera','valid','camera_info.K','camera_info.D')))]
        filters=[('frame_index','in',frames)]
        if 'episode_index' in available:filters.append(('episode_index','=',int(trace['episode'])))
        for row in pq.read_table(path,columns=columns,filters=filters).to_pylist():
            frame=int(row['frame_index'])
            if frame in rows:raise ValueError(f'ambiguous source video frame {frame}')
            rows[frame]=row
    return rows,paths


def prepare_video_geometry(sample):
    """Read each real video frame's camera; never substitute the anchor camera.

    The fixed nine-frame contract is anchor+[0,4,...,32]. Invalid/padded images
    remain masked. ``action_indices`` is [-1,3,...,31]; -1 is observed GT only.
    Camera coordinates use source-row K/E/model/distortion and exact decoded
    dimensions from the matching video alignment, preserving pixel centers.
    """
    trace=sample['trace'];anchor=int(trace['anchor']);expected=anchor+np.arange(9)*4
    indices=np.arange(9)*4-1
    output=dict(source_frames=expected.copy(),requested_source_frames=expected.copy(),action_indices=indices,
                frames=[],anchor_gt_xyz=np.full((2,3),np.nan),anchor_gt_valid=np.zeros(2,bool),reason='',
                provenance=dict(camera_policy='same source-frame camera for moving video',
                                action_alignment='frame j>0 uses action[4*j-1]; frame0 observed GT only'))
    output['frames']=[dict(source_frame=int(f),action_index=int(k),camera=None,eligible=False,video_valid=False,reason='unprepared')
                      for f,k in zip(expected,indices)]
    try:
        actual=np.asarray(trace['video_source_indices'],int)
        if actual.shape!=(9,):raise ValueError('video requires exactly nine traced source frames')
        output['source_frames']=actual
        for frame,source in zip(output['frames'],actual):frame['source_frame']=int(source)
        video_valid=_array(sample['video_mask']).astype(bool)
        if video_valid.shape!=(9,):raise ValueError('video mask must have nine entries')
        views=_array(sample.get('video_view_mask',np.ones((9,3),bool))).astype(bool)
        if views.shape!=(9,3):raise ValueError('video view mask must be [9,3]')
        video_valid &= views[:,0]
        root=Path(trace['source_repo']);info=json.loads((root/'meta/info.json').read_text())
        from openwam.dataloader.native_cameras import HEAD_CAMERAS
        family=sample['_eval']['family'];head=HEAD_CAMERAS.get(family) or next(iter(trace.get('video_alignment',{})),None)
        if head is None:raise ValueError('missing source head-camera identity')
        alignments=trace.get('video_alignment',{}).get(head,{}).get('alignment',{})
        rows,paths=_video_source_rows(sample,actual)
        qa=trace.get('calibration_eligibility')
        if qa is None:qa=sample.get('report_target',{}).get('calibration_eligibility')
        eligible=qa is None or bool(qa.get('strict_head_camera_eef_eligible',False))
        output['provenance'].update(head=head,source_parquets=paths,calibration_eligibility=qa)
        anchor_geometry=prepare_geometry([sample])[0]
        output['provenance']['native_anchor_audit']=anchor_geometry['provenance'].get('native_anchor_audit',{})
        output['provenance']['anchor_reference_reason']=anchor_geometry['reason']
        if anchor_geometry['eligible']:
            state=_array(trace['raw_state'])[0];mask=_array(trace['raw_state_mask'])[0].astype(bool)
            for side,start in enumerate((0,10)):
                xyz=state[start:start+3]
                output['anchor_gt_xyz'][side]=xyz
                output['anchor_gt_valid'][side]=mask[start:start+3].all() and np.isfinite(xyz).all()
        for j,frame in enumerate(output['frames']):
            try:
                if not video_valid[j]:raise ValueError('source video/head view masked or padded')
                frame['video_valid']=True
                if actual[j]!=expected[j]:raise ValueError('video/action source frame alignment mismatch')
                alignment=alignments.get(int(actual[j]),alignments.get(str(actual[j]),{}))
                if int(alignment.get('source_frame_index',-1))!=int(actual[j]):raise ValueError('missing/mismatched decoded source frame identity')
                if 'native_wh' not in alignment:raise ValueError('missing same-frame decoded dimensions')
                if int(actual[j]) not in rows:raise ValueError('same-frame source row unavailable')
                camera,provenance=goal_camera(family,rows[int(actual[j])],head,info,alignment['native_wh'],geometry_eligible=eligible)
                frame['provenance']=dict(camera=provenance,video_alignment=alignment)
                if camera is None:raise ValueError('; '.join(provenance['missing_reasons']))
                k,e,model,d,wh=camera
                frame.update(camera=dict(K=k,E=e,model=model,D=d,wh=tuple(wh)),eligible=True,reason='')
            except (KeyError,ValueError,TypeError) as exc:frame['reason']=str(exc)
        output['reason']='; '.join(dict.fromkeys(f['reason'] for f in output['frames'] if f['reason']))
    except (KeyError,ValueError,TypeError,OSError) as exc:
        output['reason']=str(exc)
        for frame in output['frames']:frame.update(camera=None,eligible=False,reason=str(exc))
    return output


def video_trajectories(fixed_trajectory,sample,video_geometry):
    """Project decoded world trajectories on nine moving-camera video frames.

    ``pred_uv``/``gt_uv`` are 320x256 tile pixels, never clamped. Frame zero has
    observed GT and no prediction/metric. Every future frame uses its own K/E/
    distortion; unavailable calibration stays masked with a per-frame reason.
    Generated-video overlays use the matched real source-camera trajectory as
    their reference, not camera poses estimated from the generated images.
    """
    vg=video_geometry;frames=vg['frames'];n=len(frames)
    if n!=9:raise ValueError('video geometry must contain nine frames')
    source_frames=np.asarray(vg['source_frames'],int);action_indices=np.asarray(vg['action_indices'],int)
    if action_indices.tolist()!=[-1,3,7,11,15,19,23,27,31]:raise ValueError('unsupported video action alignment')
    result=dict(source_frames=source_frames.copy(),action_indices=action_indices.copy(),
                reason='; '.join(dict.fromkeys(x for x in (vg.get('reason',''),fixed_trajectory.get('reason','')) if x)),
                frame_reasons=[f['reason'] for f in frames],
                camera_eligible=np.array([f['eligible'] for f in frames],bool),
                video_valid=np.array([f['video_valid'] for f in frames],bool),
                reference='real same-frame source camera; generated image camera is not independently estimated')
    anchor=int(sample['trace']['anchor']);fixed_frames=np.asarray(fixed_trajectory['source_frames'],int)
    for name in ('direct','fk'):
        source=fixed_trajectory[name];pred=np.full((9,2,3),np.nan);gt=np.full_like(pred,np.nan)
        valid=np.zeros((9,2),bool);gt_valid=valid.copy();metric=valid.copy();eligible=valid.copy()
        for j,(k,frame) in enumerate(zip(action_indices,frames)):
            if not frame['eligible'] or frame['camera'] is None or not frame['video_valid']:continue
            if j==0:
                gt[0]=vg['anchor_gt_xyz'];gt_valid[0]=vg['anchor_gt_valid']
                if name=='fk':gt_valid[0]&=np.asarray(fixed_trajectory['coverage']['fk'],bool)
                continue
            if k>=len(fixed_frames):
                result['frame_reasons'][j]='action trajectory shorter than selected video frame';continue
            if fixed_frames[k]!=source_frames[j] or fixed_frames[k]!=anchor+4*j:
                raise ValueError(f'action/source frame alignment mismatch at video frame {j}')
            pred[j]=source['pred_xyz'][k];gt[j]=source['gt_xyz'][k]
            valid[j]=source['valid'][k]
            gt_valid[j]=source['gt_valid'][k]
            eligible[j]=source.get('metric_gt_valid',source['metric_valid'] if name=='fk' else source['gt_valid'])[k]
            if name=='fk':gt_valid[j]&=eligible[j]
            metric[j]=source['metric_valid'][k] &gt_valid[j]
        pred_uv=np.full((9,2,2),np.nan);gt_uv=np.full_like(pred_uv,np.nan)
        projected=np.zeros((9,2),bool);inside=projected.copy();gt_projected=projected.copy();gt_inside=projected.copy()
        for j,frame in enumerate(frames):
            camera=frame['camera'] if frame['eligible'] and frame['video_valid'] else None
            _,uv,projected[j],inside[j]=_project(pred[j],valid[j],camera);pred_uv[j]=uv*[319,255]
            _,uv,gt_projected[j],gt_inside[j]=_project(gt[j],gt_valid[j],camera);gt_uv[j]=uv*[319,255]
        pixel_valid=metric &projected &gt_inside
        eligible_gt=eligible &gt_inside
        masks=dict(eligible_gt=eligible_gt,invalid_pred=eligible_gt &~projected,
                   outside_pred=eligible_gt &projected &~inside,scored=pixel_valid)
        result[name]=dict(pred_xyz=pred,gt_xyz=gt,world_xyz=pred,gt_world_xyz=gt,valid=valid,gt_valid=gt_valid,
            metric_valid=metric,metric_gt_valid=eligible,pred_uv=pred_uv,gt_uv=gt_uv,
            pred_projectable=projected,pred_in_frame=inside,gt_projection_valid=gt_projected,gt_in_frame=gt_inside,
            pixel_error_valid=pixel_valid,pixel_error=np.where(pixel_valid,np.linalg.norm(pred_uv-gt_uv,axis=-1),np.nan),
            position_error_m=np.where(metric,np.linalg.norm(pred-gt,axis=-1),np.nan),
            counts={key:int(mask.sum()) for key,mask in masks.items()},
            counts_by_side={key:mask.sum(0) for key,mask in masks.items()},
            observed_gt_count=int(gt_inside[0].sum()),
            reason=('no matched/eligible FK payload' if name=='fk' and not np.asarray(fixed_trajectory['coverage']['fk']).any()
                    else result['reason'] if not eligible.any() else ''),
            label=(FK_LABEL if name=='fk' else 'native head-frame TCP delta')+'; same-frame source camera')
    return result
