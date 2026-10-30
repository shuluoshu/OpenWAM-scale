#!/usr/bin/env python3
"""Build subtask-end candidates and exact pre-VAE review assets; never train."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from openwam.dataloader.subtask_goal import FORMAT,contiguous_subtasks,SubtaskGoalDataset
from openwam.model.goal.heads import point_mask,target_uv_pixels
from openwam.train.goal_overfit import _display_image
from openwam.train.goal_trainer import overlay_goal_targets
from scripts.data.prepare_goal_labels import read_episode,task_text,goal_camera
from scripts.data.trajectory_projection import goal_source_tcp,project_goal_tcp


def clean_json(value):
    if isinstance(value,np.ndarray):return clean_json(value.tolist())
    if isinstance(value,np.generic):return clean_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean_json(v) for v in value]
    return value


def dump(path,value):path.write_text(json.dumps(clean_json(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recorded_tcp_transform(family,info):
    """Invert only this source's recorded contact->wrist rectification."""
    if family not in ('abc_130k','robochallenge_v2'):return None
    histories=info.get('starvla_metadata',{}).get('rectification_history',[])
    matches=[h for h in histories if h.get('schema_version')=='eef_tcp_to_wrist_0804_v1']
    if not matches:return None
    if len(matches)!=1:raise ValueError('ambiguous TCP rectification history')
    basis=np.eye(4)
    if family=='abc_130k' and any(h.get('schema_version')=='abc_130k_eef_axis_x_forward_z_up_v1' for h in histories):
        basis[:3,:3]=np.eye(3)[:,[2,0,1]]
    transforms={}
    for record in matches[0]['transforms']:
        side=record['arm']
        if side not in ('left','right') or side in transforms:raise ValueError('invalid rectified arm')
        source=np.asarray(record['T_source_wrist_row_major'],float).reshape(4,4)
        if not np.isfinite(source).all() or not np.allclose(source[3],[0,0,0,1]):raise ValueError('invalid source TCP transform')
        transforms[side]=(np.linalg.inv(basis)@np.linalg.inv(source)).tolist()
    if set(transforms)!= {'left','right'}:raise ValueError('incomplete TCP transform history')
    return transforms


def camera_eligibility(family,root,episode):
    if family!='abc_130k':return True,None
    path=root/'meta/camera_geometry_eligibility.jsonl'
    records=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    record=next((r for r in records if r.get('episode_index')==episode),None)
    return bool(record and record.get('broadcast_qa_status')=='passed'),record


def decode_frames(path,offset_s,fps,indices,*,sequential=False,video_ordinals=None):
    """Exact CFR PTS seeks, or sequential presentation ordinals for VFR.

    video_ordinals maps episode row indices to explicit raw-video frame indices.
    Sequential decoding never estimates a frame number from nominal FPS or PTS.
    """
    import av
    from itertools import chain
    if sequential and offset_s!=0:
        raise ValueError('VFR nonzero time offset needs an explicit ordinal origin')
    if video_ordinals is not None and not sequential:
        raise ValueError('explicit video ordinals require sequential decode')
    indices=sorted(set(indices));base=offset_s*fps
    if abs(base-round(base))>1e-3:raise ValueError('nonintegral video offset')
    base=round(base);requested={base+i:i for i in indices};images={};alignment={}
    if video_ordinals is not None:
        try:ordinals=[video_ordinals[i] for i in indices]
        except (KeyError,IndexError) as exc:raise ValueError('incomplete explicit video ordinal mapping') from exc
        if (any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or v<0 for v in ordinals)
                or any(a>=b for a,b in zip(ordinals,ordinals[1:]))):
            raise ValueError('explicit video ordinals must be nonnegative increasing integers')
        requested={int(v):i for i,v in zip(indices,ordinals)}
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=2
        # average_rate is a container timing statistic, not an ordinal contract.
        # Only the CFR path derives ordinals from FPS and must match that rate.
        if not sequential and (stream.average_rate is None or abs(float(stream.average_rate)-fps)>1e-4):
            raise ValueError('source/video FPS mismatch')
        decoded=container.decode(stream)
        first=next(decoded);first_pts=first.pts;tb=float(stream.time_base)
        if first_pts is None:raise ValueError('missing first PTS')
        def save(frame,ordinal,method):
            from openwam.dataloader.utils.video_color import canonical_video_rgb, video_color_policy
            index=requested[ordinal];raw=frame.to_image().convert('RGB')
            images[index]=canonical_video_rgb(raw,path)
            alignment[index]=dict(source_frame_index=index,video_ordinal=ordinal,pts=frame.pts,
                first_pts=first_pts,time_base=str(stream.time_base),pts_seconds=frame.pts*tb,
                offset_seconds=offset_s,offset_frames=base,decode_method=method,native_wh=list(images[index].size),
                ordinal_mapping=('explicit_source_frame_index' if video_ordinals is not None else 'episode_frame_index'),
                source_fps=fps,video_average_rate=str(stream.average_rate),
                source_rgb_sha256=hashlib.sha256(np.asarray(images[index]).tobytes()).hexdigest(),
                decoded_rgb_sha256=hashlib.sha256(np.asarray(raw).tobytes()).hexdigest(),
                image_color_policy=video_color_policy(path))
        if sequential:
            # Count every decoded frame from stream start, including the first;
            # do not seek, round timestamps, or discard frames on a PTS filter.
            for ordinal,frame in enumerate(chain((first,),decoded)):
                if frame.pts is None:raise ValueError('missing decoded PTS')
                if ordinal in requested:save(frame,ordinal,'sequential_decoded_ordinal')
                if ordinal>=max(requested):break
        else:
            remaining=set(requested)
            while remaining:
                target=min(remaining);container.seek(first_pts+round(target/fps/tb),stream=stream,backward=True)
                for frame in container.decode(stream):
                    ordinal_float=(frame.pts-first_pts)*tb*fps;ordinal=round(ordinal_float)
                    if abs(ordinal_float-ordinal)>1e-3:raise ValueError('nonuniform PTS requires sequential decode')
                    if ordinal in remaining:save(frame,ordinal,'exact_cfr_pts_including_first_pts');remaining.remove(ordinal)
                    if not remaining or (ordinal>=target and min(remaining)-ordinal>60):break
                    if ordinal>max(requested):break
                if target in remaining:raise ValueError('requested frame unavailable')
        if set(images)!=set(indices):raise ValueError('incomplete video extraction')
    return images,alignment


def read_selection(reference):
    audit=json.loads(reference.read_text())['episodes']
    selections=[dict(id=r['id'],family=r['family'],repo=r['repo'],episode=r['episode'],camera=r['camera'],
        reference_segments=r['segments'],reference_video=str(reference.parent/r['full_video'])) for r in audit]
    # The reference ABC cases have rejected camera QA; include the current valid one explicitly.
    current=json.loads((ROOT/'outputs/g1_per_dataset_overfit_20261018_tcpfix/manifest.json').read_text())
    good=next(r for r in current if r['family']=='abc_130k')
    selections.append(dict(id='P0_abc_130k_passed',family='abc_130k',repo=good['source_repo'],episode=good['source_episode'],camera='observation.images.cam_high'))
    included={r['family'] for r in selections}
    for r in current:
        if r['family'] not in included and (Path(r['source_repo'])/'meta/sub_tasks.parquet').is_file():
            selections.append(dict(id='P1_'+r['family'],family=r['family'],repo=r['source_repo'],
                episode=r['source_episode'],camera=r.get('head','observation.images.cam_high')))
    return selections


def prepare_episode(selection,out):
    from openwam.dataloader.episode_data import episode_data_path,read_episode_table,read_episode_rows
    family=selection['family'];root=Path(selection['repo']);episode=selection['episode'];head=selection['camera']
    folder=out/'episodes'/selection['id'];folder.mkdir(parents=True,exist_ok=True)
    info_path=root/'meta/info.json';info=json.loads(info_path.read_text());ep=read_episode(root,episode);n=int(ep['length']);fps=float(info['fps'])
    data=episode_data_path(root,info,ep)
    names=pq.ParquetFile(data).schema_arrow.names
    cols=['frame_index','task_index','subtask_index']
    if family=='mecka':cols+=['source_frame_index']
    if 'subtask_unannotated' in names:cols.append('subtask_unannotated')
    table,data_files=read_episode_table(root,info,ep,cols)
    labels=table.to_pydict()
    ann_path=root/'meta/sub_tasks.parquet';annotations=pq.read_table(ann_path,filters=[('episode_index','=',episode)]).to_pylist()
    segments=contiguous_subtasks(labels,annotations,n)
    if 'reference_segments' in selection:
        expected=[(s['start'],s['end_exclusive'],s['subtask_index'],s['text'],s['valid']) for s in selection['reference_segments']]
        actual=[(s['start'],s['end_exclusive'],s['subtask_index'],s['text'],s['valid']) for s in segments]
        if actual!=expected:raise ValueError('fresh source subtask labels disagree with reference audit')
    selected=[];excluded=[]
    for s in segments:
        if not s['valid'] or s['end_exclusive']-s['start']<2:
            excluded.append({**s,'exclusion_reasons':s['exclusion_reasons'] or ['no_future_frame_in_single_frame_segment']})
        else:selected.append(s)
    indices={0,n-1}
    for s in selected:indices.update([s['start'],s['end_frame']])
    prefix='videos/'+head+'/'
    video=root/info['video_path'].format(video_key=head,chunk_index=int(ep[prefix+'chunk_index']),file_index=int(ep[prefix+'file_index']))
    offset=float(ep[prefix+'from_timestamp'])
    from openwam.dataloader.video_alignment import mecka_video_ordinals
    ordinals=mecka_video_ordinals(labels,ep) if family=='mecka' else None
    images,alignment=decode_frames(video,offset,fps,indices,sequential=family=='mecka',video_ordinals=ordinals)
    projection_fields=[k for k in names if k.startswith(('observation.states.','states.','source.','canonical.'))]
    values=read_episode_rows(dict(data=data,data_files=data_files),episode,columns=['frame_index']+projection_fields,
        filters=[('frame_index','in',sorted(indices))]).to_pylist()
    by_frame={int(r['frame_index']):r for r in values}
    if len(by_frame)!=len(indices):raise ValueError('geometry frame row mismatch')
    transforms=recorded_tcp_transform(family,info);eligible,qa=camera_eligibility(family,root,episode)
    if selection.get('reference_video'):
        shutil.copyfile(selection['reference_video'],folder/'episode.mp4')
    else:
        import av
        with av.open(str(video)) as c:
            if offset!=0 or c.streams.video[0].frames!=n:raise ValueError('cannot copy supplemental full-episode video without trimming')
        shutil.copyfile(video,folder/'episode.mp4')
    images[n-1].resize((320,256),Image.Resampling.BILINEAR).save(folder/'episode_terminal.png')
    images[0].resize((320,256),Image.Resampling.BILINEAR).save(folder/'episode_initial.png')
    task_mapping={k:task_text(root,k) for k in set(labels['task_index'])}
    records=[]
    for s in selected:
        ident=f'{selection["id"]}_seg{s["number"]:03d}';assets=folder/ident;assets.mkdir(exist_ok=True)
        anchor=s['start'];goal=s['end_frame'];row=by_frame[goal]
        endpoint=goal_source_tcp(family,row,tcp_from_eef=transforms)
        camera,cam_meta=goal_camera(family,row,head,info,images[goal].size,geometry_eligible=eligible)
        projection=dict(uv=np.zeros((2,2),np.float32),projection_valid=np.zeros(2,bool),in_frame=np.zeros(2,bool),visibility=np.full(2,-1,np.int8))
        if camera is not None:
            k,e,model,d,wh=camera
            projection=project_goal_tcp(endpoint['tcp_world'],e,k,wh,camera_model=model,distortion=d,tcp_valid=endpoint['tcp_valid'])
        for frame,name in [(anchor,'observed.png'),(goal,'goal.png')]:
            images[frame].resize((320,256),Image.Resampling.BILINEAR).save(assets/name)
        np.savez_compressed(assets/'targets.npz',**{k:projection[k] for k in ('uv','projection_valid','in_frame','visibility')})
        reasons=list(endpoint['missing_reasons'])+cam_meta['missing_reasons']
        record=dict(format=FORMAT,id=ident,family=family,source_repo=str(root),source_episode=episode,
            source_episode_length=n,segment_number=s['number'],subtask_index=s['subtask_index'],task_index=s['task_index'],
            instruction=s['text'],parent_task=task_mapping[s['task_index']],instruction_start=anchor,instruction_end_exclusive=s['end_exclusive'],
            anchor_source_frame=anchor,goal_source_frame=goal,label_kind='subtask_end',success_verified=False,
            review_status='pending_review',split='diagnostic',fps=fps,head=head,
            observed_image_path=str((assets/'observed.png').relative_to(out)),goal_image_path=str((assets/'goal.png').relative_to(out)),
            target_path=str((assets/'targets.npz').relative_to(out)),episode_id=selection['id'],
            sampling_weight_within_episode=1/len(selected),
            provenance=dict(goal_source_frame=goal,source_parquet=str(data),source_parquets=[str(p) for p in data_files],source_parquet_size=data.stat().st_size,
                source_repo_resolved=str(root.resolve()),source_info_sha256=sha(info_path),annotation_file=str(ann_path),annotation_sha256=sha(ann_path),
                frame_labels_sha256=hashlib.sha256(json.dumps(labels,sort_keys=True).encode()).hexdigest(),
                source_goal_geometry_sha256=hashlib.sha256(json.dumps(clean_json(row),sort_keys=True).encode()).hexdigest(),
                observed_video=alignment[anchor],goal_video=alignment[goal],source_video=str(video),
                endpoint=endpoint,camera=cam_meta,calibration_eligibility_record=qa,missing_reasons=reasons,
                geometry_review_status='pending_review',parent_task_used_as_condition=False,
                goal_policy='contiguous subtask end_exclusive-1; no semantic success assertion',
                image_transform='native RGB -> PIL bilinear resize 320x256 -> /127.5-1 -> BF16; no crop or augmentation'))
        records.append(clean_json(record))
    episode_record=dict(id=selection['id'],family=family,repo=str(root),episode=episode,length=n,fps=fps,
        task_texts=list(task_mapping.values()),segments=segments,excluded_segments=excluded,pairs=len(records),
        calibration_eligibility=qa,playback=str((folder/'episode.mp4').relative_to(out)),
        initial=str((folder/'episode_initial.png').relative_to(out)),terminal=str((folder/'episode_terminal.png').relative_to(out)))
    dump(folder/'episode.json',episode_record)
    print(json.dumps(dict(episode=selection['id'],pairs=len(records),excluded=len(excluded),camera_eligible=eligible)),flush=True)
    return records,episode_record


def validate_preview_batch(row,batch):
    t=batch['targets']
    if batch['instruction']!=row['instruction']:raise ValueError('subtask instruction mismatch')
    if batch['observed'].shape!=t.image.shape or tuple(t.image.shape)!=(1,3,1,256,320):
        raise ValueError('model image shape mismatch')
    if batch['observed'].dtype!=torch.bfloat16 or t.image.dtype!=torch.bfloat16:
        raise ValueError('preview images must match BF16 model input')
    if t.uv.dtype!=torch.float32:raise ValueError('TCP UV must remain float32')
    if not torch.isfinite(batch['observed']).all() or not torch.isfinite(t.image).all():raise ValueError('nonfinite image')
    p=row['provenance']
    if (p['goal_video']['source_frame_index']!=row['goal_source_frame'] or
            p['observed_video']['source_frame_index']!=row['anchor_source_frame'] or
            p['goal_source_frame']!=row['goal_source_frame']):raise ValueError('source frame alignment mismatch')
    return dict(id=row['id'],instruction_matches=True,image_shape_dtype_valid=True,uv_float32=True,
        target_frame_matches=True,observed_frame_matches=True,valid_points=point_mask(t)[0].tolist())


def prepare(reference,out,*,resume=False):
    if out.exists() and any(out.iterdir()) and not resume:raise ValueError('use a new empty output directory or --resume for interrupted preparation')
    if (out/'manifest.jsonl').exists():raise ValueError('completed/assembled manifest exists; choose a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    selections=read_selection(reference);records=[];episodes=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for rows,ep in pool.map(lambda s:prepare_episode(s,out),selections):records.extend(rows);episodes.append(ep)
    finalize_dataset(reference,out,records,episodes)


def finalize_dataset(reference,out,records,episodes):
    manifest=out/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records))
    dataset=SubtaskGoalDataset(manifest,allow_unreviewed_geometry=True)
    checks=[]
    for i,row in enumerate(records):
        batch=dataset[i];t=batch['targets'];assets=out/Path(row['goal_image_path']).parent
        valid=point_mask(t)[0].tolist();uv=target_uv_pixels(t)[0].tolist()
        _display_image(batch['observed']).save(assets/'observed_prevae.png')
        goal=_display_image(t.image);goal.save(assets/'goal_prevae.png');overlay_goal_targets(goal,uv,valid).save(assets/'goal_tcp.png')
        np.savez_compressed(assets/'model_inputs.npz',observed=batch['observed'].float().numpy(),goal=t.image.float().numpy(),
            observed_bfloat16_bits=batch['observed'].view(torch.uint16).numpy(),goal_bfloat16_bits=t.image.view(torch.uint16).numpy(),
            uv=t.uv.numpy(),point_valid=t.point_valid.numpy(),in_frame=t.in_frame.numpy(),visibility=t.visibility.numpy())
        row.update(valid_points=valid,target_uv_px=[p if ok else None for p,ok in zip(uv,valid)],
            model_input_npz=str((assets/'model_inputs.npz').relative_to(out)),observed_preview=str((assets/'observed_prevae.png').relative_to(out)),
            goal_preview=str((assets/'goal_prevae.png').relative_to(out)),goal_overlay=str((assets/'goal_tcp.png').relative_to(out)))
        checks.append(validate_preview_batch(row,batch))
    # Rich records remain the same accepted input format; preview fields are non-conditioning metadata.
    manifest.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records))
    current=json.loads((ROOT/'outputs/g1_per_dataset_overfit_20261018_tcpfix/manifest.json').read_text())
    missing=[]
    for r in current:
        if r['family'] not in {x['family'] for x in records}:
            missing.append(dict(family=r['family'],source_repo=r['source_repo'],
                checked_file=str(Path(r['source_repo'])/'meta/sub_tasks.parquet'),
                annotation_file_exists=(Path(r['source_repo'])/'meta/sub_tasks.parquet').exists(),
                reason='not in supplied annotated-family selection; no task text relabeled as subtask'))
    summary=dict(format=FORMAT,mode='data_adaptation_only',training_launched=False,model_predictions_generated=False,
        episodes=len(episodes),families=sorted({r['family'] for r in records}),pairs=len(records),
        valid_tcp_points=sum(sum(r['valid_points']) for r in records),all_source_segments=sum(len(e['segments']) for e in episodes),
        excluded_segments=sum(len(e['excluded_segments']) for e in episodes),excluded_families=missing,
        reference_chat='01a14e36-fa69-7512-84c9-580a57768fbe',reference_manifest=str(reference),
        source_hashes={p:sha(ROOT/p) for p in ['scripts/data/prepare_subtask_goals.py','openwam/dataloader/subtask_goal.py','scripts/data/trajectory_projection.py']},
        goal_policy='subtask end_exclusive-1; candidate, success unverified',
        condition='observed head image and subtask text only',splits='diagnostic only; future train/val/test must split full source episodes')
    dump(out/'summary.json',summary);dump(out/'episodes.json',episodes);dump(out/'validation.json',dict(all_passed=True,cases=checks))
    from scripts.data.subtask_goal_report import render_report
    render_report(out,summary,episodes,records)
    import yaml
    config=dict(mode='data_review_only',training_enabled=False,dataset=dict(model_class='openwam.dataloader.subtask_goal.SubtaskGoalDataset',
        kwargs=dict(manifest=str(manifest.resolve()),dtype='bfloat16',device='cpu',allow_unreviewed_geometry=True)),
        condition_text='instruction',parent_task_condition=False,goal_policy='subtask_end_exclusive_minus_one',
        tensor_contract=dict(observed=[1,3,1,256,320],goal=[1,3,1,256,320],tcp_uv=[1,2,2],tcp_dtype='float32'),
        sampling_note='one anchor per continuous run; within-episode weights stored to avoid long-episode dominance',
        semantic_review='pending; subtask ends are not verified successful completion frames')
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=ROOT/'outputs/subtask_goal_audit_20261018/comparisons.json')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true',help='rebuild interrupted preparation in its existing directory')
    args=parser.parse_args();torch.set_num_threads(2);prepare(args.reference.resolve(),args.output_dir.resolve(),resume=args.resume)
