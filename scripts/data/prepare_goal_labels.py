"""Prepare explicit goal labels from selected source episodes, never fixture ends.

Default reviewed_success mode requires a reviewed label manifest. episode_end is
an explicit weak-label mode. Rendering review assets never verifies success or
physical calibration. This command writes only its requested output directory.
"""
import argparse
from collections import Counter
from dataclasses import fields
import hashlib
import html
import json
from pathlib import Path
import sys
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openwam.model.goal.types import GoalLabel
from scripts.data.trajectory_projection import goal_source_tcp, project_goal_tcp


def source_key(source_repo, source_episode):
    return json.dumps([str(Path(source_repo).resolve()), int(source_episode)])


def validate_goal_label(label, *, allow_episode_end=False):
    if isinstance(label, dict):
        label=GoalLabel(**{f.name: label[f.name] for f in fields(GoalLabel) if f.name in label})
    if not label.instruction.strip(): raise ValueError('missing instruction text')
    if label.split not in ('train','val','test'):raise ValueError('invalid split')
    if not (0<=label.instruction_start<=label.anchor_source_frame<label.instruction_end_exclusive<=label.source_episode_length):
        raise ValueError('anchor outside instruction/source interval')
    if not (label.instruction_start<=label.goal_source_frame<label.instruction_end_exclusive):
        raise ValueError('goal outside instruction interval')
    if label.label_kind=='episode_end':
        if not allow_episode_end:raise ValueError('weak episode_end labels require explicit opt-in')
        if label.success_verified:raise ValueError('weak episode_end cannot be verified success')
        if label.goal_source_frame!=label.source_episode_length-1:raise ValueError('episode_end must be original source terminal frame')
    elif label.label_kind=='reviewed_success':
        if not label.success_verified or not label.success_evidence.get('reference') or label.success_evidence.get('type') not in ('human','environment'):
            raise ValueError('reviewed success requires human/environment evidence reference')
    else:raise ValueError('unsupported label_kind')
    return label


def split_source_episodes(rows, seed=0):
    keys=sorted({source_key(r['source_repo'],r['source_episode']) for r in rows},
                key=lambda k:hashlib.sha256(f'{seed}:{k}'.encode()).hexdigest())
    n=len(keys);nv=max(1,round(n*.1)) if n>=3 else 0;nt=nv if n>=3 else 0
    return {k:'val' if i<nv else 'test' if i<nv+nt else 'train' for i,k in enumerate(keys)}


def validate_splits(rows, *, diagnostic_override=False):
    seen={}
    for r in rows:
        key=source_key(r['source_repo'],r['source_episode']);split=r['split']
        if key in seen and seen[key]!=split:raise ValueError(f'source episode split leakage: {key}')
        seen[key]=split
        if split in ('val','test') and r.get('baseline_seen_episode') and not diagnostic_override:
            raise ValueError('baseline-seen heldout episode requires diagnostic override')
    return seen


def read_episode(root, episode):
    import pyarrow.parquet as pq
    matches=[]
    for path in sorted((root/'meta/episodes').glob('chunk-*/*.parquet')):
        matches.extend(pq.read_table(path,filters=[('episode_index','=',episode)]).to_pylist())
    if len(matches)!=1:raise ValueError(f'expected unique source episode {episode}, got {len(matches)}')
    return matches[0]


def source_rows(row, ep, info):
    import pyarrow.parquet as pq
    from openwam.dataloader.episode_data import episode_data_path,read_episode_table
    root=Path(row['source'])
    path=episode_data_path(root,info,ep)
    if path.resolve()!=Path(row['source_parquet']).resolve():raise ValueError('source parquet disagreement')
    available=pq.ParquetFile(path).schema_arrow.names
    table,data_files=read_episode_table(root,info,ep,['frame_index','task_index'])
    ids=table.to_pydict()
    task=ids['task_index'];segments=[];start=0
    for i in range(1,len(task)+1):
        if i==len(task) or task[i]!=task[start]:segments.append((start,i,task[start]));start=i
    # Do not concatenate disjoint task runs or infer goals across task boundaries.
    return path,available,segments,data_files


def task_text(root,task):
    import pyarrow.parquet as pq
    df=pq.read_table(root/'meta/tasks.parquet').to_pandas()
    indices=df['task_index'].to_numpy()
    texts=df['task'].to_numpy() if 'task' in df.columns else df.index.to_numpy()
    matches=[text for idx,text in zip(indices,texts) if int(idx)==int(task)]
    if not matches or len(set(matches))!=1:raise ValueError('missing/conflicting task mapping')
    value=matches[0]
    if not isinstance(value,str) or not value.strip():raise ValueError('missing/ambiguous task mapping')
    return value


def decode_goal(video_path, offset_s, frame_index):
    """Decode exact file ordinal in one bounded pass; PTS need not start at zero.

    Sequential enumeration handles nonuniform timing without guessing an index
    from average FPS. Nonzero metadata time offsets require CFR cadence before
    the target: converting a VFR time offset to an ordinal would be ambiguous.
    """
    import av
    with av.open(str(video_path)) as container:
        stream=container.streams.video[0];fps=float(stream.average_rate)
        if not np.isfinite(fps) or fps<=0:raise ValueError('invalid actual FPS')
        offset=offset_s*fps
        if offset_s<0 or frame_index<0 or abs(offset-round(offset))>1e-3:
            raise ValueError('invalid/nonintegral shard frame offset')
        ordinal=round(offset)+int(frame_index)
        stream.thread_type='AUTO'
        first_pts=None;previous_time=None;nonuniform=False
        for decoded,frame in enumerate(container.decode(stream)):
            if frame.pts is None:raise ValueError('missing video PTS')
            pts_s=float(frame.pts*stream.time_base)
            if first_pts is None:first_pts=frame.pts
            if previous_time is not None:
                delta=pts_s-previous_time
                if delta<=0:raise ValueError('nonmonotonic video PTS')
                nonuniform |= abs(delta-1/fps)>min(.25/fps,max(float(stream.time_base)*1.1,1e-5))
            previous_time=pts_s
            if decoded!=ordinal:continue
            if offset_s and nonuniform:raise ValueError('VFR time offset has no verified ordinal mapping')
            image=frame.to_image().convert('RGB')
            return image,dict(video_path=str(video_path),video_ordinal=ordinal,video_pts=frame.pts,
                              video_pts_seconds=pts_s,video_fps=fps,video_offset_s=offset_s,
                              decoded_wh=list(image.size),decode_alignment='sequential_ordinal',
                              video_first_pts=first_pts,stream_start_time=stream.start_time,
                              video_time_base=str(stream.time_base),nonuniform_pts_seen=bool(nonuniform))
    raise ValueError('goal video frame unavailable')


def goal_camera(family, values, head, info, decoded_wh, *, geometry_eligible=True):
    """Resolve goal-row source-reference-to-head-camera with explicit model."""
    cam=head.split('.')[-1]
    kk=next((k for k in (f'observation.states.cameras.{cam}.intrinsic',f'observation.states.camera.calibration.{cam}.intrinsic','observation.states.camera.intrinsic') if k in values),None)
    ek=next((k for k in (f'observation.states.cameras.{cam}.world_to_camera','observation.states.camera.world_to_camera') if k in values),None)
    provenance=dict(intrinsic_field=kk,extrinsic_field=ek,model='pinhole',missing_reasons=[])
    try:
        if not geometry_eligible:raise ValueError('source camera geometry QA rejected or unavailable')
        if kk is None or ek is None:raise ValueError('missing intrinsic or world_to_camera')
        raw=np.asarray(values[kk],float).ravel();d=None;model='pinhole'
        if raw.size==9:k=raw.reshape(3,3)
        elif raw.size==8 and family=='mecka':
            k=np.array([[raw[0],0,raw[2]],[0,raw[1],raw[3]],[0,0,1]])
            d=raw[4:];model='brown_conrady'
        else:raise ValueError(f'unsupported intrinsic layout {raw.shape}')
        if family=='Genrobot':
            # Native raw double sphere; the endpoint adapter supplies the
            # selected finger-device TCP convention and independent validity.
            prefix='raw.robot0.sensor.camera2.camera_info.'
            k=np.asarray(values[prefix+'K'],float).reshape(3,3)
            ds=np.asarray(values[prefix+'D'],float)
            if not np.allclose(ds[:4],k[[0,1,0,1],[0,1,2,2]],atol=1e-3):raise ValueError('double sphere K/D disagree')
            model='double_sphere';d=ds[4:6]
        shape=info['features'][head]['shape'];native_wh=(int(shape[1]),int(shape[0]))
        # Explicitly reject a mismatch instead of guessing an intrinsic resize.
        if tuple(decoded_wh)!=native_wh:raise ValueError('calibration/video dimensions mismatch')
        for key in ('source.camera.world_to_camera_valid','observation.states.camera.valid'):
            if key in values and not bool(np.asarray(values[key]).reshape(-1)[0]>.5):raise ValueError(f'{key} false')
        e=np.asarray(values[ek],float).reshape(4,4)
        # Present fields can still mean unavailable calibration (all-NaN rows).
        # Use the projection routines' own checks before returning a camera;
        # the caller then keeps image supervision and masks only TCP targets.
        # Empty points validate calibration without inventing a test endpoint.
        from scripts.data.trajectory_projection import transform_points,project_points
        transform_points(e,np.empty((0,3)))
        project_points(np.empty((0,3)),k,model=model,distortion=d)
        provenance.update(model=model,native_wh=list(native_wh),K=k.tolist(),world_to_camera=e.tolist(),distortion=None if d is None else d.tolist(),
                          resize_affine_pixel_centers={'scale':[320/native_wh[0],256/native_wh[1]],'translation':[.5*320/native_wh[0]-.5,.5*256/native_wh[1]-.5]})
        return (k,e,model,d,native_wh),provenance
    except (KeyError,ValueError,TypeError) as exc:
        provenance['missing_reasons'].append(str(exc));return None,provenance


def prepare(manifest, output_dir, *, label_kind='reviewed_success', reviewed_manifest=None,
            diagnostic_override=False, seed=0, anchors_per_episode=3):
    import pyarrow.parquet as pq
    from PIL import Image,ImageDraw
    import yaml
    manifest=Path(manifest).resolve();out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    rows=json.loads(manifest.read_text())
    assignments=split_source_episodes([dict(source_repo=r['source'],source_episode=r['episode']) for r in rows],seed)
    reviewed={}
    if reviewed_manifest:
        for line in Path(reviewed_manifest).read_text().splitlines():
            item=json.loads(line);reviewed[(source_key(item['source_repo'],item['source_episode']),item['instruction_id'])]=item
    if label_kind=='reviewed_success' and not reviewed:raise ValueError('reviewed_success requires reviewed labels with success evidence')
    labels=[];exclusions=[];geometry=[];cards=[]
    for row in rows:
        try:
            root=Path(row['source']);info=json.loads((root/'meta/info.json').read_text());ep=read_episode(root,row['episode'])
            data_path,columns,segments,data_files=source_rows(row,ep,info)
            if len(segments)!=1:raise ValueError('ambiguous/multi-instruction episode: explicit segment review required')
            start,end,task=segments[0];instruction=task_text(root,task);instruction_id=f'task:{task}:{start}'
            identity=source_key(row['source'],row['episode']);split=assignments[identity]
            selected=reviewed.get((identity,instruction_id))
            if label_kind=='reviewed_success':
                if selected is None:raise ValueError('missing reviewed success label')
                if selected.get('label_kind')!='reviewed_success' or selected.get('success_verified') is not True:
                    raise ValueError('formal mode requires an originally verified reviewed_success label; weak/unverified input cannot be upgraded')
            goal=int(ep['length'])-1 if label_kind=='episode_end' else int(selected['goal_source_frame'])
            evidence={} if selected is None else selected.get('success_evidence',{})
            from openwam.dataloader.episode_data import read_episode_rows
            values=read_episode_rows(dict(data=data_path,data_files=data_files),row['episode'],filters=[('frame_index','=',goal)]).to_pylist()
            if len(values)!=1:raise ValueError('goal row is not unique')
            values=values[0]
            prefix=f'videos/{row["head"]}/'
            video=root/info['video_path'].format(video_key=row['head'],chunk_index=int(ep[prefix+'chunk_index']),file_index=int(ep[prefix+'file_index']))
            image,video_meta=decode_goal(video,float(ep[prefix+'from_timestamp']),goal)
            assets=out/'goal_targets'/row['id'];assets.mkdir(parents=True,exist_ok=True)
            image_path=assets/'goal.png';image.save(image_path)
            config_path=manifest.parent/'data'/row['id']/'data.yaml';cfg=yaml.safe_load(config_path.read_text())
            endpoint=goal_source_tcp(row['family'],values,tcp_from_eef=cfg.get('tcp_from_eef'))
            eligible=True;eligibility_record=None
            if row['family']=='abc_130k':
                eligibility_path=root/'meta/camera_geometry_eligibility.jsonl'
                if eligibility_path.exists():
                    eligibility_record=next((json.loads(x) for x in eligibility_path.read_text().splitlines() if json.loads(x).get('episode_index')==row['episode']),None)
                eligible=bool(eligibility_record and eligibility_record.get('broadcast_qa_status')=='passed')
            camera,cam_meta=goal_camera(row['family'],values,row['head'],info,image.size,geometry_eligible=eligible)
            projection=dict(uv=np.zeros((2,2),np.float32),projection_valid=np.zeros(2,bool),in_frame=np.zeros(2,bool),visibility=np.full(2,-1,np.int8))
            if camera is not None:
                k,e,model,d,wh=camera
                try:projection=project_goal_tcp(endpoint['tcp_world'],e,k,wh,camera_model=model,distortion=d,tcp_valid=endpoint['tcp_valid'])
                except ValueError as exc:cam_meta['missing_reasons'].append(str(exc))
            np.savez(assets/'targets.npz',**{k:projection[k] for k in ('uv','projection_valid','in_frame','visibility')})
            overlay=image.resize((320,256));draw=ImageDraw.Draw(overlay)
            for hand,color in enumerate(('red','cyan')):
                if projection['in_frame'][hand]:
                    x,y=projection['uv'][hand]*[319,255];draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
            overlay.save(assets/'overlay.png')
            metadata=dict(source_episode_id=identity,source_parquet=str(data_path),source_parquets=[str(p) for p in data_files],source_image_path=str(image_path),head_image_path=str(image_path),
                          target_path=str(assets/'targets.npz'),goal_source_frame=goal,video=video_meta,endpoint=endpoint,camera=cam_meta,
                          source_curation={k:v for k,v in ep.items() if k.startswith('data_curation/')},
                          geometry_review_status='pending_review',calibration_eligibility_record=eligibility_record,
                          terminal_semantics=('released-source episode end; not raw capture end or verified success' if label_kind=='episode_end' else 'reviewed success frame supported by explicit human/environment evidence'))
            # JSON-safe geometry provenance; no NaN payload in reports.
            metadata['endpoint']['tcp_world']=np.where(np.isfinite(endpoint['tcp_world']),endpoint['tcp_world'],0).tolist()
            metadata['endpoint']['tcp_valid']=endpoint['tcp_valid'].tolist()
            episode_geometry=dict(id=row['id'],family=row['family'],
                                 numerical_projection_valid=projection['projection_valid'].tolist(),in_frame=projection['in_frame'].tolist(),
                                 visibility=projection['visibility'].tolist(),review_status='pending_review',human_reviewed=False,**metadata)
            checks=row.get('checks',[])
            anchors=sorted(set(int(c['anchor_source']) for c in checks))[:anchors_per_episode]
            if not anchors:anchors=[int(row['start'])]
            episode_labels=[];episode_cards=[]
            for anchor in anchors:
                label=GoalLabel(source_repo=row['source'],source_episode=int(row['episode']),instruction_id=instruction_id,
                                anchor_source_frame=anchor,goal_source_frame=goal,instruction_start=start,instruction_end_exclusive=end,
                                success_verified=label_kind=='reviewed_success',label_kind=label_kind,success_evidence=evidence,
                                image_path=str(image_path),calibration_id=f'{row["id"]}:goal:{goal}',tcp_definition_id=endpoint['definition_id'],
                                split=split,instruction=instruction,source_episode_length=int(ep['length']),baseline_seen_episode=True,metadata=metadata)
                validate_goal_label(label,allow_episode_end=label_kind=='episode_end');episode_labels.append(label.to_dict())
                # Cached canvases are previews only; source provenance is explicit.
                canvas_path=manifest.parent/'data'/row['id']/'canvases.npy'
                canvas=np.load(canvas_path,mmap_mode='r');local=anchor-int(row['start'])
                anchor_path=assets/f'anchor_{anchor}.jpg';Image.fromarray(np.asarray(canvas[local])).crop((0,0,320,256)).save(anchor_path)
                episode_cards.append(f'<article><h3>{html.escape(row["id"])} · {split}</h3><p>{html.escape(instruction)}</p><p>Source episode {row["episode"]}; anchor {anchor} (fixture local {local}) → goal {goal}; {label_kind}; success verified={label.success_verified}; pending review</p><img src="{anchor_path.relative_to(out)}"><img src="{(assets/"overlay.png").relative_to(out)}"><pre>{html.escape(json.dumps({k:projection[k].tolist() for k in ("projection_valid","in_frame","visibility")}))}</pre></article>')
            labels.extend(episode_labels);cards.extend(episode_cards);geometry.append(episode_geometry)
            print(f'{row["id"]}: goal={goal}, pairs={len(anchors)}, numerical_points={int(projection["in_frame"].sum())}',flush=True)
        except Exception as exc:
            exclusions.append(dict(id=row['id'],family=row['family'],reason=f'{type(exc).__name__}: {exc}'))
            print(f'{row["id"]}: excluded: {exc}',flush=True)
    validate_splits(labels,diagnostic_override=diagnostic_override)
    (out/'goal_manifest.jsonl').write_text(''.join(json.dumps(l,ensure_ascii=False,allow_nan=False)+'\n' for l in labels))
    family={}
    for g in geometry:
        entry=family.setdefault(g['family'],dict(episodes=0,numerical_valid_points=0,in_frame_points=0,total_points=0,human_reviewed_points=0))
        entry['episodes']+=1;entry['numerical_valid_points']+=sum(g['numerical_projection_valid']);entry['in_frame_points']+=sum(g['in_frame']);entry['total_points']+=2
    for v in family.values():v['numerical_valid_ratio']=v['numerical_valid_points']/v['total_points']
    def write(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write('split_manifest.json',dict(seed=seed,unit='full source root + episode',diagnostic_override=diagnostic_override,assignments=assignments,all_baseline_seen=True))
    write('geometry_review.json',dict(status='pending_review',human_reviewed=False,families=family,episodes=geometry))
    write('pair_review.json',dict(status='pending_review',pairs=len(labels),unique_source_episodes=len(geometry),success_verified_pairs=sum(l['success_verified'] for l in labels),exclusions=exclusions,families=family))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>G1 pending goal review</title><style>body{font:16px sans-serif;max-width:1000px;margin:auto}article{border:1px solid #aaa;padding:12px;margin:20px 0}img{width:320px;max-width:48%}</style><h1>Pending goal-pair review</h1><p>Bounded selected-source diagnostic. Label kind: '+html.escape(label_kind)+'. Episode-end labels mean termination, not verified task success; reviewed_success requires explicit evidence. All selected episodes were seen by baseline. Geometry is numerical eligibility, not human calibration approval. Red=left, cyan=right; unknown visibility=-1.</p>'+''.join(cards))
    return dict(pairs=len(labels),unique_source_episodes=len(geometry),exclusions=len(exclusions))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--label-kind',choices=('reviewed_success','episode_end'),default='reviewed_success')
    parser.add_argument('--reviewed-manifest');parser.add_argument('--diagnostic-override',action='store_true');parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--anchors-per-episode',type=int,default=3);args=parser.parse_args()
    print(json.dumps(prepare(**vars(args))))


if __name__=='__main__':main()
