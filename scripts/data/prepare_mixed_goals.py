#!/usr/bin/env python3
"""Combine corrected representative task episodes with their annotated subtask pairs."""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from openwam.dataloader.mixed_goal import FORMAT,load_mixed_manifest,episode_key
from openwam.dataloader.subtask_goal import load_subtask_manifest
from openwam.train.goal_overfit import validate_overfit_manifest
from scripts.data.prepare_subtask_goals import (read_episode,task_text,decode_frames,recorded_tcp_transform,
    camera_eligibility,goal_source_tcp,goal_camera,project_goal_tcp,clean_json,sha)


def rebuild_task_for_release(task,sub,out):
    root=Path(sub['source_repo']);epid=sub['source_episode'];info=json.loads((root/'meta/info.json').read_text())
    ep=read_episode(root,epid);n=int(ep['length']);head=sub['head'];fps=float(info['fps']);goal=n-1;anchor=task['anchor_source_frame']
    data=root/info['data_path'].format(episode_index=epid,chunk_index=int(ep['data/chunk_index']),file_index=int(ep['data/file_index']))
    labels=pq.read_table(data,columns=['frame_index','task_index'],filters=[('episode_index','=',epid)]).sort_by('frame_index').to_pydict()
    if labels['frame_index']!=list(range(n)) or len(set(labels['task_index']))!=1:raise ValueError('task must have dense frames and one episode-level instruction')
    instruction=task_text(root,labels['task_index'][0]);prefix='videos/'+head+'/'
    video=root/info['video_path'].format(video_key=head,chunk_index=int(ep[prefix+'chunk_index']),file_index=int(ep[prefix+'file_index']))
    images,alignment=decode_frames(video,float(ep[prefix+'from_timestamp']),fps,[anchor,goal],sequential=sub['family']=='mecka')
    fields=[k for k in pq.ParquetFile(data).schema_arrow.names if k.startswith(('observation.states.','states.','source.','canonical.'))]
    rr=pq.read_table(data,columns=['frame_index']+fields,filters=[('episode_index','=',epid),('frame_index','=',goal)]).to_pylist()
    if len(rr)!=1:raise ValueError('nonunique geometry row')
    row=rr[0];eligible,qa=camera_eligibility(sub['family'],root,epid)
    endpoint=goal_source_tcp(sub['family'],row,tcp_from_eef=recorded_tcp_transform(sub['family'],info))
    camera,cam_meta=goal_camera(sub['family'],row,head,info,images[goal].size,geometry_eligible=eligible)
    projection=dict(uv=np.zeros((2,2),np.float32),projection_valid=np.zeros(2,bool),in_frame=np.zeros(2,bool),visibility=np.full(2,-1,np.int8))
    if camera is not None:
        k,e,model,d,wh=camera
        projection=project_goal_tcp(endpoint['tcp_world'],e,k,wh,camera_model=model,distortion=d,tcp_valid=endpoint['tcp_valid'])
    folder=out/'rebuilt'/sub['family'];folder.mkdir(parents=True)
    for f,name in [(anchor,'observed.png'),(goal,'goal.png')]:images[f].save(folder/name)
    np.savez_compressed(folder/'targets.npz',**{k:projection[k] for k in ('uv','projection_valid','in_frame','visibility')})
    return {**task,'source_repo':str(root),'source_episode_length':n,'anchor_source_frame':anchor,'goal_source_frame':goal,
        'clip_start':task['clip_start'],'clip_end_exclusive':task['clip_end_exclusive'],'instruction_start':0,'instruction_end_exclusive':n,'instruction':instruction,
        'observed_image_path':str(folder/'observed.png'),'goal_image_path':str(folder/'goal.png'),'target_path':str(folder/'targets.npz'),
        'provenance':clean_json(dict(source_parquet=str(data),goal_source_frame=goal,observed_video=alignment[anchor],goal_video=alignment[goal],
            source_video=str(video),source_info_sha256=sha(root/'meta/info.json'),endpoint=endpoint,camera=cam_meta,
            calibration_eligibility_record=qa,missing_reasons=endpoint['missing_reasons']+cam_meta['missing_reasons'],
            geometry_review_status='pending_review',release_choice='same annotated release as paired subtasks'))}


def prepare(output):
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('use empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    task_manifest=ROOT/'outputs/g1_per_dataset_overfit_20261018_tcpfix/manifest.json'
    sub_manifest=ROOT/'outputs/g1_subtask_data_20261018/manifest.jsonl'
    tasks=validate_overfit_manifest(task_manifest,diagnostic_override=True,allow_unreviewed_geometry=True)
    subs=load_subtask_manifest(sub_manifest);records=[];coverage=[]
    for original in tasks:
        t=copy.deepcopy(original)
        candidates=[r for r in subs if r['family']==t['family'] and r['source_episode']==t['source_episode'] and r['source_episode_length']==t['source_episode_length']]
        # Numeric episode IDs alone are not sufficient: match release-relative repo.
        relative=lambda p:str(p).split('/pretrain_datasetsV0_')[-1].split('/',1)[-1]
        candidates=[r for r in candidates if relative(r['source_repo'])==relative(t['source_repo'])]
        if candidates and episode_key(candidates[0])!=episode_key(t):t=rebuild_task_for_release(t,candidates[0],output)
        if candidates and len({episode_key(s) for s in candidates})!=1:raise ValueError('ambiguous source release')
        t.update(id=t['family']+'_task',granularity='task',instruction_start=0,instruction_end_exclusive=t['source_episode_length'],split='diagnostic')
        selected=[t]+[{**s,'granularity':'subtask','clip_start':s['instruction_start'],'clip_end_exclusive':s['instruction_end_exclusive']} for s in candidates]
        for r in selected:
            r=copy.deepcopy(r);r['format']=FORMAT;r['metric_key']=r['family']+' / '+r['granularity']+' / '+r['id']
            folder=output/'assets'/r['id'];folder.mkdir(parents=True)
            for k,name in [('observed_image_path','observed.png'),('goal_image_path','goal.png'),('target_path','targets.npz')]:
                source=Path(r[k]);target=folder/name;shutil.copyfile(source,target);r[k]=str(target.relative_to(output))
                r.setdefault('asset_provenance',{})[k]={'source':str(source),'sha256':sha(source)}
            # Do not carry stale paths to earlier report previews into the new manifest.
            for k in ('model_input_npz','observed_preview','goal_preview','goal_overlay'):r.pop(k,None)
            records.append(r)
        coverage.append(dict(family=t['family'],source_repo=t['source_repo'],episode=t['source_episode'],task_samples=1,subtask_samples=len(candidates)))
    path=output/'manifest.json';path.write_text(json.dumps(clean_json(records),ensure_ascii=False,indent=2))
    loaded=load_mixed_manifest(path,diagnostic_override=True,allow_unreviewed_geometry=True)
    # Source text/index availability is checked for every selected representative episode.
    for c in coverage:
        root=Path(c['source_repo']);ann=root/'meta/sub_tasks.parquet'
        c['annotation_file_exists']=ann.exists()
        if ann.exists():
            rows=pq.read_table(ann,filters=[('episode_index','=',c['episode'])]).to_pylist()
            c['annotation_text_rows']=sum(bool(r.get('subtask','').strip()) for r in rows)
            if c['annotation_text_rows'] and not c['subtask_samples']:raise ValueError('annotated selected episode omitted from mixed manifest')
    result=dict(families=len(coverage),samples=len(records),task_samples=len(tasks),subtask_samples=len(records)-len(tasks),coverage=coverage,
        selection='current corrected one representative episode per YAML family; all available subtask occurrences for selected episode',
        sampling='uniform family; 1:1 task/subtask where available; uniform occurrences within level',
        source_task_manifest_sha256=sha(task_manifest),source_subtask_manifest_sha256=sha(sub_manifest))
    (output/'coverage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
    return loaded


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);prepare(p.parse_args().output_dir)
