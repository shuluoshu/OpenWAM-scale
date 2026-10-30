"""Independent, streaming audit of the complete-instruction two-row report.

Does not import the renderer or alter samples, predictions, or report videos.
Pixel evidence proves visible overlays, not physical contact-center calibration.
"""
from __future__ import annotations
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

CODE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(CODE))
from scripts.experiments.infer_stage1_samecase import sha,dump,sample_identity,tensor_sha
from scripts.experiments.fullspan_samples import source_rows
ROLES=('head','left_wrist','right_wrist')
VALIDATOR_SOURCE_SHA=sha(Path(__file__))


def require(condition,message):
    if not condition:raise ValueError(message)


def identity_contains(actual,expected):
    if isinstance(expected,dict):
        return isinstance(actual,dict) and all(key in actual and identity_contains(actual[key],value) for key,value in expected.items())
    return actual==expected


def schedule(start,end):
    require(0<=start<end,'invalid interval')
    return sorted(set([*range(start,end,10),end-1]))


def mask(value,name):
    a=np.asarray(value)
    require(a.shape==(32,2) and a.dtype==np.bool_,f'{name}: expected bool[32,2]')
    return a


def validate_raw_outputs(uv,actions,metadata_uv):
    uv=torch.as_tensor(uv).float().cpu().numpy();actions=torch.as_tensor(actions).float().cpu().numpy()
    require(uv.shape==(2,2),'Stage1 UV output shape')
    require(actions.shape==(32,112),'prediction is not a fresh 32-action chunk')
    require(np.isfinite(actions).all(),'nonfinite model action output')
    require(np.array_equal(uv,np.asarray(metadata_uv,float),equal_nan=True),'Stage1 metadata differs from raw predicted UV')


def validate_branch(branch,*,anchor,end):
    arrays={name:np.asarray(branch[name],float) for name in ('world_xyz','gt_world_xyz','pred_uv','gt_uv')}
    for name,a in arrays.items():require(a.shape==(32,2,3 if 'world' in name else 2),f'{name} shape')
    masks={name:mask(branch[name],name) for name in ('valid','gt_valid','pred_projectable','gt_projection_valid',
        'pred_in_frame','gt_in_frame','metric_valid','metric_gt_valid')}
    for prefix,valid,projected,inside,xyz,uv in [('pred','valid','pred_projectable','pred_in_frame','world_xyz','pred_uv'),
                                                ('gt','gt_valid','gt_projection_valid','gt_in_frame','gt_world_xyz','gt_uv')]:
        require(not (masks[valid]&~np.isfinite(arrays[xyz]).all(-1)).any(),f'{prefix}: valid NULL/nonfinite world point')
        require(not (masks[projected]&~masks[valid]).any(),f'{prefix}: projected invalid physical point')
        require(not (masks[inside]&~masks[projected]).any(),f'{prefix}: in-frame without projection')
        require(not (masks[projected]&~np.isfinite(arrays[uv]).all(-1)).any(),f'{prefix}: projected NULL/nonfinite UV')
        in_bounds=(arrays[uv]>=0).all(-1)&(arrays[uv]<=[319,255]).all(-1)
        require(not (masks[inside]&~in_bounds).any(),f'{prefix}: in-frame UV outside tile')
    outside=anchor+np.arange(1,33)>=end
    require(not masks['gt_valid'][outside].any(),'GT beyond instruction boundary')
    require(not masks['metric_valid'][outside].any(),'scored future beyond instruction boundary')
    origin=np.asarray(branch.get('gt_source',np.full((32,2),'unavailable')))
    require(origin.shape==(32,2),'GT source label shape')
    require(not (masks['gt_valid']&(origin=='unavailable')).any(),'valid GT uses unavailable/placeholder provenance')
    if 'native_gt_valid' in branch:
        source_only=(origin=='source_endpoint_annotation')&~mask(branch['native_gt_valid'],'native_gt_valid')
        require(not (source_only&masks['metric_gt_valid']).any(),'source-only GT promoted to native metric eligibility')
    if 'gt_native_disagreement' in branch:
        disagree=mask(branch['gt_native_disagreement'],'gt_native_disagreement')
        require(not (disagree&(masks['metric_valid']|masks['metric_gt_valid'])).any(),'conflicting GT scored')
    return dict(gt_points=masks['gt_in_frame'].sum(0).tolist(),pred_points=masks['pred_in_frame'].sum(0).tolist(),
        gt_valid=int(masks['gt_valid'].sum()),pred_valid=int(masks['valid'].sum()))


def marker_pixels(tile,xy,color,radius=7):
    x,y=np.asarray(xy,float)
    if not np.isfinite([x,y]).all():return 0
    x0=max(0,int(np.floor(x))-radius);x1=min(tile.shape[1],int(np.ceil(x))+radius+1)
    y0=max(0,int(np.floor(y))-radius);y1=min(tile.shape[0],int(np.ceil(y))+radius+1)
    if x0>=x1 or y0>=y1:return 0
    rgb=tile[y0:y1,x0:x1].astype(np.int16);r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]
    if color=='green':found=(g>100)&(g>r+40)&(g>b+20)
    elif color=='red':found=(r>125)&(r>g+40)&(r>b+40)
    elif color=='orange':found=(r>145)&(g>85)&(r>g+15)&(g>b+30)
    else:raise ValueError(color)
    return int(found.sum())


def black_region(tile):
    interior=np.asarray(tile)[8:-8,8:-8].astype(float)
    bright=float((interior.max(-1)>24).mean());mean=float(interior.mean())
    return dict(passed=bright<=1e-4 and mean<=3,mean_rgb=mean,bright_fraction=bright)


def background_error(tile,image):
    expected=np.asarray(image.convert('RGB').resize((320,256),Image.Resampling.BILINEAR),float)
    return float(np.abs(np.asarray(tile,float)-expected).mean())


def current_images(sample):
    pixels=np.rint((sample['goal_observed_multiview'][:,0].permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
    canvas=Image.fromarray(pixels)
    return [canvas.crop(box).resize((320,256),Image.Resampling.BILINEAR) for box in
        [(0,0,320,256),(0,256,160,384),(160,256,320,384)]]


def check_camera_source(provenance,source_row):
    k=np.asarray(provenance['K'],float);e=np.asarray(provenance['world_to_camera'],float)
    kk=provenance['intrinsic_field'];ek=provenance['extrinsic_field'];model=provenance['model']
    if model=='double_sphere' and 'raw.robot0.sensor.camera2.camera_info.K' in source_row:
        kk='raw.robot0.sensor.camera2.camera_info.K'
        packed=np.asarray(source_row['raw.robot0.sensor.camera2.camera_info.D'],float)
        require(np.allclose(packed[:4],k[[0,1,0,1],[0,1,2,2]],atol=1e-3,rtol=0),'double-sphere K/D source mismatch')
        require(np.allclose(packed[4:6],provenance['distortion'],atol=1e-8,rtol=0),'double-sphere source distortion mismatch')
    require(kk in source_row and ek in source_row,'projectable camera lacks recorded source fields')
    raw=np.asarray(source_row[kk],float).ravel()
    if raw.size==8 and model=='brown_conrady':
        expected=np.array([[raw[0],0,raw[2]],[0,raw[1],raw[3]],[0,0,1]])
        require(np.allclose(raw[4:],provenance['distortion'],atol=1e-8,rtol=0),'packed source distortion mismatch')
    else:
        require(raw.size==9,'unsupported independently audited intrinsic layout')
        expected=raw.reshape(3,3)
    require(np.allclose(expected,k,atol=1e-6,rtol=0),'intrinsic differs from current source row')
    require(np.allclose(np.asarray(source_row[ek],float).reshape(4,4),e,atol=1e-6,rtol=0),'extrinsic differs from current source row')
    return dict(checked=2,intrinsic_source_field=kk,extrinsic_source_field=ek)


def verify_reprojection(branch,view,source_row):
    """Recompute tile projection from world coordinates and recorded current K/E."""
    if not view['projection_available']:
        require(not np.asarray(branch['gt_projection_valid']).any() and not np.asarray(branch['pred_projectable']).any(),
            'projection marked valid without available calibration')
        require(bool(view.get('reason')),'unavailable projection lacks explanation')
        return dict(projected_points=0,source_camera_fields=0)
    from scripts.data.trajectory_projection import project_points,transform_points
    provenance=view['provenance']['camera'];k=np.asarray(provenance['K'],float)
    e=np.asarray(provenance['world_to_camera'],float);wh=np.asarray(provenance['native_wh'],float)
    source_check=check_camera_source(provenance,source_row)
    projected_points=0
    for xyz_key,valid_key,uv_key,projected_key,in_frame_key in [
        ('gt_world_xyz','gt_valid','gt_uv','gt_projection_valid','gt_in_frame'),
        ('world_xyz','valid','pred_uv','pred_projectable','pred_in_frame')]:
        world=np.asarray(branch[xyz_key],float);valid=np.asarray(branch[valid_key],bool)
        native,projectable=project_points(transform_points(e,world),k,model=provenance['model'],distortion=provenance.get('distortion'))
        projectable &= valid
        tile=(native+.5)*([320,256]/wh)-.5
        inside=projectable&(native>=0).all(-1)&(native<=wh-1).all(-1)&(tile>=0).all(-1)&(tile<=[319,255]).all(-1)
        require(np.array_equal(projectable,np.asarray(branch[projected_key],bool)),f'{uv_key}: camera projectability mismatch')
        require(np.array_equal(inside,np.asarray(branch[in_frame_key],bool)),f'{uv_key}: camera in-frame mismatch')
        require(np.allclose(tile[projectable],np.asarray(branch[uv_key],float)[projectable],rtol=0,atol=1e-3),f'{uv_key}: current-camera coordinate mismatch')
        projected_points+=int(projectable.sum())
    return dict(projected_points=projected_points,source_camera_fields=source_check['checked'],source=source_check)


def source_gt_audit(sample,sequence,coordinate):
    """Bounded source-row audit, preserving NULL/invalid annotations as invalid."""
    from scripts.data.trajectory_projection import goal_source_tcp
    from scripts.data.prepare_subtask_goals import recorded_tcp_transform
    anchor=coordinate['anchor'];end=min(anchor+33,sequence['end_exclusive'])
    rows=source_rows(sequence,start=anchor,end_exclusive=end).to_pylist()
    by_frame={int(row['frame_index']):row for row in rows}
    require(anchor in by_frame,'missing source anchor row')
    info=json.loads((Path(sequence['source_folder'])/'meta/info.json').read_text())
    transform=recorded_tcp_transform(sequence['family'],info)
    expected=np.full((32,2,3),np.nan);valid=np.zeros((32,2),bool)
    for j in range(max(0,end-anchor-1)):
        frame=anchor+j+1;require(frame in by_frame,'missing source future row')
        endpoint=goal_source_tcp(sequence['family'],by_frame[frame],tcp_from_eef=transform)
        expected[j]=endpoint['tcp_world'];valid[j]=endpoint['tcp_valid']
    count=0
    for role in ROLES:
        branch=coordinate['views'][role]['direct'];declared=np.asarray(branch['gt_valid'],bool)
        actual=np.asarray(branch['gt_world_xyz'],float);both=valid&declared
        require(not (valid&~declared).any(),f'{role}: available source GT silently omitted')
        require(np.allclose(actual[both],expected[both],atol=1e-4,rtol=0),f'{role}: GT differs from source endpoint rows')
        origin=np.asarray(branch['gt_source'])
        require(not ((origin=='source_endpoint_annotation')&declared&~valid).any(),'source-only GT not valid in source annotations')
        count+=int(both.sum())
    return by_frame[anchor],count


def read_json(path):return json.loads(Path(path).read_text())


def dependencies(root,video,records):
    paths=[root/'report'/video['video'],root/'report'/Path(video['video']).with_suffix('.json')]
    for row,record in zip(video['rows'],records):
        paths.append(root/'report'/row['coordinates'])
        for stage in ('stage1',video['mode']):
            folder=root/'inference'/stage/f'{record["id"]:06d}';paths.append(folder/'metadata.json')
            paths.append(folder/'prediction.pt')
            if stage=='stage1':paths.append(folder/'prediction.png')
    hashes={str(path.relative_to(root)):sha(path) for path in paths}
    hashes['validator_source']=VALIDATOR_SOURCE_SHA
    hashes['renderer_source']=hashlib.sha256((CODE/'scripts/experiments/fullspan_video.py').read_bytes()+
        (CODE/'scripts/experiments/fullspan_projection.py').read_bytes()).hexdigest()
    hashes['sample_manifest']=sha(root/'data/manifest.json')
    return hashes,hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()


def validate_video(root,sequence,video,records,dependency_hashes):
    import av
    expected=schedule(sequence['start'],sequence['end_exclusive'])
    for key in ('sequence_id','family','episode','scope','prompt','start','end_exclusive','target_frame'):
        require(video[key]==sequence[key],f'sequence metadata mismatch: {key}')
    require(video['source_frames']==expected==[r['anchor'] for r in records],'source frame schedule mismatch')
    require(video['frame_count']==len(expected)==len(video['rows']),'frame count metadata mismatch')
    require(video['kind']=='direct' and video['mode'] in ('joint','none'),'unexpected video branch')
    require(video['stride_source_frames']==10,'stride metadata mismatch')
    path=root/'report'/video['video'];require(sha(path)==video['sha256'],'video SHA mismatch')
    identities=[]
    for record in records:
        identities.append(dict(id=record['id'],sample_sha256=record['sha256'],
            stage1_sha=sha(root/'inference/stage1'/f'{record["id"]:06d}'/'metadata.json'),
            stage2_sha=sha(root/'inference'/video['mode']/f'{record["id"]:06d}'/'metadata.json')))
    expected_render=hashlib.sha256(json.dumps(dict(renderer_sha=dependency_hashes['renderer_source'],inputs=identities,
        sequence=sequence['sequence_id'],mode=video['mode'],kind='direct'),sort_keys=True).encode()).hexdigest()
    if expected_render!=video['render_identity']:
        return dict(status='stale',reason='renderer/input fingerprint changed; rerender required')
    summaries=[];gt_pixels=pred_pixels=stage1_pixels=source_checks=0
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        require(stream.codec_context.name=='h264','video codec must be H264')
        require(stream.codec_context.format.name=='yuv420p','video pixels must be yuv420p')
        require((stream.width,stream.height)==(1280,704),'video dimensions must be 1280x704')
        require(abs(float(stream.average_rate)-sequence['fps']/10)<1e-5,'video FPS mismatch')
        count=0
        for number,frame in enumerate(container.decode(stream)):
            require(number<len(records),'encoded video has extra frames');count+=1
            require(frame.pts is not None and abs(float(frame.time)-number/(sequence['fps']/10))<1e-4,'encoded frame timestamp mismatch')
            record=records[number];row=video['rows'][number];anchor=record['anchor']
            require(row['id']==record['id'] and row['input_frame']==anchor,'row/input identity mismatch')
            require(row['target_frame']==sequence['end_exclusive']-1 and row['chunk_horizon']==32,'target/chunk identity mismatch')
            require(row['mode']==video['mode'] and row['kind']=='direct','row mode mismatch')
            require(sha(record['sample_path'])==record['sha256'],'frozen sample SHA changed')
            sample=torch.load(record['sample_path'],map_location='cpu',weights_only=False)
            require(sample_identity(sample)==record['identity'],'conditioning identity mismatch')
            require(sample['report_target']['goal_source_frame']==row['target_frame'],'sample goal target mismatch')
            s1meta=read_json(root/'inference/stage1'/f'{record["id"]:06d}'/'metadata.json')
            s2meta=read_json(root/'inference'/video['mode']/f'{record["id"]:06d}'/'metadata.json')
            for meta in (s1meta,s2meta):
                ident=meta['identity'];require(ident['sample_sha256']==record['sha256'] and ident['anchor']==anchor and
                    ident['sequence_id']==sequence['sequence_id'],'prediction/sample identity mismatch')
                require(identity_contains(meta['input_identity'],record['identity']),'prediction conditioning differs')
                extra=meta['input_identity']
                for key in ('proprio','proprio_mask'):
                    require(extra['conditioning_sha256'][key]==tensor_sha(sample[key]),key+' input hash mismatch')
                require(extra['conditioning_sha256']['active_action_mask']==tensor_sha(torch.as_tensor(sample['_eval']['active_action_mask'])),'active action mask input hash mismatch')
                require(extra['first_frame_sha256']==[hashlib.sha256(im.tobytes()).hexdigest() for im in sample['first_frame_image']],'first-frame input hash mismatch')
                require(extra['goal_seed']==sample['goal_seed'] and extra['goal_head_valid']==sample['goal_head_valid'],'goal input identity mismatch')
                folder=root/'inference'/ident['stage']/f'{record["id"]:06d}'
                for artifact in ('prediction.pt','prediction.png'):
                    if artifact in meta['artifacts_sha256']:
                        require(sha(folder/artifact)==meta['artifacts_sha256'][artifact],'prediction artifact SHA mismatch')
            require(s1meta['identity']['checkpoint_sha256']==s2meta['identity']['stage1_sha256'],'Stage1 teacher checkpoint mismatch')
            s2=torch.load(root/'inference'/video['mode']/f'{record["id"]:06d}'/'prediction.pt',map_location='cpu',weights_only=False)
            s1=torch.load(root/'inference/stage1'/f'{record["id"]:06d}'/'prediction.pt',map_location='cpu',weights_only=False)
            validate_raw_outputs(s1['predicted_uv'],s2['normalized_actions'],s1meta['predicted_uv'])
            coordinate=read_json(root/'report'/row['coordinates'])
            require(coordinate['anchor']==anchor and coordinate['source_frames']==list(range(anchor+1,anchor+33)),'coordinate source frame schedule')
            source_anchor,checked=source_gt_audit(sample,sequence,coordinate);source_checks+=checked
            rgb=frame.to_ndarray(format='rgb24');images=current_images(sample);details={}
            for col,role in enumerate(ROLES):
                view=coordinate['views'][role];declared=row['views'][role]
                available=bool(sample['goal_view_valid'][col])
                require(view['image_available']==available==declared['image_available'],'image availability mismatch')
                require(view['source_frame']==anchor,'camera belongs to a different source frame')
                if available:
                    align=view['provenance']['video_alignment'];require(align['source_frame_index']==anchor,'decoded camera/source mismatch')
                    require(view['camera_key']==sequence['cameras'][col],'camera role mismatch')
                branch=view['direct'];counts=validate_branch(branch,anchor=anchor,end=sequence['end_exclusive'])
                require(counts['gt_points']==declared['gt_points'] and counts['pred_points']==declared['pred_points'],'advertised marker count mismatch')
                calibration=verify_reprojection(branch,view,source_anchor)
                pixel={}
                for label,y,color,key in [('gt',96,'green','gt'),('pred',400,'red','pred')]:
                    tile=rgb[y:y+256,col*320:(col+1)*320]
                    if not available:
                        black=black_region(tile);require(black['passed'],f'{role}/{label}: missing camera region not black')
                        pixel[label]=dict(black=black,markers=0);continue
                    error=background_error(tile,images[col]);require(error<12,f'{role}/{label}: current image/background mismatch ({error:.3f})')
                    pts=np.asarray(branch[key+'_uv'],float);valid=np.asarray(branch[key+'_in_frame'],bool)
                    marks=[marker_pixels(tile,point,color) for point in pts[valid]]
                    require(all(n>=2 for n in marks),f'{role}/{label}: eligible coordinate has no visible {color} marker')
                    pixel[label]=dict(markers=len(marks),minimum_marker_pixels=min(marks) if marks else None,background_mae=error)
                    if label=='gt':gt_pixels+=len(marks)
                    else:pred_pixels+=len(marks)
                details[role]=dict(coordinates=counts,calibration=calibration,pixels=pixel)
            goal=sample['_eval']['report_goal'];prediction=Image.open(root/'inference/stage1'/f'{record["id"]:06d}'/'prediction.png')
            gt_tile=rgb[96:352,960:1280];pred_tile=rgb[400:656,960:1280]
            if goal is None:
                require(black_region(gt_tile)['passed'],'unavailable instruction-end GT image must stay black')
            else:
                require(background_error(gt_tile,goal)<12,'instruction-end GT image mismatch')
            require(background_error(pred_tile,prediction)<12,'Stage1 image differs from same-anchor prediction')
            projection=sample['report_target'].get('projection',{});uv=np.asarray(projection.get('uv',np.zeros((2,2))),float)
            valid=(goal is not None)&np.asarray(projection.get('projection_valid',[False,False]),bool)&np.asarray(projection.get('in_frame',[False,False]),bool)&np.isfinite(uv).all(-1)
            pred=np.asarray(s1meta['predicted_uv'],float).reshape(2,2);stage1_counts=[]
            for side in range(2):
                if valid[side]:
                    require(marker_pixels(gt_tile,uv[side]*[319,255],'green')>=2,'Stage1 GT endpoint marker absent');stage1_pixels+=1
                if np.isfinite(pred[side]).all() and (pred[side]>=0).all() and (pred[side]<=1).all():
                    color='red' if valid[side] else 'orange'
                    require(marker_pixels(pred_tile,pred[side]*[319,255],color)>=2,'Stage1 predicted endpoint marker absent');stage1_pixels+=1
                stage1_counts.append(dict(gt_valid=bool(valid[side]),prediction_has_gt=bool(valid[side])))
            summaries.append(dict(id=record['id'],source_frame=anchor,target_frame=row['target_frame'],views=details,
                source_gt_checked_view_points=checked,stage1=stage1_counts))
            del sample,s1,s2
        require(count==len(expected),'encoded frame count mismatch')
    return dict(status='passed',decoded_frames=count,gt_markers=gt_pixels,pred_markers=pred_pixels,
        stage1_markers=stage1_pixels,source_gt_checked_view_points=source_checks,frames=summaries,
        codec='h264',pixel_format='yuv420p',width=1280,height=704)


class Page(HTMLParser):
    def __init__(self):super().__init__();self.sections=set();self.videos={}
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='section':self.sections.add(a.get('id'))
        if tag=='video':self.videos[a.get('src')]='controls' in a


def validate_once(root):
    manifest=read_json(root/'data/manifest.json');report=root/'report';expected=[(s['sequence_id'],mode) for s in manifest['sequences'] for mode in ('joint','none')]
    previous=read_json(report/'validation.json') if (report/'validation.json').exists() else {}
    old={r['key']:r for r in previous.get('videos',[])} if previous.get('schema')=='openwam.fullspan_video_validation.v2' else {}
    coverage=read_json(report/'coverage.json');page=Page();page.feed((report/'index.html').read_text())
    result=dict(schema='openwam.fullspan_video_validation.v2',status='running',expected_sequences=len(manifest['sequences']),
        expected_videos=len(expected),completed_videos=coverage['completed_videos'],verified_videos=0,decoded_frames=0,videos=[],errors=[],
        validation_scope='all encoded frames; source/target identities, physical coordinate arithmetic, actual pixel markers; not physical contact-center certification',
        cache_policy='reuse only unchanged verified video/sidecar/coordinate/prediction SHA dependencies and frozen sample-manifest SHA')
    if page.sections!={s['sequence_id'] for s in manifest['sequences']}:result['errors'].append('HTML expected sequence sections mismatch')
    seen=set()
    for video in coverage['videos']:
        key=video['sequence_id']+'/'+video['mode'];pair=(video['sequence_id'],video['mode']);record=dict(key=key,video=video['video'])
        try:
            require(pair in expected and pair not in seen,'unexpected or duplicate sequence/mode');seen.add(pair)
            require(page.videos.get(video['video'],False),'video missing HTML controls/link')
            seq=next(s for s in manifest['sequences'] if s['sequence_id']==video['sequence_id'])
            records=[r for r in manifest['anchors'] if r['sequence_id']==seq['sequence_id']]
            sidecar=read_json((report/video['video']).with_suffix('.json'))
            if sidecar!=video:
                record.update(status='stale',reason='coverage/sidecar publication in progress')
            else:
                deps,digest=dependencies(root,video,records)
                if key in old and old[key].get('status')=='passed' and old[key].get('dependency_sha256')==digest:
                    record=dict(old[key],reused_verified_sha=True)
                else:
                    record.update(validate_video(root,seq,video,records,deps),dependency_sha256=digest,reused_verified_sha=False)
                    record['verified_dependency_hashes']=deps
        except Exception as exc:record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        result['videos'].append(record)
        print(json.dumps(dict(event='video_validated',key=key,status=record['status'],error=record.get('error'),reused=record.get('reused_verified_sha',False))),flush=True)
        if record['status']=='passed':result['verified_videos']+=1;result['decoded_frames']+=record['decoded_frames']
        if record['status']=='failed':result['errors'].append(key+': '+record['error'])
    result['missing_videos']=[sid+'/'+mode for sid,mode in expected if (sid,mode) not in seen]
    if coverage['completed_videos']!=len(coverage['videos']):result['errors'].append('coverage completed count mismatch')
    result['status']='failed' if result['errors'] else 'complete' if result['verified_videos']==len(expected) else 'running'
    result['updated_at_unix']=time.time();dump(report/'validation.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--watch',action='store_true');p.add_argument('--poll-seconds',type=float,default=20)
    args=p.parse_args();torch.set_num_threads(2)
    while True:
        result=validate_once(args.root.resolve());print(json.dumps({k:result[k] for k in ('status','expected_videos','completed_videos','verified_videos','decoded_frames','errors')}),flush=True)
        if result['status']=='complete' or not args.watch:break
        time.sleep(max(1,args.poll_seconds))
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
