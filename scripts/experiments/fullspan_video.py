"""Two-row complete-instruction videos: current three views + Stage1 keyframe."""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import html
import json
from pathlib import Path
import sys
import textwrap
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

CODE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(CODE))
from scripts.experiments.infer_stage1_samecase import dump,sha,target_data
from scripts.experiments.fullspan_projection import prepare_projection_geometry,project_action_chunks

ROLES=('head','left_wrist','right_wrist')
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
GREEN='#00e676'; RED='#ff4747'; ORANGE='#ffaf35'


def complete_anchors(start,end):
    if end<=start:raise ValueError('empty instruction interval')
    return sorted(set([*range(start,end,10),end-1]))


def current_views(sample):
    value=sample['goal_observed_multiview'][:,0].float().permute(1,2,0).numpy()
    image=Image.fromarray(np.rint((value+1)*127.5).clip(0,255).astype(np.uint8))
    return [image.crop(box).resize((320,256),Image.Resampling.BILINEAR)
        for box in [(0,0,320,256),(0,256,160,384),(160,256,320,384)]]


def draw_path(image,points,mask,color):
    draw=ImageDraw.Draw(image);font=ImageFont.truetype(FONT,11)
    points=np.asarray(points,float);mask=np.asarray(mask,bool)
    counts=[]
    for side in range(2):
        valid=mask[:,side]&np.isfinite(points[:,side]).all(axis=-1)
        for j in range(1,len(points)):
            if valid[j-1] and valid[j]:draw.line([tuple(points[j-1,side]),tuple(points[j,side])],fill=color,width=2)
        ids=np.flatnonzero(valid);counts.append(len(ids))
        for j in ids:
            x,y=points[j,side];draw.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2)
        if len(ids):
            x,y=points[ids[-1],side];draw.text((min(295,max(0,x+4)),min(240,max(0,y))),('L','R')[side],font=font,fill=color,stroke_width=1,stroke_fill='black')
    return counts


def compose_frame(sample,predicted_image,tcp,projection,mode,kind='direct'):
    """Draw overlays in 320x256 tile coordinates before assembling the grid."""
    canvas=Image.new('RGB',(1280,704),'#142238');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(FONT,14);small=ImageFont.truetype(FONT,12)
    anchor=int(sample['trace']['anchor']);target=int(sample['instruction_end_exclusive'])-1
    text=f"{sample['granularity'].upper()}: {sample['prompt']}"
    for i,line in enumerate(textwrap.wrap(text,width=155)[:3]):draw.text((8,5+i*18),line,font=font,fill='white')
    draw.text((8,60),f'Input f{anchor} | Goal f{target} | {mode} | {kind.upper()} action chunk: f{anchor+1}..f{anchor+32} | green GT / red prediction',font=font,fill='white')
    masks=np.asarray(sample['goal_view_valid'],bool).reshape(3);views=current_views(sample)
    goal_available=sample['_eval'].get('report_goal') is not None
    meta=dict(input_frame=anchor,target_frame=target,chunk_horizon=32,mode=mode,kind=kind,views={},goal_image_available=goal_available)
    for row,y in [(0,96),(1,400)]:
        label='GT' if row==0 else 'PRED';color=GREEN if row==0 else RED
        for col,role in enumerate(ROLES):
            info=projection['views'][role];available=bool(masks[col] and info['image_available'])
            tile=views[col].copy() if available else Image.new('RGB',(320,256),'black')
            branch=info.get(kind,{})
            points=branch.get('gt_uv' if row==0 else 'pred_uv',np.full((32,2,2),np.nan))
            valid=branch.get('gt_in_frame' if row==0 else 'pred_in_frame',np.zeros((32,2),bool))
            counts=draw_path(tile,points,valid,color) if available else [0,0]
            canvas.paste(tile,(col*320,y))
            draw.text((col*320+6,y-19),f'{label} | {role} | current f{anchor}',font=font,fill=color)
            if sum(counts):status=f'future points L/R: {counts[0]}/{counts[1]}'
            elif not available:status='NO CAMERA'
            elif not info.get('projection_available',False):status='CAMERA PROJECTION UNAVAILABLE'
            elif np.asarray(branch.get('gt_projection_valid' if row==0 else 'pred_projectable',False)).any():status='FUTURE POINTS OUTSIDE VIEW'
            else:status='NO VALID SOURCE TCP' if row==0 else 'NO VALID PREDICTED TCP MAPPING'
            draw.text((col*320+6,y+259),status,font=small,fill='white' if sum(counts) else ORANGE)
            if row==0:meta['views'][role]=dict(image_available=available,gt_points=counts,reason=info.get('reason',''))
            else:meta['views'][role]['pred_points']=counts
        source=sample['_eval'].get('report_goal') if row==0 else predicted_image
        goal=source.convert('RGB').resize((320,256),Image.Resampling.BILINEAR) if source is not None else Image.new('RGB',(320,256),'black')
        gd=ImageDraw.Draw(goal)
        for arm in tcp['arms']:
            gt_valid=bool(arm['gt_valid'] and goal_available)
            valid=gt_valid if row==0 else arm['pred_drawable']
            if not valid:continue
            x,y0=arm['gt_px'] if row==0 else arm['pred_px'];c=GREEN if row==0 else RED if gt_valid else ORANGE
            if row==0:gd.rectangle((x-4,y0-4,x+4,y0+4),outline=c,width=2)
            else:gd.ellipse((x-4,y0-4,x+4,y0+4),outline=c,width=2)
            gd.text((min(295,x+5),min(240,y0+5)),arm['side'],font=small,fill=c,stroke_width=1,stroke_fill='black')
        canvas.paste(goal,(960,y));draw.text((966,y-19),f'{label} | Stage1 keyframe f{target}',font=font,fill=color)
        subtitle=(f'Source GT TCP: {tcp["valid_gt_count"]}/2' if goal_available else 'NO TARGET HEAD IMAGE') if row==0 else 'Learned 2D TCP; orange = no valid GT'
        draw.text((966,y+259),subtitle,font=small,fill='white' if tcp['valid_gt_count'] else ORANGE)
    valid_steps=np.asarray(sample['action_mask']).reshape(32,-1).any(-1).sum()
    draw.text((8,682),f'Action shape 32 x 112 | source-supervised steps {valid_steps}/32 | current cameras project the whole future chunk | missing views stay black',font=small,fill='white')
    meta['valid_action_steps']=int(valid_steps);return canvas,meta


def inference_folder(root,stage,record):return root/'inference'/stage/f'{record["id"]:06d}'


def ready(root,stage,record):return (inference_folder(root,stage,record)/'metadata.json').exists()


def checked_prediction(root,stage,record):
    folder=inference_folder(root,stage,record);meta=json.loads((folder/'metadata.json').read_text())
    identity=meta['identity']
    if identity['anchor']!=record['anchor'] or identity['sequence_id']!=record['sequence_id'] or identity['sample_sha256']!=record['sha256']:
        raise ValueError('prediction identity does not match current sample')
    for name,expected in meta['artifacts_sha256'].items():
        if sha(folder/name)!=expected:raise ValueError('prediction artifact changed')
    return folder,torch.load(folder/'prediction.pt',map_location='cpu',weights_only=False)


def render_sequence(root,sequence,records,mode,kind='direct'):
    import av
    out=root/'report';dest=out/'videos';dest.mkdir(parents=True,exist_ok=True)
    name=f'{sequence["sequence_id"]}_{mode}_{kind}'
    if [r['anchor'] for r in records]!=complete_anchors(sequence['start'],sequence['end_exclusive']):
        raise ValueError('incomplete instruction span')
    path=dest/(name+'.mp4');sidecar=dest/(name+'.json')
    renderer_sha=hashlib.sha256(Path(__file__).read_bytes()+(CODE/'scripts/experiments/fullspan_projection.py').read_bytes()).hexdigest()
    input_identities=[dict(id=r['id'],sample_sha256=r['sha256'],stage1_sha=sha(inference_folder(root,'stage1',r)/'metadata.json'),
        stage2_sha=sha(inference_folder(root,mode,r)/'metadata.json')) for r in records]
    render_identity=hashlib.sha256(json.dumps(dict(renderer_sha=renderer_sha,inputs=input_identities,sequence=sequence['sequence_id'],mode=mode,kind=kind),sort_keys=True).encode()).hexdigest()
    if path.exists() and sidecar.exists():
        old=json.loads(sidecar.read_text())
        if old.get('render_identity')==render_identity and old.get('sha256')==sha(path):return old
    pending=dest/('.'+name+'.mp4');rows=[]
    fps=Fraction(str(sequence['fps']))/10
    with av.open(str(pending),'w',options={'movflags':'+faststart'}) as container:
        stream=container.add_stream('libx264',rate=fps);stream.width=1280;stream.height=704;stream.pix_fmt='yuv420p'
        stream.options={'crf':'21','preset':'fast'}
        for j,record in enumerate(records):
            sample=torch.load(record['sample_path'],map_location='cpu',weights_only=False)
            sf,s1=checked_prediction(root,'stage1',record);_,s2=checked_prediction(root,mode,record)
            cache=root/'projection'/f'{record["id"]:06d}';cache.mkdir(parents=True,exist_ok=True)
            gp=cache/'geometry.pt'
            cached=torch.load(gp,map_location='cpu',weights_only=False) if gp.exists() else {}
            if cached.get('sample_sha256')==record['sha256'] and cached.get('renderer_sha')==renderer_sha:
                geometry=cached['geometry']
            else:
                geometry=prepare_projection_geometry(sample)
                torch.save(dict(sample_sha256=record['sha256'],renderer_sha=renderer_sha,geometry=geometry),gp)
            projection=project_action_chunks(sample,s2['normalized_actions'].numpy(),geometry)
            coordinate_dir=out/'coordinates';coordinate_dir.mkdir(exist_ok=True)
            coordinate_path=coordinate_dir/f'{mode}_{record["id"]:06d}.json'
            dump(coordinate_path,projection)
            tcp=target_data(sample,s1['predicted_uv'].numpy())
            frame,meta=compose_frame(sample,Image.open(sf/'prediction.png'),tcp,projection,mode,kind)
            rows.append(dict(id=record['id'],coordinates=str(coordinate_path.relative_to(out)),endpoint=projection.get('endpoint'),**meta))
            if j in {0,len(records)//2,len(records)-1}:frame.save(dest/f'{name}_{j:04d}.png')
            for packet in stream.encode(av.VideoFrame.from_image(frame)):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    pending.replace(path)
    result=dict(sequence_id=sequence['sequence_id'],family=sequence['family'],episode=sequence['episode'],scope=sequence['scope'],prompt=sequence['prompt'],mode=mode,kind=kind,
        start=sequence['start'],end_exclusive=sequence['end_exclusive'],target_frame=sequence['target_frame'],
        source_frames=[r['anchor'] for r in records],stride_source_frames=10,fps=float(fps),frame_count=len(rows),
        video='videos/'+path.name,sha256=sha(path),rows=rows,render_identity=render_identity,
        timeline='Full instruction interval sampled every10 source frames plus final frame; last short interval is displayed at nominal sampled FPS.')
    dump(sidecar,result);return result


def publish_index(root,manifest,results):
    out=root/'report';out.mkdir(exist_ok=True)
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>GT / 预测：完整指令视频</title><style>body{font:16px system-ui;max-width:1320px;margin:24px auto;padding:12px}video{width:100%}section{margin:32px 0}p{line-height:1.6}</style><h1>完整 subtask / task：GT 与预测</h1><p>上排 GT，下排预测；每排依次为当前头图、左腕、右腕、Stage1 关键帧。每10源帧重新预测一次，补齐区间最后一帧。当前三视图各投影未来32步；缺腕图为黑图，无有效标定/端点时说明无法投影。绿色为源GT，红色为预测，橙色为无有效GT的二维输出。</p>']
    for sequence in manifest['sequences']:
        parts.append(f'<section id="{sequence["sequence_id"]}"><h2>{html.escape(sequence["family"])} · episode {sequence["episode"]} · {sequence["scope"]}</h2><p>{html.escape(sequence["prompt"])}</p><p>完整范围 f{sequence["start"]}–f{sequence["end_exclusive"]-1}；{len(sequence["anchors"])} 次预测。源视频 {sequence["fps"]:g} fps；采样预览 {sequence["fps"]/10:g} fps。</p>')
        for mode in ('joint','none'):
            row=next((r for r in results if r['sequence_id']==sequence['sequence_id'] and r['mode']==mode and r['kind']=='direct'),None)
            if row:
                evidence=row['video'].replace('.mp4','.json')
                parts.append(f'<h3>{mode}</h3><video controls preload="metadata" src="{row["video"]}"></video><details><summary>投影说明</summary><p>源端点可能是人手指尖中点、采集设备TCP或机器人声明的工具点；不默认认证为夹爪接触中心。缺标定、出画或缺模型通道分别记录。</p><a href="{evidence}">逐帧坐标来源与有效范围</a></details>')
            else:parts.append(f'<p>{mode}：完整区间正在生成。</p>')
        parts.append('</section>')
    parts.append('<a href="coverage.json">帧号与投影覆盖记录</a></html>')
    tmp=out/'index.pending.html';tmp.write_text('\n'.join(parts));tmp.replace(out/'index.html')
    dump(out/'coverage.json',dict(expected_sequences=len(manifest['sequences']),completed_videos=len(results),status='complete' if len(results)==2*len(manifest['sequences']) else 'running',videos=results))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--watch',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);root=a.root.resolve();results=[]
    while True:
        manifest=json.loads((root/'data/manifest.json').read_text())
        for sequence in manifest['sequences']:
            records=[r for r in manifest['anchors'] if r['sequence_id']==sequence['sequence_id']]
            for mode in ('joint','none'):
                if any(r['sequence_id']==sequence['sequence_id'] and r['mode']==mode for r in results):continue
                if not all(r['status']=='ready' and ready(root,'stage1',r) and ready(root,mode,r) for r in records):continue
                result=render_sequence(root,sequence,records,mode);results.append(result)
                print(json.dumps(dict(event='video_complete',sequence=sequence['sequence_id'],mode=mode,frames=result['frame_count'])),flush=True)
        publish_index(root,manifest,results)
        if len(results)==2*len(manifest['sequences']) or not a.watch:break
        time.sleep(10)


if __name__=='__main__':main()
