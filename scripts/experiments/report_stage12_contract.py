"""Build the complete same-case Stage1/Stage2 report without changing training assets."""
from __future__ import annotations
import argparse
import html
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

CODE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CODE))
from scripts.experiments.infer_stage1_samecase import sample_identity, dump, sha
from scripts.experiments.report_stage12_timed import video
E=html.escape


def read(path):
    return json.loads(path.read_text())


def frame_contract(sample):
    """Selected instruction boundaries and declared source timeline, never text guesses."""
    trace=sample['trace'];anchor=int(trace['anchor'])
    start=int(sample['instruction_start']);end=int(sample['instruction_end_exclusive'])
    frames=[int(x) for x in trace['video_source_indices']]
    valid=np.asarray(sample['video_mask'],bool)
    if not start<=anchor<end:raise ValueError('input frame lies outside selected instruction interval')
    if sample['report_target']['goal_source_frame']!=end-1:raise ValueError('instruction target must equal end_exclusive - 1')
    if not frames or frames[0]!=anchor:raise ValueError('source timeline does not begin at actual input frame')
    if valid.shape!=(len(frames),) or not valid[0]:raise ValueError('source frame validity does not match timeline')
    if any(b<=a for a,b in zip(frames,frames[1:])):raise ValueError('source timeline must be increasing')
    if frames[-1]!=anchor+len(sample['action']):raise ValueError('source timeline differs from declared action horizon')
    kind=sample['granularity']
    if kind not in ('task','subtask'):raise ValueError('unknown selected instruction kind')
    return dict(input_frame=anchor,instruction_kind=kind,instruction_start_frame=start,
        instruction_end_exclusive=end,target_frame=end-1,stage2_horizon_last_frame=frames[-1],
        stage2_valid_action_last_frame=min(frames[-1],int(trace['last_source'])),
        stage2_last_valid_video_frame=max(f for f,v in zip(frames,valid) if v),
        stage2_video_source_frames=frames,stage2_video_valid=valid.tolist(),
        stage2_horizon_semantics='nominal action horizon; masked/padded frames are not valid source observations')


def frame_label_lines(frame):
    return [f"INPUT f{frame['input_frame']} -> {frame['instruction_kind']} TARGET f{frame['target_frame']}",
        f"Selected interval [f{frame['instruction_start_frame']}, f{frame['instruction_end_exclusive']})",
        f"end_exclusive f{frame['instruction_end_exclusive']} (boundary)",
        f"Stage2 horizon f{frame['stage2_horizon_last_frame']} (nominal)",
        f"valid action <= f{frame['stage2_valid_action_last_frame']} | video <= f{frame['stage2_last_valid_video_frame']}"]


def label_display_image(source,destination,frame,*,role,valid_gt_count=None):
    """Append a banner after rendered pixels; never move/recompute TCP coordinates."""
    if Path(source).resolve()==Path(destination).resolve():raise ValueError('display banner must not overwrite its source')
    image=Image.open(source).convert('RGB');width,height=image.size
    labels=[role,*frame_label_lines(frame)]
    if valid_gt_count is not None:
        labels+=([f'NO VALID GT {valid_gt_count}/2','Untrusted 2D model output; NOT physical TCP']
            if valid_gt_count==0 else [f'SOURCE-VALID GT {valid_gt_count}/2 | 2D prediction','Physical contact midpoint NOT certified'])
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    regular=ImageFont.truetype(font_path,11);large=ImageFont.truetype(font_path,17)
    result=Image.new('RGB',(width,height+22+19*len(labels)),'#142238');result.paste(image,(0,0))
    draw=ImageDraw.Draw(result);y=height+9
    for line in labels:
        emph=line.startswith(('NO VALID GT','SOURCE-VALID GT'))
        font=large if emph else regular
        if draw.textbbox((0,0),line,font=font)[2]>width-14:
            font=ImageFont.truetype(font_path,10)
        draw.text((7,y),line,font=font,fill='#ffb347' if line.startswith(('NO VALID GT','Untrusted')) else 'white')
        y+=19
    result.save(destination)
    return dict(source=str(source),source_sha256=sha(source),asset=str(destination),asset_sha256=sha(destination),
        labels=labels,frame_contract=frame,image_region_xywh=[0,0,width,height],banner_origin_xy=[0,height],
        coordinate_policy='Original rendered pixels unchanged; banner appended below; UV origin stays at image (0,0).')


def discover_variants(root):
    result=[]
    for mode in ('joint','none'):
        dirs=sorted((root/mode).glob('eval_[0-9]*'))
        if not dirs: raise ValueError(f'No evaluation directories for {mode}')
        for directory in dict.fromkeys((dirs[0],dirs[-1])):
            step=int(directory.name.split('_')[1])
            result.append(dict(mode=mode,step=step,key=f'{mode}_step{step}',directory=directory))
    return result


def checked_stage1(folder,sample,identity):
    metadata=folder/'metadata.json'
    if not metadata.exists():
        return dict(status='missing',reason='Same-case Stage1 forward metadata not yet completed.')
    meta=read(metadata);actual=meta.get('identity',{});tr=sample['trace']
    expected=dict(source_repo=tr['source_repo'],episode=tr['episode'],anchor=tr['anchor'],prompt=sample['prompt'])
    if any(actual.get(k)!=v for k,v in expected.items()):
        raise ValueError(f'{folder}: Stage1 identity differs from actual sample')
    if actual!=sample_identity(sample):raise ValueError(f'{folder}: Stage1 conditioning identity/hash mismatch')
    if meta.get('goal_frame')!=sample['instruction_end_exclusive']-1:
        raise ValueError(f'{folder}: wrong Stage1 instruction-end target')
    if meta.get('anchor_frame')!=tr['anchor']:raise ValueError(f'{folder}: wrong Stage1 anchor')
    paths=['prediction.png','tcp_overlay.png','target_tcp_overlay.png']
    if not all((folder/p).is_file() for p in paths):
        return dict(status='missing',reason='Stage1 metadata exists but prediction/TCP image is missing.')
    count=meta['tcp']['valid_gt_count']
    return dict(status='ready',status_semantics='artifact availability only; does not certify prediction trust or physical calibration',
        coordinate_output_kind='stage1_2d_model_output',gt_coordinate_output_kind='physical_gt_projection',
        prediction_status='unverified_no_gt' if count==0 else 'compared_with_valid_gt',
        physical_tcp_definition={k:meta.get('report_target',{}).get(k) for k in ('endpoint','camera','calibration_eligibility')},
        identity=identity,assets=paths,evidence=['metadata.json'],goal_frame=meta['goal_frame'],checkpoint_sha256=meta.get('checkpoint_sha256'),mean_tcp_error_px=meta.get('mean_tcp_error_px'),valid_gt_count=count)


def projection_lane(folder,root,branch,payload,sample):
    evidence=[str((folder/name).relative_to(root)) for name in ('projection.json','video_projection.json','video_camera_geometry.json')]
    if payload is None:return dict(status='missing',reason='Projection coordinate payload not rendered yet.',evidence=[p for p in evidence if (root/p).exists()])
    valid=np.asarray(payload[branch].get('valid',[]),bool).any()
    name=f'video_{branch}_tcp.mp4';path=folder/name
    if valid:
        if not path.is_file():return dict(status='missing',reason='Prediction trajectory exists; required moving-background overlay video is missing.',evidence=evidence)
        return dict(status='ready',assets=[str(path.relative_to(root))],evidence=evidence)
    masks=sample['trace'].get('mask_reasons',[])
    camera=bool(np.asarray(payload.get('camera_eligible',[])).any())
    if branch=='fk':reason='No matched, verified FK payload/robot-to-camera registration for these joint outputs.'
    elif 'missing_tcp_action_mapping' in masks:
        reason='No reviewed TCP endpoint mapping; native TCP action/anchor channels are masked. Raw source EEF may exist but is not a verified TCP conversion.'
    else:reason=payload.get('reason') or 'Native TCP trajectory unavailable under the source/calibration masks.'
    reason+= ' Head camera available.' if camera else ' Head camera calibration unavailable/ineligible.'
    return dict(status='unavailable',reason=reason,evidence=evidence)


def media(path):
    if path.endswith('.mp4'):
        return f'<video controls preload="metadata" src="{E(path)}"></video>'
    return f'<img loading="lazy" src="{E(path)}">'


def lane_html(title,lane):
    cls='lane stage1' if 'valid_gt_count' in lane else 'lane'
    body=f'<div class="{cls}"><h4>{E(title)} — {E(lane["status"])}</h4>'
    points='源数据有效 GT 投影 + 二维模型预测（物理接触中点未认证）' if lane.get('valid_gt_count',0)>0 else '无有效 GT：未验证二维模型输出'
    captions={'prediction.png':'Stage1 预测图像（展示副本附帧号）','tcp_overlay.png':'预测图像背景：'+points,'target_tcp_overlay.png':'真实 instruction-end 图像背景：'+points}
    for path in lane.get('assets',[]):
        role=lane.get('asset_roles',{}).get(path,Path(path).name)
        caption=captions.get(role)
        body+=('<figure>'+media(path)+'<figcaption>'+caption+'</figcaption></figure>') if caption else media(path)
    if 'valid_gt_count' in lane:
        if lane['valid_gt_count']==0:body+='<p class="no-gt"><strong>无有效 GT · 0/2</strong><br>橙色点是未验证二维模型输出，不是物理 TCP 投影；ready 仅表示展示资产已生成。</p>'
        error=lane.get('mean_tcp_error_px')
        metric=f'{error:.2f} px' if error is not None else 'N/A（无有效 GT，不计误差）'
        body+=f'<p>目标源帧 {lane["goal_frame"]} · 源数据有效 GT {lane["valid_gt_count"]}/2 · TCP 平均像素误差 {metric}。物理接触中点未认证；误差仅比较源定义坐标。</p>'
    if lane.get('reason'):body+=f'<p class="notice">{E(lane["reason"])}</p>'
    body+=' '.join(f'<a href="{E(p)}">{E(Path(p).name)}</a>' for p in lane.get('evidence',[]))
    return body+'</div>'


def endpoint_html(details,audit_path=None):
    """Show recorded endpoint semantics verbatim; do not infer a gripper center."""
    endpoint=details.get('endpoint') or {}
    body='<section class="endpoint-definition"><h3>物理 TCP 定义 · 源数据端点约定</h3>'
    body+='<p>以下保留源姿态、设备端点或人手代理点的实际定义；Stage1 二维模型输出本身不建立物理端点或相机标定。</p><dl>'
    for key in ('definition_id','endpoint_semantics','used_fields','transforms','missing_reasons'):
        value=endpoint.get(key)
        text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)
        body+=f'<dt>{E(key)}</dt><dd><code>{E(text)}</code></dd>'
    body+='</dl><details><summary>端点完整来源、相机参数与标定资格</summary><pre>'+E(json.dumps(details,ensure_ascii=False,indent=2))+'</pre></details>'
    if audit_path:body+=f'<p><a href="{E(audit_path)}">端点与坐标链核验：endpoint_audit_v3.json</a>（计算链一致性与物理标定认证分别记录。）</p>'
    return body+'</section>'


def physical_contact_review(audit,case_id,sample,root):
    """Physical acceptance is independent of source validity and projection math."""
    suspects=audit.get('visual_suspects',{});entry=suspects.get('case'+case_id)
    review=dict(status='not_independently_certified',physical_contact_certification='not_established',
        source_gt_validity_semantics='Source annotation/projection validity; not physical contact-midpoint certification',
        audit='endpoint_audit_v3.json' if audit else None)
    if not entry:return review
    if entry.get('goal_frame')!=sample['report_target']['goal_source_frame']:
        raise ValueError('physical review target does not match this case')
    for key,target in [('source_repo',sample['trace'].get('source_repo')),('episode',sample['trace'].get('episode'))]:
        if key in entry and entry[key]!=target:raise ValueError('physical review source identity differs')
    review.update(status='visual_suspect_unresolved',visual_observation=entry['visual_observation'],
        recommended_report_semantics=entry.get('recommended_report_semantics'),audit_details=entry,native_images=[])
    for path in suspects.get('viewed_native_images',[]):
        path=Path(path)
        if path.name.startswith('case'+case_id+'_'):
            if not path.is_file():raise FileNotFoundError(path)
            review['native_images'].append(str(path.resolve().relative_to(root.resolve())))
    return review


def physical_review_html(review,stage):
    suspect=review['status']=='visual_suspect_unresolved'
    css='physical-review suspect' if suspect else 'physical-review'
    title=f'{stage}：物理接触点疑点未解决' if suspect else f'{stage}：物理接触中点未独立认证'
    body=f'<aside class="{css}"><h4>{E(title)}</h4><p>源数据有效 GT 只表示源标注／投影 mask 有效，物理接触中点未认证。像素计算一致性检查不能证明指垫中点正确。</p>'
    if suspect:
        body+='<p><strong>'+E(review['visual_observation'])+'</strong></p>'
        details=review['audit_details']
        arms=details.get('arms',{})
        if arms:
            distances=[f'{side}: {data["source_EE_to_TCP_distance_m"]:.3f} m' for side,data in arms.items() if 'source_EE_to_TCP_distance_m' in data]
            if distances:body+='<p>源记录 flange → TCP 距离：'+E('; '.join(distances))+'。可见指垫中点与声明源 TCP 的疑点仍未解决；当前证据不足以指定新的固定偏移。</p>'
        camera=details.get('camera',{});device=camera.get('device',{})
        if device.get('calibration_source_device_sn'):
            body+='<p>头相机标定来源 '+E(device['calibration_source_device_sn'])+' → 当前设备 '+E(device['device_sn'])+'；重现部署矩阵不代表本 episode 的独立标定认证。</p>'
        if review.get('recommended_report_semantics'):body+='<p>'+E(review['recommended_report_semantics'])+'</p>'
        for path in review.get('native_images',[]):body+=f'<a href="{E(path)}">原分辨率物理端点核验图：{E(Path(path).name)}</a>'
    if review.get('audit'):body+=f'<p><a href="{E(review["audit"])}">物理疑点与计算证据</a></p>'
    return body+'</aside>'


def contexts(run,samples):
    import yaml
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    cfg=yaml.safe_load((run/'comparison_config.yaml').read_text())['dataloader']
    cfg['normalization_registry']=None;cfg['metadata_only']=True
    ds=CuratedStage2Dataset.from_config(cfg);repos={r['repo']:r for r in ds.repos}
    result=[]
    for s in samples:
        t=s['trace'];m=ds._episode(repos[t['identity'][0]],t['episode']);row={}
        for key in ('tasks','subtasks'):
            row[key]=[x for x in m[key] if x['start']<=t['anchor']<x['end_exclusive']]
        row['fps']=m['info']['fps'];result.append(row)
    return result


STYLE='''.physical-review{padding:14px;margin:16px 0;background:#fff6e6;border:2px solid #b77b20}.physical-review.suspect{border:3px solid #a02d21;background:#fff0ea;font-size:17px}.physical-review h4{font-size:21px;margin:4px 0}.contact-sheet img{width:100%;max-width:960px;max-height:550px;object-fit:contain}.contact-sheet{margin:10px 0}body{font:15px system-ui;margin:24px;background:#f5f6f8;color:#17202a}article{background:white;border:1px solid #ccd3db;padding:20px;margin:24px 0}img,video{max-width:100%;width:320px;vertical-align:top}video{background:#111;width:640px}.row{display:flex;gap:18px;flex-wrap:wrap}.lane{padding:12px;border:1px solid #d9dfe7;max-width:670px}.stage1{max-width:none}.lane figure{display:inline-block;margin:8px;vertical-align:top}.lane figcaption{max-width:320px}.notice{color:#8b4d00;max-width:650px}.no-gt{font-size:19px;color:#773f00;background:#fff0d8;border:2px solid #dd8b17;padding:12px}.frame-contract{font-size:18px;font-weight:600;background:#edf4fc;padding:12px}pre{white-space:pre-wrap}button,input{margin:5px}.variant{margin-top:22px;border-top:2px solid #bbc6d6}a{margin-right:10px} .controls{position:sticky;top:0;background:#edf4fc;padding:10px;z-index:2}'''
SCRIPT='''document.querySelectorAll('article').forEach(c=>{const vids=[...c.querySelectorAll('video')],slider=c.querySelector('input'),label=c.querySelector('.frame'),frames=JSON.parse(c.dataset.frames),valid=JSON.parse(c.dataset.valid);let busy=false;function show(t){const j=Math.min(8,Math.max(0,Math.floor(t*4+.0001)));slider.value=j;label.textContent='sample '+j+'/8 | '+(valid[j]?'source frame ':'MASKED/PADDED nominal frame ')+frames[j]+' | '+(valid[j]?(j?'action['+(4*j-1)+']':'observed anchor; no prediction'):'repeated display; no valid source observation');}function seek(t,origin){if(busy)return;busy=true;vids.forEach(v=>{if(v!==origin&&Math.abs(v.currentTime-t)>.015)v.currentTime=t;});show(t);busy=false;}slider.oninput=()=>{vids.forEach(v=>v.pause());seek(Number(slider.value)/4+.001);};c.querySelector('[data-back]').onclick=()=>{slider.value=Math.max(0,Number(slider.value)-1);slider.oninput();};c.querySelector('[data-next]').onclick=()=>{slider.value=Math.min(8,Number(slider.value)+1);slider.oninput();};c.querySelector('[data-play]').onclick=()=>{let pause=vids.some(v=>!v.paused);vids.forEach(v=>pause?v.pause():v.play().catch(()=>{}));};vids.forEach(v=>{v.addEventListener('seeking',()=>seek(v.currentTime,v));v.addEventListener('play',()=>{if(busy)return;busy=true;vids.forEach(w=>{if(w!==v){w.currentTime=v.currentTime;w.play().catch(()=>{});}});busy=false;});v.addEventListener('pause',()=>{if(busy)return;busy=true;vids.forEach(w=>w.pause());busy=false;});v.addEventListener('timeupdate',()=>{if(!v.paused)show(v.currentTime);});});show(0);});'''


def build(run):
    run=Path(run).resolve();root=run/'report';root.mkdir(exist_ok=True)
    samples=torch.load(run/'data/samples.pt',weights_only=False,map_location='cpu');variants=discover_variants(run/'stage2')
    frozen=read(run/'stage1_identity.json')
    if not frozen.get('frozen_for_both_arms') or sha(Path(frozen['path']))!=frozen['sha256']:
        raise ValueError('Stage1 frozen checkpoint identity is not verified')
    endpoint_audit=read(root/'endpoint_audit_v3.json') if (root/'endpoint_audit_v3.json').is_file() else {}
    text=contexts(run,samples);cases=[]
    for i,(s,ctx) in enumerate(zip(samples,text)):
        case_id=f'{i:03d}';tr=s['trace'];folder=root/'contract_assets'/f'case_{case_id}';folder.mkdir(parents=True,exist_ok=True)
        frames=frame_contract(s);display_records=[]
        current=folder/'current.png';source=folder/'source.mp4';goal=folder/'instruction_end_gt.png'
        rgb=s['goal_observed_multiview'][:,0].float().permute(1,2,0).numpy()
        Image.fromarray(np.rint((rgb+1)*127.5).clip(0,255).astype(np.uint8)).save(current)
        if not source.exists():video(source,s['video'],rate=4)
        s['_eval']['report_goal'].save(goal)
        rel=lambda p:str(p.relative_to(root))
        current_labeled=folder/'current_labeled.png';goal_labeled=folder/'instruction_end_gt_labeled.png'
        goal_tile=folder/'instruction_end_gt_tile.png'
        Image.open(goal).convert('RGB').resize((320,256),Image.Resampling.BILINEAR).save(goal_tile)
        display_records.append(label_display_image(current,current_labeled,frames,role='Actual current three-view model input'))
        display_records.append(label_display_image(goal_tile,goal_labeled,frames,role='Actual selected instruction-end source image'))
        identity=dict(dataset=tr['identity'][0],episode=str(tr['episode']),anchor_frame=int(tr['anchor']),conditioning_text=s['prompt'])
        c=dict(case_id=case_id,identity=identity,task=ctx['tasks'],subtask=ctx['subtasks'],granularity=s['granularity'],
            selected_instruction=s['prompt'],frame_contract=frames,current_image=rel(current_labeled),current_image_raw=rel(current),
            view_valid=s['goal_view_valid'].tolist(),source_fps=ctx['fps'],source_frames=tr['video_source_indices'],
            source_frame_valid=frames['stage2_video_valid'],goal_frame=s['report_target']['goal_source_frame'],goal_image=rel(goal_labeled),
            goal_image_raw=rel(goal),source_video=dict(status='ready',assets=[rel(source)]),variants={})
        c['physical_contact_review']=physical_contact_review(endpoint_audit,case_id,s,root)
        sf=root/'stage1_samecase'/f'case_{case_id}';c['stage1']=checked_stage1(sf,s,identity)
        c['physical_tcp_definition']=c['stage1'].get('physical_tcp_definition',{
            k:s.get('report_target',{}).get(k) for k in ('endpoint','camera','calibration_eligibility')})
        if (root/'endpoint_audit_v3.json').is_file():c['endpoint_audit']='endpoint_audit_v3.json'
        if c['stage1']['status']=='ready' and c['stage1']['checkpoint_sha256']!=frozen['sha256']:
            raise ValueError('Same-case Stage1 forward used different weights from Stage2 conditioning')
        c['stage1']['frame_contract']=frames
        originals=c['stage1'].get('assets',[]);c['stage1']['assets']=[];c['stage1']['asset_roles']={}
        for name in originals:
            display=folder/('stage1_'+Path(name).stem+'_labeled.png')
            role={'prediction.png':'Stage1 generated image','tcp_overlay.png':'Generated image + 2D model output',
                  'target_tcp_overlay.png':'Instruction-end source + 2D model output'}[name]
            display_records.append(label_display_image(sf/name,display,frames,role=role,valid_gt_count=c['stage1']['valid_gt_count']))
            c['stage1']['assets'].append(rel(display));c['stage1']['asset_roles'][rel(display)]=name
        c['stage1']['original_assets']=[rel(sf/name) for name in originals]
        c['stage1']['evidence']=[rel(sf/p) for p in c['stage1'].get('evidence',[])]
        audit=sf/'source_audit.json'
        if audit.exists():
            aud=read(audit)
            if aud['identity']!=sample_identity(s):raise ValueError(f'case {i}: source audit identity mismatch')
            c['source_audit']=rel(audit);c['stage1']['evidence'].append(rel(audit))
        for variant in variants:
            d=root/'assets'/f'{variant["mode"]}_{variant["step"]:08d}_{case_id}'
            pp=d/'video_projection.json';payload=read(pp) if pp.exists() else None
            geom=d/'video_camera_geometry.json'
            if geom.exists():
                frames=read(geom)['source_frames']
                if list(frames)!=list(c['source_frames']):raise ValueError(f'case {i}: mismatched source video frame mapping')
            row=dict(step=variant['step'],mode=variant['mode'],stage1_checkpoint_sha256=frozen['sha256'])
            for branch in ('direct','fk'):row[branch]=projection_lane(d,root,branch,payload,s)
            if variant['mode']=='none':row['generated']=dict(status='not_applicable',reason='none mode has no generated video branch; real source video remains available above.')
            elif (d/'video.mp4').exists():row['generated']=dict(status='ready',assets=[rel(d/'video.mp4')])
            else:row['generated']=dict(status='missing',reason='joint generated video asset is missing.')
            c['variants'][variant['key']]=row
        for record in display_records:
            record['source']=rel(Path(record['source']));record['asset']=rel(Path(record['asset']))
        dump(folder/'display_provenance.json',dict(frame_contract=frames,selected_instruction=s['prompt'],displays=display_records))
        c['display_provenance']=rel(folder/'display_provenance.json')
        c['stage1']['evidence'].append(c['display_provenance'])
        dump(folder/'input_provenance.json',dict(identity=sample_identity(s),task=ctx['tasks'],subtask=ctx['subtasks'],
            selected_instruction=s['prompt'],frame_contract=frames,source_frames=c['source_frames'],view_valid=c['view_valid'],goal_frame=c['goal_frame'],source_fps=c['source_fps'],preview_fps=4,mask_reasons=tr['mask_reasons']))
        c['input_provenance']=rel(folder/'input_provenance.json');cases.append(c)
    contract=dict(schema_version=1,page='stage12_comparison.html',expected_cases=[f'{i:03d}' for i in range(len(samples))],expected_variants=[v['key'] for v in variants],cases=cases)
    parts=['<!doctype html><html><head><meta charset="utf-8"><title>Same-case Stage1 / Stage2 comparison</title><style>'+STYLE+'</style></head><body><h1>同案例 Stage1 / Stage2 预测对比</h1>']
    contact=endpoint_audit.get('visual_suspects',{}).get('contact_sheet')
    if contact:
        contact=Path(contact).resolve()
        if not contact.is_file():raise FileNotFoundError(contact)
        contact=str(contact.relative_to(root));contract['gt_only_contact_sheet']=contact
        parts.append('<section class="physical-review suspect"><h2>Source GT／源数据端点对照：物理接触中点 NOT certified</h2><p>绿色方框仅为源定义 GT，不能凭数值重投影一致就认定为指垫或夹爪物理中点。Case018 的声明源 TCP 与可见指垫位置有疑点；Case012 的端点及跨设备标定适用性仍待核验。未据此修改坐标或采用新的固定偏移。</p><figure class="contact-sheet"><a href="'+E(contact)+'">'+media(contact)+'</a><figcaption>仅显示 Source GT 的接触点核验总览；点击查看原图。物理中点未认证。</figcaption></figure><a href="endpoint_audit_v3.json">完整物理疑点与数值核验</a></section>')
    n=sum(any(v['direct']['status']=='ready' for v in c['variants'].values()) for c in cases)
    nf=sum(any(v['fk']['status']=='ready' for v in c['variants'].values()) for c in cases)
    parts+=[f'<p>{len(cases)} 个真实评估案例，{len(variants)} 个方法/快照。可恢复 direct TCP：{n}/{len(cases)}；匹配 FK：{nf}/{len(cases)}。不可用栏目保留具体原因及坐标证据。Stage1 是同锚点实际 forward；缺 GT 的二维预测不等于已标定三维 TCP。</p>', '<p>GT 绿色方框；预测红圆圈；L/R 区分左右。Stage1 无 GT 的二维预测为橙圈。视频是 9 个采样帧的 4 fps 预览，不是原视频实时速度。投影只对应头相机，腕图不伪造投影。</p><a href="stage12_contract.json">完整合同 JSON</a>']
    parts.append('<nav aria-label="案例目录"><h2>案例目录</h2>'+''.join(f'<a href="#case-{c["case_id"]}">Case {c["case_id"]}</a>' for c in cases)+'</nav>')
    for c in cases:
        ident=c['identity'];parts.append(f'<article id="case-{c["case_id"]}" data-frames="{E(json.dumps(c["source_frames"]))}" data-valid="{E(json.dumps(c["source_frame_valid"]))}"><h2>Case {c["case_id"]} · {E(ident["dataset"])}</h2><p>episode {ident["episode"]} · 当前帧 {ident["anchor_frame"]} · granularity {E(c["granularity"])}</p>')
        f=c['frame_contract']
        parts.append(f'<p class="frame-contract">输入 f{f["input_frame"]} → 所选 {E(f["instruction_kind"])} 目标 f{f["target_frame"]}；指令区间 [f{f["instruction_start_frame"]}, f{f["instruction_end_exclusive"]})，end_exclusive 边界 f{f["instruction_end_exclusive"]}。<br>Stage2 名义 horizon 至 f{f["stage2_horizon_last_frame"]}；有效动作至 f{f["stage2_valid_action_last_frame"]}，最后有效视频采样帧 f{f["stage2_last_valid_video_frame"]}。</p>')
        parts.append('<pre>Task: '+E(json.dumps(c['task'],ensure_ascii=False))+'\nSubtask: '+E(json.dumps(c['subtask'],ensure_ascii=False) if c['subtask'] else '未提供')+'\n实际 conditioning: '+E(ident['conditioning_text'])+'</pre>')
        parts.append(f'<p>视图有效 mask（head/left/right）：{c["view_valid"]} · 源 FPS {c["source_fps"]} · 源帧 {c["source_frames"]}</p><a href="{c["input_provenance"]}">输入身份/文本/采样证据</a>')
        parts.append('<div class="row"><div><h3>实际当前三视图输入</h3>'+media(c['current_image'])+'</div><div><h3>真实 instruction-end 目标图像 · 源帧 '+str(c['goal_frame'])+'</h3>'+media(c['goal_image'])+'</div></div>')
        parts.append(physical_review_html(c['physical_contact_review'],'Stage1'))
        parts.append('<h3>Stage1 同案例预测</h3><p>二维模型坐标输出与物理 GT 投影分开记录；ready 表示资产已生成，不代表预测可信或物理标定通过。四个 Stage2 变体共享的冻结 Stage1 权重 SHA256：'+E(str(c['stage1'].get('checkpoint_sha256','missing')))+'</p>'+lane_html('预测图像 + 二维模型坐标',c['stage1']))
        parts.append(endpoint_html(c['physical_tcp_definition'],c.get('endpoint_audit')))
        parts.append(physical_review_html(c['physical_contact_review'],'Stage2'))
        parts.append('<h3>Stage2：32 步 action horizon</h3><div class="controls"><button data-play>共享播放 / 暂停</button><button data-back>上一采样帧</button><button data-next>下一采样帧</button><input type="range" min="0" max="8" step="1" value="0"><span class="frame"></span></div>'+lane_html('真实源视频（所有变体共享）',c['source_video']))
        for name,row in sorted(c['variants'].items(),key=lambda item:item[1]['step']==0):
            initial=row['step']==0
            if initial:parts.append(f'<details class="initial"><summary>{E(name)} · 初始 step 0 对照（展开查看）</summary>')
            layout='joint 视频：左侧是真实逐帧背景，右侧是生成视频；右侧复用源相机 UV，只供对照。' if row['mode']=='joint' else 'none 视频：真实逐帧背景上的动作预测；无生成视频分支。'
            parts.append(f'<section class="variant"><h3>{E(name)} · checkpoint step {row["step"]}</h3><p>{layout}</p><div class="row">'+lane_html('Direct action → TCP · 真实逐帧背景',row['direct'])+lane_html('FK joint → TCP · 独立轨迹',row['fk'])+'</div><details><summary>额外生成视频（不替代真实背景投影）</summary>'+lane_html('原始对照：左 GT 源视频 / 右 joint 生成视频',row['generated'])+'</details></section>')
            if initial:parts.append('</details>')
        parts.append('</article>')
    parts.append('<script>'+SCRIPT+'</script></body></html>')
    (root/contract['page']).write_text('\n'.join(parts));dump(root/'stage12_contract.json',contract)
    return contract


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--validator',type=Path,required=True);p.add_argument('--decode-videos',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);contract=build(a.run)
    spec=importlib.util.spec_from_file_location('stage12_validator',a.validator);validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    result=validator.validate(contract,a.run/'report',a.decode_videos);dump(a.run/'report/stage12_contract_check.json',result)
    print(json.dumps(dict(ok=result['ok'],errors=result['errors'],cases=result['checked_cases'],limitations=len(result['limitations']))))
    raise SystemExit(0 if result['ok'] else 1)

if __name__=='__main__':main()
