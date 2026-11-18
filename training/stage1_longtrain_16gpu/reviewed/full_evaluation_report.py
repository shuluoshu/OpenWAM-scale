#!/usr/bin/env python3
"""Offline four-arm report of complete64-case native generation evaluations."""
from __future__ import annotations
import argparse
import html
import json
import os
from pathlib import Path
import time

import full_generation_eval as evaluation
base=evaluation.base
ARMS=evaluation.ARMS


def matched_identity(arms):
    signatures={}
    for arm,m in arms.items():
        cases=sorted(m['cases'],key=lambda r:r['index'])
        signature=dict(format=m['format'],parent=m['parent_manifest_sha256'],update=m['update'],steps=m['steps'],
            counts=m['counts'],cases=[dict(index=r['index'],id=r['id'],family=r.get('family'),grain=r.get('grain'),
                counts=r['counts'],ground_truth_identity=r.get('ground_truth_identity')) for r in cases])
        signatures[arm]=evaluation.shared.canonical(signature)
    return dict(matched=bool(signatures) and len(set(signatures.values()))==1,
                all_four_present=set(arms)==set(ARMS),participants=list(arms),
                source_identity_hashes={a:__import__('hashlib').sha256(s.encode()).hexdigest() for a,s in signatures.items()})


def report(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True);arms={};directories={}
    for arm in ARMS:
        directory=root/arm/'step_001000';path=directory/'manifest.json'
        if not path.is_file():continue
        m=json.loads(path.read_text())
        if m.get('format')!=evaluation.FORMAT or m.get('arm')!=arm or m.get('update')!=1000 or m.get('steps')!=20:
            raise ValueError('wrong complete full-generation manifest: '+str(path))
        evaluation.verify_complete(m['cases'],[dict(id=r['id']) for r in sorted(m['cases'],key=lambda r:r['index'])])
        if len(m['cases'])!=64 or (m['counts']['valid_head_images'],m['counts']['valid_points'])!=(57,91):
            raise ValueError('full evaluation64/57/91 counts differ')
        for name,digest in m['files'].items():
            if base.sha256(directory/name)!=digest:raise ValueError('full-generation output checksum differs: '+name)
        arms[arm]=m;directories[arm]=directory
    matching=matched_identity(arms)
    payload=dict(format='openwam.full_generation_comparison.report.v1',generated_unix=time.time(),
        expected_arms=ARMS,matching=matching,arms=arms,
        protocol='Actual trained1000-step native models;64fixed IDs;20-step image generation;original headRGB MAE and valid TCPpx. G UV clean readout;C* UV nativejoint20-step TCPpatch.',
        scope='Original source heads57, validTCPpoints91,10families,40task/24subtask. Zero-valid families remain n/a. Native objectives and diffusion schedules differ; no cross-architecture FM comparison.')
    base.atomic_json(output/'metrics.json',payload)
    def number(value):return 'n/a' if value is None else f'{value:.6g}'
    def tr(arm,m,label=None):
        counts=m['counts'];metrics=m['metrics']
        values=(label or arm,number(metrics['head_image_mae']),number(metrics['tcp_px']),counts['records'],counts['valid_head_images'],counts['valid_points'])
        return '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
    header='<tr><th>Arm/group</th><th>Original headRGB MAE[0,1]</th><th>TCPpx</th><th>Records</th><th>Valid headimages</th><th>Valid TCPpoints</th></tr>'
    summary=''.join(tr(a,m) for a,m in arms.items())
    grouped=''
    for group in sorted({g for m in arms.values() for g in m['groups']}):
        grouped+='<h3>'+html.escape(group)+'</h3><table>'+header+''.join(tr(a,m['groups'][group]) for a,m in arms.items() if group in m['groups'])+'</table>'
    timing=''
    for arm,m in arms.items():
        timing+='<h3>'+arm+'</h3><p>Weights SHA256 '+m['weights']['sha256']+'</p><table><tr><th>Rank</th><th>Cases</th><th>GPU</th><th>Bootstrap seconds</th><th>Generation seconds</th><th>Peak allocatedGiB</th></tr>'
        for r in m['ranks']:
            values=(r['rank'],r['records'],r['device'],number(r['bootstrap_seconds']),number(r['generation_seconds']),number(r['peak_allocated_bytes']/2**30))
            timing+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
        timing+='</table>'
    gallery=''
    if matching['matched']:
        first=next(iter(arms.values()),{})
        for i,case in enumerate(first.get('cases',[])):
            gallery+='<article><h3>'+html.escape(f'{i:02d} {case["family"]} {case["grain"]} {case["id"]}')+'</h3>'
            gallery+='<p>'+html.escape(case.get('instruction',''))+'</p>'
            for arm,m in arms.items():
                record=m['cases'][i];gallery+='<figure><figcaption>'+arm+' headMAE '+number(record['metrics']['head_image_mae'])+' TCPpx '+number(record['metrics']['tcp_px'])+'</figcaption>'
                if record.get('image'):
                    image=os.path.relpath(directories[arm]/record['image'],output)
                    gallery+='<img loading="lazy" width="960" src="'+html.escape(image)+'">'
                gallery+='</figure>'
            gallery+='</article>'
    page='''<!doctype html><meta charset="utf-8"><title>Full64-case native G / causal C* generation</title>
<style>body{font:15px system-ui;max-width:1250px;margin:24px auto;color:#17212b;background:#fafafa}p{line-height:1.5}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}img{max-width:100%;height:auto}figure{margin:10px 0}article{border-top:2px solid #bbb;margin-top:28px;padding-top:12px}</style>
<h1>Full64-case effect comparison at trained update1000</h1>
<p>All eight GPUs per arm generate the same64fixed validation cases using actual full30-layer native weights and actual VAE. Each prediction uses20image denoising steps. G supplies its clean G UV readout; causal C* jointly generates its192D TCPpatch. Only original-headRGB MAE and valid TCPpx are compared. The model objectives/schedules differ.</p>
<p>Completed participants: PARTICIPANTS. All four present: ALLFOUR. Exact source/GT/count identities match: MATCHED. Unfinished arms have no committed manifest and are omitted. Original coverage is57validheads,91TCPpoints,10families and40task/24subtask; Genrobot has no validhead/TCP and remains n/a.</p>
<p><a href="metrics.json">All machine metrics, source/weight hashes, every case and rank timings</a></p>
<h2>Generated head/TCP metrics</h2><table>HEADERROWS</table>
<h2>By family, grain and their intersection</h2>GROUPS
<h2>Eight-rank timing and memory</h2>TIMING
<h2>Every fixed source case: current head | original GT | prediction</h2>GALLERY'''
    page=page.replace('PARTICIPANTS',html.escape(', '.join(arms))).replace('ALLFOUR',str(matching['all_four_present'])).replace('MATCHED',str(matching['matched']))
    page=page.replace('HEADER',header).replace('ROWS',summary).replace('GROUPS',grouped).replace('TIMING',timing).replace('GALLERY',gallery)
    base.atomic_text(output/'index.html',page);return payload


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();result=report(args.root,args.output)
    print(json.dumps(dict(output=args.output,participants=list(result['arms']),matching=result['matching']),indent=2))
