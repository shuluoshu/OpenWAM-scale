#!/usr/bin/env python3
"""Parallel native20-step generation over every frozen validation record."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import rfc_distributed as shared
base=shared.base
ARMS=('default','no_history','no_g','causal_cstar')
FORMAT='openwam.full_generation_evaluation.v1'
PARENT_SHA='987b385a32586300e4c1ad6f49e0d1d5a49d7cbfaa0a3917992e74b983e10704'


def partition_indices(count,rank,world):
    if count<1 or world<1 or not 0<=rank<world:raise ValueError('invalid evaluation partition')
    return list(range(rank,count,world))


def valid_targets(row):
    head=bool(row['target_image_valid']) and bool(row['target_view_valid'][0])
    uv=row['target_uv'].float()
    valid=row['target_point_valid'].bool() & row['target_in_frame'].bool() & torch.isfinite(uv).all(-1)
    return head,valid & head


def summarize(sums):
    sums=dict(sums);rgb=int(sums.get('rgb_den',0));points=int(sums.get('points',0))
    return dict(metrics=dict(head_image_mae=sums.get('rgb_sum',0)/rgb if rgb else None,
        tcp_px=sums.get('tcp_sum',0)/points if points else None,
        uv_mse=sums.get('uv_sum',0)/(2*points) if points else None),
        counts=dict(records=int(sums.get('records',0)),valid_head_images=int(sums.get('heads',0)),
                    valid_points=points,head_rgb_values=rgb),sums=sums)


def measure(prediction,uv,row):
    if prediction.shape!=(3,1,256,320) or uv.shape!=(2,2):raise ValueError('native generated head/UV shape differs')
    if not torch.isfinite(prediction).all() or not torch.isfinite(uv).all():raise FloatingPointError('nonfinite native prediction')
    head,valid=valid_targets(row);truth=row['target_uv'].double()
    prediction=prediction.detach().cpu().double();uv=uv.detach().cpu().double()
    target=row['raw_target_rgb'][:,:,:256].double()/255.
    error=uv[valid]-truth[valid]
    sums=dict(records=1,heads=int(head),points=int(valid.sum()),rgb_den=3*256*320 if head else 0,
        rgb_sum=float((prediction-target).abs().sum()) if head else 0.,uv_sum=float(error.square().sum()),
        tcp_sum=float((error*torch.tensor([319.,255.],dtype=torch.float64)).norm(dim=-1).sum()))
    return summarize(sums)


def aggregate(records):
    total=defaultdict(float);groups=defaultdict(lambda:defaultdict(float))
    for record in records:
        for key,value in record['sums'].items():
            total[key]+=value
            for name in ('grain:'+record['grain'],'family:'+record['family'],
                         'family_grain:'+record['family']+'/'+record['grain']):groups[name][key]+=value
    result=summarize(total);result['groups']={name:summarize(sums) for name,sums in groups.items()}
    weighted={}
    for metric in ('head_image_mae','tcp_px','uv_mse'):
        values=[result['groups'].get('grain:'+grain,{}).get('metrics',{}).get(metric) for grain in ('task','subtask')]
        weighted[metric]=.3*values[0]+.7*values[1] if all(v is not None for v in values) else None
    result['grain_3_7']=dict(weights=dict(task=.3,subtask=.7),metrics=weighted,
        policy='separate weighted grain means; pooled64 validation has40task/24subtask')
    return result


def verify_complete(records,entries):
    if len(records)!=len(entries) or sorted(r['index'] for r in records)!=list(range(len(entries))):
        raise ValueError('all-rank generation records missing/duplicated')
    for record in records:
        if record['id']!=entries[record['index']]['id']:raise ValueError('generated source identity differs')


def verify_cstar_record(parent,child):
    if (parent['record']['id']!=child['record']['id'] or parent['instruction']!=child['instruction'] or
            parent['granularity']!=child['granularity']):raise ValueError('C* parent record/instruction/grain differs')
    head,valid=valid_targets(parent)
    if bool(child['target_image_valid'])!=head:raise ValueError('C* head-valid mask differs')
    for key in ('target_uv','target_point_valid','target_in_frame','observed_view_valid'):
        if key not in parent and key=='observed_view_valid':continue
        a,b=parent[key],child[key]
        same=torch.allclose(a,b,rtol=0,atol=0,equal_nan=True) if a.is_floating_point() else torch.equal(a,b)
        if not same:raise ValueError('C* parent labels differ: '+key)


def safe_metadata(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:safe_metadata(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe_metadata(v) for v in value]
    return value


def ground_truth_identity(row):
    return dict(source_record_sha256=hashlib.sha256(shared.canonical(safe_metadata(base.jsonable(row['record']))).encode()).hexdigest(),
        instruction_sha256=hashlib.sha256(row['instruction'].encode()).hexdigest(),
        tensors={name:shared.tensor_digest(value) for name,value in dict(
            head_rgb=row['raw_target_rgb'][:,:,:256],uv=row['target_uv'],image_valid=row['target_image_valid'],
            view_valid=row['target_view_valid'],point_valid=row['target_point_valid'],in_frame=row['target_in_frame']).items()})


def resolve_checkpoint(path):
    path=Path(path).resolve();payload=json.loads(path.read_text())
    if 'generation' in payload:
        manifest=Path(payload['generation'])/'manifest.json'
        if base.sha256(manifest)!=payload['manifest_sha256']:raise ValueError('training checkpoint manifest SHA differs')
        payload=json.loads(manifest.read_text());path=manifest
    if payload.get('update')!=1000:raise ValueError('full evaluation requires completed1000 checkpoint')
    weights=path.parent/payload['weights']['file']
    return weights,payload['weights']['sha256'],base.sha256(path)


def checked_metadata(weights,*,arm,parent_sha,cstar_sha=None):
    from safetensors import safe_open
    with safe_open(str(weights),framework='pt',device='cpu') as source:metadata=source.metadata() or {}
    key='visual_history' if arm=='causal_cstar' else 'goal_memory_bundle'
    if key not in metadata:raise ValueError('wrong native trained bundle representation')
    metadata=json.loads(metadata[key]);training=metadata.get('training',{});contract=training.get('contract',{})
    if training.get('update')!=1000:raise ValueError('native bundle is not update1000')
    dit=metadata['dit_config']
    if dit.get('num_layers')!=30 or dit.get('dim')!=3072:raise ValueError('full native30-layer/dim3072 model required')
    if arm=='causal_cstar':
        if metadata.get('format')!='openwam.visual_history.v3' or not metadata.get('history',{}).get('causal_history') or not metadata['history'].get('language_conditioning'):
            raise ValueError('native causal text v3 model required')
        if contract.get('arm')!=arm or contract.get('parent_manifest_sha256')!=parent_sha or contract.get('prepared_manifest_sha256')!=cstar_sha:
            raise ValueError('C* trained arm/prepared identity differs')
    else:
        scientific=contract.get('scientific',{})
        if metadata.get('format')!='openwam.goal_memory.v1' or scientific.get('comparison',{}).get('arm')!=arm:
            raise ValueError('G trained arm differs (wrong ablation/no_text)')
        if scientific.get('population',{}).get('manifest_sha256')!=parent_sha:raise ValueError('G trained prepared identity differs')
    return metadata


def save_image(path,prediction,uv,row,*,arm,index):
    from PIL import Image,ImageDraw
    head,valid=valid_targets(row);truth=row['target_uv'].float()
    def pil(value):return Image.fromarray(value[:,0].permute(1,2,0).contiguous().numpy())
    current=pil(row['raw_current_rgb'][:,:,:256]);target=pil(row['raw_target_rgb'][:,:,:256])
    generated=pil((prediction.clamp(0,1)*255).round().byte())
    for image,coords in ((target,truth),(generated,uv)):
        draw=ImageDraw.Draw(image)
        for hand,color in ((0,'cyan'),(1,'magenta')):
            if valid[hand]:
                x,y=(coords[hand]*torch.tensor([319.,255.])).tolist();draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
    canvas=Image.new('RGB',(960,296),'white');draw=ImageDraw.Draw(canvas)
    draw.text((5,5),f'{arm} update1000 case{index}: current head | original terminal head | native20-step prediction',fill='black')
    draw.text((5,21),f'{row["record"]["id"]} head-valid={head} TCP-points={int(valid.sum())}',fill='black')
    for x,image in ((0,current),(320,target),(640,generated)):canvas.paste(image,(x,40))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def evaluate(args):
    if int(os.environ.get('WORLD_SIZE','1'))!=8 or args.steps!=20:raise ValueError('production full generation requires8ranks/20steps')
    if args.arm=='causal_cstar' and not args.cstar_data:raise ValueError('--cstar-data required')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
    device=torch.device('cuda',local);dist.init_process_group('nccl',timeout=timedelta(minutes=60))
    try:
        start=time.monotonic();code=base.bind_code(args.code_dir);parent=Path(args.parent_data).resolve()
        if base.sha256(parent/'manifest.json')!=PARENT_SHA:raise ValueError('original frozen parent manifest differs')
        manifest=base.load_manifest(parent,verify=rank==0)
        if code['python_tree_sha256']!=manifest['code']['python_tree_sha256']:raise ValueError('pinned source tree differs')
        entries=manifest['records']['val']
        if len(entries)!=64:raise ValueError('all64 fixed validation records required')
        cstar_manifest=None;cstar_sha=None
        if args.arm=='causal_cstar':
            sys.path.insert(0,str(ROOT/'baseline_cstar'))
            import train as baseline_train
            cstar_manifest=baseline_train.load_manifest(args.cstar_data,verify=rank==0)
            cstar_sha=base.sha256(Path(args.cstar_data)/'manifest.json')
            if [e['id'] for e in cstar_manifest['records']['val']]!=[e['id'] for e in entries]:raise ValueError('C* validation source IDs differ')
        weights,weight_sha,checkpoint_sha=resolve_checkpoint(args.checkpoint)
        if rank==0 and base.sha256(weights)!=weight_sha:raise ValueError('completed trained weights SHA256 differs')
        metadata=checked_metadata(weights,arm=args.arm,parent_sha=PARENT_SHA,cstar_sha=cstar_sha)
        if metadata['dependencies']['vae']!=manifest['dependencies']['vae'] or metadata['dependencies']['text']!=manifest['dependencies']['text']:
            raise ValueError('trained model VAE/text dependencies differ')
        config=manifest['config']
        if rank==0 and base.sha256(config['baseline_checkpoint'])!=manifest['baseline']['sha256']:raise ValueError('frozen VAE foundation checkpoint differs')
        output=Path(args.output).resolve()
        if rank==0:
            if output.exists() and any(output.iterdir()):raise ValueError('full evaluation output nonempty; use fresh output')
            (output/'records').mkdir(parents=True)
        dist.barrier();base.seed_all(42)
        from openwam.model.goal.codec import GoalCodec,GoalCodecSpec
        from openwam.model.goal.dependencies import module_identity
        vae=base.construct_component(config,'vae',device=device,dtype=torch.bfloat16).eval().requires_grad_(False)
        actual_vae=module_identity(vae)
        if actual_vae!=manifest['dependencies']['vae']:raise ValueError('actual decode VAE identity differs')
        if args.arm=='causal_cstar':
            import cstar
            from openwam.model.video_backbone.wan.models.dit import WanModel
            from safetensors import safe_open
            model=cstar.PreparedVisualHistoryExpert(WanModel(**metadata['dit_config']),cstar.PreparedGoalCodec(vae),
                dit_config=metadata['dit_config'],shift=metadata['shift'],lambda_image=metadata['lambda_image'],
                lambda_tcp=metadata['lambda_tcp'],**metadata['history'])
            with safe_open(str(weights),framework='pt',device='cpu') as source:
                model.load_state_dict({name:source.get_tensor(name) for name in source.keys()},strict=True)
            model=model.to(device=device,dtype=torch.bfloat16).freeze();codec=GoalCodec(vae)
        else:
            from openwam.train.goal_memory_checkpoint import load_memory_bundle
            model=load_memory_bundle(weights,dependencies=manifest['dependencies'],device=device,dtype=torch.bfloat16).eval().requires_grad_(False)
            codec=GoalCodec(vae,GoalCodecSpec(name='wan22_multiview_t1',image_shape=(3,1,384,320),latent_shape=(48,1,24,20)))
        torch.cuda.synchronize(device);bootstrap_seconds=time.monotonic()-start;dist.barrier()
        torch.cuda.reset_peak_memory_stats(device);generation_start=time.monotonic();records=[];files={}
        with torch.inference_mode():
            for index in partition_indices(64,rank,8):
                case_start=time.monotonic();entry=entries[index]
                row=torch.load(parent/entry['file'],map_location='cpu',weights_only=False)
                if args.arm=='causal_cstar':
                    child=torch.load(Path(args.cstar_data)/cstar_manifest['records']['val'][index]['file'],map_location='cpu',weights_only=False)
                    verify_cstar_record(row,child)
                    batch=cstar.collate([child],args.cstar_data,device=device,dtype=torch.bfloat16)
                    prediction=model.generate(**{k:v for k,v in batch.items() if k!='targets'},num_steps=20,seed=42+index+1000000)
                    head=((codec.decode(prediction.latent)[0].float().clamp(-1,1)+1)/2).cpu();uv=prediction.uv[0].float().cpu()
                else:
                    batch,_=base.collate_prepared([row],parent,device=device,dtype=torch.bfloat16)
                    prediction=base.generate_latents(model,batch,arm=args.arm,seed=42,index=index,steps=20)
                    # Decode the entire multiview latent before taking the RGB
                    # head tile. VAE encoding/decoding is spatially contextual.
                    decoded=codec.decode(prediction['latents'])[0].float()
                    head=((decoded.clamp(-1,1)+1)/2)[:,:,:256].cpu();uv=prediction['uv'][0].float().cpu()
                torch.cuda.synchronize(device);metrics=measure(head,uv,row)
                record=dict(index=index,id=entry['id'],family=entry['family'],grain=entry['granularity'],arm=args.arm,
                    update=1000,steps=20,rank=rank,seconds=time.monotonic()-case_start,
                    inference_protocol='native-joint20-step-image-and-TCP' if args.arm=='causal_cstar' else 'native20-step-Y-clean-G-UV-forced-ablation-mask',
                    source_record=safe_metadata(base.jsonable(row['record'])),instruction=row['instruction'],
                    ground_truth_identity=ground_truth_identity(row),generated_uv=uv.tolist(),**metrics)
                if not args.no_images:
                    image=f'images/{index:04d}.png';save_image(output/image,head,uv,row,arm=args.arm,index=index)
                    files[image]=base.sha256(output/image);record['image']=image
                filename=f'records/{index:04d}.json';base.atomic_json(output/filename,record);files[filename]=base.sha256(output/filename)
                records.append(record)
                print(json.dumps(dict(event='generated_case',arm=args.arm,rank=rank,index=index,id=entry['id'],seconds=record['seconds'],metrics=metrics['metrics']),allow_nan=False),flush=True)
                del batch,prediction,head,uv,row
        torch.cuda.synchronize(device)
        rank_receipt=dict(rank=rank,local_rank=local,device=torch.cuda.get_device_name(device),records=len(records),
            indices=[r['index'] for r in records],bootstrap_seconds=bootstrap_seconds,generation_seconds=time.monotonic()-generation_start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            weights_sha256=weight_sha,torch=torch.__version__)
        rank_file=f'rank_{rank:02d}.json';base.atomic_json(output/rank_file,rank_receipt);files[rank_file]=base.sha256(output/rank_file)
        gathered=[None]*8;dist.all_gather_object(gathered,dict(records=records,files=files,receipt=rank_receipt))
        if rank==0:
            records=sorted([r for part in gathered for r in part['records']],key=lambda r:r['index']);verify_complete(records,entries)
            result=aggregate(records)
            if (result['counts']['valid_head_images'],result['counts']['valid_points'])!=(57,91) or len({r['family'] for r in records})!=10:
                raise ValueError('full validation head/TCP/family coverage differs from57/91/10')
            files={name:digest for part in gathered for name,digest in part['files'].items()}
            runner={name:base.sha256(ROOT/name) for name in ('full_generation_eval.py','full_evaluation_report.py')}
            final=dict(format=FORMAT,arm=args.arm,update=1000,steps=20,seed=42,world=8,
                parent_manifest_sha256=PARENT_SHA,cstar_manifest_sha256=cstar_sha,pinned_code=code,
                weights=dict(path=str(weights),sha256=weight_sha,checkpoint_manifest_sha256=checkpoint_sha),
                dependencies=dict(vae=actual_vae,text=manifest['dependencies']['text']),runner=runner,
                cases=records,ranks=[part['receipt'] for part in gathered],files=files,
                generation_wall_seconds=time.monotonic()-generation_start,**result)
            # Manifest is the completion marker, committed after all eight rank
            # shards/cases/images have completed and the exact64 IDs validate.
            base.atomic_json(output/'manifest.json',final)
            from full_evaluation_report import report
            report(args.report_root,Path(args.report_root)/'report')
        dist.barrier();return dict(output=str(output),arm=args.arm,records=64,update=1000)
    finally:dist.destroy_process_group()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-dir',required=True);parser.add_argument('--parent-data',required=True)
    parser.add_argument('--arm',choices=ARMS,required=True);parser.add_argument('--cstar-data')
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--report-root',required=True);parser.add_argument('--steps',type=int,default=20);parser.add_argument('--no-images',action='store_true')
    args=parser.parse_args(argv);result=evaluate(args)
    if int(os.environ.get('RANK','0'))==0:print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
