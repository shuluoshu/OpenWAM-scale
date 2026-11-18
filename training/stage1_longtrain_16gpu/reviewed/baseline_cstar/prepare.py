"""Prepare native causal C* head targets/history for an immutable RFC cohort.

The parent multiview observations, contexts, labels and exact selected records
are reused. Only head RGB target/history frames enter the native head GoalCodec.
This command never edits the parent data, source metadata, videos or runtimes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import datetime
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

FORMAT='openwam.causal_cstar.prepared.v1'
PARENT_FORMAT='openwam.rfc_stage1_comparison.prepared.v1'
PARENT_SHA256='987b385a32586300e4c1ad6f49e0d1d5a49d7cbfaa0a3917992e74b983e10704'
COUNTS={'train':256,'val':64}
HEAD_SHAPE=(48,1,16,20)
OBSERVED_SHAPE=(48,1,24,20)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def atomic_torch(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    torch.save(value,temporary);temporary.replace(path)


def source_stamp(path):
    try:
        info=Path(path).stat()
        return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns
    except FileNotFoundError:return None


def validate_source_snapshot(snapshot,*,verify_small_hashes=False):
    for path,expected,digest in snapshot['files']:
        current=source_stamp(path)
        # Linux st_dev identifies this process's mount and differs between DSW
        # and DLC pods. Inode/size/mtime/ctime identify the same CPFS generation.
        matches=(current is None and expected is None) or (current is not None and expected is not None
                                                          and current[1:]==tuple(expected)[1:])
        if not matches:
            raise ValueError('source dependency changed: '+path)
        if verify_small_hashes and digest is not None and sha256(path)!=digest:
            raise ValueError('source metadata content differs: '+path)


def bind_native(code_dir,expected):
    code=Path(code_dir).resolve()
    if not (code/'openwam/model/goal/visual_history.py').is_file():
        raise ValueError('--code-dir must contain the native VisualHistoryGoalExpert')
    sys.path.insert(0,str(code));import openwam
    if Path(openwam.__file__).resolve().parent != code/'openwam':
        raise ValueError('another openwam was imported before the selected native runtime')
    digest=hashlib.sha256()
    for path in sorted((code/'openwam').rglob('*.py')):
        digest.update(str(path.relative_to(code)).encode());digest.update(sha256(path).encode())
    if digest.hexdigest()!=expected['python_tree_sha256']:
        raise ValueError('native source differs from the frozen parent code')
    return dict(directory=str(code),python_tree_sha256=digest.hexdigest())


def history_config():
    from openwam.dataloader.head_history import HistorySamplingConfig
    return HistorySamplingConfig(enabled=True,num_frames=10,mean_interval_s=1.,
                                 jitter_fraction=0.,frame_dropout_prob=0.)


def _tensor(row,name,shape,dtype=None):
    value=row[name]
    if not torch.is_tensor(value) or tuple(value.shape)!=shape or (dtype is not None and value.dtype!=dtype):
        raise ValueError(f'wrong frozen {name} shape/dtype')
    return value


def head_target_valid(row):
    return bool(row['target_image_valid']) and bool(row['target_view_valid'][0])


def validate_frozen_row(row,entry):
    record=row['record'];grain=row['granularity']
    if grain not in ('task','subtask') or grain!=entry['granularity'] or record['granularity']!=grain:
        raise ValueError('frozen grain differs')
    for key in ('id','family','source_episode'):
        if record[key]!=entry[key]:raise ValueError('frozen record identity differs: '+key)
    if Path(record['source_repo']).resolve()!=Path(entry['source_repo']).resolve():
        raise ValueError('frozen source repository differs')
    if row['instruction']!=record['instruction'] or record['label_kind']!=grain+'_end':
        raise ValueError('frozen instruction/endpoint label differs')
    begin,anchor,goal,end,length=(int(record[key]) for key in ('instruction_start','anchor_source_frame',
                'goal_source_frame','instruction_end_exclusive','source_episode_length'))
    if not 0<=begin<=anchor<goal==end-1<length:
        raise ValueError('frozen anchor or matching-grain endpoint is invalid')
    observed=_tensor(row,'current_latents',OBSERVED_SHAPE)
    if not observed.is_floating_point() or not torch.isfinite(observed).all():
        raise ValueError('nonfinite frozen observed latent')
    _tensor(row,'target_latents',OBSERVED_SHAPE)
    _tensor(row,'observed_view_valid',(3,),torch.bool)
    _tensor(row,'target_view_valid',(3,),torch.bool)
    _tensor(row,'target_image_valid',(),torch.bool)
    _tensor(row,'target_uv',(2,2),torch.float32)
    _tensor(row,'target_point_valid',(2,),torch.bool)
    _tensor(row,'target_in_frame',(2,),torch.bool)
    if head_target_valid(row)!=bool(record['goal_image_valid']):
        raise ValueError('frozen head supervision validity differs from native record')


def select_frozen_repo(repos,row):
    record=row['record'];source=Path(record['source_repo']).resolve()
    episode=int(record['source_episode']);anchor=int(record['anchor_source_frame'])
    candidates=[repo for repo in repos if Path(repo['path']).resolve()==source and repo['family']==record['family']
                and any(int(e)==episode for e,_,_ in repo['ranges'])]
    if len(candidates)!=1:raise ValueError('frozen source repository is missing or ambiguous')
    repo=candidates[0]
    # Exact native CuratedGoalDataset rule: first containing selected range.
    for selected_episode,start,stop in repo['ranges']:
        if int(selected_episode)==episode and int(start)<=anchor<int(stop):return repo,int(start)
    raise ValueError('frozen anchor is outside every selected observation range')


def validate_history_trace(trace,fields,*,anchor,start):
    valid=fields['history_valid'];ages=fields['history_ages']
    if valid.shape!=(10,) or valid.dtype!=torch.bool or ages.shape!=(10,) or ages.dtype!=torch.float32:
        raise ValueError('native history mask/ages contract differs')
    ids=np.asarray(trace['indices'],dtype=np.int64)
    if ids.shape!=(10,) or trace['anchor']!=anchor or trace['selected_start']!=start:
        raise ValueError('native history selection identity differs')
    mask=valid.cpu().numpy();chosen=ids[mask];actual_ages=ages.cpu().numpy()
    if not ((chosen>=start)&(chosen<anchor)).all():
        raise ValueError('history must be strictly past and within the original selected range')
    if len(set(chosen.tolist()))!=len(chosen) or (len(chosen)>1 and not (np.diff(chosen)>0).all()):
        raise ValueError('history contains duplicates or is not oldest-first')
    if (ids[~mask]!=-1).any() or (actual_ages[~mask]!=0).any():
        raise ValueError('unavailable history must use -1 indices and zero ages')
    expected=(anchor-chosen)/float(trace['fps'])
    if not np.allclose(actual_ages[mask],expected,rtol=0,atol=1e-6):
        raise ValueError('history ages differ from source ordinals/FPS')
    if trace['valid']!=valid.tolist() or not np.allclose(trace['ages_s'],actual_ages,rtol=0,atol=1e-6):
        raise ValueError('native history trace differs from saved masks/ages')


def validation_head_pixels(row):
    raw=_tensor(row,'raw_target_rgb',(3,1,384,320),torch.uint8)
    return raw[:,:,:256,:].float().div(127.5).sub(1.).contiguous()


def encode_head_frames(codec,target,history,valid,*,device,dtype):
    if tuple(target.shape)!=(3,1,256,320) or tuple(history.shape)!=(10,3,1,256,320):
        raise ValueError('head encoding requires raw normalized head RGB, never multiview latents')
    if valid.shape!=(10,) or valid.dtype!=torch.bool:raise ValueError('history validity must be bool[10]')
    frames=torch.cat((target[None],history[valid]),0).to(device=device,dtype=dtype)
    # Independent frames stay on B; T remains one, including cache hits.
    with torch.no_grad():encoded=codec.encode(frames,device=device).detach().cpu()
    if tuple(encoded.shape)!=(len(frames),*HEAD_SHAPE):raise ValueError('native head codec latent shape differs')
    output=torch.zeros((10,*HEAD_SHAPE),dtype=encoded.dtype)
    output[valid]=encoded[1:]
    return encoded[0].contiguous(),output


def build_prepared_row(row,*,parent_dir,target_latent,history_latents,history_fields,history_trace):
    context=(Path(parent_dir)/row['context_file']).resolve()
    if not context.is_file():raise FileNotFoundError(context)
    result=dict(observed_latent=row['current_latents'].detach().cpu().contiguous(),
        target_latent=target_latent,history_latents=history_latents,
        history_ages=history_fields['history_ages'].detach().cpu(),
        history_valid=history_fields['history_valid'].detach().cpu(),
        history_episode_id=history_fields['history_episode_id'].detach().cpu(),
        target_uv=row['target_uv'].detach().cpu(),target_image_valid=torch.tensor(head_target_valid(row),dtype=torch.bool),
        target_point_valid=row['target_point_valid'].detach().cpu(),target_in_frame=row['target_in_frame'].detach().cpu(),
        observed_view_valid=row['observed_view_valid'].detach().cpu(),context_file=str(context),
        record=copy.deepcopy(row['record']),instruction=row['instruction'],granularity=row['granularity'],
        history_trace=history_trace,parent_memory_source_identity=row['memory_source_identity'],
        parent_population_identity=row['population_identity'])
    if 'raw_current_rgb' in row:
        result['raw_current_rgb']=row['raw_current_rgb'].detach().cpu().contiguous()
        result['raw_target_rgb']=row['raw_target_rgb'][:,:,:256,:].detach().cpu().contiguous()
    return result


def load_parent(parent_dir,expected_sha256):
    parent=Path(parent_dir).resolve();path=parent/'manifest.json'
    if sha256(path)!=expected_sha256:raise ValueError('parent manifest SHA256 differs')
    manifest=json.loads(path.read_text())
    if manifest['format']!=PARENT_FORMAT or manifest['seed']!=42 or manifest['dtype']!='bfloat16':
        raise ValueError('parent format/seed/dtype differs from matched comparison')
    if {key:len(value) for key,value in manifest['records'].items()}!=COUNTS:
        raise ValueError('matched comparison requires exactly256 training and64 validation records')
    for name,digest in manifest['files'].items():
        path=(parent/name).resolve()
        if not path.is_relative_to(parent):raise ValueError('parent file escapes prepared directory')
        if sha256(path)!=digest:raise ValueError('parent prepared file SHA256 differs: '+name)
    return parent,manifest


def frozen_source_snapshots(parent,manifest):
    result={}
    for split in COUNTS:
        source=json.loads((parent/(split+'_sources.json')).read_text())
        values={item['identity']:item for item in source['sources']}
        if len(values)!=len(source['sources']):raise ValueError('duplicate source snapshot identities')
        for snapshot in values.values():validate_source_snapshot(snapshot,verify_small_hashes=True)
        result[split]=values
    return result


def audit_rows(parent,manifest,snapshots):
    result={};all_physical={}
    for split in COUNTS:
        ids=[];grains=Counter();families=Counter();heads=0;physical=set()
        for entry in manifest['records'][split]:
            row=torch.load(parent/entry['file'],map_location='cpu',weights_only=False)
            validate_frozen_row(row,entry)
            if row['memory_source_identity'] not in snapshots[split]:raise ValueError('frozen row source snapshot is absent')
            if row['context_file'] not in manifest['files']:raise ValueError('frozen context is absent from parent hash contract')
            ids.append(entry['id']);grains[row['granularity']]+=1;families[entry['family']]+=1;heads+=int(head_target_valid(row))
            physical.add((str(Path(entry['source_repo']).resolve()),int(entry['source_episode'])))
        # Repeated source anchors may legitimately occur in a finite frozen draw.
        result[split]=dict(records=len(ids),unique_ids=len(set(ids)),grains=dict(grains),families=dict(families),valid_head_targets=heads,
                           ordered_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest())
        all_physical[split]=physical
    if all_physical['train']&all_physical['val']:raise ValueError('frozen train/validation physical episodes overlap')
    mappings={}
    for split in snapshots.values():
        for snapshot in split.values():
            for path,expected,_ in snapshot['files']:
                current=source_stamp(path)
                if current is not None and expected is not None:
                    mappings.setdefault(str(expected[0]),set()).add(current[0])
    result['source_mount_device_mapping']={key:sorted(values) for key,values in mappings.items()}
    result['source_generation_contract']='same resolved path/inode/size/mtime_ns/ctime_ns and saved small-file SHA256; st_dev is mount-local'
    return result


def decode_terminal_head(row,metadata,*,sequential):
    from openwam.model.goal.runtime import clean_goal_head
    from scripts.data.prepare_subtask_goals import decode_frames
    record=row['record'];head=metadata['head'];goal=int(record['goal_source_frame'])
    if head is None:return torch.full((3,1,256,320),-1.),dict(available=False,mask_reason='missing_head_camera')
    prefix='videos/'+head+'/';ep=metadata['ep'];info=metadata['info']
    path=metadata['root']/info['video_path'].format(video_key=head,episode_index=int(record['source_episode']),
        chunk_index=int(ep[prefix+'chunk_index']),file_index=int(ep[prefix+'file_index']))
    images,alignment=decode_frames(path,float(ep[prefix+'from_timestamp']),float(info['fps']),[goal],
        sequential=sequential,video_ordinals=metadata.get('video_ordinals') if sequential else None)
    actual=alignment[goal];expected=record['provenance'].get('goal_video',{})
    for name in ('source_frame_index','video_ordinal','source_rgb_sha256','decoded_rgb_sha256'):
        if name in expected and actual.get(name)!=expected[name]:
            raise ValueError('terminal head re-decode differs from frozen RGB provenance: '+name)
    return clean_goal_head(images[goal]),actual


def prepare(args):
    parent,manifest=load_parent(args.parent_data,args.expected_parent_manifest_sha256)
    code=bind_native(args.code_dir,manifest['code'])
    snapshots=frozen_source_snapshots(parent,manifest)
    audit=audit_rows(parent,manifest,snapshots)
    if args.audit_only:
        return dict(event='parent_audit_passed',parent_manifest_sha256=args.expected_parent_manifest_sha256,
                    parent_data=str(parent),counts=audit,model_loaded=False,source_files_changed=False)
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    from openwam.dataloader.head_history import materialize_head_history
    from openwam.model.goal.codec import GoalCodec
    from openwam.model.goal.dependencies import module_identity
    from openwam.train.goal_trainer import _construct,load_component_weights
    output=Path(args.output).resolve()
    if output==parent or output.is_relative_to(parent):raise ValueError('output must be outside frozen parent data')
    if output.exists() and any(output.iterdir()):raise ValueError('C* preparation requires a fresh empty output')
    output.mkdir(parents=True,exist_ok=True)
    if args.dtype!=manifest['dtype']:raise ValueError('C* codec dtype must match frozen parent')
    dtype=getattr(torch,args.dtype);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    config=copy.deepcopy(manifest['config'])
    vae=_construct(config['components']['vae'])
    load_component_weights(vae,config['baseline_checkpoint'],'video_backbone.vae.')
    vae.to(device=args.device,dtype=dtype).eval().requires_grad_(False)
    vae_identity=module_identity(vae)
    if vae_identity['fingerprint']!=manifest['dependencies']['vae']['fingerprint']:
        raise ValueError('native head VAE identity differs from frozen parent representation')
    codec=GoalCodec(vae,max_encode_batch_size=args.vae_batch_size)
    codec.enable_cache(max_bytes=args.codec_cache_mb*1024*1024)
    readers={};records={split:[] for split in COUNTS};files={};external_files={};history_valid_counts=Counter()
    cfg=history_config()
    for split in COUNTS:
        options=config['dataset'];source=manifest['sources'][split]
        if sha256(source['path'])!=source['sha256']:raise ValueError('original selection YAML changed')
        reader=CuratedGoalDataset(source['path'],output/'reader_cache'/split,seed=42,
            subtask_probability=.7,max_cached_episodes=options.get('max_cached_episodes',2),dtype=torch.float32,
            data_root=options.get('data_root'),path_aliases=options.get('path_aliases'),observation_mode='multiview',history=cfg)
        readers[split]=reader;(output/split).mkdir()
        for index,entry in enumerate(manifest['records'][split]):
            row=torch.load(parent/entry['file'],map_location='cpu',weights_only=False);validate_frozen_row(row,entry)
            snapshot=snapshots[split][row['memory_source_identity']];validate_source_snapshot(snapshot)
            repo,start=select_frozen_repo(reader.repos,row);record=row['record']
            metadata=reader._episode(repo,int(record['source_episode']))
            if metadata['head']!=record['head'] or metadata['length']!=record['source_episode_length']:
                raise ValueError('native episode metadata differs from frozen record')
            native_valid=metadata['head'] is not None and bool(metadata['source_valid'][int(record['goal_source_frame'])])
            if native_valid!=head_target_valid(row):raise ValueError('native terminal head validity changed')
            segment_pool=metadata['subtasks' if row['granularity']=='subtask' else 'tasks']
            matching=[segment for segment in segment_pool if int(segment['start'])==int(record['instruction_start'])
                and int(segment['end_exclusive'])==int(record['instruction_end_exclusive']) and segment['text']==row['instruction']]
            if len(matching)!=1:raise ValueError('frozen matching-grain instruction interval changed')
            anchor=int(record['anchor_source_frame']);sequential=record['family']=='mecka'
            fields,trace=materialize_head_history(metadata['root'],metadata['info'],metadata['ep'],metadata['head'],
                anchor,start,cfg,np.random.default_rng(np.random.SeedSequence([42,index])),split=='train',
                sequential=sequential,dtype=torch.float32,source_valid=metadata['source_valid'],
                video_ordinals=metadata.get('video_ordinals') if sequential else None)
            validate_history_trace(trace,fields,anchor=anchor,start=start)
            if split=='val':
                target=validation_head_pixels(row);target_decode=dict(source='frozen_raw_target_rgb_head_tile',parent_file=entry['file'])
            else:target,target_decode=decode_terminal_head(row,metadata,sequential=sequential)
            target_latent,history_latents=encode_head_frames(codec,target,fields['history_images'],fields['history_valid'],
                device=args.device,dtype=dtype)
            prepared=build_prepared_row(row,parent_dir=parent,target_latent=target_latent,history_latents=history_latents,
                                       history_fields=fields,history_trace=trace)
            prepared['target_head_encoding']=dict(source=target_decode,independent_t1=True,
                native_codec=codec.metadata(),parent_multiview_target_latent_reused=False)
            validate_source_snapshot(snapshot)
            relative=f'{split}/{index:06d}.pt';atomic_torch(output/relative,prepared);files[relative]=sha256(output/relative)
            context=row['context_file'];external_files[str((parent/context).resolve())]=manifest['files'][context]
            records[split].append(dict(entry,file=relative,parent_file=str(parent/entry['file'])))
            history_valid_counts[int(fields['history_valid'].sum())]+=1
            print(json.dumps(dict(event='prepared_cstar_record',split=split,index=index,id=entry['id'],
                anchor=anchor,goal=int(record['goal_source_frame']),valid_history=int(fields['history_valid'].sum()),head_target_valid=native_valid)),flush=True)
            del row,prepared,target,target_latent,history_latents,fields
    atomic_json(output/'config.json',config);files['config.json']=sha256(output/'config.json')
    atomic_json(output/'parent_audit.json',audit);files['parent_audit.json']=sha256(output/'parent_audit.json')
    dependencies=copy.deepcopy(manifest['dependencies'])
    dependencies['causal_cstar_contract']=dict(format='openwam.visual_history.v3',causal_history=True,language_conditioning=True,
        head_codec=codec.metadata(),history=asdict(cfg),history_frame_policy='native-containing-range-strict-past-oldest-first',
        observed='exact_parent_multiview_latents',text='exact_parent_trimmed_contexts',target='fresh_head_rgb_native_GoalCodec',
        native_vae_fingerprint=vae_identity['fingerprint'])
    result=dict(format=FORMAT,parent_manifest_sha256=args.expected_parent_manifest_sha256,parent_data=str(parent),
        code=code,preparer=dict(path=str(Path(__file__).resolve()),sha256=sha256(__file__)),seed=42,dtype=args.dtype,
        config=config,dependencies=dependencies,records=records,files=files,external_files=external_files,
        baseline=manifest['baseline'],sources=manifest['sources'],created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        preparation=dict(independent_t1=True,temporal_compression=False,vae_batch_size=args.vae_batch_size,
            codec_cache_mb=args.codec_cache_mb,device=args.device,dtype=args.dtype,head_rgb_not_latent_crop=True,
            train_val_episode_disjoint=True,selection='exact-parent-record-order',valid_history_counts=dict(history_valid_counts)))
    # Manifest is the final readiness marker. Incomplete output has no manifest.
    atomic_json(output/'manifest.json',result)
    return dict(event='prepared_cstar_complete',output=str(output),manifest_sha256=sha256(output/'manifest.json'),
                counts={split:len(rows) for split,rows in records.items()})


def parser():
    value=argparse.ArgumentParser(description=__doc__)
    value.add_argument('--code-dir',required=True)
    value.add_argument('--parent-data',required=True)
    value.add_argument('--output',required=True)
    value.add_argument('--expected-parent-manifest-sha256',default=PARENT_SHA256)
    value.add_argument('--device',default='cuda')
    value.add_argument('--dtype',choices=('bfloat16','float32'),default='bfloat16')
    value.add_argument('--vae-batch-size',type=int,default=4)
    value.add_argument('--codec-cache-mb',type=int,default=512)
    value.add_argument('--audit-only',action='store_true',help='CPU source/identity audit; no output files or model allocation')
    return value


if __name__=='__main__':
    options=parser().parse_args()
    if options.vae_batch_size<1 or options.codec_cache_mb<0:raise ValueError('invalid VAE batch/cache limit')
    print(json.dumps(prepare(options),ensure_ascii=False,allow_nan=False),flush=True)
