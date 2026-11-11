"""Stage2 construction, strict native initialization and source-aligned batches."""
from dataclasses import asdict
import json
from pathlib import Path
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import save_file
from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.dual_system.goal_memory import GoalMemoryPolicy


def _policy_options(config):
    p=config.get('policy',{})
    return dict(attention_mask_mode=p.get('attention_mode','isolated'),future_video_mode=p.get('future_video_mode','none'),
        lambda_keyframe=p.get('keyframe_aux_weight',.1),freeze_video=p.get('freeze_video',False),goal_dropout=p.get('goal_dropout',0.),
        allow_approximate_low_precision_cache=config.get('inference',{}).get('allow_approximate_low_precision_cache',False))


def _action_config(video,hidden=1024,ffn=None,freq_dim=256):
    heads=video.dit.blocks[0].num_heads
    return dict(action_dim=112,dim=hidden,ffn_dim=ffn or hidden*4,num_heads=heads,num_layers=len(video.dit.blocks),
        video_dim=video.dit.dim,bridge_layers=list(range(len(video.dit.blocks))),variant='joint_self_attn',
        attn_head_dim=video.dit.dim//heads,text_dim=video.context_width,freq_dim=freq_dim)


def make_smoke_policy(video,config):
    ac=_action_config(video,video.dit.dim,freq_dim=16)
    p=GoalMemoryPolicy(video,ActionDiT(**ac),nn.Linear(112,video.context_width),**_policy_options(config))
    p.action_config=ac;p.initialization_metadata={'kind':'random smoke only'}
    return p.to(device=video.goal_queries.device,dtype=video.goal_queries.dtype)


def build_stage2(config,*,dit,dependencies,device,dtype):
    from openwam.train.goal_memory_checkpoint import load_memory_bundle
    path=config.get('stage1_checkpoint')
    if not path:raise ValueError('explicit Stage1 goal_memory checkpoint required')
    # Caller retains its own baseline reference until this function returns;
    # the policy never stores it or a second teacher.
    del dit
    video=load_memory_bundle(path,dependencies=dependencies,device='cpu',dtype=dtype)
    ab,proprio,initialization=build_native_action(video,config)
    p=GoalMemoryPolicy(video,ab,proprio,**_policy_options(config))
    p.action_config=initialization['action_config']
    p.initialization_metadata=initialization|{'stage1_checkpoint':str(path)}
    return p.to(device=device,dtype=dtype)


def build_native_action(video,config):
    """Strictly load native ActionDiT/proprio without allocating a second VideoDiT."""
    from openwam.model.checkpoint_utils import expand_action_state_dict
    baseline=config['baseline_checkpoint']
    with safe_open(str(baseline),framework='pt',device='cpu') as file:
        state={k:file.get_tensor(k) for k in file.keys() if k.startswith(('action_backbone.','proprio_encoder.'))}
    if 'action_backbone.action_encoder.weight' not in state:raise ValueError('baseline has no native action weights')
    hidden,source_dim=state['action_backbone.action_encoder.weight'].shape
    # Native FastWAM defaults; frequency width is recoverable from the checkpoint.
    freq_key='action_backbone.time_embedding.mlp.0.weight'
    ac=_action_config(video,hidden,freq_dim=state[freq_key].shape[1] if freq_key in state else 256)
    ab=ActionDiT(**ac);proprio=nn.Linear(112,video.context_width)
    expected={'action_backbone.'+k:tuple(v.shape) for k,v in ab.state_dict().items()}
    expected.update({'proprio_encoder.'+k:tuple(v.shape) for k,v in proprio.state_dict().items()})
    if source_dim!=112:
        if source_dim!=80 or not config.get('initialization',{}).get('expand_action_dim_from_checkpoint',False):raise ValueError('explicit known 80->112 migration required')
        state=expand_action_state_dict(state,expected,source_dim=80,target_dim=112)
    ab.load_state_dict({k.removeprefix('action_backbone.'):v for k,v in state.items() if k.startswith('action_backbone.')},strict=True)
    proprio.load_state_dict({k.removeprefix('proprio_encoder.'):v for k,v in state.items() if k.startswith('proprio_encoder.')},strict=True)
    metadata=dict(kind='strict native baseline',checkpoint=str(baseline),source_dim=source_dim,target_dim=112,
        migration='known projections zero padded; inherited axes preserved' if source_dim==80 else 'none',action_config=ac)
    return ab,proprio,metadata


def save_policy_bundle(policy,path,*,dependencies,training=None):
    from openwam.train.goal_memory_checkpoint import _atomic
    metadata=dict(format='openwam.goal_memory.policy.v1',dependencies=dependencies,dit_config=policy.video.dit_config,
        goal_memory=asdict(policy.video.config),action_config=policy.action_config,
        policy_options={k:getattr(policy,k) for k in ('attention_mask_mode','future_video_mode','lambda_keyframe','freeze_video','allow_approximate_low_precision_cache','goal_dropout')},
        initialization=policy.initialization_metadata,training=training or {})
    weights={k:v.detach().cpu().contiguous() for k,v in policy.state_dict().items()}
    _atomic(path,lambda filename:save_file(weights,filename,metadata={'goal_memory_policy':json.dumps(metadata,sort_keys=True)}))
    return metadata


def load_policy_bundle(path,*,dependencies,device='cpu',dtype=torch.float32):
    from openwam.model.video_backbone.wan.models.dit import WanModel
    from openwam.model.goal.goal_memory import GoalMemoryExpert,GoalMemoryConfig
    with safe_open(str(path),framework='pt',device='cpu') as file:
        m=json.loads((file.metadata() or {}).get('goal_memory_policy','{}'))
        if m.get('format')!='openwam.goal_memory.policy.v1' or m.get('dependencies')!=dependencies:raise ValueError('policy bundle format/dependencies differ')
        video=GoalMemoryExpert(WanModel(**m['dit_config']),GoalMemoryConfig(**m['goal_memory']),dit_config=m['dit_config'])
        p=GoalMemoryPolicy(video,ActionDiT(**m['action_config']),nn.Linear(112,video.context_width),**m['policy_options'])
        p.load_state_dict({k:file.get_tensor(k) for k in file.keys()},strict=True)
    p.action_config=m['action_config'];p.initialization_metadata=m['initialization'];p.bundle_metadata=m
    return p.to(device=device,dtype=dtype)


def stage2_provider(config,base_provider,dataset,codec,*,device,dtype):
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    options=dict(config.get('action_data',{}))
    if not options.get('normalization_registry'):raise ValueError('explicit immutable normalization_registry required')
    for key in ('source_yaml','data_root','path_aliases'):
        if key in config['dataset']:options.setdefault(key,config['dataset'][key])
    options.setdefault('seed',config.get('seed',42))
    mode=config.get('policy',{}).get('future_video_mode','none')
    if options.setdefault('future_video_mode',mode)!=mode:raise ValueError('policy/data future modes differ')
    native=CuratedStage2Dataset(**options)
    dataset.restrict_stage2(native)
    registry_identity=native.normalization_identity()
    registry_hash=registry_identity['registry_sha256']
    def stamp():
        st=Path(native.registry_path).stat()
        return st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns
    registry_stamp=stamp()

    def provide(update):
        # No retry/redraw: both branches use the very same deterministic selection.
        if stamp()!=registry_stamp:
            raise ValueError('immutable normalization_registry changed')
        epoch,index=divmod(update,len(dataset))
        dataset.set_epoch(epoch)
        selected=dataset.selection_at(index)
        matches=[r for r in native.repos if r['repo']==selected['repo']['repo']]
        if len(matches)!=1:raise ValueError('source repo alignment failed')
        repo=matches[0];episode=selected['episode'];anchor=selected['anchor'];level=selected['level']
        m=native._episode(repo,episode);segment=selected['segment']
        option=selected['stage2_option']
        if not option['start']<=anchor<option['end_exclusive'] or option['instruction']!=segment:
            raise ValueError('native action interval/anchor alignment failed')
        sample=native._sample_selected(repo,episode,m,option,anchor,level,draw_id=update)
        batch=base_provider(update)
        if batch['_metadata']['instruction']!=sample['instruction'] or batch['_metadata']['granularity']!=level:
            raise ValueError('RGB/history/action instruction alignment failed')
        batch.update(actions=sample['action'][None].to(device=device,dtype=dtype),action_mask=sample['action_mask'][None].to(device),
                     proprio=sample['proprio'].to(device=device,dtype=dtype))
        batch['_metadata']['action_source']={k:sample['trace'][k] for k in ('source_repo','episode','anchor','selected_range')}
        batch['_metadata']['normalization_sha256']=registry_hash
        if mode=='joint':
            import numpy as np
            # Native future uses temporal VAE9->3, unlike independent T1 history.
            frames=torch.stack([torch.from_numpy(np.asarray(im).copy()).permute(2,0,1) for im in sample['video']],1)
            frames=frames.to(device=device,dtype=dtype)/127.5-1
            with torch.no_grad():latents=codec.vae.encode([frames],device=device)
            if tuple(latents.shape)!=(1,48,3,24,20):raise ValueError('native future VAE must encode nine RGB slots to three latent frames')
            batch['future_latents']=latents[:,:,1:].to(dtype=dtype)
            vm=sample['video_view_mask']
            # Conservative temporal receptive fields: padded/invalid frames cannot
            # supervise compressed future slots; causal prefix includes current.
            batch['future_view_valid']=torch.stack((vm[:5].all(0),vm[:9].all(0)))[None].to(device)
        return batch
    provide.native_dataset=native
    provide.normalization_identity=registry_identity
    return provide
