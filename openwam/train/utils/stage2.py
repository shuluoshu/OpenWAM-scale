"""CPU-safe Stage 2 launch, optimizer ownership and committed draw contracts."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import Sampler


def is_stage2(cfg):
    return bool(OmegaConf.select(cfg, 'goal.enabled', default=False))


def required_draws(*, max_updates, accumulation, batch_size, world_size):
    values=[max_updates,accumulation,batch_size,world_size]
    if any(int(v)<1 for v in values):raise ValueError('Stage2 draw dimensions must be positive')
    return int(max_updates)*int(accumulation)*int(batch_size)*int(world_size)


class GlobalDrawSampler(Sampler):
    """Identical global stream on every rank; Accelerate alone shards batches.

    Cursor advancement by DataLoader prefetch is intentionally never recorded.
    Checkpoints supply the committed global offset after consumed microbatches.
    """
    version='openwam.stage2.global_draw.v1'

    def __init__(self, draws_per_epoch, *, seed, selection_identity):
        self.draws_per_epoch=int(draws_per_epoch)
        if self.draws_per_epoch<1:raise ValueError('draws_per_epoch must be positive')
        self.seed=int(seed);self.selection_identity=selection_identity
        self.epoch=0;self.offset=0

    def __iter__(self):
        base=self.epoch*self.draws_per_epoch
        return iter(range(base+self.offset,base+self.draws_per_epoch))

    def __len__(self):return self.draws_per_epoch-self.offset

    def set_epoch(self,epoch):
        epoch=int(epoch)
        if epoch!=self.epoch:self.epoch=epoch;self.offset=0

    def state_dict(self,*,committed_offset):
        offset=int(committed_offset)
        if not 0<=offset<=self.draws_per_epoch:raise ValueError('invalid committed draw offset')
        return dict(version=self.version,seed=self.seed,selection_identity=self.selection_identity,
                    draws_per_epoch=self.draws_per_epoch,epoch=self.epoch,offset=offset)

    def load_state_dict(self,state):
        for key in ('version','seed','selection_identity','draws_per_epoch'):
            if state.get(key)!=getattr(self,key):raise ValueError(f'Stage2 sampler {key} mismatch')
        epoch=int(state['epoch']);offset=int(state['offset'])
        if epoch<0 or not 0<=offset<=self.draws_per_epoch:raise ValueError('invalid sampler epoch/offset')
        self.epoch=epoch;self.offset=offset


@dataclass
class Stage2Progress:
    """Microsteps, successful updates and globally committed source draws."""
    batch_size:int
    world_size:int
    accumulation:int
    micro_step:int=0
    opt_step:int=0
    consumed_draws:int=0
    epoch:int=0
    epoch_offset:int=0

    def consume(self,*,sync_gradients,optimizer_step_skipped):
        self.micro_step+=1
        count=self.batch_size*self.world_size
        self.consumed_draws+=count;self.epoch_offset+=count
        updated=bool(sync_gradients and not optimizer_step_skipped)
        if updated:self.opt_step+=1
        return updated

    def begin_epoch(self,epoch):
        if int(epoch)!=self.epoch:self.epoch=int(epoch);self.epoch_offset=0

    def state_dict(self):return asdict(self)

    def load_state_dict(self,state):
        for key in ('batch_size','world_size','accumulation'):
            if state.get(key)!=getattr(self,key):raise ValueError(f'Stage2 resume {key} mismatch')
        for key in ('micro_step','opt_step','consumed_draws','epoch','epoch_offset'):
            if type(state.get(key)) is not int or state[key]<0:raise ValueError(f'invalid Stage2 {key}')
        if state['micro_step']%self.accumulation:raise ValueError('Stage2 resume requires an update boundary')
        if state['consumed_draws']!=state['micro_step']*self.batch_size*self.world_size:
            raise ValueError('Stage2 consumed draw count mismatch')
        if state['opt_step']>state['micro_step']//self.accumulation:raise ValueError('Stage2 optimizer count mismatch')
        if state['epoch_offset']%(self.batch_size*self.world_size*self.accumulation):
            raise ValueError('Stage2 sampler offset requires an update boundary')
        for key,value in state.items():
            if key in asdict(self):setattr(self,key,value)


def build_stage2_parameter_groups(architecture,training):
    from openwam.train.goal_contract import build_g1_parameter_groups, G1_LRS
    configured=OmegaConf.to_container(training.parameter_lrs,resolve=True)
    if configured!=G1_LRS:raise ValueError(f'Stage2 learning rate contract mismatch: {configured} != {G1_LRS}')
    groups=build_g1_parameter_groups(architecture)
    if {g['name'] for g in groups}!=set(G1_LRS):raise ValueError('Stage2 missing required optimizer ownership group')
    for group in groups:
        if group['lr']!=float(configured[group['name']]):raise ValueError('Stage2 actual optimizer learning rate mismatch')
    return groups


def validate_stage2_config(cfg):
    from openwam.model.goal.runtime import goal_observation_mode
    goal_observation_mode(OmegaConf.to_container(cfg.goal,resolve=True))
    required={'goal.enabled':True,'goal.conditioning':'image_memory',
        'goal.mode':'predicted','goal.num_steps':20,
        'model.architecture.action_dim':112,'model.architecture.state_dim':112,
        'model.architecture.use_proprioception':True,'model.architecture.attention_mask_mode':'mutual',
        'model.architecture.video_attention_mask_mode':'first_frame_causal',
        'dataloader.type':'curated_stage2','training.step_unit':'optimizer_updates',
        'training.expand_action_dim_from_checkpoint':False}
    for key,expected in required.items():
        if OmegaConf.select(cfg,key)!=expected:raise ValueError(f'Stage2 contract requires {key}={expected!r}')
    for key in ('goal.cache_size','goal.cache_max_bytes','dataloader.max_cached_episodes','dataloader.max_cache_bytes'):
        value=OmegaConf.select(cfg,key)
        if value is None or int(value)<0:raise ValueError(f'explicit bounded {key} required')
    for path in ('video_backbone.vae','video_backbone.text_encoder'):
        if path not in cfg.model.freeze:raise ValueError(f'Stage2 must freeze {path}')
    if cfg.project.seed is None:raise ValueError('Stage2 requires project.seed for resumable draws')
    if cfg.training.num_epochs is not None or int(cfg.training.max_steps)<1:
        raise ValueError('Stage2 uses positive max_steps optimizer updates and num_epochs=null')
    if int(cfg.evaluation.every_updates)!=500:raise ValueError('Stage2 evaluation every_updates must be 500')


def validate_stage2_checkpoint(cfg,path):
    """Inspect projection shapes without loading multi-gigabyte tensors or paths."""
    from safetensors import safe_open
    axes={'action_backbone.action_encoder.weight':1,'action_backbone.action_decoder.weight':0,
          'action_backbone.action_decoder.bias':0,'proprio_encoder.weight':1}
    with safe_open(str(path),framework='pt',device='cpu') as handle:
        for key,axis in axes.items():
            if key not in handle.keys():raise ValueError(f'Stage2 baseline projection missing: {key}')
            shape=handle.get_slice(key).get_shape()
            if len(shape)!=(1 if key.endswith('.bias') else 2) or shape[axis]!=112:
                raise ValueError(f'Stage2 baseline projection must retain width112: {key}={shape}')


def validate_stage2_launch(cfg,*,metadata_only=False):
    """Validate real assets before model/Accelerator construction; CPU only."""
    from safetensors import safe_open
    from openwam.train.utils.ckpt_model_loader import _resolve_ckpt_source,find_latest_weights
    validate_stage2_config(cfg)
    resume = OmegaConf.select(cfg, 'training.resume_ckpt_path')
    finetune = OmegaConf.select(cfg, 'training.finetune_ckpt_path')
    if resume and finetune:
        raise ValueError('Stage2 resume_ckpt_path and finetune_ckpt_path are mutually exclusive')
    baseline = resume or finetune
    if not baseline:
        raise ValueError('STAGE2_BASELINE_CHECKPOINT is required for warm-start')
    directory,weights = _resolve_ckpt_source(str(baseline))
    baseline_cfg = OmegaConf.load(str(Path(directory)/'config.yaml'))
    goal_directory = directory if is_stage2(baseline_cfg) else None
    if goal_directory:
        from openwam.model.goal.runtime import goal_observation_mode
        if goal_observation_mode(OmegaConf.to_container(cfg.goal,resolve=True)) != goal_observation_mode(OmegaConf.to_container(baseline_cfg.goal,resolve=True)):
            raise ValueError('Stage2 configured goal policy/mode disagrees with saved checkpoint')
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.goal = baseline_cfg.goal
    if resume:
        from omegaconf import open_dict
        from openwam.train.utils.stage2_assets import resolve_resume_normalization
        from openwam.train.utils.checkpointing import find_latest_accel_state, validate_stage2_full_state
        saved = OmegaConf.load(str(Path(resume)/'config.yaml'))
        if not is_stage2(saved):
            raise ValueError('Stage2 resume requires a Stage2 checkpoint; use warm-start')
        state_dir = find_latest_accel_state(resume)
        if state_dir is None:
            raise ValueError('Stage2 resume requires full states; use warm-start')
        with open(Path(state_dir)/'trainer_state.json') as stream:
            full_metadata = json.load(stream)
        validate_stage2_full_state(state_dir, full_metadata)
        with open_dict(cfg):
            cfg.goal = saved.goal
        resolve_resume_normalization(cfg, resume)
    validate_stage2_config(cfg)
    from openwam.model.goal.runtime import resolve_goal_asset
    goal=resolve_goal_asset(OmegaConf.to_container(cfg.goal, resolve=True), goal_directory)
    if not goal.is_file():raise FileNotFoundError('STAGE1_GOAL_CHECKPOINT must name a patch bundle')
    with safe_open(str(goal),framework='pt',device='cpu') as handle:
        metadata=json.loads((handle.metadata() or {}).get('patch_goal_bundle','{}'))
    from openwam.model.goal.runtime import goal_observation_mode
    from openwam.model.goal.observation import observation_metadata
    mode=goal_observation_mode(OmegaConf.to_container(cfg.goal,resolve=True))
    if metadata.get('format')!=cfg.goal.format:raise ValueError('Stage1 patch bundle metadata format mismatch')
    if metadata.get('observation_mode', 'head')!=mode:raise ValueError('Stage1 patch bundle observation mode mismatch')
    if mode=='multiview' and metadata.get('observation')!=json.loads(json.dumps(observation_metadata(mode))):
        raise ValueError('Stage1 patch bundle multiview observation contract mismatch')
    if metadata.get('codec',{}).get('latent_shape')!=[48,1,16,20]:raise ValueError('Stage1 patch bundle codec mismatch')
    if not metadata.get('dependencies'):raise ValueError('Stage1 patch bundle dependencies missing')
    for key in ('action_dim','state_dim','attention_mask_mode','video_attention_mask_mode'):
        if OmegaConf.select(baseline_cfg,'model.architecture.'+key)!=OmegaConf.select(cfg,'model.architecture.'+key):
            raise ValueError(f'Stage2 baseline structural contract mismatch: {key}')
    validate_stage2_checkpoint(cfg, (str(Path(directory)/full_metadata["unified_weights"]) if resume else weights or find_latest_weights(directory)))
    if not cfg.dataloader.source_yaml:raise ValueError('STAGE2_SOURCE_YAML is required')
    source=Path(cfg.dataloader.source_yaml)
    if not source.is_file():raise FileNotFoundError(f'STAGE2_SOURCE_YAML missing: {source}')
    if not cfg.project.output_dir:raise ValueError('STAGE2_OUTPUT_DIR is required')
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import audit_selection
    dl=OmegaConf.to_container(cfg.dataloader,resolve=True)
    if metadata_only and not dl.get('normalization_registry'):dl['metadata_only']=True
    dataset=CuratedStage2Dataset.from_config(dl)
    audit=audit_selection(dataset)
    if audit['errors']:raise ValueError(f'Stage2 source metadata errors: {audit["errors"]}')
    return dict(dataset.scope_summary(),global_batch=int(cfg.training.batch_size)*int(os.environ.get('WORLD_SIZE','1'))*int(cfg.training.gradient_accumulation_steps),
                state=('resume requested' if cfg.training.resume_ckpt_path else 'warm-start')+'; full resume/relocation validation required',metadata_only=metadata_only)


def stage2_resume_contract(cfg):
    """Fields that change the next update cannot silently change on full resume."""
    paths = ['training.' + key for key in (
        'max_steps', 'lr_scheduler', 'warmup_ratio', 'lr_min_ratio', 'mixed_precision',
        'parameter_lrs', 'adam_betas', 'weight_decay', 'max_grad_norm', 'zero_stage',
        'lambda_video', 'lambda_action', 'lambda_fk_position', 'lambda_fk_rotation',
        'max_timestep_boundary', 'min_timestep_boundary', 'debug')]
    paths += ['dataloader.observation_augmentation']
    result = {}
    for path in paths:
        value = OmegaConf.select(cfg, path)
        result[path] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return result
