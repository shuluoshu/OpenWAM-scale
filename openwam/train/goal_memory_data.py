"""Bind frozen native dependencies, source-reader batches and the RFC expert."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def make_dataset(config, codec=None):
    from openwam.dataloader.rfc_goal import RFCGoalDataset
    options = dict(config['dataset'])
    options.pop('observation_mode', None)
    options.setdefault('epoch_size', config['max_updates'])
    options.setdefault('require_cache', config.get('memory_cache', {}).get('require_cache', True))
    options.setdefault('seed', config.get('seed', 42))
    options.setdefault('recent_frames', config.get('history', {}).get('recent_frames', 2))
    return RFCGoalDataset(codec=codec, **options)


def preflight(config, output_dir):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('preflight requires an empty output directory')
    dataset = make_dataset(config)
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    for index in range(min(10, len(dataset))):
        sample = dataset.sample(index)
        samples.append({key: value.tolist() if torch.is_tensor(value) else value
                        for key, value in sample.items()
                        if key in ('instruction','granularity','goal_horizon_s','memory_ages',
                                   'memory_frame_ids','memory_valid','memory_view_valid',
                                   'observed_view_valid','target_view_valid','target_uv',
                                   'target_image_valid','target_point_valid','target_in_frame','record')})
    result = dict(scope=dataset.scope_summary(), samples=samples, model_loaded=False)
    (output/'preflight.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    return result


class LatentBatchProvider:
    """Encode current/terminal images together as independent T=1 frames."""
    def __init__(self,dataset,codec,text_cache,*,device,dtype):
        self.dataset,self.codec,self.text_cache = dataset,codec,text_cache
        self.device,self.dtype = device,dtype

    def __call__(self,update):
        from openwam.dataloader.rfc_goal import collate_goal_memory
        sample = self.dataset.sample(update)
        batch = collate_goal_memory([sample])
        observed,target = batch['observed'],batch['target_image']
        images = torch.cat((observed,target),0)
        # Separate T=1 videos in the batch: no temporal VAE leakage between them.
        latents = self.codec.encode_images(images)
        context,lens = self.text_cache.get(sample['instruction'])
        context = context.to(device=self.device,dtype=self.dtype)
        lens = torch.as_tensor(lens,device=self.device).reshape(-1)
        text_mask = torch.arange(context.shape[1],device=self.device)[None] < lens[:,None]
        fields = ('memory_latents','memory_valid','memory_view_valid','memory_ages','memory_frame_ids',
                  'observed_view_valid','target_uv','target_image_valid','target_view_valid',
                  'target_point_valid','target_in_frame','goal_horizon_s')
        result = {key: batch[key].to(device=self.device,
                  dtype=self.dtype if key=='memory_latents' else batch[key].dtype) for key in fields}
        result.update(current_latents=latents[:1].to(device=self.device,dtype=self.dtype),
                      target_latents=latents[1:].to(device=self.device,dtype=self.dtype),
                      text=context,text_mask=text_mask,
                      _metadata=dict(granularity=sample['granularity'],instruction=sample['instruction'],
                                     source=sample.get('record',{})))
        return result


def build_training(config, *, device, dtype):
    from openwam.train.goal_trainer import bootstrap_components, _seed
    from openwam.train.goal_curated import TextContextCache
    from openwam.model.goal.dependencies import dependency_identity
    from openwam.model.goal.goal_memory import GoalMemoryExpert,GoalMemoryConfig
    from openwam.model.goal.memory_codec import GoalMemoryCodec
    from openwam.model.video_backbone.wan.encode import encode_text
    if config['stage']==2:
        if not config.get('stage1_checkpoint') or not Path(config['stage1_checkpoint']).is_file():
            raise ValueError('explicit existing Stage1 goal_memory checkpoint required')
        registry = config.get('action_data',{}).get('normalization_registry')
        if not registry or not Path(registry).is_file():
            raise ValueError('explicit existing native normalization_registry required')
    if not config.get('dataset',{}).get('source_yaml') or not Path(config['dataset']['source_yaml']).is_file():
        raise ValueError('existing source_yaml required')
    _seed(config.get('seed',42))
    original,text_encoder,tokenizer = bootstrap_components(config,device='cpu',dtype=dtype,validate_codec=False)
    vae,dit = original.codec.vae,original.dit
    del original
    # Hash once on CPU at bootstrap, never in the sample/denoising loop.
    dependencies = dependency_identity(vae,text_dim=dit.text_embedding[0].in_features,
                                       text_encoder=text_encoder,tokenizer=tokenizer)
    cache_config = config['memory_cache']
    vae.to(device=device,dtype=dtype).eval().requires_grad_(False)
    codec = GoalMemoryCodec(vae,cache_config['directory'],
                            vae_identity=dependencies['vae']['fingerprint'],device=device,dtype=dtype)
    dependencies['memory_contract'] = dict(fps=1,temporal_vae=False,views='head+left_wrist+right_wrist',
                                           labels='matching-granularity terminal',
                                           preprocess=getattr(codec,'preprocess_identity',None))
    dataset = make_dataset(config,codec)
    text_device = config.get('text_device','cpu')
    text_encoder.to(device=text_device,dtype=dtype).eval().requires_grad_(False)
    def encode(prompt):
        return encode_text([prompt],tokenizer=tokenizer,text_encoder=text_encoder,device=text_device)
    text_cache = TextContextCache(encode,max_entries=config.get('text_cache_size',32))
    dit_config = config['components']['dit']['extra_kwargs']
    if config['stage']==1:
        model = GoalMemoryExpert(dit,GoalMemoryConfig(**config.get('memory',{})),dit_config=dit_config).to(device=device,dtype=dtype)
    else:
        # Stage2 attaches the same Stage1-initialized VideoDiT; it never creates
        # an independent frozen teacher or reuses an incompatible old C* bundle.
        from openwam.train.goal_memory_stage2 import build_stage2
        model = build_stage2(config,dit=dit,dependencies=dependencies,device=device,dtype=dtype)
    provider = LatentBatchProvider(dataset,codec,text_cache,device=device,dtype=dtype)
    if config['stage']==2:
        from openwam.train.goal_memory_stage2 import stage2_provider
        provider = stage2_provider(config,provider,dataset,codec,device=device,dtype=dtype)
    if config['stage']==2:
        normalization = provider.normalization_identity
        dependencies['action_normalization'] = {k:v for k,v in normalization.items()
                                                 if k not in ('registry_path','assets')}
    population = dict(source_yaml_sha256=hashlib.sha256(Path(config['dataset']['source_yaml']).read_bytes()).hexdigest(),
                      scope=dataset.scope_summary())
    return model,provider,dit_config,dependencies,population
