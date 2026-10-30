"""Bound host loading memory per node without serializing the whole job."""
import json
import gc
import time

import torch.distributed as dist


def bootstrap_plan(world_size, local_world_size, workers_per_node):
    if any(type(n) is not int or n < 1 for n in (world_size,local_world_size,workers_per_node)):
        raise ValueError('bootstrap sizes must be positive integers')
    if world_size % local_world_size or workers_per_node > local_world_size:
        raise ValueError('bootstrap requires uniform local world size and a valid per-node limit')
    return [tuple(rank for rank in range(world_size)
                  if (rank % local_world_size)//workers_per_node == wave)
            for wave in range((local_world_size+workers_per_node-1)//workers_per_node)]


def run_bootstrap_in_waves(bootstrap, *, local_rank, local_world_size, workers_per_node=2, event=None):
    """Run a callback once per rank. Catchable failures stop every live rank.

    Homogeneous torchrun ranks are contiguous within a node. All ranks validate
    that topology and the same schedule before any heavy allocation. Each wave
    may run on every node; error gathering is also its completion barrier.
    """
    distributed=dist.is_initialized()
    rank=dist.get_rank() if distributed else 0
    world=dist.get_world_size() if distributed else 1
    emit=event or (lambda row: print(json.dumps(row),flush=True))
    error=None;plan=None
    try:
        plan=bootstrap_plan(world,local_world_size,workers_per_node)
        if type(local_rank) is not int or local_rank!=rank%local_world_size:
            raise ValueError('LOCAL_RANK does not match contiguous torchrun node ranks')
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    specification=dict(local_world_size=local_world_size,workers_per_node=workers_per_node,error=error)
    specifications=[specification]
    if distributed:
        specifications=[None]*world
        dist.all_gather_object(specifications,specification)
    failures=[f'rank {r}: {s["error"]}' for r,s in enumerate(specifications) if s['error']]
    if failures or any(s!=specifications[0] for s in specifications):
        raise ValueError('bootstrap configuration mismatch: '+str(failures or specifications))
    emit(dict(event='bootstrap_plan',rank=rank,local_rank=local_rank,world_size=world,
              local_world_size=local_world_size,workers_per_node=workers_per_node,waves=len(plan)))
    result=None
    for wave, active in enumerate(plan):
        error=None
        if rank in active:
            started=time.monotonic()
            emit(dict(event='bootstrap_started',rank=rank,wave=wave))
            try:
                result=bootstrap()
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
            emit(dict(event='bootstrap_finished' if error is None else 'bootstrap_failed',
                      rank=rank,wave=wave,elapsed_s=time.monotonic()-started,error=error))
        errors=[error]
        if distributed:
            errors=[None]*world
            dist.all_gather_object(errors,error)
        failures=[f'rank {r}: {e}' for r,e in enumerate(errors) if e]
        if failures:raise RuntimeError('bootstrap failed: '+'; '.join(failures))
    return result


def bootstrap_query_components(config, *, device, dtype):
    """Load bounded CPU components, then validate the VAE on its final device."""
    import torch
    from openwam.train.goal_trainer import bootstrap_components, _seed
    from openwam.model.goal.query_expert import QueryGoalExpert
    from openwam.model.goal.dependencies import dependency_identity
    _seed(config.get('seed',42))
    original,text_encoder,tokenizer=bootstrap_components(config,device='cpu',dtype=dtype,validate_codec=False)
    visual=config['model'].get('goal_representation')=='visual_history_masked_sa'
    if visual:
        from openwam.model.goal.visual_history import VisualHistoryGoalExpert
        expert=VisualHistoryGoalExpert(original.dit,original.codec,dit_config=original.dit_config,
            **config['model'].get('visual_history',{}),observation_mode=config['observation_mode'],
            shift=config.get('shift',5.),lambda_image=config.get('lambda_image',.5),lambda_tcp=config.get('lambda_tcp',1.),
            use_gradient_checkpointing=config.get('gradient_checkpointing',True)).to(device=device,dtype=dtype)
    else:
        expert=QueryGoalExpert(original.dit,original.codec,dit_config=original.dit_config,
        query_config=config['model'].get('query_goal'),history_config=config['model'].get('query_history'),
        conditioning_mode=config['model'].get('query_conditioning','legacy_text'),
        observation_mode=config['observation_mode'],shift=config.get('shift',5.),
        lambda_image=config.get('lambda_image',.5),lambda_tcp=config.get('lambda_tcp',1.),
        use_gradient_checkpointing=config.get('gradient_checkpointing',True)).to(device=device,dtype=dtype)
    del original
    expert.codec.vae.to(device=device,dtype=dtype).eval().requires_grad_(False)
    with torch.no_grad():
        expert.codec.encode(torch.zeros(1,3,1,256,320,device=device,dtype=dtype))
    text_device=device if config.get('text_device','cuda')=='cuda' else torch.device(config['text_device'])
    text_encoder.to(device=text_device,dtype=dtype).eval().requires_grad_(False)
    dependencies=dependency_identity(expert.codec.vae,text_dim=expert.context_width if visual else expert.text_dim,text_encoder=text_encoder,tokenizer=tokenizer)
    gc.collect()
    return expert,text_encoder,tokenizer,dependencies
