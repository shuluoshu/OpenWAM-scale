"""One finite Stage1 epoch followed by the existing frozen-query Stage2 trainer."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from openwam.train.goal_epoch import atomic_json,verify_completion,sha256
ROOT=Path(__file__).resolve().parents[2]


def single_node_torchrun(python,world,*,visible_devices=None):
    """One worker per local GPU; the same launch contract is used for both stages."""
    if isinstance(world,bool) or not isinstance(world,int) or world<1:
        raise ValueError('world_size must be a positive integer')
    if visible_devices is not None and visible_devices<world:
        raise ValueError(f'world_size={world} requires {world} visible CUDA devices on this node; found {visible_devices}')
    return [python,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}']


def epoch_identity(config,population,world):
    from openwam.train.epoch_order import sampling_contract
    code=hashlib.sha256()
    for folder in ('openwam/model/goal','openwam/dataloader','openwam/train'):
        for path in sorted((ROOT/folder).rglob('*.py')):
            code.update(str(path.relative_to(ROOT)).encode());code.update(path.read_bytes())
    return dict(code_sha256=code.hexdigest(),**{k:population[k] for k in ('source_yaml_sha256','index_sha256','total_samples')},
        world_size=world,seed=config.get('seed',42),sampling=sampling_contract(config,population['total_samples']),
        config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())


def execute_stages(output,identity,stage1_command,stage2_command,*,runner=None,resume=False,stage2_preflight=None,pipeline_identity=None):
    """Subprocess exit and durable verified coverage are both necessary to hand off."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if runner is None:runner=lambda command:subprocess.run(command,cwd=ROOT,check=True)
    with (output/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=output/'pipeline_state.json'
        if state.exists() and not resume:raise ValueError('pipeline exists; use --resume')
        previous=json.loads(state.read_text()) if state.exists() else {}
        if previous.get('identity',identity)!=identity:raise ValueError('pipeline identity mismatch')
        if previous.get('pipeline_identity',pipeline_identity)!=pipeline_identity:raise ValueError('Stage2 pipeline identity mismatch')
        phase='stage1'
        try:
            if not (resume and (output/'stage1/epoch_complete.json').exists()):
                atomic_json(state,dict(phase=phase,identity=identity,pipeline_identity=pipeline_identity));runner(stage1_command)
            proof=verify_completion(output/'stage1',identity)
            if previous.get('phase')=='complete':return proof
            phase='stage2'
            atomic_json(state,dict(phase=phase,identity=identity,pipeline_identity=pipeline_identity,stage1=proof))
            substitute=lambda command:[arg.replace('{goal_checkpoint}',proof['checkpoint']) for arg in command]
            if stage2_preflight:runner(substitute(stage2_preflight))
            runner(substitute(stage2_command))
            atomic_json(state,dict(phase='complete',identity=identity,pipeline_identity=pipeline_identity,stage1=proof))
            return proof
        except BaseException as exc:
            atomic_json(state,dict(phase=phase+'_failed',identity=identity,pipeline_identity=pipeline_identity,error=f'{type(exc).__name__}: {exc}'))
            raise


def load_pipeline(path):
    from omegaconf import OmegaConf
    config=OmegaConf.to_container(OmegaConf.load(path),resolve=True)
    for key in ('stage1_config','source_yaml','output_dir','stage2_baseline','stats_registry'):
        if config.get(key):
            p=Path(config[key]).expanduser();config[key]=str(p if p.is_absolute() else ROOT/p)
    return config


def stage2_path_overrides(stage1,stage2):
    """Pin Stage2 to the same source paths checked against the Stage1 index."""
    aliases=stage1.get('dataset',{}).get('path_aliases')
    if not aliases:return []
    if not isinstance(aliases,str):raise ValueError('pipeline path_aliases must be a JSON file path')
    stage2['dataloader']['path_aliases']=aliases
    return ['++dataloader.path_aliases='+json.dumps(aliases)]


def validate_stage2_overrides(extra):
    protected=('goal.', 'model.goal_representation', 'model.query_', 'dataloader.source_yaml', 'dataloader.history',
               'dataloader.normalization_registry', 'dataloader.path_aliases', 'training.resume_ckpt_path', 'training.finetune_ckpt_path', 'project.output_dir')
    if any(str(v).lstrip('+~').startswith(protected) for v in extra):
        raise ValueError('stage2_overrides cannot replace the shared source/query/history/checkpoint contract')


def main(argv=None, *, _runner=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/goal/stage12_query_history_posttrain.yaml'))
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true',help='CPU finite population + bounded normalization preparation')
    modes.add_argument('--train',action='store_true',help='launch Stage1 then automatically frozen Stage2')
    modes.add_argument('--check',action='store_true',help='validate prepared inputs; no model or GPU training')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stage2-resume',help='explicit Stage2 full-state checkpoint if Stage2 was interrupted')
    args=parser.parse_args(argv)
    config=load_pipeline(args.config);output=Path(config['output_dir']);output.mkdir(parents=True,exist_ok=True)
    world=config.get('world_size',8)
    from openwam.train.stage12_distributed import topology,torchrun_command,coordinate_pipeline
    nodes,_=topology(config)
    if args.train:
        import torch
        validate=lambda:torchrun_command(sys.executable,config,visible_devices=torch.cuda.device_count())
        if nodes>1 and _runner is None:
            return coordinate_pipeline(lambda runner: main(argv,_runner=runner),config,startup_validation=validate)
        validate()
    source=config['source_yaml'];index=output/'population.sqlite';stage1_output=output/'stage1'
    python=sys.executable
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    def run(command):
        if _runner is not None:return _runner(command,cwd=ROOT,env=env)
        return subprocess.run(command,cwd=ROOT,env=env,check=True)
    common=[str(ROOT/'scripts/train_goal_epoch.py'),'--config',config['stage1_config'],'--source-yaml',source,'--index',str(index),'--output-dir',str(stage1_output)]
    from openwam.train.goal_curated import load_config
    stage1=load_config(config['stage1_config'],source_yaml=source)
    from hydra import initialize_config_dir,compose
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):
        stage2=OmegaConf.to_container(compose(config_name=config['stage2_config']),resolve=True)
    stage2['dataloader']['source_yaml']=source
    stage2['dataloader']['metadata_only']=True
    stage2['dataloader']['normalization_registry']=None
    path_overrides=stage2_path_overrides(stage1,stage2)
    if stage1['model']['query_goal']!=stage2['model']['query_goal'] or stage1['model']['query_history']!=stage2['model']['query_history']:
        raise ValueError('Stage1/Stage2 query or history configuration mismatch')
    if args.prepare:
        run([python,*common,'--prepare'])
        from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
        from openwam.dataloader.native_stage2_stats import audit_selection,prepare_native_stats
        dataset=CuratedStage2Dataset.from_config(stage2['dataloader'])
        audit=audit_selection(dataset);atomic_json(output/'stage2_metadata_audit.json',audit)
        if audit['errors']:raise ValueError('Stage2 metadata audit failed; see stage2_metadata_audit.json')
        registry=prepare_native_stats(dataset,output/'statistics',audit=audit,
            max_episodes_per_profile=config.get('stats_episodes_per_profile',4),max_windows_per_profile=config.get('stats_windows_per_profile',128))
        atomic_json(output/'prepared.json',dict(source_yaml=source,stats_registry=registry,index=str(index)))
        print(json.dumps(dict(index=str(index),stats_registry=registry),indent=2));return 0
    prepared=json.loads((output/'prepared.json').read_text())
    registry=config.get('stats_registry') or prepared['stats_registry']
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    from openwam.dataloader.goal_epoch import validate_index,epoch_sample_fps
    ds=dict(stage1['dataset']);ds['cache_dir']=str(output/'check_cache')
    dataset=CuratedGoalDataset(**ds,seed=stage1.get('seed',42))
    population=validate_index(dataset,index,sample_fps=epoch_sample_fps(stage1));identity=epoch_identity(stage1,population,world)
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    stage2['dataloader']['normalization_registry']=registry
    # Construction validates immutable registry bytes and exact selection/config contract.
    CuratedStage2Dataset.from_config(stage2['dataloader'])
    for p in (stage1['baseline_checkpoint'],config.get('stage2_baseline')):
        if not p or not Path(p).exists():raise FileNotFoundError('baseline checkpoint missing: '+str(p))
    if args.check:
        print(json.dumps(dict(identity=identity,sample_fps=population['sample_fps'],
            stage1_updates=(population['total_samples']+world-1)//world,
            torchrun=torchrun_command(python,config,require_env=False),stats_registry=registry,stage2_config=config['stage2_config']),indent=2));return 0
    if not args.resume and (output/'pipeline_state.json').exists():raise ValueError('existing pipeline requires --resume')
    old=json.loads((output/'pipeline_state.json').read_text()) if (output/'pipeline_state.json').exists() else {}
    if old.get('phase') in ('stage2','stage2_failed') and not args.stage2_resume:
        raise ValueError('interrupted Stage2 requires --stage2-resume full-state checkpoint; refusing implicit restart')
    torchrun=torchrun_command(python,config)
    command1=[*torchrun,*common,'--train']+(['--resume'] if args.resume else [])
    overrides=[f'dataloader.source_yaml={source}',f'dataloader.normalization_registry={registry}',
        'goal.checkpoint={goal_checkpoint}',f'project.output_dir={output / "stage2"}',
        f'training.max_steps={config.get("stage2_max_steps",100000)}']
    if args.stage2_resume:overrides += ['training.finetune_ckpt_path=null',f'training.resume_ckpt_path={Path(args.stage2_resume).resolve()}']
    else:overrides += [f'training.finetune_ckpt_path={config["stage2_baseline"]}']
    extra=config.get('stage2_overrides',[])
    validate_stage2_overrides(extra)
    overrides+=path_overrides+extra
    common2=[str(ROOT/'scripts/train.py'),'--config-name',config['stage2_config'],*overrides]
    baseline=Path(config['stage2_baseline'])
    if baseline.is_file():
        assets=[baseline,baseline.parent/'config.yaml']
        assets+=sorted(p for p in (baseline.parent/'tokenizer').rglob('*') if p.is_file())
    else:assets=sorted(p for p in baseline.rglob('*') if p.is_file())
    contract=dict(stage2_composed=stage2,stage2_max_steps=config.get('stage2_max_steps',100000),
        stage2_overrides=extra,baseline={str(p):sha256(p) for p in assets},
        registry_sha256=sha256(registry),output=str(output/'stage2'))
    if path_overrides:contract['path_aliases_sha256']=sha256(stage1['dataset']['path_aliases'])
    pipeline_identity=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    atomic_json(output/f'stage2_contract_{pipeline_identity}.json',contract)
    execute_stages(output,identity,command1,[*torchrun,*common2,'goal.preflight=false'],runner=run,resume=args.resume,pipeline_identity=pipeline_identity,
        stage2_preflight=[python,*common2,'goal.preflight=true'])
    return 0
