"""Evaluate exact saved Stage2 weight steps or render their immutable history.

python scripts/eval_stage2.py --run-dir RUN --steps 500 1000
python scripts/eval_stage2.py --run-dir RUN --history-only

For explicitly bounded single-GPU diagnostics use --diagnostic-subset --max-cases 2
and a separate --output-dir. Default selection covers every family task/subtask.
No training or optimizer construction occurs. CPU is supported for tiny fixtures;
real production generation requires the normal model device/memory budget.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,help='Self-contained Stage2 checkpoint directory')
    parser.add_argument('--config',help='Same training config; default is RUN/config.yaml')
    parser.add_argument('--output-dir',help='Default RUN; use separate directory for diagnostic subsets')
    parser.add_argument('--steps',nargs='+',type=int,help='Exact numbered checkpoint steps; default latest saved step')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--num-steps',type=int,help='Joint video/action denoise budget; frozen Stage1 budget remains saved goal.num_steps')
    parser.add_argument('--max-cases',type=int)
    parser.add_argument('--diagnostic-subset',action='store_true')
    parser.add_argument('--history-only',action='store_true',help='Render existing complete reports without loading model or source data')
    args=parser.parse_args(argv)
    from openwam.train.stage2_eval import (build_fixed_cases,evaluate_stage2,render_stage2_history,completed_evaluation)
    run=Path(args.run_dir).resolve();output=Path(args.output_dir).resolve() if args.output_dir else run
    if args.history_only:
        print(render_stage2_history(output));return
    import torch
    from omegaconf import OmegaConf,open_dict
    from scripts.data.preflight_stage2 import load_config
    from openwam.train.utils.stage2_assets import resolve_resume_normalization
    from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
    config=OmegaConf.create(load_config(args.config)) if args.config else OmegaConf.load(run/'config.yaml')
    if not OmegaConf.select(config,'goal.enabled',default=False):raise ValueError('Stage2 checkpoint/config required')
    resolve_resume_normalization(config,run)
    with open_dict(config):
        if args.max_cases is not None:OmegaConf.update(config,'evaluation.max_cases',args.max_cases,force_add=True)
        if args.diagnostic_subset:OmegaConf.update(config,'evaluation.diagnostic_subset',True,force_add=True)
        if args.num_steps is not None:OmegaConf.update(config,'evaluation.num_steps',args.num_steps,force_add=True)
    if args.steps is None:
        available=[int(m.group(1)) for path in run.glob('checkpoint_step_*.safetensors')
                   if (m:=re.fullmatch(r'checkpoint_step_(\d+)\.safetensors',path.name))]
        if not available:raise FileNotFoundError('no numbered Stage2 checkpoints')
        steps=[max(available)]
    else:steps=sorted(set(args.steps))
    cases=build_fixed_cases(config,output);architecture=None
    seed=int(config.project.seed);num_steps=int(OmegaConf.select(config,'evaluation.num_steps',default=20))
    from openwam.model.goal.runtime import goal_observation_mode
    observation_mode=goal_observation_mode(OmegaConf.to_container(config.goal,resolve=True))
    for step in steps:
        if completed_evaluation(output,cases,step=step,seed=seed,num_steps=num_steps,observation_mode=observation_mode):continue
        weights=run/f'checkpoint_step_{step}.safetensors'
        if not weights.is_file():raise FileNotFoundError(f'exact saved checkpoint required: {weights}')
        if architecture is None:
            _,architecture,_=build_architecture_from_ckpt_dir(str(weights),weights_required=True)
            precision=str(OmegaConf.select(config,'training.mixed_precision',default='bf16'))
            dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'no':torch.float32}[precision]
            architecture.set_dtype_device(dtype,torch.device(args.device));architecture.move_frozen_to_device(torch.device(args.device))
        else:architecture.load_checkpoint(str(weights))
        evaluate_stage2(architecture,cases,output,step=step,seed=seed,num_steps=num_steps)
    print(render_stage2_history(output))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO);main()
