"""Run unified Stage1 then two native Stage2 arms with an immutable frozen teacher.

Preparation supplies source.yaml, stage1_config.yaml, comparison_config.yaml and
native data/samples.pt+manifest.json. All paths and exact update counts are logged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

CODE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(CODE))
from openwam.train.goal_epoch import atomic_json, sha256, verify_completion


def verify_stage1(run, expected_updates):
    folder=Path(run)/'stage1'
    identity=json.loads((folder/'identity.json').read_text())
    proof=verify_completion(folder,identity)
    if proof.get('updates')!=expected_updates or proof.get('representation')!='visual_history_masked_sa':
        raise ValueError('Stage1 representation/update budget mismatch')
    path=folder/'visual_history.safetensors'
    if sha256(path)!=proof['checkpoint_sha256']:raise ValueError('Stage1 convenience checkpoint differs from completion proof')
    return dict(path=str(path.resolve()),sha256=proof['checkpoint_sha256'],updates=expected_updates,
                identity=identity,frozen_for_both_arms=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--stage1-updates',type=int,default=1000);p.add_argument('--stage2-updates',type=int,default=500)
    p.add_argument('--wait-stage1-pid',type=int,help='Wait for an already-running unified Stage1 process; never launch a second copy')
    p.add_argument('--safety-hours',type=float,default=24.)
    a=p.parse_args();run=a.run.resolve()
    if min(a.stage1_updates,a.stage2_updates)<1 or a.safety_hours<=0:raise ValueError('positive budgets required')
    state_path=run/'supervisor.json'
    if state_path.exists():raise FileExistsError('supervisor output already exists')
    state=dict(status='starting',pid=os.getpid(),started_unix=time.time(),commands=[],stage1_updates=a.stage1_updates,stage2_updates=a.stage2_updates)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    def update(**kwargs):
        state.update(kwargs);state['updated_unix']=time.time();atomic_json(state_path,state)
    def command(argv,label):
        row=dict(label=label,argv=argv,started_unix=time.time());state['commands'].append(row);update(status=label)
        with (run/(label+'_supervised.log')).open('x') as log:
            child=subprocess.Popen(argv,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
            row['pid']=child.pid;update()
            try:child.wait(timeout=a.safety_hours*3600)
            except BaseException:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        row.update(finished_unix=time.time(),returncode=child.returncode);update()
        if child.returncode:raise RuntimeError(f'{label} failed with status {child.returncode}')
    try:
        update()
        proof=run/'stage1/training_complete.json'
        if a.wait_stage1_pid:
            update(status='waiting_existing_stage1',stage1_pid=a.wait_stage1_pid)
            deadline=time.monotonic()+a.safety_hours*3600
            while not proof.exists():
                os.kill(a.wait_stage1_pid,0)
                if time.monotonic()>deadline:raise TimeoutError('existing Stage1 exceeded safety budget')
                time.sleep(5)
        elif not proof.exists():
            common=[sys.executable,'-B',str(CODE/'scripts/train_goal_epoch.py'),'--config',str(run/'stage1_config.yaml'),
                    '--index',str(run/'population.sqlite'),'--output-dir',str(run/'stage1')]
            if not (run/'population.sqlite').exists():command(common+['--prepare'],'stage1_prepare')
            command(common+['--train','--max-updates',str(a.stage1_updates),'--repeat-population'],'stage1_train')
        teacher=verify_stage1(run,a.stage1_updates);atomic_json(run/'stage1_identity.json',teacher)
        manifest=json.loads((run/'data/manifest.json').read_text())
        if sha256(run/'data/samples.pt')!=manifest['samples_sha256']:raise ValueError('prepared samples changed')
        cfg=yaml.safe_load((run/'comparison_config.yaml').read_text())
        cfg['goal']['checkpoint']=teacher['path'];cfg['goal']['precomputed_text_fingerprint']=json.loads((run/'stage1/dependencies.json').read_text())['text']['fingerprint']
        cfg['training']['max_steps']=a.stage2_updates
        config_path=run/'stage2_config.yaml';config_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        common=[sys.executable,'-B',str(CODE/'scripts/compare_z_action_stage2.py'),'--config',str(config_path),
            '--samples',str(run/'data/samples.pt'),'--output',str(run/'stage2'),'--single-gpu-master',
            '--max-updates',str(a.stage2_updates),'--seconds-per-arm',str(a.safety_hours*3600),
            '--eval-every',str(max(1,a.stage2_updates//2)),'--denoise-steps','20','--warmup-updates','25',
            '--no-legacy-report','--eval-case-indices',*[str(i) for i in manifest['evaluation_indices']]]
        for mode in ('joint','none'):
            if sha256(teacher['path'])!=teacher['sha256']:raise ValueError('frozen Stage1 changed')
            command(common+['--arm',mode],'stage2_'+mode)
            result=json.loads((run/'stage2'/mode/'status.json').read_text())
            if result.get('status')!='complete' or result.get('step')!=a.stage2_updates or not result.get('teacher_frozen'):
                raise ValueError(f'{mode} did not finish requested frozen-teacher updates')
        if sha256(teacher['path'])!=teacher['sha256']:raise ValueError('frozen Stage1 changed')
        update(status='complete',finished_unix=time.time(),teacher_sha256=teacher['sha256'])
        atomic_json(run/'completion.json',state)
    except BaseException as exc:
        update(status='failed',error=repr(exc));raise


if __name__=='__main__':main()
