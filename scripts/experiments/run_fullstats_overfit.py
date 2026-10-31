"""Finish unified Stage1, require full relative stats, train timed arms and render."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import yaml

CODE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CODE))
from openwam.train.goal_epoch import atomic_json,sha256
from scripts.experiments.run_three_episode_stage12 import verify_stage1
SKILL=Path('/root/.codex-chatgpt/skills/openwam-visualization')


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def alive(pid):
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def checked_readiness(path):
    ready=read(path)
    if ready.get('status')!='complete':raise ValueError('statistics not complete')
    registry=Path(ready['stats_registry']);digest=sha256(registry)
    if digest!=ready['registry_sha256'] or registry.name!=f'registry-{digest}.json':
        raise ValueError('statistics hash mismatch')
    payload=read(registry)
    if not payload.get('preparation',{}).get('complete') or (ready.get('horizon'),ready.get('dimensions'))!=(32,112):
        raise ValueError('statistics population/representation incomplete')
    return ready


def stage2_command(run,indices,seconds,mode):
    return [sys.executable,'-B',str(CODE/'scripts/compare_z_action_stage2.py'),
        '--config',str(run/'stage2_config.yaml'),'--samples',str(run/'data/samples.pt'),
        '--output',str(run/'stage2'),'--single-gpu-master','--seconds-per-arm',str(seconds),
        '--eval-every','1000','--denoise-steps','20','--warmup-updates','25','--no-legacy-report',
        '--eval-case-indices',*[str(i) for i in indices],'--arm',mode]


def verify_arm(result,seconds):
    if result.get('status')!='complete' or result.get('step',0)<1 or not result.get('teacher_frozen') or result.get('budget_seconds')!=seconds:
        raise ValueError('Stage2 did not complete its frozen-teacher timed budget')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args(argv);run=args.run.resolve();plan=read(run/'plan.json')
    lock=(run/'.supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (run/'supervisor.json').exists():raise FileExistsError('supervisor exists; inspect before restarting')
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    state=dict(status='waiting_stage1_and_statistics',pid=os.getpid(),started_unix=time.time(),commands=[],
        stage1_updates=plan['stage1_updates'],seconds_per_arm=plan['seconds_per_arm'],stage1_pid=plan['stage1_pid'])
    active=None;report=None;last_report=0.
    report_command=[sys.executable,'-m','scripts.experiments.report_fullstats_overfit','--run',str(run)]
    prior_run=read(run/'selection.json').get('prior_run')
    if prior_run:report_command+=['--prior-run',prior_run]
    def update(**values):
        state.update(values);state['updated_unix']=time.time();atomic_json(run/'supervisor.json',state)
    def refresh(force=False):
        nonlocal report,last_report
        if report is not None and report.poll() is None:return
        if not force and time.monotonic()-last_report<60:return
        if not (CODE/'scripts/experiments/report_fullstats_overfit.py').exists():return
        with (run/'report.log').open('a') as log:
            report=subprocess.Popen(report_command,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
        last_report=time.monotonic()
    def command(cmd,label,timeout):
        nonlocal active
        row=dict(label=label,argv=cmd,started_unix=time.time());state['commands'].append(row);update(status=label)
        with (run/(label+'.log')).open('x') as log:
            active=subprocess.Popen(cmd,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        row['pid']=active.pid;update(active_pid=active.pid)
        deadline=time.monotonic()+timeout
        while active.poll() is None:
            refresh()
            if time.monotonic()>deadline:
                os.killpg(active.pid,signal.SIGTERM)
                try:active.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
                raise TimeoutError(f'{label} exceeded safety timeout')
            time.sleep(5)
        row.update(returncode=active.returncode,finished_unix=time.time());update(active_pid=None)
        if active.returncode:raise RuntimeError(f'{label} failed; see {label}.log')
        active=None;refresh(True)
    def skill(name,extra,label,timeout):
        command([sys.executable,str(SKILL/'scripts/tools.py'),'--repo',str(CODE),'--python',sys.executable,
            'run',name,'--',*extra],label,timeout)
    def stop(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        update();refresh(True)
        readiness=Path(plan['statistics_readiness']);prepared=False;teacher=None
        waiting_deadline=time.monotonic()+8*3600
        while teacher is None or not prepared:
            if time.monotonic()>waiting_deadline:raise TimeoutError('Stage1/statistics handoff safety deadline exceeded')
            if not prepared and readiness.exists():
                ready=checked_readiness(readiness);atomic_json(run/'statistics_identity.json',ready)
                command([sys.executable,str(CODE/'scripts/experiments/prepare_fullstats_overfit.py'),
                    '--run',str(run),'--stats-readiness',str(readiness)],'prepare_fullstats_samples',3600)
                manifest=read(run/'data/manifest.json')
                if sha256(run/'data/samples.pt')!=manifest['samples_sha256']:raise ValueError('prepared samples changed')
                prepared=True
            if teacher is None and not alive(plan['stage1_pid']):
                teacher=verify_stage1(run,plan['stage1_updates']);atomic_json(run/'stage1_identity.json',teacher)
            if teacher is None and time.time()-plan['stage1_started_unix']>3*3600:
                raise TimeoutError('Stage1 exceeded its three-hour safety limit')
            if not readiness.exists():
                migration=read(readiness.parent/'migration_progress.json')
                if migration.get('status')=='failed':raise RuntimeError('statistics finalization failed: '+str(migration.get('error')))
            if teacher is None or not prepared:
                update(status='waiting_stage1' if prepared else 'waiting_statistics' if teacher else 'waiting_stage1_and_statistics')
                refresh();time.sleep(10)
        cfg=yaml.safe_load((run/'comparison_config.yaml').read_text());cfg['goal']['checkpoint']=teacher['path']
        cfg['goal']['precomputed_text_fingerprint']=read(run/'stage1/dependencies.json')['text']['fingerprint']
        (run/'stage2_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        manifest=read(run/'data/manifest.json')
        for mode in ('joint','none'):
            if sha256(teacher['path'])!=teacher['sha256']:raise ValueError('frozen Stage1 changed')
            checked_readiness(readiness)
            command(stage2_command(run,manifest['evaluation_indices'],plan['seconds_per_arm'],mode),
                'stage2_'+mode,plan['seconds_per_arm']+1800)
            verify_arm(read(run/'stage2'/mode/'status.json'),plan['seconds_per_arm'])
        if sha256(teacher['path'])!=teacher['sha256']:raise ValueError('frozen Stage1 changed')
        atomic_json(run/'completion.json',dict(status='complete',finished_unix=time.time(),teacher_sha256=teacher['sha256'],
            statistics_sha256=ready['registry_sha256'],stage1_updates=teacher['updates'],
            stage2={mode:read(run/'stage2'/mode/'status.json') for mode in ('joint','none')},commands=state['commands']))
        full=run/'fullspan';base=['--run',str(run),'--manifest',str(full/'data/manifest.json'),'--output',str(full),'--steps','20','--anchor-major']
        skill('fullspan-inference',base+['--limit','1'],'first_anchor_inference',1800)
        progress=read(full/'inference_status.json')
        if any(progress.get('completed',{}).get(stage,0)<1 for stage in ('stage1','joint','none')):
            raise ValueError('first anchor did not produce all three model outputs')
        skill('fullspan-inference',base,'fullspan_inference',7200)
        skill('fullspan-video',['--root',str(full)],'fullspan_video',1800)
        skill('fullspan-validate',['--root',str(full)],'fullspan_validation',1800)
        command([sys.executable,'-m','scripts.experiments.projection_coverage','--root',str(full)],'projection_coverage',1800)
        command([sys.executable,str(SKILL/'scripts/validate_report.py'),str(full/'report/index.html'),
            '--root',str(run),'--require-video','--require-offline','--decode-videos','6',
            '--output-json',str(run/'report/delivery_check.json')],'delivery_validation',1800)
        update(status='complete',finished_unix=time.time(),teacher_sha256=teacher['sha256'])
        if report is not None and report.poll() is None:report.wait(timeout=120)
        subprocess.run(report_command,cwd=CODE,env=env,check=True,timeout=180)
    except BaseException as exc:
        if active is not None and active.poll() is None:
            os.killpg(active.pid,signal.SIGTERM)
            try:active.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
        update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time());refresh(True);raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
