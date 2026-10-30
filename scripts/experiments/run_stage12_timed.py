"""Durable single-GPU supervisor: Stage1 freeze, native smoke, matched Stage2 arms."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import yaml

CODE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(CODE))
from openwam.train.z_action_report import atomic_json


def alive(pid):
    try:
        os.kill(int(pid),0)
        return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except (ProcessLookupError,FileNotFoundError):return False

def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--samples',type=Path,required=True)
    a=p.parse_args();plan=read(a.run/'plan.json');deadline=float(plan['deadline_unix'])
    if (a.run/'supervisor.json').exists():raise FileExistsError('supervisor already started')
    state=dict(status='waiting_stage1',pid=os.getpid(),started_unix=time.time(),deadline_unix=deadline,children=[])
    report=None;last_report=0
    def update(**kw):
        state.update(kw);state['updated_unix']=time.time();atomic_json(a.run/'supervisor.json',state)
        atomic_json(a.run/'report/supervisor.json',state)
    def refresh(force=False):
        nonlocal report,last_report
        if report is not None and report.poll() is None:return
        if not force and time.time()-last_report<120:return
        log=(a.run/'report.log').open('a')
        report=subprocess.Popen([sys.executable,str(CODE/'scripts/experiments/report_stage12_timed.py'),
            '--run',str(a.run),'--samples',str(a.samples)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT)
        log.close();last_report=time.time()
    def finish_report():
        if report is None:return
        remaining=deadline-time.time()
        if remaining<=0:
            report.terminate();report.wait(timeout=10)
            raise TimeoutError('report deadline reached')
        try:report.wait(timeout=min(600,remaining))
        except subprocess.TimeoutExpired:
            report.terminate()
            try:report.wait(timeout=10)
            except subprocess.TimeoutExpired:report.kill();report.wait()
            raise
        if report.returncode:raise RuntimeError('report failed; inspect report.log')
    def child(command,label,timeout):
        log=(a.run/(label+'.log')).open('x')
        process=subprocess.Popen(command,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();started=time.time();state['children'].append(dict(label=label,pid=process.pid,command=command,started_unix=started))
        update(active=label,active_pid=process.pid)
        while process.poll() is None:
            refresh()
            if time.time()-started>timeout or time.time()>deadline-120:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=60)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                state['children'][-1]['timed_out']=True
                break
            time.sleep(5)
        state['children'][-1].update(returncode=process.returncode,finished_unix=time.time())
        update();refresh(True)
        return process.returncode
    try:
        update();refresh(True)
        while read(a.run/'stage1/status.json').get('status')!='complete':
            if not alive(plan['stage1_pid']):raise RuntimeError('Stage1 exited before its complete marker; inspect stage1.log')
            if time.time()>deadline-4*3600:raise RuntimeError('Stage1 exceeded safe handoff time')
            refresh();time.sleep(5)
        checkpoint=a.run/'stage1/visual_history.safetensors'
        identity=dict(path=str(checkpoint),sha256=sha(checkpoint),bytes=checkpoint.stat().st_size,
            stage1_status=read(a.run/'stage1/status.json'),frozen_for_both_arms=True)
        atomic_json(a.run/'stage1_identity.json',identity)
        update(status='stage2_preflight',teacher_sha256=identity['sha256'])
        cfg=a.run/'comparison_config.yaml'
        common=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',
            str(CODE/'scripts/compare_z_action_stage2.py'),'--config',str(cfg),'--samples',str(a.samples),
            '--lr-scale','1.5','--warmup-updates','100','--no-legacy-report']
        base=common+['--single-gpu-master'];selected=None
        for backend in ['single_gpu_master','cpu_offload']:
            if backend=='cpu_offload':
                fallback=yaml.safe_load(cfg.read_text())
                fallback['training'].update(offload_optimizer_device='cpu',adamw_fused=False)
                fallback_path=a.run/'comparison_config_cpu.yaml';fallback_path.write_text(yaml.safe_dump(fallback,sort_keys=False))
                base=common.copy();base[base.index('--config')+1]=str(fallback_path)
                update(optimizer_fallback='GPU master preflight failed; validating BOTH arms with identical CPU offload')
            all_pass=True
            for mode in ['joint','none']:
                smoke_root=a.run/('smoke_'+backend)
                command=base+['--output',str(smoke_root),'--arm',mode,'--seconds-per-arm','1200',
                    '--smoke','--smoke-updates','2','--smoke-case-indices','0','18','--denoise-steps','4','--eval-every','100']
                if child(command,'smoke_'+backend+'_'+mode,1800):all_pass=False;break
                smoke=read(smoke_root/mode/'status.json')
                if smoke.get('status')!='complete' or not smoke.get('teacher_frozen'):all_pass=False;break
            if all_pass:selected=backend;break
        if selected is None:raise RuntimeError('Stage2 preflight failed with both optimizer paths')
        # Equal native worker budgets. Retain headroom for BOTH final evaluations,
        # initialization/save overhead and final CPU projection/video rendering.
        budget=min(4*3600,max(0,(deadline-time.time()-3600)/2))
        if budget<1800:raise RuntimeError('insufficient remaining time for meaningful matched arms')
        update(status='stage2_training',seconds_per_arm=budget,peak_lr_scale=1.5,warmup_updates=100,optimizer_backend=selected)
        for mode in ['joint','none']:
            command=base+['--output',str(a.run/'stage2'),'--arm',mode,'--seconds-per-arm',str(budget),
                '--denoise-steps','20','--eval-every','250']
            if child(command,'train_'+mode,budget+1200):raise RuntimeError('Stage2 '+mode+' training failed')
        update(status='rendering',active='report')
        finish_report()
        refresh(True)
        finish_report()
        if sha(checkpoint)!=identity['sha256']:raise RuntimeError('shared frozen checkpoint changed')
        update(status='complete',finished_unix=time.time(),active=None)
        refresh(True)
        finish_report()
    except BaseException as exc:
        update(status='failed',error=repr(exc),finished_unix=time.time())
        if report is not None and report.poll() is None:
            report.terminate()
            try:report.wait(timeout=10)
            except subprocess.TimeoutExpired:report.kill();report.wait()
        if time.time()<deadline-60:
            refresh(True)
            try:finish_report()
            except Exception:pass
        raise


if __name__=='__main__':main()
