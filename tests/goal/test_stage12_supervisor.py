"""Exercise actual supervisor sequencing without starting model processes."""
import json
from pathlib import Path
import sys
import time

import pytest
import yaml

from scripts.experiments import run_stage12_timed as runner


def fixture(tmp_path, monkeypatch, *, master_failure=False, final_render_failure=False):
    run=tmp_path/'run';(run/'stage1').mkdir(parents=True)
    (run/'stage1/status.json').write_text(json.dumps(dict(status='complete',step=100)))
    (run/'stage1/visual_history.safetensors').write_bytes(b'frozen-checkpoint')
    deadline=time.time()+4*3600
    (run/'plan.json').write_text(json.dumps(dict(stage1_pid=1234,deadline_unix=deadline)))
    (run/'comparison_config.yaml').write_text(yaml.safe_dump(dict(training={})))
    samples=tmp_path/'samples.pt';samples.write_bytes(b'samples');commands=[]

    class Process:
        def __init__(self, command, **kw):
            self.command=command;commands.append(command);self.pid=10000+len(commands)
            self.returncode=0
            if '--arm' in command:
                mode=command[command.index('--arm')+1]
                out=Path(command[command.index('--output')+1])/mode;out.mkdir(parents=True)
                if master_failure and '--single-gpu-master' in command:self.returncode=1
                (out/'status.json').write_text(json.dumps(dict(status='complete' if not self.returncode else 'failed',teacher_frozen=not self.returncode)))
            elif final_render_failure and runner.read(run/'supervisor.json').get('status')=='complete':
                self.returncode=1
        def poll(self):return self.returncode
        def wait(self,timeout=None):return self.returncode
        def terminate(self):self.returncode=-15
        def kill(self):self.returncode=-9

    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    monkeypatch.setattr(sys,'argv',['run_stage12_timed','--run',str(run),'--samples',str(samples)])
    return run,commands,deadline


def test_complete_pipeline_freezes_once_and_assigns_equal_bounded_budgets(tmp_path,monkeypatch):
    run,commands,deadline=fixture(tmp_path,monkeypatch)
    runner.main();state=runner.read(run/'supervisor.json')
    assert state['status']=='complete' and state['optimizer_backend']=='single_gpu_master'
    workers=[c for c in commands if '--arm' in c]
    assert [c[c.index('--arm')+1] for c in workers]==['joint','none','joint','none']
    assert all('--single-gpu-master' in c for c in workers)
    assert all('--smoke' in c for c in workers[:2])
    budgets=[float(c[c.index('--seconds-per-arm')+1]) for c in workers[2:]]
    assert budgets[0]==budgets[1] and 2*budgets[0]+3600<=deadline-state['started_unix']
    assert runner.read(run/'stage1_identity.json')['frozen_for_both_arms']
    assert runner.read(run/'report/supervisor.json')['status']=='complete'


def test_failed_master_smoke_revalidates_both_arms_on_same_cpu_backend(tmp_path,monkeypatch):
    run,commands,_=fixture(tmp_path,monkeypatch,master_failure=True)
    runner.main();state=runner.read(run/'supervisor.json')
    assert state['status']=='complete' and state['optimizer_backend']=='cpu_offload'
    workers=[c for c in commands if '--arm' in c]
    assert len(workers)==5 and '--single-gpu-master' in workers[0]
    assert all('--single-gpu-master' not in c for c in workers[1:])
    assert all('--smoke' in c for c in workers[1:3])
    assert [c[c.index('--arm')+1] for c in workers[1:]]==['joint','none','joint','none']
    assert all(c[c.index('--config')+1].endswith('comparison_config_cpu.yaml') for c in workers[1:])


def test_failed_final_render_cannot_leave_complete_status(tmp_path,monkeypatch):
    run,_,_=fixture(tmp_path,monkeypatch,final_render_failure=True)
    with pytest.raises(RuntimeError,match='report failed'):runner.main()
    assert runner.read(run/'supervisor.json')['status']=='failed'
    assert runner.read(run/'report/supervisor.json')['status']=='failed'
